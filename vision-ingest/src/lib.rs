#![deny(unsafe_code)]
#![warn(clippy::pedantic, clippy::cargo)]

//! Bounded, snapshot-driven ingest path for compressed visual tokens.
//!
//! The crate covers the tail of the pipeline: a fixed-capacity cyclic cache
//! ([`ring_buffer::IngestRingBuffer`]) fed over an mpsc channel by a
//! supervisor task ([`orchestrator::AsyncIngestOrchestrator`]), plus a length
//! check ([`StreamGuard`]) applied to every frame on the way in.
//!
//! Encoder hidden-state extraction and resampler compression are upstream of
//! this crate and are not implemented here; [`IngestConfig::PATCH_COUNT`] and
//! [`IngestConfig::ENCODER_DIM`] are published for those producers to size
//! against.

pub mod orchestrator;
pub mod ring_buffer;

use std::time::Instant;

/// System dimension primitives tailored to native open-weight backbones (e.g., SigLIP-SO400M).
pub struct IngestConfig;
impl IngestConfig {
    /// Patch sequence length of one encoded snapshot (27x27 grid at 384px, patch 14).
    pub const PATCH_COUNT: usize = 729;
    /// Hidden width of the visual encoder.
    pub const ENCODER_DIM: usize = 1152;
    /// Token count emitted per frame by the resampler. This is a sequence
    /// length, not a vector width: a frame carries `COMPRESSED_DIM * target_dim`
    /// elements.
    pub const COMPRESSED_DIM: usize = 64;
    /// Depth of the timeline cache (10s at 3 FPS).
    pub const MAX_FIFO_FRAMES: usize = 30;
}

/// `CompressedFrame` encapsulates the downsampled visual tokens ready for LLM context injection.
#[derive(Clone, Debug)]
pub struct CompressedFrame {
    pub timestamp: Instant,
    /// Row-major token block of `IngestConfig::COMPRESSED_DIM * dimensions` elements.
    pub token_data: Vec<f32>,
    /// Target dimension width of the text model this frame was projected for.
    pub dimensions: usize,
}

/// Length gate applied to every frame entering the cache.
pub struct StreamGuard;
impl StreamGuard {
    /// Rejects a token block whose element count does not match the configured
    /// geometry, so a malformed frame cannot be linearized into the context array.
    ///
    /// This is a length check only. It does not inspect frame contents and does
    /// not detect a block that is correctly sized but projected for a different
    /// target width.
    ///
    /// # Errors
    ///
    /// Returns an error when `incoming_len != expected_len`.
    #[inline]
    pub fn verify_tensor_bounds(
        incoming_len: usize,
        expected_len: usize,
    ) -> Result<(), &'static str> {
        if incoming_len != expected_len {
            return Err(
                "StreamGuard Violation: Structural dimensionality mismatch. Visual array rejected.",
            );
        }
        Ok(())
    }
}

#[cfg(test)]
// Every float compared below is a marker value copied verbatim through the
// pipeline, never computed, so exact comparison is the intended check.
#[allow(clippy::float_cmp)]
mod tests {
    use super::orchestrator::AsyncIngestOrchestrator;
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_pipeline_concurrency_and_bounds() {
        let target_dim = 2048; // Edge node validation dimension (e.g., Qwen3-VL/Gemma widths)
        let (orchestrator, tx, buffer) = AsyncIngestOrchestrator::new(target_dim, 10);

        // Disengage the orchestration tracker into the background.
        tokio::spawn(orchestrator.start_orchestration_loop());

        // Well-formed payloads pass the guard.
        let valid_frame = CompressedFrame {
            timestamp: Instant::now(),
            token_data: vec![1.337f32; IngestConfig::COMPRESSED_DIM * target_dim],
            dimensions: target_dim,
        };
        assert!(tx.send(valid_frame).await.is_ok());

        // Wait for the loop to drain the channel rather than assuming a fixed
        // delay is enough.
        let linear_data = await_payload(&buffer, IngestConfig::COMPRESSED_DIM * target_dim).await;
        assert_eq!(linear_data.len(), IngestConfig::COMPRESSED_DIM * target_dim);
        assert_eq!(linear_data[0], 1.337f32);
    }

    #[tokio::test]
    async fn test_malformed_frame_is_rejected_without_stalling_loop() {
        let target_dim = 128;
        let (orchestrator, tx, buffer) = AsyncIngestOrchestrator::new(target_dim, 10);
        tokio::spawn(orchestrator.start_orchestration_loop());

        // Short block: the guard must drop it and the loop must keep running.
        tx.send(CompressedFrame {
            timestamp: Instant::now(),
            token_data: vec![0.0f32; 16],
            dimensions: target_dim,
        })
        .await
        .expect("channel open");

        tx.send(CompressedFrame {
            timestamp: Instant::now(),
            token_data: vec![2.0f32; IngestConfig::COMPRESSED_DIM * target_dim],
            dimensions: target_dim,
        })
        .await
        .expect("channel open");

        let linear_data = await_payload(&buffer, IngestConfig::COMPRESSED_DIM * target_dim).await;
        assert_eq!(linear_data.len(), IngestConfig::COMPRESSED_DIM * target_dim);
        assert_eq!(linear_data[0], 2.0f32);
    }

    #[test]
    fn test_cache_evicts_oldest_beyond_capacity() {
        let target_dim = 4;
        let frame_len = IngestConfig::COMPRESSED_DIM * target_dim;
        let mut cache = ring_buffer::IngestRingBuffer::new(target_dim);

        // Overfill by 5 frames; the payload must stay capped at MAX_FIFO_FRAMES.
        let overshoot = 5;
        for i in 0..IngestConfig::MAX_FIFO_FRAMES + overshoot {
            #[allow(clippy::cast_precision_loss)]
            let marker = i as f32;
            cache
                .push_snapshot(CompressedFrame {
                    timestamp: Instant::now(),
                    token_data: vec![marker; frame_len],
                    dimensions: target_dim,
                })
                .expect("well-formed frame");
        }

        let payload = cache.export_linearized_payload();
        assert_eq!(payload.len(), IngestConfig::MAX_FIFO_FRAMES * frame_len);
        // Oldest surviving frame is the one at index `overshoot`.
        #[allow(clippy::cast_precision_loss)]
        let oldest = overshoot as f32;
        assert_eq!(payload[0], oldest);
        #[allow(clippy::cast_precision_loss)]
        let newest = (IngestConfig::MAX_FIFO_FRAMES + overshoot - 1) as f32;
        assert_eq!(payload[payload.len() - 1], newest);
    }

    #[test]
    fn test_guard_bounds() {
        assert!(StreamGuard::verify_tensor_bounds(64, 64).is_ok());
        assert!(StreamGuard::verify_tensor_bounds(63, 64).is_err());
    }

    /// Polls the shared cache until it holds `expected_len` elements, so the
    /// tests do not depend on a fixed sleep being long enough.
    async fn await_payload(
        buffer: &std::sync::Arc<tokio::sync::RwLock<ring_buffer::IngestRingBuffer>>,
        expected_len: usize,
    ) -> Vec<f32> {
        for _ in 0..500 {
            let payload = buffer.read().await.export_linearized_payload();
            if payload.len() >= expected_len {
                return payload;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("ingest loop did not commit the frame within the timeout");
    }
}
