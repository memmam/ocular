use crate::{CompressedFrame, IngestConfig, StreamGuard};

/// Fixed-capacity cyclic cache of the most recent frames.
///
/// The slot vector is allocated once at construction and never grows. Frame
/// payloads themselves are owned `Vec<f32>` handed over by the producer, so
/// pushing a frame takes ownership of an existing allocation and dropping the
/// evicted frame frees one; the hot path is bounded in memory, not
/// allocation-free.
pub struct IngestRingBuffer {
    cache: Vec<Option<CompressedFrame>>,
    head: usize,
    target_dim: usize,
}

impl IngestRingBuffer {
    /// Pre-sizes the slot vector at boot time so the push path never reallocates.
    #[must_use]
    pub fn new(target_dim: usize) -> Self {
        Self {
            cache: vec![None; IngestConfig::MAX_FIFO_FRAMES],
            head: 0,
            target_dim,
        }
    }

    /// Drops the stale frame at the targeted cyclical cell index and binds the incoming payload.
    ///
    /// # Errors
    ///
    /// Returns a [`StreamGuard`] violation when `frame.token_data` is not
    /// exactly `IngestConfig::COMPRESSED_DIM * target_dim` elements long. The
    /// frame is discarded and the cache is left untouched.
    pub fn push_snapshot(&mut self, frame: CompressedFrame) -> Result<(), &'static str> {
        let expected_size = IngestConfig::COMPRESSED_DIM * self.target_dim;
        StreamGuard::verify_tensor_bounds(frame.token_data.len(), expected_size)?;

        let idx = self.head % IngestConfig::MAX_FIFO_FRAMES;
        self.cache[idx] = Some(frame);
        self.head = self.head.wrapping_add(1);
        Ok(())
    }

    /// Serializes the retained frames oldest-first into one flat block for context insertion.
    ///
    /// Allocates a fresh buffer sized to the retained window on every call.
    #[must_use]
    pub fn export_linearized_payload(&self) -> Vec<f32> {
        let mut out_buffer = Vec::with_capacity(
            IngestConfig::MAX_FIFO_FRAMES * IngestConfig::COMPRESSED_DIM * self.target_dim,
        );

        let start = self.head.saturating_sub(IngestConfig::MAX_FIFO_FRAMES);

        for i in start..self.head {
            let idx = i % IngestConfig::MAX_FIFO_FRAMES;
            if let Some(ref frame) = self.cache[idx] {
                out_buffer.extend_from_slice(&frame.token_data);
            }
        }
        out_buffer
    }

    /// Number of frames currently retained in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.head.min(IngestConfig::MAX_FIFO_FRAMES)
    }

    /// Returns `true` while no frame has been accepted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head == 0
    }

    /// Target text-model width this cache validates incoming frames against.
    #[must_use]
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }
}
