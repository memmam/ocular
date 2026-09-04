//! Integration coverage for the ingest path as a consumer sees it.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use vision_ingest::orchestrator::IngestHandles;
use vision_ingest::pool::BufferPool;
use vision_ingest::ring_buffer::IngestRingBuffer;
use vision_ingest::{CaptureTimestamp, IngestConfig, IngestError, IngestStats, StreamGuard};

const TEST_DIM: usize = 128;
const FRAME_LEN: usize = IngestConfig::TOKENS_PER_FRAME * TEST_DIM;

/// Leases a buffer, fills it with `marker`, stamps the header and publishes it.
fn publish(
    pool: &Arc<BufferPool<f32>>,
    sender: &vision_ingest::orchestrator::FrameSender<f32>,
    marker: f32,
    nanos: u64,
    declared_dim: usize,
) -> bool {
    let mut lease = pool.acquire();
    lease.fill(marker);
    lease.set_header(CaptureTimestamp::from_nanos(nanos), declared_dim);
    sender.try_publish(lease)
}

async fn await_frames(stats: &Arc<IngestStats>, accepted: u64, rejected: u64) {
    for _ in 0..500 {
        if stats.accepted() >= accepted && stats.rejected_total() >= rejected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!(
        "timed out: accepted {} (want {accepted}), rejected {} (want {rejected})",
        stats.accepted(),
        stats.rejected_total()
    );
}

#[tokio::test]
async fn valid_frame_reaches_the_cache() {
    let handles = IngestHandles::<f32>::new(TEST_DIM, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    assert!(publish(&pool, &handles.sender, 1.337, 42, TEST_DIM));
    await_frames(&handles.stats, 1, 0).await;

    let cache = handles.buffer.read().await;
    let payload = cache.export_linearized_payload();
    assert_eq!(payload.len(), FRAME_LEN);
    assert!(payload.iter().all(|v| (*v - 1.337).abs() < f32::EPSILON));

    let meta: Vec<_> = cache.frame_metadata().collect();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].capture, CaptureTimestamp::from_nanos(42));
    assert_eq!(meta[0].sequence, 0);
}

#[tokio::test]
async fn width_mismatch_is_counted_and_does_not_stall_the_loop() {
    let handles = IngestHandles::<f32>::new(TEST_DIM, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    // Correct length, wrong declared width: the sensor device is projecting
    // for a model this node is not running.
    assert!(publish(&pool, &handles.sender, 9.0, 1, TEST_DIM * 2));
    // A well-formed frame still gets through afterwards.
    assert!(publish(&pool, &handles.sender, 2.0, 2, TEST_DIM));
    await_frames(&handles.stats, 1, 1).await;

    assert_eq!(handles.stats.rejected_dimension(), 1);
    assert_eq!(handles.stats.rejected_length(), 0);
    assert_eq!(handles.stats.accepted(), 1);

    let payload = handles.buffer.read().await.export_linearized_payload();
    assert_eq!(payload.len(), FRAME_LEN);
    assert!((payload[0] - 2.0).abs() < f32::EPSILON);
}

#[tokio::test]
async fn non_finite_frame_is_rejected() {
    let handles = IngestHandles::<f32>::new(TEST_DIM, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    let mut lease = pool.acquire();
    lease.fill(0.5);
    lease[FRAME_LEN / 2] = f32::NAN;
    lease.set_header(CaptureTimestamp::from_nanos(7), TEST_DIM);
    assert!(handles.sender.try_publish(lease));

    await_frames(&handles.stats, 0, 1).await;
    assert_eq!(handles.stats.rejected_non_finite(), 1);
    assert_eq!(handles.stats.accepted(), 0);
    assert!(handles.buffer.read().await.is_empty());
}

#[tokio::test]
async fn queue_full_drops_rather_than_blocking() {
    // No orchestration loop spawned, so nothing drains the queue.
    let handles = IngestHandles::<f32>::new(TEST_DIM, 1);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 4);

    assert!(publish(&pool, &handles.sender, 1.0, 1, TEST_DIM));
    assert!(!publish(&pool, &handles.sender, 1.0, 2, TEST_DIM));
    assert!(!publish(&pool, &handles.sender, 1.0, 3, TEST_DIM));

    assert_eq!(handles.stats.dropped_queue_full(), 2);
    // Dropped leases returned their buffers: one is still held by the queue.
    assert_eq!(pool.available(), 3);
}

#[test]
fn window_evicts_oldest_and_stays_capped() {
    let mut cache = IngestRingBuffer::<f32>::new(TEST_DIM);
    let overshoot = 5u64;
    let total = IngestConfig::MAX_FIFO_FRAMES as u64 + overshoot;

    for i in 0..total {
        let marker = i as f32;
        let block = vec![marker; FRAME_LEN];
        let evicted = cache
            .push_snapshot(&block, CaptureTimestamp::from_nanos(i), TEST_DIM)
            .expect("well-formed frame");
        assert_eq!(evicted, i >= IngestConfig::MAX_FIFO_FRAMES as u64);
    }

    let payload = cache.export_linearized_payload();
    assert_eq!(payload.len(), cache.window_capacity());
    assert_eq!(cache.len(), IngestConfig::MAX_FIFO_FRAMES);
    assert!((payload[0] - overshoot as f32).abs() < f32::EPSILON);
    assert!((payload[payload.len() - 1] - (total - 1) as f32).abs() < f32::EPSILON);

    let sequences: Vec<u64> = cache.frame_metadata().map(|m| m.sequence).collect();
    assert_eq!(sequences.first().copied(), Some(overshoot));
    assert_eq!(sequences.last().copied(), Some(total - 1));
}

#[test]
fn export_into_reuses_the_caller_buffer() {
    let mut cache = IngestRingBuffer::<f32>::new(TEST_DIM);
    for i in 0..IngestConfig::MAX_FIFO_FRAMES {
        let block = vec![i as f32; FRAME_LEN];
        cache
            .push_snapshot(&block, CaptureTimestamp::from_nanos(i as u64), TEST_DIM)
            .expect("well-formed frame");
    }

    let mut out = Vec::with_capacity(cache.window_capacity());
    let capacity_before = out.capacity();
    for _ in 0..16 {
        cache.export_into(&mut out);
        assert_eq!(out.len(), cache.window_capacity());
    }
    // A correctly sized buffer is never reallocated by repeated exports.
    assert_eq!(out.capacity(), capacity_before);
}

#[test]
fn pool_recycles_buffers() {
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 3);
    assert_eq!(pool.available(), 3);
    {
        let _a = pool.acquire();
        let _b = pool.acquire();
        assert_eq!(pool.available(), 1);
    }
    assert_eq!(pool.available(), 3);
}

#[test]
fn guard_reports_cause() {
    let block = vec![0.0f32; FRAME_LEN];
    assert_eq!(
        StreamGuard::verify(&block, FRAME_LEN, TEST_DIM, TEST_DIM),
        Ok(())
    );
    assert_eq!(
        StreamGuard::verify(&block, FRAME_LEN + 1, TEST_DIM, TEST_DIM),
        Err(IngestError::LengthMismatch {
            expected: FRAME_LEN + 1,
            actual: FRAME_LEN
        })
    );
    assert_eq!(
        StreamGuard::verify(&block, FRAME_LEN, 64, TEST_DIM),
        Err(IngestError::DimensionMismatch {
            expected: TEST_DIM,
            actual: 64
        })
    );
    let mut bad = block.clone();
    bad[3] = f32::INFINITY;
    assert_eq!(
        StreamGuard::verify(&bad, FRAME_LEN, TEST_DIM, TEST_DIM),
        Err(IngestError::NonFiniteElement { index: 3 })
    );
}

#[tokio::test]
async fn cache_is_readable_while_ingest_runs() {
    let handles = IngestHandles::<f32>::new(TEST_DIM, 16);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 16);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    let reader: Arc<RwLock<IngestRingBuffer<f32>>> = Arc::clone(&handles.buffer);
    let reader_task = tokio::spawn(async move {
        let mut out = Vec::with_capacity(FRAME_LEN * IngestConfig::MAX_FIFO_FRAMES);
        for _ in 0..20 {
            reader.read().await.export_into(&mut out);
            assert_eq!(out.len() % FRAME_LEN, 0);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    });

    for i in 0..12u64 {
        while !publish(&pool, &handles.sender, i as f32, i, TEST_DIM) {
            tokio::task::yield_now().await;
        }
    }
    await_frames(&handles.stats, 12, 0).await;
    reader_task.await.expect("reader task");
    assert_eq!(handles.stats.accepted(), 12);
    assert_eq!(handles.stats.rejected_total(), 0);
}
