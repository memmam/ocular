//! Integration coverage for the ingest path as a consumer sees it.
//!
//! Frame geometry and capacity here are arbitrary small values chosen for
//! the tests. Nothing in the crate depends on them.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use vision_ingest::orchestrator::{FrameSender, IngestHandles};
use vision_ingest::pool::BufferPool;
use vision_ingest::ring_buffer::{ExportPolicy, IngestRingBuffer};
use vision_ingest::{CaptureTimestamp, FrameShape, IngestError, IngestStats, StreamGuard};

const SHAPE: FrameShape = FrameShape::new(8, 16);
const FRAME_LEN: usize = 128;
const CAP: usize = 12;

/// Same element count as `SHAPE`, different geometry.
const SWAPPED: FrameShape = FrameShape::new(16, 8);

fn ts(nanos: u64) -> CaptureTimestamp {
    CaptureTimestamp::from_nanos(nanos)
}

fn block(marker: f32) -> Vec<f32> {
    vec![marker; FRAME_LEN]
}

/// Leases a buffer, fills it with `marker`, stamps the header and publishes it.
fn publish(
    pool: &Arc<BufferPool<f32>>,
    sender: &FrameSender<f32>,
    marker: f32,
    nanos: u64,
    declared: FrameShape,
) -> bool {
    let mut lease = pool.acquire();
    lease.fill(marker);
    lease.set_header(ts(nanos), declared);
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

fn cache() -> IngestRingBuffer<f32> {
    IngestRingBuffer::<f32>::new(SHAPE, CAP)
}

/// Pushes `count` frames whose marker equals their index.
fn fill(cache: &mut IngestRingBuffer<f32>, count: u64) {
    for i in 0..count {
        cache
            .push_snapshot(&block(i as f32), ts(i), SHAPE)
            .expect("well-formed frame");
    }
}

fn marker_at(out: &[f32], frame: usize) -> f32 {
    out[frame * FRAME_LEN]
}

// --- ingest path end to end ---

#[tokio::test]
async fn valid_frame_reaches_the_cache() {
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    assert!(publish(&pool, &handles.sender, 1.337, 42, SHAPE));
    await_frames(&handles.stats, 1, 0).await;

    let cache = handles.buffer.read().await;
    let payload = {
        let mut out = Vec::new();
        cache.export_into(&mut out);
        out
    };
    assert_eq!(payload.len(), FRAME_LEN);
    assert!(payload.iter().all(|v| (*v - 1.337).abs() < f32::EPSILON));

    let meta: Vec<_> = cache.frame_metadata().collect();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].capture, ts(42));
    assert_eq!(meta[0].sequence, 0);
}

#[tokio::test]
async fn shape_mismatch_is_counted_and_does_not_stall_the_loop() {
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    // Right length, wrong declared geometry: the sensor side is emitting a
    // block this node was not configured for.
    assert!(publish(&pool, &handles.sender, 9.0, 1, SWAPPED));
    assert!(publish(&pool, &handles.sender, 2.0, 2, SHAPE));
    await_frames(&handles.stats, 1, 1).await;

    assert_eq!(handles.stats.rejected_shape(), 1);
    assert_eq!(handles.stats.rejected_length(), 0);
    assert_eq!(handles.stats.accepted(), 1);

    let mut out = Vec::new();
    handles.buffer.read().await.export_into(&mut out);
    assert_eq!(out.len(), FRAME_LEN);
    assert!((out[0] - 2.0).abs() < f32::EPSILON);
}

#[tokio::test]
async fn non_finite_frame_is_rejected() {
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    let mut lease = pool.acquire();
    lease.fill(0.5);
    lease[FRAME_LEN / 2] = f32::NAN;
    lease.set_header(ts(7), SHAPE);
    assert!(handles.sender.try_publish(lease));

    await_frames(&handles.stats, 0, 1).await;
    assert_eq!(handles.stats.rejected_non_finite(), 1);
    assert_eq!(handles.stats.accepted(), 0);
    assert!(handles.buffer.read().await.is_empty());
}

#[tokio::test]
async fn queue_full_drops_rather_than_blocking() {
    // No orchestration loop spawned, so nothing drains the queue.
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 1);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 4);

    assert!(publish(&pool, &handles.sender, 1.0, 1, SHAPE));
    assert!(!publish(&pool, &handles.sender, 1.0, 2, SHAPE));
    assert!(!publish(&pool, &handles.sender, 1.0, 3, SHAPE));

    assert_eq!(handles.stats.dropped_queue_full(), 2);
    // Dropped leases returned their buffers: one is still held by the queue.
    assert_eq!(pool.available(), 3);
}

#[tokio::test]
async fn cache_is_readable_while_ingest_runs() {
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 16);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 16);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    let reader: Arc<RwLock<IngestRingBuffer<f32>>> = Arc::clone(&handles.buffer);
    let reader_task = tokio::spawn(async move {
        let mut out = Vec::with_capacity(FRAME_LEN * CAP);
        for _ in 0..20 {
            let _ = reader.read().await.export_into(&mut out);
            assert_eq!(out.len() % FRAME_LEN, 0);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    });

    for i in 0..CAP as u64 {
        while !publish(&pool, &handles.sender, i as f32, i, SHAPE) {
            tokio::task::yield_now().await;
        }
    }
    await_frames(&handles.stats, CAP as u64, 0).await;
    reader_task.await.expect("reader task");
    assert_eq!(handles.stats.accepted(), CAP as u64);
    assert_eq!(handles.stats.rejected_total(), 0);
}

// --- the guard ---

#[test]
fn guard_reports_cause() {
    let good = block(0.0);
    assert_eq!(StreamGuard::verify(&good, SHAPE, SHAPE), Ok(()));

    let short = vec![0.0f32; FRAME_LEN - 1];
    assert_eq!(
        StreamGuard::verify(&short, SHAPE, SHAPE),
        Err(IngestError::LengthMismatch {
            expected: FRAME_LEN,
            actual: FRAME_LEN - 1
        })
    );
    assert_eq!(
        StreamGuard::verify(&good, SHAPE, SWAPPED),
        Err(IngestError::ShapeMismatch {
            expected: SHAPE,
            actual: SWAPPED
        })
    );
    let mut bad = good.clone();
    bad[3] = f32::INFINITY;
    assert_eq!(
        StreamGuard::verify(&bad, SHAPE, SHAPE),
        Err(IngestError::NonFiniteElement { index: 3 })
    );
}

// --- the window ---

#[test]
fn window_evicts_oldest_and_stays_capped() {
    let mut cache = cache();
    let overshoot = 5u64;
    let total = CAP as u64 + overshoot;

    for i in 0..total {
        let evicted = cache
            .push_snapshot(&block(i as f32), ts(i), SHAPE)
            .expect("well-formed frame");
        assert_eq!(evicted, i >= CAP as u64);
    }

    let mut out = Vec::new();
    cache.export_into(&mut out);
    assert_eq!(out.len(), cache.window_capacity());
    assert_eq!(cache.len(), CAP);
    assert!((marker_at(&out, 0) - overshoot as f32).abs() < f32::EPSILON);
    assert!((marker_at(&out, CAP - 1) - (total - 1) as f32).abs() < f32::EPSILON);

    let sequences: Vec<u64> = cache.frame_metadata().map(|m| m.sequence).collect();
    assert_eq!(sequences.first().copied(), Some(overshoot));
    assert_eq!(sequences.last().copied(), Some(total - 1));
}

#[test]
fn capacity_is_explicit_and_bounds_the_window() {
    let mut cache = IngestRingBuffer::<f32>::new(SHAPE, 4);
    assert_eq!(cache.capacity(), 4);
    assert_eq!(cache.shape(), SHAPE);
    assert_eq!(cache.window_capacity(), 4 * FRAME_LEN);

    fill(&mut cache, 10);
    assert_eq!(cache.len(), 4);

    let mut out = Vec::new();
    cache.export_into(&mut out);
    assert_eq!(out.len(), 4 * FRAME_LEN);
    assert!((marker_at(&out, 0) - 6.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 3) - 9.0).abs() < f32::EPSILON);
}

#[tokio::test]
async fn orchestrator_honours_the_configured_capacity() {
    let handles = IngestHandles::<f32>::new(SHAPE, 2, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 8);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    for i in 0..5u64 {
        while !publish(&pool, &handles.sender, i as f32, i, SHAPE) {
            tokio::task::yield_now().await;
        }
    }
    await_frames(&handles.stats, 5, 0).await;

    let cache = handles.buffer.read().await;
    assert_eq!(cache.capacity(), 2);
    assert_eq!(cache.len(), 2);
    let mut out = Vec::new();
    assert_eq!(cache.export_into(&mut out).frames, 2);
}

// --- allocation ---

#[test]
fn export_into_reuses_the_caller_buffer() {
    let mut cache = cache();
    fill(&mut cache, CAP as u64);

    let mut out = Vec::with_capacity(cache.window_capacity());
    let capacity_before = out.capacity();
    for _ in 0..16 {
        cache.export_into(&mut out);
        assert_eq!(out.len(), cache.window_capacity());
    }
    assert_eq!(out.capacity(), capacity_before);
}

#[test]
fn export_is_allocation_free_at_a_large_capacity() {
    // A capacity well past anything the other tests use, to show the export
    // path holds nothing per frame and does not fall back to allocating as
    // the window grows.
    let shape = FrameShape::new(2, 2);
    let mut cache = IngestRingBuffer::<f32>::new(shape, 512);
    let frame_len = cache.frame_len();
    for i in 0..600u64 {
        cache
            .push_snapshot(&vec![i as f32; frame_len], ts(i), shape)
            .expect("well-formed frame");
    }

    let mut out = Vec::with_capacity(cache.window_capacity());
    let capacity_before = out.capacity();
    for _ in 0..8 {
        assert_eq!(
            cache.export_with(&mut out, &ExportPolicy::all()).frames,
            512
        );
        assert_eq!(
            cache
                .export_with(&mut out, &ExportPolicy::all().with_stride(7))
                .frames,
            74
        );
    }
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

#[tokio::test]
async fn pool_exhaustion_is_counted_and_published_but_never_fails() {
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 1);
    let mut watcher = pool.watch_fallbacks();
    assert_eq!(*watcher.borrow(), 0);

    let a = pool.acquire();
    let b = pool.acquire();
    assert_eq!(a.len(), FRAME_LEN);
    assert_eq!(b.len(), FRAME_LEN);
    assert_eq!(pool.fallback_allocations(), 1);

    watcher.changed().await.expect("sender alive");
    assert_eq!(*watcher.borrow_and_update(), 1);

    drop(a);
    drop(b);
    assert_eq!(pool.available(), 2);
}

// --- export selection ---

#[test]
fn export_policy_limits_frames_and_preserves_order() {
    let mut cache = cache();
    fill(&mut cache, 10);

    let mut out = Vec::with_capacity(cache.window_capacity());
    let policy = ExportPolicy {
        max_frames: 3,
        max_age: None,
        stride: 1,
        min_interval: None,
    };
    assert_eq!(cache.export_with(&mut out, &policy).frames, 3);
    // Newest three, still written oldest-first.
    assert!((marker_at(&out, 0) - 7.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 1) - 8.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 2) - 9.0).abs() < f32::EPSILON);
}

#[test]
fn export_stride_thins_the_selection() {
    let mut cache = cache();
    fill(&mut cache, 9);

    let mut out = Vec::with_capacity(cache.window_capacity());
    let policy = ExportPolicy::all().with_stride(3);
    assert_eq!(cache.export_with(&mut out, &policy).frames, 3);
    // Back from the newest in steps of 3: 8, 5, 2 -> emitted 2, 5, 8.
    assert!((marker_at(&out, 0) - 2.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 1) - 5.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 2) - 8.0).abs() < f32::EPSILON);
}

#[test]
fn min_interval_thins_by_sensor_clock_not_by_position() {
    // Bursty capture times, as after sensor-side dedup.
    let captures_ms = [0u64, 50, 100, 1000, 1010, 2000, 2005, 2010, 3000];
    let mut cache = cache();
    for (i, ms) in captures_ms.iter().enumerate() {
        cache
            .push_snapshot(&block(i as f32), ts(ms * 1_000_000), SHAPE)
            .expect("well-formed frame");
    }

    let mut out = Vec::with_capacity(cache.window_capacity());
    let policy = ExportPolicy::all().with_min_interval(Duration::from_millis(500));
    let outcome = cache.export_with(&mut out, &policy);

    // Back from 3000: keep 3000; 2010 keep; 2005, 2000 skip; 1010 keep;
    // 1000 skip; 100 keep; 50, 0 skip -> emitted 100, 1010, 2010, 3000.
    assert_eq!(outcome.frames, 4);
    let markers: Vec<f32> = (0..4).map(|k| marker_at(&out, k)).collect();
    assert_eq!(markers, vec![2.0, 4.0, 7.0, 8.0]);
}

#[test]
fn min_interval_and_stride_compose() {
    let mut cache = cache();
    for i in 0..10u64 {
        cache
            .push_snapshot(&block(i as f32), ts(i * 100_000_000), SHAPE)
            .expect("well-formed frame");
    }
    let mut out = Vec::with_capacity(cache.window_capacity());

    let outcome = cache.export_with(&mut out, &ExportPolicy::all().with_stride(2));
    assert_eq!(outcome.frames, 5);
    assert!((marker_at(&out, 0) - 1.0).abs() < f32::EPSILON);

    let policy = ExportPolicy::all()
        .with_stride(2)
        .with_min_interval(Duration::from_millis(350));
    let outcome = cache.export_with(&mut out, &policy);
    assert_eq!(outcome.frames, 3);
    let markers: Vec<f32> = (0..3).map(|k| marker_at(&out, k)).collect();
    assert_eq!(markers, vec![1.0, 5.0, 9.0]);
}

#[test]
fn export_matching_accepts_an_arbitrary_predicate() {
    let mut cache = cache();
    fill(&mut cache, 8);
    let mut out = Vec::with_capacity(cache.window_capacity());

    let outcome = cache.export_matching(&mut out, |meta| meta.sequence % 2 == 0);
    assert_eq!(outcome.frames, 4);
    assert!((marker_at(&out, 0) - 0.0).abs() < f32::EPSILON);
    assert!((marker_at(&out, 1) - 2.0).abs() < f32::EPSILON);
}

// --- event-correlated export ---

/// Frames one second apart on the sensor clock.
fn spaced(count: u64) -> IngestRingBuffer<f32> {
    let mut cache = cache();
    for i in 0..count {
        cache
            .push_snapshot(&block(i as f32), ts(i * 1_000_000_000), SHAPE)
            .expect("well-formed frame");
    }
    cache
}

#[test]
fn export_around_brackets_an_event_time() {
    let cache = spaced(10);
    let mut out = Vec::with_capacity(cache.window_capacity());

    let outcome = cache.export_around(
        &mut out,
        ts(5_000_000_000),
        Duration::from_secs(2),
        Duration::from_secs(2),
    );
    assert_eq!(outcome.frames, 5);
    for (position, expected) in (3..=7).enumerate() {
        assert!((marker_at(&out, position) - expected as f32).abs() < f32::EPSILON);
    }
}

#[test]
fn export_around_is_asymmetric_and_clamps_at_the_window_edge() {
    let cache = spaced(10);
    let mut out = Vec::with_capacity(cache.window_capacity());

    let outcome = cache.export_around(
        &mut out,
        ts(4_000_000_000),
        Duration::from_secs(3),
        Duration::ZERO,
    );
    assert_eq!(outcome.frames, 4);
    assert!((marker_at(&out, 0) - 1.0).abs() < f32::EPSILON);

    // An event older than anything retained yields nothing rather than the
    // nearest frames, which would silently misattribute context.
    let outcome = cache.export_around(
        &mut out,
        ts(500_000_000_000),
        Duration::from_secs(1),
        Duration::from_secs(1),
    );
    assert_eq!(outcome.frames, 0);
    assert!(out.is_empty());
}

// --- intermittent residency ---

#[test]
fn stale_frames_are_excluded_by_max_age() {
    let mut cache = cache();
    fill(&mut cache, 1);
    let mut out = Vec::with_capacity(cache.window_capacity());

    let fresh = ExportPolicy::recent(CAP, Duration::from_secs(60));
    assert_eq!(cache.export_with(&mut out, &fresh).frames, 1);

    std::thread::sleep(Duration::from_millis(2));
    let strict = ExportPolicy::recent(CAP, Duration::from_millis(1));
    assert_eq!(cache.export_with(&mut out, &strict).frames, 0);
    assert!(out.is_empty());
    assert!(cache.newest_age().expect("one frame retained") >= Duration::from_millis(2));
}

#[test]
fn clear_drops_retained_frames_without_reallocating() {
    let mut cache = cache();
    fill(&mut cache, 5);
    assert_eq!(cache.len(), 5);

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert!(cache.newest_age().is_none());
    assert_eq!(cache.frame_metadata().count(), 0);

    // Still usable, and sequence numbering carries across the gap.
    cache
        .push_snapshot(&block(99.0), ts(99), SHAPE)
        .expect("well-formed frame");
    let meta: Vec<_> = cache.frame_metadata().collect();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].sequence, 5);
}

#[test]
fn clear_overwrites_the_token_store() {
    let mut cache = cache();
    cache
        .push_snapshot(&block(7.5), ts(1), SHAPE)
        .expect("well-formed frame");

    let mut out = Vec::with_capacity(cache.window_capacity());
    assert_eq!(cache.export_into(&mut out).frames, 1);
    assert!((out[0] - 7.5).abs() < f32::EPSILON);

    cache.clear();

    // A rejected push leaves the slot alone; the old contents must not resurface.
    let short = vec![0.0f32; 8];
    assert!(cache.push_snapshot(&short, ts(2), SHAPE).is_err());
    assert!(cache.export_into(&mut out).is_blind());
    assert!(out.is_empty());

    cache
        .push_snapshot(&block(0.25), ts(3), SHAPE)
        .expect("well-formed frame");
    assert_eq!(cache.export_into(&mut out).frames, 1);
    assert!(out.iter().all(|v| (*v - 0.25).abs() < f32::EPSILON));
}

#[tokio::test]
async fn suspended_ingest_drops_without_queueing() {
    let handles = IngestHandles::<f32>::new(SHAPE, CAP, 8);
    let pool = BufferPool::<f32>::with_buffers(FRAME_LEN, 4);
    tokio::spawn(handles.orchestrator.start_orchestration_loop());

    handles.control.suspend();
    assert!(!handles.control.is_active());
    assert!(!publish(&pool, &handles.sender, 1.0, 1, SHAPE));
    assert!(!publish(&pool, &handles.sender, 1.0, 2, SHAPE));

    assert_eq!(handles.stats.dropped_suspended(), 2);
    assert_eq!(handles.stats.accepted(), 0);
    assert_eq!(pool.available(), 4);
    assert!(handles.buffer.read().await.is_empty());

    handles.control.resume();
    assert!(publish(&pool, &handles.sender, 3.0, 3, SHAPE));
    await_frames(&handles.stats, 1, 0).await;
    assert_eq!(handles.stats.accepted(), 1);
    assert_eq!(handles.stats.dropped_suspended(), 2);
}

// --- staleness is a deployment property ---

#[test]
fn staleness_bound_cannot_be_widened_by_a_caller() {
    let mut cache = cache();
    cache.set_staleness_bound(Some(Duration::from_millis(1)));
    fill(&mut cache, 1);

    let mut out = Vec::with_capacity(cache.window_capacity());
    assert_eq!(cache.export_with(&mut out, &ExportPolicy::all()).frames, 1);

    std::thread::sleep(Duration::from_millis(3));

    assert_eq!(cache.export_with(&mut out, &ExportPolicy::all()).frames, 0);
    let generous = ExportPolicy::recent(CAP, Duration::from_secs(3600));
    assert_eq!(cache.export_with(&mut out, &generous).frames, 0);
    assert!(out.is_empty());
}

#[test]
fn staleness_bound_applies_to_event_correlated_export() {
    let mut cache = cache();
    cache.set_staleness_bound(Some(Duration::from_millis(1)));
    cache
        .push_snapshot(&block(1.0), ts(5_000_000_000), SHAPE)
        .expect("well-formed frame");

    let mut out = Vec::with_capacity(cache.window_capacity());
    let around = |cache: &IngestRingBuffer<f32>, out: &mut Vec<f32>| {
        cache.export_around(
            out,
            ts(5_000_000_000),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
    };
    assert_eq!(around(&cache, &mut out).frames, 1);

    std::thread::sleep(Duration::from_millis(3));

    assert_eq!(around(&cache, &mut out).frames, 0);
    assert_eq!(cache.export_matching(&mut out, |_| true).frames, 0);
}

#[test]
fn a_caller_can_still_narrow_below_the_bound() {
    let mut cache = cache();
    cache.set_staleness_bound(Some(Duration::from_secs(3600)));
    assert_eq!(cache.staleness_bound(), Some(Duration::from_secs(3600)));
    fill(&mut cache, 1);

    let mut out = Vec::with_capacity(cache.window_capacity());
    std::thread::sleep(Duration::from_millis(3));
    let strict = ExportPolicy::recent(CAP, Duration::from_millis(1));
    assert_eq!(cache.export_with(&mut out, &strict).frames, 0);
    assert_eq!(cache.export_with(&mut out, &ExportPolicy::all()).frames, 1);
}

// --- the person cannot see how much context the agent had ---

#[test]
fn an_empty_cache_is_distinguishable_from_a_stale_one() {
    let mut cache = cache();
    cache.set_staleness_bound(Some(Duration::from_millis(1)));
    let mut out = Vec::with_capacity(cache.window_capacity());

    let outcome = cache.export_with(&mut out, &ExportPolicy::all());
    assert!(outcome.is_blind());
    assert!(!outcome.is_stale());
    assert_eq!(outcome.retained, 0);

    fill(&mut cache, 3);

    let outcome = cache.export_with(&mut out, &ExportPolicy::all());
    assert_eq!(outcome.frames, 3);
    assert_eq!(outcome.excluded_stale, 0);
    assert!(!outcome.is_blind() && !outcome.is_stale());

    std::thread::sleep(Duration::from_millis(3));

    let outcome = cache.export_with(&mut out, &ExportPolicy::all());
    assert!(outcome.is_empty());
    assert!(outcome.is_stale());
    assert!(!outcome.is_blind());
    assert_eq!(outcome.retained, 3);
    assert_eq!(outcome.excluded_stale, 3);
}

#[test]
fn outcome_reports_staleness_on_event_correlated_export() {
    let mut cache = cache();
    cache.set_staleness_bound(Some(Duration::from_millis(1)));
    cache
        .push_snapshot(&block(1.0), ts(5_000_000_000), SHAPE)
        .expect("well-formed frame");

    let mut out = Vec::with_capacity(cache.window_capacity());
    std::thread::sleep(Duration::from_millis(3));
    let outcome = cache.export_around(
        &mut out,
        ts(5_000_000_000),
        Duration::from_secs(1),
        Duration::from_secs(1),
    );
    assert!(outcome.is_stale());
    assert_eq!(outcome.excluded_stale, 1);
}
