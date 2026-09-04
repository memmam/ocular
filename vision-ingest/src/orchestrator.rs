use crate::pool::FrameLease;
use crate::ring_buffer::IngestRingBuffer;
use crate::{FrameShape, IngestStats, TokenElement};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Suspend/resume gate for the ingest path.
///
/// The compute node is a resident agent, not a dedicated vision pipeline: it
/// only needs eyes for some of its work. While suspended, publishing costs one
/// relaxed atomic load and the leased buffer goes straight back to the pool --
/// nothing is queued, copied or locked. Suspending here does not stop the
/// sensor device transmitting; gate that at the wire too if battery matters.
#[derive(Debug)]
pub struct IngestControl {
    active: AtomicBool,
}

impl IngestControl {
    /// Starts active.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: AtomicBool::new(true),
        }
    }

    /// Accepts frames again. Does not clear frames already retained -- call
    /// [`crate::ring_buffer::IngestRingBuffer::clear`] first if the previous
    /// window belongs to a finished task.
    pub fn resume(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Discards frames at the sender until resumed.
    pub fn suspend(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Whether frames are currently being accepted.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Default for IngestControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Producer handle for the ingest queue.
///
/// Publishing never blocks and never applies backpressure to the sensor
/// device. The cache holds a fixed recent window by design, so a compute node
/// that falls behind discards the frames it cannot keep up with rather than
/// stalling capture upstream. Drops are counted, not silent.
pub struct FrameSender<T: TokenElement> {
    tx: mpsc::Sender<FrameLease<T>>,
    stats: Arc<IngestStats>,
    control: Arc<IngestControl>,
}

impl<T: TokenElement> FrameSender<T> {
    /// Publishes a frame if the queue has room, otherwise drops it.
    ///
    /// Returns `true` when the frame was queued. A dropped lease returns its
    /// buffer to the pool immediately; check the result if the caller needs to
    /// know it is falling behind.
    #[must_use]
    pub fn try_publish(&self, frame: FrameLease<T>) -> bool {
        if !self.control.is_active() {
            self.stats.record_dropped_suspended();
            return false;
        }
        if self.tx.try_send(frame).is_ok() {
            return true;
        }
        self.stats.record_dropped_queue_full();
        false
    }

    /// Shared counters for this ingest path.
    #[must_use]
    pub fn stats(&self) -> &Arc<IngestStats> {
        &self.stats
    }

    /// Suspend/resume gate for this ingest path.
    #[must_use]
    pub fn control(&self) -> &Arc<IngestControl> {
        &self.control
    }
}

impl<T: TokenElement> Clone for FrameSender<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            stats: Arc::clone(&self.stats),
            control: Arc::clone(&self.control),
        }
    }
}

/// The four handles that make up one ingest path.
pub struct IngestHandles<T: TokenElement> {
    /// Drain loop. Spawn [`AsyncIngestOrchestrator::start_orchestration_loop`].
    pub orchestrator: AsyncIngestOrchestrator<T>,
    /// Producer handle for the receive task.
    pub sender: FrameSender<T>,
    /// Shared cache, readable while ingest runs.
    pub buffer: Arc<RwLock<IngestRingBuffer<T>>>,
    /// Counters covering acceptance, rejection and drops.
    pub stats: Arc<IngestStats>,
    /// Suspend/resume gate.
    pub control: Arc<IngestControl>,
}

impl<T: TokenElement> IngestHandles<T> {
    /// Builds the cache, the queue and the producer handle.
    ///
    /// `shape` and `capacity` go straight to [`IngestRingBuffer::new`].
    /// `max_queue` bounds the in-flight depth: because publishing drops
    /// rather than blocks, this sets how much jitter the loop absorbs before
    /// frames start being discarded, not how far the sensor device can run
    /// ahead.
    #[must_use]
    pub fn new(shape: FrameShape, capacity: usize, max_queue: usize) -> Self {
        let buffer = Arc::new(RwLock::new(IngestRingBuffer::new(shape, capacity)));
        let stats = Arc::new(IngestStats::new());
        let control = Arc::new(IngestControl::new());
        let (tx, rx) = mpsc::channel(max_queue);
        Self {
            orchestrator: AsyncIngestOrchestrator {
                buffer: Arc::clone(&buffer),
                frame_rx: rx,
                stats: Arc::clone(&stats),
            },
            sender: FrameSender {
                tx,
                stats: Arc::clone(&stats),
                control: Arc::clone(&control),
            },
            buffer,
            stats,
            control,
        }
    }
}

/// Drains the ingest queue into the cache.
pub struct AsyncIngestOrchestrator<T: TokenElement> {
    buffer: Arc<RwLock<IngestRingBuffer<T>>>,
    frame_rx: mpsc::Receiver<FrameLease<T>>,
    stats: Arc<IngestStats>,
}

impl<T: TokenElement> AsyncIngestOrchestrator<T> {
    /// Drains the queue into the cache until every sender is dropped.
    ///
    /// A frame refused by the guard is counted by cause and skipped; the loop
    /// continues. Panics are not caught here — one terminates the task and
    /// surfaces through the caller's `JoinHandle`, which is what a supervisor
    /// should be watching.
    pub async fn start_orchestration_loop(mut self) {
        while let Some(frame) = self.frame_rx.recv().await {
            let capture = frame.capture;
            let declared = frame.declared_shape;
            let mut cache = self.buffer.write().await;
            match cache.push_snapshot(&frame, capture, declared) {
                Ok(evicted) => {
                    self.stats.record_accepted();
                    if evicted {
                        self.stats.record_evicted();
                    }
                }
                Err(err) => self.stats.record_rejection(err),
            }
        }
    }
}
