use crate::pool::FrameLease;
use crate::ring_buffer::IngestRingBuffer;
use crate::{IngestStats, TokenElement};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Producer handle for the ingest queue.
///
/// Publishing never blocks and never applies backpressure to the sensor
/// device. The cache holds a fixed recent window by design, so a compute node
/// that falls behind discards the frames it cannot keep up with rather than
/// stalling capture upstream. Drops are counted, not silent.
pub struct FrameSender<T: TokenElement> {
    tx: mpsc::Sender<FrameLease<T>>,
    stats: Arc<IngestStats>,
}

impl<T: TokenElement> FrameSender<T> {
    /// Publishes a frame if the queue has room, otherwise drops it.
    ///
    /// Returns `true` when the frame was queued. A dropped lease returns its
    /// buffer to the pool immediately; check the result if the caller needs to
    /// know it is falling behind.
    #[must_use]
    pub fn try_publish(&self, frame: FrameLease<T>) -> bool {
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
}

impl<T: TokenElement> Clone for FrameSender<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            stats: Arc::clone(&self.stats),
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
}

impl<T: TokenElement> IngestHandles<T> {
    /// Builds the cache, the queue and the producer handle.
    ///
    /// `max_queue` bounds the in-flight depth. Because publishing drops rather
    /// than blocks, this sets how much jitter the loop absorbs before frames
    /// start being discarded, not how far the sensor device can run ahead.
    #[must_use]
    pub fn new(target_dim: usize, max_queue: usize) -> Self {
        let buffer = Arc::new(RwLock::new(IngestRingBuffer::new(target_dim)));
        let stats = Arc::new(IngestStats::new());
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
            },
            buffer,
            stats,
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
            let declared_dim = frame.declared_dim;
            let mut cache = self.buffer.write().await;
            match cache.push_snapshot(&frame, capture, declared_dim) {
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
