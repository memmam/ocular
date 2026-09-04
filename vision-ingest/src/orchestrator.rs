use crate::ring_buffer::IngestRingBuffer;
use crate::CompressedFrame;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Supervisor task that decouples the capture thread from the cache.
pub struct AsyncIngestOrchestrator {
    buffer: Arc<RwLock<IngestRingBuffer>>,
    frame_rx: mpsc::Receiver<CompressedFrame>,
}

impl AsyncIngestOrchestrator {
    /// Builds the orchestrator, returning it alongside the producer handle and a
    /// shared reference to the cache for readers.
    ///
    /// `max_queue` bounds the channel: once it is full, producers await capacity
    /// rather than growing the queue.
    #[must_use]
    pub fn new(
        target_dim: usize,
        max_queue: usize,
    ) -> (
        Self,
        mpsc::Sender<CompressedFrame>,
        Arc<RwLock<IngestRingBuffer>>,
    ) {
        let buffer = Arc::new(RwLock::new(IngestRingBuffer::new(target_dim)));
        let (tx, rx) = mpsc::channel(max_queue);
        let reader_handle = Arc::clone(&buffer);
        (
            Self {
                buffer,
                frame_rx: rx,
            },
            tx,
            reader_handle,
        )
    }

    /// Drains the channel into the cache until every sender is dropped.
    ///
    /// A frame rejected by the guard is logged and skipped; the loop continues.
    /// A panic inside this loop is not caught here — it terminates the task and
    /// the caller's `JoinHandle` is what observes it.
    pub async fn start_orchestration_loop(mut self) {
        while let Some(frame) = self.frame_rx.recv().await {
            let mut write_lock = self.buffer.write().await;
            if let Err(err) = write_lock.push_snapshot(frame) {
                eprintln!("[STREAM_GUARD_WARN] Ingestion pipeline anomaly bypassed: {err}");
            }
        }
    }
}
