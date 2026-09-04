//! Reusable wire buffers, so receiving a frame allocates nothing.
//!
//! The receive task leases a buffer, decodes into it, and publishes the lease.
//! The ingest loop copies the block into a cache slot and drops the lease,
//! which returns the buffer to the pool. In steady state the same buffers
//! circulate for the life of the process.

use crate::{CaptureTimestamp, FrameShape, TokenElement};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tokio::sync::watch;

/// Fixed set of pre-allocated frame buffers.
///
/// Running out is a sizing signal, not a fault. [`Self::acquire`] allocates a
/// replacement so the frame is never lost, counts the event, and publishes the
/// running count on a [`watch`] channel so a supervisor can surface it to the
/// operator -- a notice that the pool wants to be bigger, not something that
/// needs acting on in the moment.
pub struct BufferPool<T: TokenElement> {
    free: Mutex<Vec<Vec<T>>>,
    frame_len: usize,
    fallbacks: AtomicU64,
    fallback_tx: watch::Sender<u64>,
}

impl<T: TokenElement> BufferPool<T> {
    /// Pre-allocates `count` buffers of `frame_len` elements.
    ///
    /// Size `count` to the channel bound plus the number of frames that can be
    /// in flight in the receive task, so [`Self::acquire`] never has to fall
    /// back to allocating.
    #[must_use]
    pub fn with_buffers(frame_len: usize, count: usize) -> Arc<Self> {
        let free = (0..count).map(|_| vec![T::default(); frame_len]).collect();
        let (fallback_tx, _) = watch::channel(0);
        Arc::new(Self {
            free: Mutex::new(free),
            frame_len,
            fallbacks: AtomicU64::new(0),
            fallback_tx,
        })
    }

    /// Takes a buffer from the pool.
    ///
    /// If the pool is exhausted this allocates a replacement rather than
    /// failing -- the frame is worth more than the allocation -- and records
    /// it via [`Self::fallback_allocations`] and [`Self::watch_fallbacks`].
    /// A non-zero count means `count` was sized below the in-flight depth.
    #[must_use]
    pub fn acquire(self: &Arc<Self>) -> FrameLease<T> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        let pooled = free.pop();
        drop(free);
        let buf = pooled.unwrap_or_else(|| {
            let total = self.fallbacks.fetch_add(1, Ordering::Relaxed) + 1;
            self.fallback_tx.send_replace(total);
            vec![T::default(); self.frame_len]
        });
        FrameLease {
            buf: Some(buf),
            pool: Arc::clone(self),
            capture: CaptureTimestamp::default(),
            declared_shape: FrameShape::new(0, 0),
        }
    }

    /// Buffers currently available.
    #[must_use]
    pub fn available(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Elements in each pooled buffer.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Times [`Self::acquire`] found the pool empty and allocated instead.
    #[must_use]
    pub fn fallback_allocations(&self) -> u64 {
        self.fallbacks.load(Ordering::Relaxed)
    }

    /// Receiver that wakes each time the fallback count changes.
    ///
    /// Intended for a supervisor task that turns the event into an operator
    /// notice. The value carried is the running total.
    #[must_use]
    pub fn watch_fallbacks(&self) -> watch::Receiver<u64> {
        self.fallback_tx.subscribe()
    }

    fn release(&self, buf: Vec<T>) {
        if buf.len() == self.frame_len {
            self.free
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(buf);
        }
    }
}

/// A pooled buffer plus the frame header the sensor device sent with it.
///
/// Derefs to `[T]`. Returns itself to the pool when dropped.
pub struct FrameLease<T: TokenElement> {
    buf: Option<Vec<T>>,
    pool: Arc<BufferPool<T>>,
    /// Capture time reported by the sensor device.
    pub capture: CaptureTimestamp,
    /// Frame geometry the sensor device says this block has.
    pub declared_shape: FrameShape,
}

impl<T: TokenElement> FrameLease<T> {
    /// Stamps the frame header carried alongside the token block.
    pub fn set_header(&mut self, capture: CaptureTimestamp, declared_shape: FrameShape) {
        self.capture = capture;
        self.declared_shape = declared_shape;
    }
}

impl<T: TokenElement> Deref for FrameLease<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.buf.as_deref().unwrap_or(&[])
    }
}

impl<T: TokenElement> DerefMut for FrameLease<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.buf.as_deref_mut().unwrap_or(&mut [])
    }
}

impl<T: TokenElement> Drop for FrameLease<T> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf);
        }
    }
}
