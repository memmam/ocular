//! Reusable wire buffers, so receiving a frame allocates nothing.
//!
//! The receive task leases a buffer, decodes into it, and publishes the lease.
//! The ingest loop copies the block into a cache slot and drops the lease,
//! which returns the buffer to the pool. In steady state the same buffers
//! circulate for the life of the process.

use crate::{CaptureTimestamp, TokenElement};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError};

/// Fixed set of pre-allocated frame buffers.
pub struct BufferPool<T: TokenElement> {
    free: Mutex<Vec<Vec<T>>>,
    frame_len: usize,
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
        Arc::new(Self {
            free: Mutex::new(free),
            frame_len,
        })
    }

    /// Takes a buffer from the pool.
    ///
    /// Allocates a replacement only if the pool is exhausted, which means
    /// `count` was sized too small for the in-flight depth.
    #[must_use]
    pub fn acquire(self: &Arc<Self>) -> FrameLease<T> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        let buf = free
            .pop()
            .unwrap_or_else(|| vec![T::default(); self.frame_len]);
        drop(free);
        FrameLease {
            buf: Some(buf),
            pool: Arc::clone(self),
            capture: CaptureTimestamp::default(),
            declared_dim: 0,
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
    /// Target model width the sensor device projected this frame for.
    pub declared_dim: usize,
}

impl<T: TokenElement> FrameLease<T> {
    /// Stamps the frame header carried alongside the token block.
    pub fn set_header(&mut self, capture: CaptureTimestamp, declared_dim: usize) {
        self.capture = capture;
        self.declared_dim = declared_dim;
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
