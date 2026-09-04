#![deny(unsafe_code)]
#![warn(clippy::pedantic, clippy::cargo)]

//! Bounded ingest path for visual token blocks arriving from a remote sensor
//! device, feeding a resident agent that only sometimes needs to see.
//!
//! # Topology
//!
//! Perception and language run on separate machines. The sensor device
//! encodes a frame and ships a fixed-shape token block; this crate runs on
//! the compute node and owns everything downstream of the wire.
//!
//! ```text
//! [sensor device: encoder] --wire--> [receive task]
//!     -> FrameSender::try_publish        (drop-oldest, never blocks the wire)
//!     -> AsyncIngestOrchestrator         (drain loop)
//!     -> StreamGuard                     (shape + finiteness)
//!     -> IngestRingBuffer                (fixed slots, copy-in, no allocation)
//!     -> export_with / export_around     (into a caller-owned buffer)
//! ```
//!
//! The crate never sees a pixel and makes no assumption about frame rate,
//! model, or token count. Frame geometry is a [`FrameShape`] the deployment
//! supplies; the retention window is a capacity the deployment supplies.
//!
//! # Allocation
//!
//! Every buffer is allocated during construction. In steady state the ingest
//! path performs no heap allocation: frames are copied into pre-sized slots,
//! wire buffers are leased from a [`pool::BufferPool`] and returned on drop,
//! and exports fill a caller-owned buffer.
//!
//! # Element type
//!
//! Storage is generic over [`TokenElement`]. `f32` and `f64` are implemented
//! here; implement it for `half::f16` or `half::bf16` to keep the wire, the
//! cache and the model at one width with no conversion.

pub mod orchestrator;
pub mod pool;
pub mod ring_buffer;

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Geometry of one frame's token block.
///
/// Supplied by the deployment to match whatever the sensor side emits. This
/// crate does not care what produced it, only that every frame has the same
/// shape and that both machines agree on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameShape {
    /// Tokens per frame.
    pub tokens: usize,
    /// Elements per token, i.e. the width of the model the block was
    /// projected for.
    pub dim: usize,
}

impl FrameShape {
    #[must_use]
    pub const fn new(tokens: usize, dim: usize) -> Self {
        Self { tokens, dim }
    }

    /// Elements in one frame.
    ///
    /// # Panics
    ///
    /// Panics if the product overflows `usize`.
    #[must_use]
    pub const fn elements(self) -> usize {
        match self.tokens.checked_mul(self.dim) {
            Some(n) => n,
            None => panic!("frame shape overflows usize"),
        }
    }
}

/// Element type of a token block.
///
/// Implemented here for `f32` and `f64`. Implement it for `half::f16` or
/// `half::bf16` to match the model's native width.
pub trait TokenElement: Copy + Default + Send + Sync + 'static {
    /// Returns `false` for NaN and infinities, which must never reach the
    /// model's context.
    fn is_finite_token(self) -> bool;
}

impl TokenElement for f32 {
    #[inline]
    fn is_finite_token(self) -> bool {
        f32::is_finite(self)
    }
}

impl TokenElement for f64 {
    #[inline]
    fn is_finite_token(self) -> bool {
        f64::is_finite(self)
    }
}

/// Capture time as measured by the sensor device.
///
/// Nanoseconds against an epoch the two machines agree on. Deliberately not
/// [`std::time::Instant`]: an `Instant` is opaque, process-local and has no
/// constructor from a raw value, so it cannot cross a wire.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CaptureTimestamp {
    pub nanos: u64,
}

impl CaptureTimestamp {
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }
}

/// Per-frame bookkeeping retained alongside a slot's token block.
#[derive(Clone, Copy, Debug)]
pub struct FrameMeta {
    /// Capture time reported by the sensor device.
    pub capture: CaptureTimestamp,
    /// Local monotonic arrival time, stamped on this machine. Valid for
    /// recency and staleness checks; not comparable with `capture`.
    pub arrival: Instant,
    /// Monotonic sequence number assigned on acceptance.
    pub sequence: u64,
}

/// Reason a frame was refused entry to the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IngestError {
    /// Token block length did not match the configured shape. Usually means
    /// the sensor device is emitting a different geometry than this node
    /// was configured for.
    LengthMismatch { expected: usize, actual: usize },
    /// Frame declared a shape other than the one this cache validates
    /// against, even though its length happened to match.
    ShapeMismatch {
        expected: FrameShape,
        actual: FrameShape,
    },
    /// Block contained a NaN or an infinity. Left unchecked these propagate
    /// through attention and silently poison every subsequent token.
    NonFiniteElement { index: usize },
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "token block length {actual} does not match expected {expected}"
            ),
            Self::ShapeMismatch { expected, actual } => write!(
                f,
                "frame declares shape {}x{}, cache validates against {}x{}",
                actual.tokens, actual.dim, expected.tokens, expected.dim
            ),
            Self::NonFiniteElement { index } => {
                write!(f, "non-finite element at index {index}")
            }
        }
    }
}

impl std::error::Error for IngestError {}

/// Validation gate between the wire and the model's context.
pub struct StreamGuard;

impl StreamGuard {
    /// Checks a token block's geometry and numeric sanity before it is copied
    /// into the cache.
    ///
    /// The finiteness scan is the reason this is not just a length check: a
    /// single NaN reaching the context corrupts the forward pass with no error
    /// anywhere, which is expensive to diagnose after the fact.
    ///
    /// # Errors
    ///
    /// [`IngestError::LengthMismatch`] when the block is not exactly
    /// `expected.elements()` long, [`IngestError::ShapeMismatch`] when the
    /// frame's declared shape disagrees with the cache's, and
    /// [`IngestError::NonFiniteElement`] on the first NaN or infinity found.
    pub fn verify<T: TokenElement>(
        tokens: &[T],
        expected: FrameShape,
        declared: FrameShape,
    ) -> Result<(), IngestError> {
        let expected_len = expected.elements();
        if tokens.len() != expected_len {
            return Err(IngestError::LengthMismatch {
                expected: expected_len,
                actual: tokens.len(),
            });
        }
        if declared != expected {
            return Err(IngestError::ShapeMismatch {
                expected,
                actual: declared,
            });
        }
        for (index, value) in tokens.iter().enumerate() {
            if !value.is_finite_token() {
                return Err(IngestError::NonFiniteElement { index });
            }
        }
        Ok(())
    }
}

/// Observable counters for the ingest path.
///
/// A shape disagreement between the sensor device and this node rejects every
/// frame. That must be visible to a supervisor rather than printed to stderr,
/// so rejections are counted here by cause.
#[derive(Debug, Default)]
pub struct IngestStats {
    accepted: AtomicU64,
    rejected_length: AtomicU64,
    rejected_shape: AtomicU64,
    rejected_non_finite: AtomicU64,
    dropped_queue_full: AtomicU64,
    dropped_suspended: AtomicU64,
    evicted: AtomicU64,
}

impl IngestStats {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record_accepted(&self) {
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_evicted(&self) {
        self.evicted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_dropped_queue_full(&self) {
        self.dropped_queue_full.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_dropped_suspended(&self) {
        self.dropped_suspended.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_rejection(&self, err: IngestError) {
        let counter = match err {
            IngestError::LengthMismatch { .. } => &self.rejected_length,
            IngestError::ShapeMismatch { .. } => &self.rejected_shape,
            IngestError::NonFiniteElement { .. } => &self.rejected_non_finite,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Frames copied into the cache.
    #[must_use]
    pub fn accepted(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Frames overwritten because the window advanced past them.
    #[must_use]
    pub fn evicted(&self) -> u64 {
        self.evicted.load(Ordering::Relaxed)
    }

    /// Frames discarded at the sender because the queue was full.
    #[must_use]
    pub fn dropped_queue_full(&self) -> u64 {
        self.dropped_queue_full.load(Ordering::Relaxed)
    }

    /// Frames discarded because ingest was suspended.
    #[must_use]
    pub fn dropped_suspended(&self) -> u64 {
        self.dropped_suspended.load(Ordering::Relaxed)
    }

    /// Frames refused for a length mismatch.
    #[must_use]
    pub fn rejected_length(&self) -> u64 {
        self.rejected_length.load(Ordering::Relaxed)
    }

    /// Frames refused for a declared-shape mismatch.
    #[must_use]
    pub fn rejected_shape(&self) -> u64 {
        self.rejected_shape.load(Ordering::Relaxed)
    }

    /// Frames refused for containing a NaN or infinity.
    #[must_use]
    pub fn rejected_non_finite(&self) -> u64 {
        self.rejected_non_finite.load(Ordering::Relaxed)
    }

    /// Total refused frames across all causes.
    #[must_use]
    pub fn rejected_total(&self) -> u64 {
        self.rejected_length() + self.rejected_shape() + self.rejected_non_finite()
    }
}
