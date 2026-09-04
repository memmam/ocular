use crate::{CaptureTimestamp, FrameMeta, IngestConfig, IngestError, StreamGuard, TokenElement};
use std::time::Instant;

/// Maps a monotonic sequence number onto a slot index.
///
/// The remainder is strictly less than `MAX_FIFO_FRAMES` (30), so it always
/// fits in a `usize` regardless of pointer width.
#[allow(clippy::cast_possible_truncation)]
const fn slot_of(sequence: u64) -> usize {
    (sequence % IngestConfig::MAX_FIFO_FRAMES as u64) as usize
}

/// Fixed-capacity cyclic cache of the most recent frames.
///
/// Token storage is one flat, contiguous block sized at construction to
/// `MAX_FIFO_FRAMES * TOKENS_PER_FRAME * target_dim`. Pushing copies into a
/// slot; nothing is allocated, moved or freed on the hot path, and the
/// resident footprint is constant for the life of the cache.
pub struct IngestRingBuffer<T: TokenElement> {
    tokens: Vec<T>,
    meta: Vec<Option<FrameMeta>>,
    head: u64,
    target_dim: usize,
    frame_len: usize,
}

impl<T: TokenElement> IngestRingBuffer<T> {
    /// Allocates the full token block and slot metadata up front.
    ///
    /// # Panics
    ///
    /// Panics if `target_dim` is zero, or if the resulting window size
    /// overflows `usize`.
    #[must_use]
    pub fn new(target_dim: usize) -> Self {
        assert!(target_dim > 0, "target_dim must be non-zero");
        let frame_len = IngestConfig::TOKENS_PER_FRAME
            .checked_mul(target_dim)
            .expect("frame length overflows usize");
        let total = frame_len
            .checked_mul(IngestConfig::MAX_FIFO_FRAMES)
            .expect("window size overflows usize");
        Self {
            tokens: vec![T::default(); total],
            meta: vec![None; IngestConfig::MAX_FIFO_FRAMES],
            head: 0,
            target_dim,
            frame_len,
        }
    }

    /// Validates a token block and copies it into the next slot, overwriting
    /// the oldest frame once the window is full.
    ///
    /// Returns `true` when the write evicted a live frame.
    ///
    /// # Errors
    ///
    /// Propagates any [`StreamGuard`] rejection. On error the cache is
    /// untouched and the frame is discarded.
    pub fn push_snapshot(
        &mut self,
        tokens: &[T],
        capture: CaptureTimestamp,
        declared_dim: usize,
    ) -> Result<bool, IngestError> {
        StreamGuard::verify(tokens, self.frame_len, declared_dim, self.target_dim)?;

        let slot = slot_of(self.head);
        let start = slot * self.frame_len;
        self.tokens[start..start + self.frame_len].copy_from_slice(tokens);

        let evicted = self.meta[slot].is_some();
        self.meta[slot] = Some(FrameMeta {
            capture,
            arrival: Instant::now(),
            sequence: self.head,
        });
        self.head = self.head.wrapping_add(1);
        Ok(evicted)
    }

    /// Writes the retained window, oldest frame first, into `out`.
    ///
    /// `out` is cleared and refilled. Reuse one buffer across calls and this
    /// performs no allocation; size it with [`Self::window_capacity`].
    pub fn export_into(&self, out: &mut Vec<T>) {
        out.clear();
        out.reserve(self.len() * self.frame_len);
        for sequence in self.oldest_sequence()..self.head {
            let slot = slot_of(sequence);
            if self.meta[slot].is_some() {
                let start = slot * self.frame_len;
                out.extend_from_slice(&self.tokens[start..start + self.frame_len]);
            }
        }
    }

    /// Allocating convenience wrapper around [`Self::export_into`].
    ///
    /// Allocates a buffer sized to the retained window on every call. Prefer
    /// [`Self::export_into`] on any path that runs more than once.
    #[must_use]
    pub fn export_linearized_payload(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len() * self.frame_len);
        self.export_into(&mut out);
        out
    }

    /// Metadata for the retained frames, oldest first.
    pub fn frame_metadata(&self) -> impl Iterator<Item = FrameMeta> + '_ {
        (self.oldest_sequence()..self.head).filter_map(move |sequence| {
            let slot = slot_of(sequence);
            self.meta[slot]
        })
    }

    fn oldest_sequence(&self) -> u64 {
        self.head
            .saturating_sub(IngestConfig::MAX_FIFO_FRAMES as u64)
    }

    /// Frames currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(self.head)
            .unwrap_or(usize::MAX)
            .min(IngestConfig::MAX_FIFO_FRAMES)
    }

    /// True while no frame has been accepted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.head == 0
    }

    /// Elements in one frame: `TOKENS_PER_FRAME * target_dim`.
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Elements in a full window. Use this to size an [`Self::export_into`] buffer.
    #[must_use]
    pub fn window_capacity(&self) -> usize {
        self.frame_len * IngestConfig::MAX_FIFO_FRAMES
    }

    /// Target model width this cache validates against.
    #[must_use]
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    /// Total frames accepted since construction, including evicted ones.
    #[must_use]
    pub fn frames_accepted(&self) -> u64 {
        self.head
    }
}
