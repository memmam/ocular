use crate::{CaptureTimestamp, FrameMeta, IngestConfig, IngestError, StreamGuard, TokenElement};
use std::time::{Duration, Instant};

/// Maps a monotonic sequence number onto a slot index.
///
/// The remainder is strictly less than `MAX_FIFO_FRAMES` (30), so it always
/// fits in a `usize` regardless of pointer width.
#[allow(clippy::cast_possible_truncation)]
const fn slot_of(sequence: u64) -> usize {
    (sequence % IngestConfig::MAX_FIFO_FRAMES as u64) as usize
}

/// Selects which retained frames an export writes out.
///
/// A resident agent shares its context window with other work, so inserting
/// the whole window on every turn is rarely what you want: 30 frames is
/// `30 * TOKENS_PER_FRAME` tokens, and consecutive frames at a few FPS are
/// largely redundant.
#[derive(Clone, Copy, Debug)]
pub struct ExportPolicy {
    /// Write at most this many frames, counting back from the newest.
    pub max_frames: usize,
    /// Skip frames that arrived longer ago than this. `None` disables the
    /// check, which is only safe if ingest is known to be running.
    pub max_age: Option<Duration>,
    /// Take every `stride`-th frame working back from the newest. `1` takes
    /// every frame; `3` thins a 3 FPS stream to roughly one frame per second.
    pub stride: usize,
}

impl ExportPolicy {
    /// Every retained frame, regardless of age.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            max_frames: IngestConfig::MAX_FIFO_FRAMES,
            max_age: None,
            stride: 1,
        }
    }

    /// The newest `max_frames` frames, no older than `max_age`.
    #[must_use]
    pub const fn recent(max_frames: usize, max_age: Duration) -> Self {
        Self {
            max_frames,
            max_age: Some(max_age),
            stride: 1,
        }
    }

    /// Thins the selection to every `stride`-th frame.
    #[must_use]
    pub const fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }
}

impl Default for ExportPolicy {
    fn default() -> Self {
        Self::all()
    }
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

    /// Writes every retained frame, oldest first, into `out`.
    ///
    /// Equivalent to [`Self::export_with`] under [`ExportPolicy::all`]. This
    /// ignores frame age: on an intermittently active ingest path prefer
    /// [`Self::export_with`] with a `max_age`, or the window may be minutes
    /// stale and indistinguishable from live data.
    pub fn export_into(&self, out: &mut Vec<T>) {
        self.export_with(out, &ExportPolicy::all());
    }

    /// Writes the frames selected by `policy` into `out`, oldest first.
    ///
    /// `out` is cleared and refilled; reuse one buffer and this allocates
    /// nothing. Returns the number of frames written, which is zero when every
    /// retained frame is older than `policy.max_age`.
    pub fn export_with(&self, out: &mut Vec<T>, policy: &ExportPolicy) -> usize {
        out.clear();
        let max_frames = policy.max_frames.min(IngestConfig::MAX_FIFO_FRAMES);
        let stride = policy.stride.max(1);
        let now = Instant::now();

        // Slot indices, newest first. Bounded by the window, so it stays on the
        // stack and the export path allocates nothing.
        let mut chosen = [0usize; IngestConfig::MAX_FIFO_FRAMES];
        let mut count = 0;
        let mut live_seen = 0usize;
        let oldest = self.oldest_sequence();
        let mut sequence = self.head;

        while sequence > oldest && count < max_frames {
            sequence -= 1;
            let slot = slot_of(sequence);
            let Some(meta) = self.meta[slot] else {
                continue;
            };
            if let Some(max_age) = policy.max_age {
                // Arrival is monotonic in sequence order, so everything past
                // the first stale frame is staler still.
                if now.saturating_duration_since(meta.arrival) > max_age {
                    break;
                }
            }
            if live_seen % stride == 0 {
                chosen[count] = slot;
                count += 1;
            }
            live_seen += 1;
        }

        out.reserve(count * self.frame_len);
        for &slot in chosen[..count].iter().rev() {
            let start = slot * self.frame_len;
            out.extend_from_slice(&self.tokens[start..start + self.frame_len]);
        }
        count
    }

    /// Discards every retained frame without freeing the backing store.
    ///
    /// Call this when a visual task ends, so context from one situation cannot
    /// leak into the next. Sequence numbering continues across the clear, which
    /// leaves the gap visible in [`Self::frame_metadata`].
    pub fn clear(&mut self) {
        self.meta.iter_mut().for_each(|slot| *slot = None);
    }

    /// How long ago the newest retained frame arrived, or `None` when empty.
    ///
    /// Use this to decide whether the window is worth inserting at all.
    #[must_use]
    pub fn newest_age(&self) -> Option<Duration> {
        let now = Instant::now();
        (self.oldest_sequence()..self.head)
            .rev()
            .find_map(|sequence| self.meta[slot_of(sequence)])
            .map(|meta| now.saturating_duration_since(meta.arrival))
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
