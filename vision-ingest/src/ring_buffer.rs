use crate::{CaptureTimestamp, FrameMeta, IngestConfig, IngestError, StreamGuard, TokenElement};
use std::time::{Duration, Instant};

/// Selects which retained frames an export writes out.
///
/// A resident agent shares its context window with other work, so inserting
/// the whole window on every turn is rarely what you want: a full window is
/// `capacity * TOKENS_PER_FRAME` tokens, and consecutive frames at a few FPS
/// are largely redundant. `max_frames` is clamped to the cache's capacity.
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
            max_frames: usize::MAX,
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
/// `capacity * TOKENS_PER_FRAME * target_dim`. Pushing copies into a
/// slot; nothing is allocated, moved or freed on the hot path, and the
/// resident footprint is constant for the life of the cache.
pub struct IngestRingBuffer<T: TokenElement> {
    tokens: Vec<T>,
    meta: Vec<Option<FrameMeta>>,
    head: u64,
    target_dim: usize,
    frame_len: usize,
    capacity: usize,
}

impl<T: TokenElement> IngestRingBuffer<T> {
    /// Allocates a cache holding [`IngestConfig::DEFAULT_FIFO_FRAMES`] frames.
    ///
    /// # Panics
    ///
    /// Panics if `target_dim` is zero, or if the window size overflows `usize`.
    #[must_use]
    pub fn new(target_dim: usize) -> Self {
        Self::with_capacity(target_dim, IngestConfig::DEFAULT_FIFO_FRAMES)
    }

    /// Allocates a cache holding `capacity` frames.
    ///
    /// `capacity` is the retention window, and on a system with durable storage
    /// downstream it is really a decision deadline: how long the agent has to
    /// notice something and commit it somewhere permanent before it rolls off.
    /// At `f` frames per second the window lasts `capacity / f` seconds and
    /// costs `capacity * TOKENS_PER_FRAME * target_dim` elements resident.
    ///
    /// # Panics
    ///
    /// Panics if `target_dim` or `capacity` is zero, or if the window size
    /// overflows `usize`.
    #[must_use]
    pub fn with_capacity(target_dim: usize, capacity: usize) -> Self {
        assert!(target_dim > 0, "target_dim must be non-zero");
        assert!(capacity > 0, "capacity must be non-zero");
        let frame_len = IngestConfig::TOKENS_PER_FRAME
            .checked_mul(target_dim)
            .expect("frame length overflows usize");
        let total = frame_len
            .checked_mul(capacity)
            .expect("window size overflows usize");
        Self {
            tokens: vec![T::default(); total],
            meta: vec![None; capacity],
            head: 0,
            target_dim,
            frame_len,
            capacity,
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

        let slot = self.slot_of(self.head);
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
        if self.head == 0 {
            return 0;
        }
        let stride = policy.stride.max(1) as u64;
        let max_frames = policy.max_frames.min(self.capacity);
        let now = Instant::now();
        let oldest = self.oldest_sequence();
        let newest = self.head - 1;

        // Live frames occupy a contiguous run of sequences ending at `newest`
        // (`clear` empties the whole window), so the selection can be counted
        // walking back and then replayed forward. Nothing is stored per frame,
        // which keeps this allocation-free at any capacity.
        let mut count: u64 = 0;
        while usize::try_from(count).unwrap_or(usize::MAX) < max_frames {
            let Some(sequence) = count
                .checked_mul(stride)
                .and_then(|offset| newest.checked_sub(offset))
            else {
                break;
            };
            if sequence < oldest {
                break;
            }
            let Some(meta) = self.meta[self.slot_of(sequence)] else {
                break;
            };
            if let Some(max_age) = policy.max_age {
                // Arrival is monotonic in sequence order, so everything past
                // the first stale frame is staler still.
                if now.saturating_duration_since(meta.arrival) > max_age {
                    break;
                }
            }
            count += 1;
        }

        let written = usize::try_from(count).unwrap_or(usize::MAX);
        out.reserve(written * self.frame_len);
        for step in (0..count).rev() {
            let sequence = newest - step * stride;
            let start = self.slot_of(sequence) * self.frame_len;
            out.extend_from_slice(&self.tokens[start..start + self.frame_len]);
        }
        written
    }

    /// Writes every frame `accept` returns `true` for, oldest first, into `out`.
    ///
    /// `out` is cleared and refilled; pre-size it with [`Self::window_capacity`]
    /// and the call allocates nothing. Returns the number of frames written.
    pub fn export_matching<F>(&self, out: &mut Vec<T>, mut accept: F) -> usize
    where
        F: FnMut(&FrameMeta) -> bool,
    {
        out.clear();
        let mut count = 0;
        for sequence in self.oldest_sequence()..self.head {
            let slot = self.slot_of(sequence);
            let Some(meta) = self.meta[slot] else {
                continue;
            };
            if !accept(&meta) {
                continue;
            }
            let start = slot * self.frame_len;
            out.extend_from_slice(&self.tokens[start..start + self.frame_len]);
            count += 1;
        }
        count
    }

    /// Writes the frames bracketing `centre` in sensor-clock time, oldest first.
    ///
    /// This is the join between a fast reflex path and the deliberative one.
    /// A frame-rate detector running upstream reports an event at some capture
    /// time; the agent then pulls the visual context around that moment out of
    /// here rather than the newest frames, which by the time it is woken are
    /// not the ones the event refers to.
    ///
    /// Selection is by [`CaptureTimestamp`], the sensor device's clock, so the
    /// detector and this cache must be reading the same clock. Arrival time on
    /// this node is not comparable and is not used.
    ///
    /// Returns the number of frames written, which is zero if the event
    /// predates the retained window.
    pub fn export_around(
        &self,
        out: &mut Vec<T>,
        centre: CaptureTimestamp,
        before: Duration,
        after: Duration,
    ) -> usize {
        let span_before = u64::try_from(before.as_nanos()).unwrap_or(u64::MAX);
        let span_after = u64::try_from(after.as_nanos()).unwrap_or(u64::MAX);
        let low = centre.nanos.saturating_sub(span_before);
        let high = centre.nanos.saturating_add(span_after);
        self.export_matching(out, |meta| {
            meta.capture.nanos >= low && meta.capture.nanos <= high
        })
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
            .find_map(|sequence| self.meta[self.slot_of(sequence)])
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
            let slot = self.slot_of(sequence);
            self.meta[slot]
        })
    }

    fn oldest_sequence(&self) -> u64 {
        self.head.saturating_sub(self.capacity as u64)
    }

    /// Maps a monotonic sequence number onto a slot index. The remainder is
    /// strictly less than `capacity`, which is a `usize`, so this never truncates.
    #[allow(clippy::cast_possible_truncation)]
    fn slot_of(&self, sequence: u64) -> usize {
        (sequence % self.capacity as u64) as usize
    }

    /// Frames currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(self.head)
            .unwrap_or(usize::MAX)
            .min(self.capacity)
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
        self.frame_len * self.capacity
    }

    /// Frames this cache retains before overwriting.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
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
