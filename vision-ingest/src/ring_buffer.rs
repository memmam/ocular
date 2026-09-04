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
    /// every frame. Positional, so it assumes a regular cadence: if the sensor
    /// side skips near-duplicate frames before encoding, prefer `min_interval`.
    pub stride: usize,
    /// Skip a frame captured closer than this to the previously selected one.
    /// Measured on the sensor clock, so it holds up when frames arrive at an
    /// irregular rate. `None` disables it. Applied after `stride`.
    pub min_interval: Option<Duration>,
}

impl ExportPolicy {
    /// Every retained frame, regardless of age.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            max_frames: usize::MAX,
            max_age: None,
            stride: 1,
            min_interval: None,
        }
    }

    /// The newest `max_frames` frames, no older than `max_age`.
    #[must_use]
    pub const fn recent(max_frames: usize, max_age: Duration) -> Self {
        Self {
            max_frames,
            max_age: Some(max_age),
            stride: 1,
            min_interval: None,
        }
    }

    /// Thins the selection to every `stride`-th frame.
    #[must_use]
    pub const fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Thins the selection to frames at least `interval` apart on the sensor
    /// clock.
    #[must_use]
    pub const fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = Some(interval);
        self
    }
}

impl Default for ExportPolicy {
    fn default() -> Self {
        Self::all()
    }
}

/// What an export actually returned, and what it left out.
///
/// On a co-deployed system the person acting on the agent's output cannot see
/// how much context the agent had. A bare frame count cannot distinguish "the
/// scene is empty" from "nothing has been ingested for thirty seconds", and an
/// agent that reports the first when the second is true is confidently wrong
/// in the direction a human is least able to check. These fields let a caller
/// tell the two apart and hedge accordingly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExportOutcome {
    /// Frames written into the output buffer.
    pub frames: usize,
    /// Frames held in the cache but excluded for being older than the
    /// effective age limit.
    pub excluded_stale: usize,
    /// Frames the cache held when the call was made, before any filtering.
    pub retained: usize,
}

impl ExportOutcome {
    /// No frames were written.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// The cache held frames and every one of them was too old.
    ///
    /// Distinct from an empty cache: the agent has been looking, and what it
    /// saw is no longer current. Worth saying out loud rather than answering
    /// as though the view were live.
    #[must_use]
    pub const fn is_stale(&self) -> bool {
        self.frames == 0 && self.excluded_stale > 0
    }

    /// Nothing has been ingested, or the window was cleared.
    #[must_use]
    pub const fn is_blind(&self) -> bool {
        self.retained == 0
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
    live: usize,
    staleness_bound: Option<Duration>,
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
            live: 0,
            staleness_bound: None,
        }
    }

    /// Caps how old a frame may be and still be exported, whatever policy a
    /// caller passes.
    ///
    /// Export policies are set at the call site, so nothing stops a caller
    /// asking for the whole window with no age check. On a system that
    /// actuates, that is the difference between reasoning about where
    /// something is and where it was. Setting a bound here makes staleness a
    /// property of the deployment: a policy can narrow it, never widen it.
    ///
    /// `None` disables the bound, which is only appropriate where nothing
    /// downstream acts on the result.
    pub fn set_staleness_bound(&mut self, max_age: Option<Duration>) {
        self.staleness_bound = max_age;
    }

    /// The deployment-level staleness bound, if one is set.
    #[must_use]
    pub fn staleness_bound(&self) -> Option<Duration> {
        self.staleness_bound
    }

    /// Narrows a caller's age limit by the deployment bound. Never widens it.
    fn effective_max_age(&self, requested: Option<Duration>) -> Option<Duration> {
        match (requested, self.staleness_bound) {
            (Some(caller), Some(bound)) => Some(caller.min(bound)),
            (Some(only), None) | (None, Some(only)) => Some(only),
            (None, None) => None,
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
        if !evicted {
            self.live += 1;
        }
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
    pub fn export_into(&self, out: &mut Vec<T>) -> ExportOutcome {
        self.export_with(out, &ExportPolicy::all())
    }

    /// Writes the frames selected by `policy` into `out`, oldest first.
    ///
    /// `out` is cleared and refilled; reuse one buffer and this allocates
    /// nothing. Returns the number of frames written, which is zero when every
    /// retained frame is older than `policy.max_age`.
    pub fn export_with(&self, out: &mut Vec<T>, policy: &ExportPolicy) -> ExportOutcome {
        out.clear();
        let retained = self.len();
        if self.head == 0 {
            return ExportOutcome::default();
        }
        let max_age = self.effective_max_age(policy.max_age);
        let now = Instant::now();

        // Selection is data-dependent once `min_interval` is involved, so it
        // cannot be replayed arithmetically. Instead the walk runs twice: once
        // to count, then once to place each frame at its final position. Both
        // passes touch metadata only until the copy, and nothing is held per
        // frame, so this stays allocation-free at any capacity.
        let count = self.select(policy, max_age, now).count();
        out.resize(count * self.frame_len, T::default());
        for (index, slot) in self.select(policy, max_age, now).enumerate() {
            let dst = (count - 1 - index) * self.frame_len;
            let src = slot * self.frame_len;
            out[dst..dst + self.frame_len].copy_from_slice(&self.tokens[src..src + self.frame_len]);
        }

        ExportOutcome {
            frames: count,
            excluded_stale: retained - self.fresh_frames(max_age, now),
            retained,
        }
    }

    /// Slots selected by `policy`, newest first.
    fn select<'a>(
        &'a self,
        policy: &ExportPolicy,
        max_age: Option<Duration>,
        now: Instant,
    ) -> Selection<'a, T> {
        Selection {
            cache: self,
            sequence: self.head,
            oldest: self.oldest_sequence(),
            remaining: policy.max_frames.min(self.capacity),
            stride: policy.stride.max(1) as u64,
            seen: 0,
            max_age,
            now,
            min_interval_nanos: policy
                .min_interval
                .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
            last_capture_nanos: None,
        }
    }

    /// Live frames no older than `max_age`. Walks metadata only, so this is
    /// independent of whatever stride the selection used.
    fn fresh_frames(&self, max_age: Option<Duration>, now: Instant) -> usize {
        let Some(max_age) = max_age else {
            return self.len();
        };
        let oldest = self.oldest_sequence();
        let mut fresh = 0;
        let mut sequence = self.head;
        while sequence > oldest {
            sequence -= 1;
            let Some(meta) = self.meta[self.slot_of(sequence)] else {
                break;
            };
            if now.saturating_duration_since(meta.arrival) > max_age {
                break;
            }
            fresh += 1;
        }
        fresh
    }

    /// Writes every frame `accept` returns `true` for, oldest first, into `out`.
    ///
    /// `out` is cleared and refilled; pre-size it with [`Self::window_capacity`]
    /// and the call allocates nothing. Returns the number of frames written.
    ///
    /// Any [`Self::set_staleness_bound`] applies before `accept` is consulted,
    /// so a predicate cannot reach a frame the deployment considers too old.
    pub fn export_matching<F>(&self, out: &mut Vec<T>, mut accept: F) -> ExportOutcome
    where
        F: FnMut(&FrameMeta) -> bool,
    {
        out.clear();
        let bound = self.staleness_bound;
        let now = Instant::now();
        let retained = self.len();
        let mut excluded_stale = 0;
        let mut count = 0;
        for sequence in self.oldest_sequence()..self.head {
            let slot = self.slot_of(sequence);
            let Some(meta) = self.meta[slot] else {
                continue;
            };
            if let Some(max_age) = bound {
                if now.saturating_duration_since(meta.arrival) > max_age {
                    excluded_stale += 1;
                    continue;
                }
            }
            if !accept(&meta) {
                continue;
            }
            let start = slot * self.frame_len;
            out.extend_from_slice(&self.tokens[start..start + self.frame_len]);
            count += 1;
        }
        ExportOutcome {
            frames: count,
            excluded_stale,
            retained,
        }
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
    ) -> ExportOutcome {
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
    ///
    /// The token store is overwritten, not just unlinked. On a device worn in
    /// public the difference matters: dropping the metadata alone would leave
    /// every frame resident and recoverable through a later bug or a memory
    /// dump. The cost is one pass over the window, paid at a task boundary
    /// rather than on the ingest path.
    pub fn clear(&mut self) {
        self.meta.iter_mut().for_each(|slot| *slot = None);
        self.live = 0;
        self.tokens
            .iter_mut()
            .for_each(|value| *value = T::default());
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
    ///
    /// Counts live frames, not elapsed sequence positions, so this drops to
    /// zero after [`Self::clear`] even though sequence numbering continues.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// True when no frame is retained, whether because none has arrived or
    /// because the window was cleared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
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

/// Newest-first walk over the frames an [`ExportPolicy`] selects.
///
/// Live frames occupy a contiguous run of sequences ending at the newest
/// (`clear` empties the whole window), and arrival time is monotonic in
/// sequence order, so the first gap or the first stale frame ends the walk.
struct Selection<'a, T: TokenElement> {
    cache: &'a IngestRingBuffer<T>,
    sequence: u64,
    oldest: u64,
    remaining: usize,
    stride: u64,
    seen: u64,
    max_age: Option<Duration>,
    now: Instant,
    min_interval_nanos: Option<u64>,
    last_capture_nanos: Option<u64>,
}

impl<T: TokenElement> Iterator for Selection<'_, T> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.remaining == 0 || self.sequence <= self.oldest {
                return None;
            }
            self.sequence -= 1;
            let slot = self.cache.slot_of(self.sequence);
            let meta = self.cache.meta[slot]?;
            if let Some(max_age) = self.max_age {
                if self.now.saturating_duration_since(meta.arrival) > max_age {
                    return None;
                }
            }
            let positional = self.seen % self.stride == 0;
            self.seen += 1;
            if !positional {
                continue;
            }
            if let (Some(interval), Some(last)) = (self.min_interval_nanos, self.last_capture_nanos)
            {
                // Walking newest-first, capture times decrease.
                if last.saturating_sub(meta.capture.nanos) < interval {
                    continue;
                }
            }
            self.last_capture_nanos = Some(meta.capture.nanos);
            self.remaining -= 1;
            return Some(slot);
        }
    }
}
