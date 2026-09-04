# ocular

Rust workspace for the `vision-ingest` pipeline.

| Crate | Purpose |
| --- | --- |
| [`vision-ingest`](vision-ingest/) | Bounded ingest path for compressed visual tokens arriving from a remote sensor device. |

## Build

```sh
cargo test
cargo clippy --all-targets
```

## Topology

Perception and language run on separate machines. The sensor device encodes a
frame and projects it to the target model's width; the compute node runs the
LLM and everything in this crate.

```
[sensor device]                          [compute node]
 camera -> SigLIP encoder -> resampler
        64 x target_dim tokens
                 |
                 +---- wire ----> receive task
                                    | FrameSender::try_publish   (drop-oldest)
                                    v
                                  AsyncIngestOrchestrator        (drain loop)
                                    | StreamGuard                (geometry + finiteness)
                                    v
                                  IngestRingBuffer               (30 fixed slots)
                                    | export_into(&mut Vec<T>)
                                    v
                                  LLM context insertion
```

The encoder and resampler are out of scope for this crate — it never sees a
pixel. `IngestConfig::PATCH_COUNT` (729) and `IngestConfig::ENCODER_DIM` (1152)
are published so the sensor side can size against them.

## Design notes

**`target_dim` is a cross-device contract.** The sensor device's projection
must match the width of the model on the compute node. A mismatch rejects every
frame, so `StreamGuard` checks the declared width as well as the block length,
and `IngestStats::rejected_dimension` makes the failure visible to a supervisor
instead of printing to stderr. Negotiate the width at session start.

**Backpressure never reaches the sensor.** The cache holds a fixed recent
window by design, so a compute node that falls behind should discard frames,
not stall capture. `FrameSender::try_publish` drops when the queue is full and
counts the drop; it never blocks and never awaits.

**Timestamps cross a machine boundary.** `CaptureTimestamp` is nanoseconds
against an epoch both machines agree on. `std::time::Instant` is deliberately
not used for capture time: it is opaque, process-local, and has no constructor
from a raw value, so it cannot be sent over a wire. Arrival is stamped locally
as an `Instant` and is valid only for recency checks on this node.

**Finiteness is checked, not assumed.** A NaN or infinity reaching the model's
context corrupts the forward pass with no error raised anywhere. The guard
scans for it and rejects the frame.

**No allocation after construction.** Cache slots are one flat pre-sized block
and frames are copied into them. Wire buffers are leased from `BufferPool` and
returned when the lease drops. `export_into` fills a caller-owned buffer; size
it with `window_capacity()` and reuse it. `export_linearized_payload` allocates
per call and exists only for convenience — the docs say so.

The measured cost this avoids is allocator churn and first-touch page faults on
a ~15.7 MB window (2048-wide, `f32`) every export, which shows up as latency
jitter next to an LLM competing for the same unified memory. It is not a
bandwidth win; the copy itself is small against a unified memory bus.

**Element width is the consumer's choice.** Storage is generic over
`TokenElement`, implemented here for `f32` and `f64`. Implement it for
`half::f16` or `half::bf16` to keep the wire, the cache and the model at one
width with no conversion: `f16` halves both the resident window (to ~7.9 MB)
and the wire rate (to ~768 KB/s at 3 FPS).

**The retention window is a deployment parameter.** `with_capacity` sets how
many frames the cache holds; `IngestConfig::DEFAULT_FIFO_FRAMES` (30, ten
seconds at 3 FPS) is a starting point, not a limit. With durable storage
downstream the window is really a decision deadline -- how long the agent has
to notice something and commit it somewhere permanent before it rolls off --
so it belongs to the deployment, not to this crate. Selection during export is
computed arithmetically rather than held per frame, so a large window costs
memory but no extra allocation.

**The compute node is a resident agent, not a dedicated vision pipeline.** It
needs eyes for some of its work and not the rest, which constrains three things:

- *Suspension.* `IngestControl::suspend` gates the path at the sender. While
  suspended, publishing costs one relaxed atomic load and the leased buffer
  returns to the pool immediately -- nothing is queued, copied or locked. It
  does not stop the sensor device transmitting; gate that at the wire too if
  headset battery matters.
- *Staleness.* Intermittent ingest means a retained window can be arbitrarily
  old while looking exactly like live data. `ExportPolicy::max_age` excludes
  stale frames and `newest_age()` reports the gap. Prefer `export_with` over
  `export_into` on any path where ingest may not be running. Call `clear()`
  when a visual task ends so one situation's context cannot leak into the next.
- *Context budget.* A full window is `capacity * TOKENS_PER_FRAME` tokens
  inserted per turn, competing with the agent's other work.
  `ExportPolicy::max_frames`, `stride` and `min_interval` thin the selection.
  `stride` is positional and assumes a regular cadence; `min_interval` is
  measured on the sensor clock and holds up when frames arrive irregularly.

**Near-duplicate frames are the sensor side's problem.** Consecutive frames at
a few FPS are largely redundant, and a luma-difference check before the
encoder is the cheap way to drop them -- it saves encoder cycles, headset
battery and wire, not just work here. This crate never sees a pixel, so it
cannot make that check itself; what it does is not assume a regular frame
cadence, which is what `min_interval` is for.

**The person acting on the output cannot see how much context the agent had.**
This is the failure mode a co-deployed wearable has that a robot does not: an
actuator working from stale data fails visibly and immediately, while a person
acting on confident advice fails quietly, because nothing told them the window
was thin. So an export reports `ExportOutcome`, not a bare count --
`is_blind()` (nothing ingested, or cleared), `is_stale()` (frames held, all of
them too old) and `excluded_stale` are distinguishable, and an agent can say
"I have not had a view recently" instead of answering as though the scene were
empty.

`clear()` overwrites the token store rather than unlinking it. On a device worn
in public, dropping metadata alone would leave every frame resident and
recoverable through a later bug or a memory dump. The cost is one pass over the
window, paid at a task boundary.

**This crate is deliberative-layer only.** On a system that actuates -- a
humanoid, or anything driving hardware -- nothing in a balance, collision or
force loop may depend on this path being responsive. On a wearable there is no
such loop -- the person is the actuator and their own reflexes are the safety
layer -- but the constraint holds for any variant that drives hardware. The LLM's timescale is
hundreds of milliseconds to seconds; a control loop's is milliseconds. Ingest
is built so it cannot interfere: publishing never blocks, the cache and queue
are fixed size, and a stalled or panicking drain loop drops frames rather than
propagating backpressure. It is not built to be read from a real-time context,
and `push_snapshot` holds the write lock across a full frame copy.

`set_staleness_bound` makes staleness a property of the deployment rather than
of each call site. Export policies are chosen by callers, so without a bound
any of them can ask for the whole window with no age check -- on a system that
acts in the world, the difference between reasoning about where something is
and where it was. A policy can narrow the bound; it can never widen it, and it
applies to event-correlated and predicate exports too.

**A reflex path is a separate pipeline, and this crate is the join.** Reacting
to something in the scene cannot run through here: at 3 FPS a frame arrives
every 333 ms, already slower than human visual reaction, before inference adds
anything. A reflex path wants a much higher frame rate, a payload of a few
floats rather than a token block, and an interrupt rather than a context
insertion -- so it belongs upstream, as close to the sensor as it can run.

What this crate provides is the correlation. When a detector reports an event
at some capture time, `export_around` pulls the frames bracketing that moment;
by the time the agent is woken the newest frames are no longer the ones the
event refers to. Selection is by `CaptureTimestamp`, so the detector and this
cache must read the same sensor clock. `export_matching` takes an arbitrary
predicate over `FrameMeta` for anything else.

**Nothing here can be made to allocate by a hostile or broken sender.** The
cache, the pool and the queue are all fixed size, and a full queue drops rather
than growing. Wire framing and decode happen upstream of this crate and are
where untrusted parsing actually lives; by the time a block reaches
`StreamGuard` it is already a typed slice of known length.

**Panics are not caught in the drain loop.** One terminates the task and
surfaces through the caller's `JoinHandle`, which is what a supervisor should
watch. There is no in-loop restart.
