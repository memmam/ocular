# ocular

Rust workspace for the `vision-ingest` pipeline.

| Crate | Purpose |
| --- | --- |
| [`vision-ingest`](vision-ingest/) | Bounded ingest path for visual token blocks arriving from a remote sensor device, feeding a resident agent that only sometimes needs to see. |

## Build

```sh
cargo test
cargo clippy --all-targets
```

## Topology

Three machines, with the compute node as the hub.

```
[sensor device]  <--dedicated link-->  [compute node]  <--wired-->  [scratchpad]
 camera -> encoder                      resident agent + LLM         durable state
 emits fixed-shape token blocks         this crate lives here
```

The sensor device is worn by a person. It encodes what it sees and ships a
token block; the compute node runs the agent and everything in this crate;
the scratchpad is durable, isolated storage the agent writes to. The agent
addresses both. Nothing in this crate touches the scratchpad, and nothing in
it sees a pixel.

The sensor link is meant to be a dedicated radio with the sensor device as its
only client, with a general-purpose router carrying everything else. This
crate does not care how the link is built, only that frames can be lost on it.

## What is fixed here and what is not

The crate makes no assumption about frame rate, encoder, model, or token
count. It takes:

- a `FrameShape` (tokens per frame, elements per token) that both machines
  agree on;
- a `capacity` (frames retained);
- an element type implementing `TokenElement` (`f32` and `f64` are provided;
  implement it for `half::f16` or `half::bf16` to match the model);
- optionally a staleness bound.

The retention window's duration is `capacity` divided by whatever rate the
sensor side actually delivers. With durable storage downstream it is really a
decision deadline: how long the agent has to notice something and commit it
before it rolls off. That belongs to the deployment, not to this crate.

**Swapping the sensor device, the encoder, or the model is a reconfiguration,
not a code change.** Each of those shows up here as a different `FrameShape`
or element type. Build a new cache with the new shape and retire the old one;
there is deliberately no in-place reshape, because a shape change means the
retained frames are no longer comparable with the new ones. Until the switch
is made, the guard rejects the new geometry and `rejected_shape` climbs, which
is the signal a supervisor should act on.

Whether the projection to the model's width runs on the sensor side or here
is the one placement decision that couples a sensor to a model. This crate
handles either: it ingests whatever fixed-shape block arrives, from a wire or
from a local encoder.

## Design notes

**`FrameShape` is a cross-device contract.** The sensor side's output geometry
must match what this node is configured for. A mismatch rejects every frame,
so `StreamGuard` checks the declared shape as well as the block length, and
`IngestStats::rejected_shape` makes the failure visible to a supervisor
instead of printing to stderr. Negotiate the shape at session start.

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
returned when the lease drops. Exports fill a caller-owned buffer; size it with
`window_capacity()` and reuse it. The cost this avoids is allocator churn and
first-touch page faults on the full window every export, which shows up as
latency jitter next to an LLM sharing the same memory.

Running the pool dry is a sizing notice, not a fault. `acquire` allocates a
replacement so the frame is never lost, counts it in `fallback_allocations()`,
and publishes the running total on `watch_fallbacks()` so a supervisor task
can turn it into an operator notice. A non-zero count means the pool wants to
be bigger; nothing needs acting on in the moment.

**The compute node is a resident agent, not a dedicated vision pipeline.** It
needs eyes for some of its work and not the rest.

- *Suspension.* `IngestControl::suspend` gates the path at the sender. While
  suspended, publishing costs one relaxed atomic load and the leased buffer
  returns to the pool immediately. It does not stop the sensor device
  transmitting; gate that at the wire too if battery matters.
- *Staleness.* Intermittent ingest means a retained window can be arbitrarily
  old while looking exactly like live data. `set_staleness_bound` makes the
  limit a property of the deployment: a policy can narrow it, never widen it,
  and it applies to event-correlated and predicate exports too. Call `clear()`
  when a visual task ends so one situation's context cannot leak into the
  next.
- *Context budget.* A full window is `capacity * shape.tokens` tokens inserted
  per turn, competing with the agent's other work. `ExportPolicy::max_frames`,
  `stride` and `min_interval` thin the selection. `stride` is positional and
  assumes a regular cadence; `min_interval` is measured on the sensor clock
  and holds up when frames arrive irregularly.

**Near-duplicate frames are the sensor side's problem.** A luma-difference
check before the encoder is the cheap way to drop them, and it saves encoder
cycles, sensor-side battery and wire, not just work here. This crate cannot make
that check; what it does is not assume a regular frame cadence.

**The person acting on the output cannot see how much context the agent had.**
An export returns `ExportOutcome`, not a bare count. `is_blind()` (nothing
ingested, or cleared), `is_stale()` (frames held, all of them too old) and
`excluded_stale` are distinguishable, so an agent can say "I have not had a
view recently" instead of answering as though the scene were empty. On a
co-deployed system that is the failure a person is least able to catch.

`clear()` overwrites the token store rather than unlinking it. On a device
worn in public, dropping metadata alone would leave every frame resident and
recoverable through a later bug or a memory dump.

**A detection path is a separate pipeline, and this crate is the join.** An
LLM turn is hundreds of milliseconds to seconds; anything that has to react
to the scene cannot wait on it, whatever the frame rate. That work runs
upstream, as close to the sensor as it can, and emits small typed events.
What this crate provides is the correlation: `export_around` pulls the frames
bracketing an event's capture time, because by the time the agent is woken
the newest frames are no longer the ones the event refers to. Selection is by
`CaptureTimestamp`, so the detector and this cache must read the same sensor
clock.

**Nothing here can be made to allocate by a hostile or broken sender.** The
cache, the pool and the queue are all fixed size, and a full queue drops rather
than growing. Wire framing and decode happen upstream of this crate and are
where untrusted parsing actually lives; by the time a block reaches
`StreamGuard` it is already a typed slice of known length.

**Panics are not caught in the drain loop.** One terminates the task and
surfaces through the caller's `JoinHandle`, which is what a supervisor should
watch. There is no in-loop restart.
