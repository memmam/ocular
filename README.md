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

**Panics are not caught in the drain loop.** One terminates the task and
surfaces through the caller's `JoinHandle`, which is what a supervisor should
watch. There is no in-loop restart.
