# ocular

Rust workspace for the `vision-ingest` pipeline.

## Crates

| Crate | Purpose |
| --- | --- |
| [`vision-ingest`](vision-ingest/) | Bounded, snapshot-driven ingest buffer for compressed visual tokens feeding a local LLM context. |

## Build

```sh
cargo test
cargo clippy --all-targets
```

## Scope

`vision-ingest` implements the tail of the ingest path described in the design
spec: a fixed-capacity cyclic frame cache behind a bounded mpsc channel, with a
length gate on every frame entering the cache.

Encoder hidden-state extraction (SigLIP-class, 729 patches x 1152) and the
resampler / projection stage that compresses a frame to 64 tokens are **not**
implemented here. `IngestConfig::PATCH_COUNT` and `IngestConfig::ENCODER_DIM`
are published so upstream producers can size against them.

### Known deviations from the design spec

- **The path is memory-bounded, not allocation-free.** Frame payloads are owned
  `Vec<f32>` buffers handed over by the producer, and
  `export_linearized_payload` allocates a fresh output buffer sized to the
  retained window on every call (~15.7 MB at a 2048-wide target). A genuinely
  zero-allocation hot path requires slot reuse and a caller-supplied output
  buffer, which is a different API.
- **The orchestration loop does not catch panics.** A panic terminates the task
  and surfaces through the caller's `JoinHandle`. There is no restart.
- **`StreamGuard` validates element count only.** A correctly sized block that
  was projected for a different target width is not detected.
