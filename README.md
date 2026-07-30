<!-- markdownlint-disable MD033 MD041 -->

<p align="center">
  <a href="https://decibri.com">
    <img
      src="https://github.com/user-attachments/assets/c43894c0-aec0-49fd-b9b7-aac2563eca1d"
      alt="Decibri Audio Echo Cancellation (AEC)"
      width="100%">
  </a>
</p>

# decibri-aec

Real-time acoustic echo cancellation for the decibri audio stack.

Decibri-aec is a deterministic, real-time acoustic echo canceller written in Rust. It removes the echo of far-end audio that leaks back into a microphone, tracks the echo path as it changes during a call, and stays out of the way when there is no echo to cancel.

<a href="https://crates.io/crates/decibri-aec"><img src="https://img.shields.io/crates/v/decibri-aec.svg" alt="Crates.io"></a>
<a href="https://github.com/decibri/decibri-aec/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>

## What it does

- **Wide-delay acquisition.** Finds echo paths across short and long transport delays.
- **Resilient streaming operation.** Tracks routing changes, recovers from capture discontinuities, and re-acquires when alignment becomes unreliable.
- **Conservative double-talk protection.** Reduces adaptation when the microphone is not reliably explained by the far-end reference, helping protect near-end speech.
- **Self-contained engine.** Delay discovery, lock, continuity, and reacquisition are handled inside the engine, so you integrate one component rather than building and maintaining a chain of parts yourself.
- **Written in Rust.** Built for safety, speed, and efficiency, with no third-party FFT or canceller code.

## Performance

Every recording is scored twice: once as the raw microphone signal, the audio before decibri runs, and once after decibri's echo cancellation is applied. The scores use AECMOS, Microsoft's echo-cancellation quality metric, on a five-point scale where higher is better. Benchmarked on 800 real recordings from the public ICASSP AEC Challenge dataset.

Echo cancellation is measured across three key scenarios, each testing a different problem the canceller has to handle. Far-end covers removing echo when only the remote speaker is talking. Near-end covers staying out of the way when only the local speaker is talking. Double-talk covers the hardest case, when both talk at once and echo overlaps with wanted speech. Each is scored below.

### Reading the scores

Each recording gets two scores, both out of five, and both are reported before and after cancellation.

The echo score rates how much echo is left and how noticeable it is. A low score means the echo is obvious and distracting. A high score means little or no echo can be heard.

The speech score rates how natural the local speaker's voice sounds. It is a measure of damage rather than echo, so it catches a canceller that removes echo by cutting into the wanted voice as well.

Before is the raw microphone signal exactly as it was captured. After is the same recording once cancellation has run. Comparing the two shows what the canceller did to that recording, both what it removed and what it cost.

### Far-end

Echo from the far-end speaker leaks back into the microphone while nobody local is talking. This is the case echo cancellation exists to solve.

The raw microphone scores 2.12 for echo quality. Applying decibri raises it to 3.24, with clean-speech quality untouched at 5.00.

### Near-end

Only the local speaker is talking and there is no echo to remove. The test here is that the canceller stays out of the way.

The scores are identical before and after, 5.00 for echo quality and 4.09 for clean speech, so decibri does no harm when there is nothing to cancel.

### Double-talk

Both people talk at once, so echo and wanted speech overlap. This is the hardest case for any canceller.

The raw microphone scores 2.39 for echo quality and decibri raises it to 2.72. Clean-speech quality drops slightly, from 4.07 to 3.79*. This is a known limitation of the deterministic engine on overlapping speech.

### Performance summary

| Scenario | Recordings | Echo before | Echo after | Speech before | Speech after |
| --- | --- | --- | --- | --- | --- |
| Far-end | 300 | 2.12 | 3.24 | 5.00 | 5.00 |
| Near-end | 200 | 5.00 | 5.00 | 4.09 | 4.09 |
| Double-talk | 300 | 2.39 | 2.72 | 4.07 | 3.79* |

All figures are mean AECMOS scores.

\* During double-talk the engine removes echo at some cost to the local speaker's voice, which is why the speech score moves from 4.07 to 3.79. Separating echo from speech that overlaps it is the hardest problem in echo cancellation, and every deterministic canceller has to choose where to sit between removing more echo and preserving more speech. Decibri is tuned toward removing echo, since that is what the far end hears, and the cost to local speech is small and confined to the moments both people talk at once.

### Comparing these numbers

AECMOS scores depend heavily on the recordings they are measured on. Different datasets contain different echo paths, different noise, and different amounts of nonlinear distortion, so a score measured on one dataset cannot be compared against a score measured on another.

The same is true of ERLE, the decibel measure of how much echo energy was removed. ERLE also rises when a canceller simply suppresses its output, so a high figure on its own says nothing about whether the local speaker survived. That is why the scores above report echo and speech together.

A higher AECMOS or ERLE figure measured on different material does not show that one canceller is better than another. Two cancellers can only be compared by running both over the same source recordings, through the same scorer, in one harness.

These figures are measured against what is achievable on this material. Before running the engine, we measure how linearly predictable each recording's echo path is, which sets an upper bound on how much echo any linear canceller can remove from it.

On the recordings where that bound is meaningful, decibri captures around 80 percent of it, and on some recordings it exceeds the linear estimate entirely because the residual suppressor removes echo a linear filter cannot. The remaining headroom is small.

## Installation

```toml
[dependencies]
decibri-aec = "0.2"
```

### Cargo features

- `tracing` (off by default). Forwards the engine's diagnostic events to [`tracing`](https://crates.io/crates/tracing). Without it the crate has no dependency on `tracing` and the emit sites compile to nothing. Enable it with `features = ["tracing"]`.

There is also an `internal-tests` feature, which is internal to this repository's own test suite. It adds no public API and there is nothing in it for a consumer to enable.

## Usage

```rust
use decibri_aec::{Aec, AecConfig};

// Construct the engine from a configuration. Construction validates the
// configuration and sizes the reference ring; it is the only fallible step.
// `AecConfig` is non-exhaustive: start from the default and assign fields.
let mut config = AecConfig::default();

// Supply a measured platform latency if you have one. Without it the engine
// estimates the delay itself, which costs some audio before it locks.
config.delay_hint_ms = Some(16);

let mut aec = Aec::new(config).unwrap();

// Feed far-end reference samples (mono, at the configured rate, in played
// order, including any renderer-inserted silence) as they are rendered.
aec.feed_reference(&[0.0_f32; 256]);

// Cancel a near-end capture block, appending the echo-reduced samples to a
// caller-owned buffer. The canceller re-blocks internally, so a call may
// append fewer or more samples than it consumed.
let mut out = Vec::new();
aec.process(&[0.0_f32; 256], &mut out).unwrap();

// Drain the end-of-stream carry once, after the final `process`.
aec.flush(&mut out).unwrap();
assert_eq!(out.len(), 256);
```

## License

Apache-2.0 © 2026 [Decibri](https://decibri.com).
