<!-- markdownlint-disable MD024 -->
# Decibri AEC Changelog

## [0.2.0] - 2026-07-30

Dependency and packaging hygiene, an allocation-free steady-state hot path, and
documentation corrections. Output samples are byte-identical to 0.1.0.

### Changed

- BREAKING: `tracing` is now an off-by-default cargo feature. A default build has
  no dependency on `tracing` and the diagnostic emit sites compile to nothing.
  Migration, for a consumer that wants the events back:
  `decibri-aec = { version = "0.2", features = ["tracing"] }`.
- The real FFT owns its working buffer, allocated once at construction, so
  neither transform direction allocates per call and `Aec::process` performs no
  heap allocation in steady state. Output is byte-identical to 0.1.0.
- The crate-internal reference canceller and the golden-pair suite that validates
  the pipeline against it are behind a new `internal-tests` cargo feature and
  excluded from the published package. The feature is internal to development: it
  adds no public API and there is nothing in it for a consumer to enable.

### Documentation

- `AecConfig::delay_hint_ms`: the offset a hint seeds is measured from the far-end
  reference frontier as the caller's own feeding establishes it, not from an
  absolute platform latency. The two directions of error are not equivalent: a
  hint short of that offset is absorbed by the modelled tail, while a hint longer
  than it cancels nothing and reports no error. The previous claim that a wrong
  hint costs convergence time rather than correctness was wrong in the overshoot
  direction and is removed, here and on `Aec::new`.
- `Aec::feed_reference`: a reference at a rate other than the configured one is
  accepted silently, and the diagnostic is `AecMetrics::acquisition_parked`
  climbing while `AecMetrics::delay_samples` stays `None`.
- `Aec::feed_reference`: automatic acquisition needs broadband far-end material.
  Sustained periodic material can leave acquisition parked for a whole stream;
  `AecConfig::delay_hint_ms` is the way around it, subject to that field's
  frontier-relative caveat.
- SECURITY.md: the `tracing` dependency is opt-in.
- CONTRIBUTING.md: plain `cargo test` is not the full suite, and the command that
  is.

## [0.1.0] - 2026-07-29

### Added

- The `EchoCanceller` trait: consumes a time-aligned `(near, far)` pair and appends the echo-reduced near-end to a caller-owned buffer, with `flush`, `reset`, `latency_samples`, and `metrics`.
- The `Aec` engine: constructs from an `AecConfig`, owns the far-end reference alignment, sanitizes non-finite input, and exposes metrics through `AecMetrics`.
- The `AecConfig` configuration surface, the `#[non_exhaustive]` `AecModel` selector, and the `AecError` type.
- `AecModel` string parsing: `FromStr` parses a model name, `AecModel::as_str` renders it back, `AecModel::PUBLIC_MODEL_NAMES` lists the selectable set, and an unknown name returns `AecError::UnknownModel`.
- The `Tau` canceller behind `AecModel::Tau`, with double-talk handling and deterministic output.
- A conservative residual echo suppressor behind `Suppression::Conservative` (the default); `Suppression::Off` bypasses it.
- Automatic echo-delay estimation used when no `delay_hint_ms` is supplied, reported through `AecMetrics::delay_samples`.
- `DelayEstimate::coarse_last_resort_exhausted`, a read-only observational flag carried through `AecMetrics::delay`.
- The `OutputTransitionPolicy` setting and its `#[non_exhaustive]` enum, wired into `AecConfig::output_transition`, governing emitted audio during delay reacquisition. Default `GradedReacquisition { fade_out_ms: 100, fade_in_ms: 200 }`; `PreserveCorrection` keeps the prior behavior. Additive and, at the default, byte-identical on any stream that never reacquires.
- Capture-continuity declaration: `Aec::declare_capture_continuity` and the `#[non_exhaustive]` `CaptureContinuity` enum, through which a host reports lost or restarted capture samples. `AecMetrics` gains `capture_discontinuities`, `capture_samples_lost`, `capture_declaration_pending`, and `capture_declarations_without_decision`. Additive: a caller that never declares is byte-identical to before.

### Changed

- `AecMetrics::delay_samples` now reports the active alignment offset instead of always being `None`.
- `Aec::latency_samples` now reports the selected canceller's framing latency.
- Undeclared capture loss is now recovered and reported through `AecMetrics::reference_reanchors`, kept separate from host-declared `capture_discontinuities`.

### Removed

- `AecError::ModelUnavailable`. `AecError::UnknownModel` remains the one model error.
