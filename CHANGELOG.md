<!-- markdownlint-disable MD024 -->
# Decibri AEC Changelog

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
