# Contributing to decibri-aec

Thanks for your interest in contributing. This guide covers what you need to know.

Before making any contribution, please read our [Code of Conduct](https://github.com/decibri/decibri/blob/main/CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

## Ways to contribute

- Reporting a bug
- Submitting a fix
- Suggesting improvements
- Adding or updating documentation
- Improving test coverage
- Anything else we may have forgotten

## How to report bugs

1. Check the [issue tracker](https://github.com/decibri/decibri-aec/issues) for duplicates.
2. If the bug is new, open an issue.
3. Include:
   - Operating system and architecture
   - `rustc --version`
   - The crate version you are using
   - Steps to reproduce
   - Expected versus actual behaviour
   - Exact error messages

For an echo cancellation problem specifically, the most useful thing you can include is the configuration you used, the sample rate, and whether you supplied a `delay_hint_ms`. If you can share the microphone and far-end reference audio that triggers it, that helps enormously, but only share audio you have the right to share.

## How to request features

Open an issue explaining the problem you are trying to solve, your proposed solution if you have one, and any alternatives you considered.

Please open an issue describing your use case before submitting a large feature pull request, so we can align on scope first.

## Development setup

### Requirements

- Rust stable toolchain via [rustup](https://rustup.rs/). The minimum supported version is recorded in `Cargo.toml` under `rust-version`.

That is the whole list. Decibri-aec is pure signal processing and touches no audio hardware, so there are no platform-specific dependencies and the tests run anywhere.

### Building

```
cargo build                    # debug build
cargo build --release          # optimised release build
```

### Testing

```
cargo test --all-targets       # unit, integration, and example tests
cargo test --doc               # the README usage example
```

Both must pass. `cargo test --all-targets` does not run doctests, so the second command is not optional.

### Linting

```
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Both must be clean. The CI pipeline runs exactly these commands.

## What you should know before changing the engine

The cancellation engine is deterministic and its behaviour is pinned by tests. Several of those tests are bit-exact: they compare against stored golden vectors and fail on any change to the output, however small. This is deliberate. It means an accidental behaviour change cannot slip through unnoticed.

If your change touches anything under `src/`, expect the golden tests to tell you whether the audio output moved. If they fail, that is information, not an obstacle. Either the change was not meant to alter output and something is wrong, or it was meant to and the evidence for the new behaviour needs to accompany the pull request.

A change to cancellation quality needs measurement, not argument. The published performance figures come from a benchmark you can run yourself, described in [benchmarks/README.md](benchmarks/README.md). A claim that a change improves cancellation should be backed by that benchmark, over the same recordings, with both the echo and the speech scores reported. A change that removes more echo at the cost of the local speaker's voice is a trade, not an improvement, and the numbers need to show both sides of it.

The benchmark dataset and the scoring model are not included in this repository. They are third-party material with their own licence terms, credited in [benchmarks/ATTRIBUTIONS.md](benchmarks/ATTRIBUTIONS.md). The benchmark README explains how to obtain them.

## What we accept

- Bug fixes, with a test that demonstrates the fix
- Platform compatibility improvements
- Documentation improvements
- Performance improvements with a measurement
- Small refactors that reduce code without changing behaviour

## What we do not accept

- Changes to cancellation behaviour without benchmark evidence covering both echo removal and speech preservation
- New runtime dependencies. The crate deliberately has a very small dependency footprint and owns its own transform code
- A model runtime, inference engine, or trained weights. This crate is deterministic signal processing by design, and stays that way
- Breaking changes to the public API before 1.0.0 without a prior issue and scope discussion
- `unsafe` code. The crate forbids it

## How to contribute code

1. Fork the repository and clone it to your local machine
2. Create a branch from `main` with a descriptive name
3. Make your changes
4. Ensure the tests and linters above all pass locally
5. Commit with a clear and descriptive message
6. Push your branch to your fork
7. Open a pull request against `main`, describing what changed and why

Our team will review your pull request and provide feedback. We may ask for additional changes, so please be prepared to iterate before merging.

## Project structure

```
decibri-aec/
├── src/                       the library
├── tests/                     integration tests
│   ├── continuity.rs          capture-continuity behaviour
│   └── quality.rs             cancellation quality and API contracts
├── examples/
│   ├── cancel.rs              minimal usage example
│   ├── benchmark/             the benchmark harness
│   ├── shared/                shared example helpers
│   ├── coherence-census.rs    engine-free survey of the clip pool
│   ├── delay-probe.rs         synthetic delay coverage probe
│   └── make-split.rs          benchmark split tooling
├── benches/                   criterion benchmarks
├── benchmarks/                scoring tooling and reproduction guide
├── .github/workflows/         CI and release workflows
├── Cargo.toml
├── CHANGELOG.md
└── README.md
```

## CI pipeline

Every pull request runs an automated pipeline covering lint, format, security audit, and the test suite across Linux, macOS, Windows and ARM, plus a build at the minimum supported Rust version. Your pull request must pass CI before it can be merged. The pipeline lives in `.github/workflows/ci.yml`.

## Contributor License Agreement

Before your first contribution can be merged, we ask you to agree to the decibri Contributor License Agreement. It is a one-time step that lets the project include your work under its current and future licenses, with clear provenance, and it does not take away your copyright in what you contribute. You are welcome to read the full agreements first: the [Individual CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Individual-CLA-v1.md) and, for contributions made on behalf of a company, the [Corporate CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Corporate-CLA-v1.md).

When you open a pull request, an automated check looks at whether you are already covered. If you are not, it leaves a comment with a short sentence to agree to. Reply with that exact sentence as a comment on your own pull request, and the check turns green. That is the whole process, and once you have done it you are covered for your future contributions too. Until the check passes, the pull request cannot be merged.

If you are contributing as part of your work, your employer may need a Corporate CLA on file instead of an individual one. If that applies to you, or the check asks about it, contact the maintainers and we will sort it out.

The record we keep is deliberately minimal: your GitHub username and account ID, which version of the agreement you agreed to, and the date. How we handle that information, and how to request its removal, is set out in our [Privacy Policy](https://decibri.com/privacy).

The CLA covers your contributions across the decibri organisation's repositories, so you only need to agree once.

## License

The decibri-aec source is released under the [Apache License 2.0](https://github.com/decibri/decibri-aec/blob/main/LICENSE).

Contributions are governed by the Contributor License Agreement described above. Under the CLA you keep your copyright in what you contribute and grant the project the rights it needs to include and license your work, including under future licenses. Contributed code or content must be your own work, and you confirm that you have the right to grant those rights.
