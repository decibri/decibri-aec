#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod canceller;
pub mod error;

// The coarse-to-fine delay acquisition, used when the caller supplies no hint:
// the promotion gate between the two searches below, the tracker drive, and
// the reacquisition triggers.
mod acquire;
// The cheap downsampled global delay scan: the coarse stage.
mod coarse;
mod config;
// The sample-granular GCC-PHAT echo-delay estimator: the fine stage.
mod delay;
mod engine;
// The owned deterministic FFT primitive, consumed by the shipped
// frequency-domain canceller (Tau).
mod fft;
mod ring;
// The shipped partitioned-block frequency-domain canceller behind
// `AecModel::Tau`, built on the owned transform.
mod tau;
// The delay tracker: follows a locked delay and decides when the lock has
// gone bad. Driven by the acquirer.
mod track;

// The internal reference canceller (Rho) and the golden-pair suite that
// validates the pipeline against it. Compiled only into test builds, and only
// with the `internal-tests` feature: the reference is reachable for the harness
// and for crate-internal validation, never from the published library, a public
// selector, or a model string. The two files are excluded from the published
// package, so a build from the package has neither the files nor a declaration
// of them.
#[cfg(all(test, feature = "internal-tests"))]
mod golden;
#[cfg(all(test, feature = "internal-tests"))]
mod rho;

pub use canceller::{CancellerMetrics, EchoCanceller};
pub use config::{AecConfig, AecModel, OutputTransitionPolicy, Suppression};
pub use delay::{DelayEstimate, DelayLockSource, DelayStatus, ReacquireTrigger};
pub use engine::{Aec, AecMetrics, CaptureContinuity};
pub use error::AecError;
