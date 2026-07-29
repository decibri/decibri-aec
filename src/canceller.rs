//! The streaming [`EchoCanceller`] trait, this crate's public seam, and the
//! [`CancellerMetrics`] snapshot it exposes.

use crate::error::AecError;

/// A streaming acoustic echo canceller for mono `f32` audio: the crate's
/// swappable algorithm seam.
///
/// An implementation consumes two time-aligned mono streams at one fixed sample
/// rate: the near-end capture (the microphone signal, containing the playback
/// echo) and the far-end reference (the signal the loudspeaker played, including
/// any inserted silence). It appends the echo-cancelled near-end signal to a
/// caller-owned buffer. It is long-lived and stateful: adaptive filter state,
/// detector state, and any partial-block carry persist across calls, so chunk
/// boundaries are invisible in the output.
///
/// # Alignment contract
///
/// `near` and `far` in one [`process`](EchoCanceller::process) call have equal
/// length, and `far[i]` is the caller's estimate of the reference sample playing
/// at the capture instant of `near[i]`, accurate to within the modelled tail;
/// residual delay inside the tail is absorbed by adaptation. The
/// [`Aec`](crate::Aec) engine produces this alignment from its reference ring; a
/// consumer with its own alignment drives the trait directly. The equal-length
/// precondition is the caller's to uphold and is checked with a
/// `debug_assert!`.
///
/// # Fallibility
///
/// Construction is where the classical implementations can fail.
/// [`process`](EchoCanceller::process) and [`flush`](EchoCanceller::flush)
/// return [`Result`] so an implementation whose runtime can fail (a model
/// inference backend) reports through the same seam; the classical
/// implementations never return an error after construction and document it.
///
/// # Determinism
///
/// Determinism is a property of the implementation, not the trait. The
/// classical cancellers document bit-exact cross-platform output for identical
/// inputs; a model-backed implementation documents its own reproducibility
/// regime instead. Consumers must not assume bit-exactness across
/// implementations.
///
/// # Input assumptions
///
/// Mono `f32`, nominally in `[-1.0, 1.0]`, at the constructed rate, free of
/// non-finite samples. The [`Aec`](crate::Aec) engine sanitizes both inputs
/// before this trait sees them, because a non-finite sample reaching an adaptive
/// update would poison the learned state permanently; a consumer that drives the
/// trait directly takes on that responsibility.
///
/// Implementations are [`Send`], so an instance can live behind a mutex on a
/// consumer thread.
pub trait EchoCanceller: Send {
    /// Cancels one block: consumes `near` and the equally long, time-aligned
    /// `far`, appending the echo-cancelled samples to `out`.
    ///
    /// `out` is appended to, never cleared, and the caller owns it. A framed
    /// implementation re-blocks internally and may append fewer or more samples
    /// than it consumed; the totals balance after [`flush`](EchoCanceller::flush)
    /// except for the constant [`latency_samples`](EchoCanceller::latency_samples)
    /// lead. Implementations that never fail after construction document it.
    fn process(&mut self, near: &[f32], far: &[f32], out: &mut Vec<f32>) -> Result<(), AecError>;

    /// Drains the end-of-stream partial-block carry into `out`.
    ///
    /// Call once at close, after the final [`process`](EchoCanceller::process).
    /// Appends nothing for a sample-in, sample-out implementation.
    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AecError>;

    /// Returns the constant algorithmic delay, in samples at the configured
    /// rate, between a near-end input sample and its cancelled output sample.
    ///
    /// Zero for a time-domain canceller; one block for a framed
    /// frequency-domain one. Callers use it to account for the delay the
    /// canceller introduces.
    ///
    /// It is a latency figure for a caller's own budget, not an alignment
    /// correction. The cancelled stream aligns index-for-index with the near-end
    /// input: the framing lead this value reports is a buffering delay that
    /// [`flush`](EchoCanceller::flush) resolves, not an index offset, so a caller
    /// must not shift the returned output by it.
    fn latency_samples(&self) -> usize;

    /// Clears all streaming and adaptive state without reallocation, discarding
    /// the learned echo path.
    ///
    /// The configured geometry is kept; the next stream re-converges from
    /// scratch.
    fn reset(&mut self);

    /// Returns a snapshot of algorithm-observable state.
    ///
    /// Metadata only, never sample data: the smoothed ERLE estimate, the current
    /// double-talk flag, and the divergence-reset count. See
    /// [`CancellerMetrics`].
    fn metrics(&self) -> CancellerMetrics;
}

/// A snapshot of an [`EchoCanceller`]'s algorithm-observable state.
///
/// Metadata only, never sample data. The [`Aec`](crate::Aec) engine composes
/// this into [`AecMetrics`](crate::AecMetrics), adding the delay estimate and
/// the reference-transport counters it owns.
///
/// This struct is `#[non_exhaustive]`: it is returned to callers, never
/// constructed by them, so a future metric field is a non-breaking addition.
/// Read it by field, and include a `..` rest pattern if you match on it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CancellerMetrics {
    /// The smoothed echo-return-loss-enhancement estimate, in decibels: how much
    /// the canceller is currently reducing the echo. Zero before the filter has
    /// observed enough signal to estimate it.
    pub erle_db: f32,

    /// Whether the double-talk detector currently believes the near-end talker
    /// is active. Adaptation is held while this is `true`.
    pub double_talk: bool,

    /// The number of times the divergence guard has reset the adaptive filter
    /// since construction, converting a non-finite or explosive coefficient
    /// state into a bounded re-convergence.
    pub divergence_resets: u64,
}

impl Default for CancellerMetrics {
    fn default() -> Self {
        Self {
            erle_db: 0.0,
            double_talk: false,
            divergence_resets: 0,
        }
    }
}

/// Compile-time assertion that the trait object is [`Send`], upholding the
/// bound the trait declares. The check is type-only: the closure is never run,
/// but its body fails to compile if the bound ever regresses.
const _: fn() = || {
    fn assert_send<T: Send + ?Sized>() {}
    assert_send::<dyn EchoCanceller>();
};
