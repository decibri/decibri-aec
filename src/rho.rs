//! Rho: the crate-internal time-domain NLMS reference canceller.
//!
//! Rho is not a shipped model. It is the bit-exact numerical reference every
//! future correctness claim in this crate is checked against: the golden
//! fixtures pin Rho's output, and the production Tau canceller is validated
//! against Rho when it lands. Rho is therefore built for exactness and
//! determinism, not speed: a plain normalized least-mean-squares (NLMS)
//! adaptive FIR, updated per sample, with a Geigel double-talk detector that
//! freezes adaptation while the near-end talker is active.
//!
//! # Reachability
//!
//! Rho is deliberately unreachable from outside the crate. It is not an
//! [`AecModel`](crate::AecModel) variant, it has no string name in the model
//! parse, and this module is compiled only into test builds (`cfg(test)`), so
//! it does not exist in the published library at all. The harness reaches it
//! through the crate-internal test constructor on the engine and through this
//! module directly.
//!
//! # Determinism
//!
//! The same input produces bit-identical output on every platform and every
//! run, which is what makes Rho usable as a reference:
//!
//! - The algorithm is per-sample and sequential; every loop iterates in fixed
//!   lag order, so no floating-point reduction ever reorders.
//! - The sample path uses only IEEE-exact operations (`+`, `-`, `*`, `/`,
//!   `abs`, comparisons, and exact `f32`/`f64` conversions). There are no
//!   transcendentals, no `mul_add`/FMA, and rustc applies no fast-math
//!   reassociation at any optimization level. The one transcendental
//!   (`log10`, for the ERLE display value) runs in [`metrics`] on read and
//!   never feeds back into streaming state.
//! - There are no unordered containers, no threads, no time, and no
//!   randomness anywhere in the module.
//!
//! [`metrics`]: EchoCanceller::metrics

use crate::canceller::{CancellerMetrics, EchoCanceller};
use crate::error::AecError;

/// NLMS step size (mu). Stability requires `0 < mu < 2`; 0.5 is the
/// conservative textbook choice, trading a little convergence speed for
/// robustness to the regularized normalization.
const STEP_SIZE: f64 = 0.5;

/// Regularization epsilon added to the far-end window energy in the
/// normalized step, so a near-silent reference cannot blow the update up.
/// Sized against nominal `[-1.0, 1.0]` input, where a fully active window's
/// energy is orders of magnitude above it.
const REGULARIZATION: f64 = 1e-3;

/// Geigel double-talk threshold: near-end speech is declared when the
/// microphone magnitude exceeds this fraction of the recent far-end peak.
/// 0.5 is the classic setting, assuming at least about 6 dB of echo-return
/// loss through the acoustic path.
const GEIGEL_THRESHOLD: f32 = 0.5;

/// Double-talk hangover in milliseconds: how long adaptation stays frozen
/// after the last Geigel trigger, bridging the gaps between near-end speech
/// peaks.
const DTD_HANGOVER_MS: u32 = 15;

/// One-pole smoothing coefficient for the ERLE power estimates, applied per
/// sample. A fixed rational constant (about a 60 ms time constant at 16 kHz)
/// rather than a rate-derived exponential, so no transcendental ever runs at
/// construction.
const ERLE_SMOOTHING: f64 = 0.999;

/// Power floor for the ERLE estimate: below this smoothed near-end power the
/// estimate reads zero (not enough signal observed), and the residual power is
/// floored by it so the reported ratio stays finite.
const ERLE_POWER_FLOOR: f64 = 1e-10;

/// The crate-internal NLMS reference canceller. See the module documentation
/// for its role and its determinism guarantees.
pub(crate) struct RhoCanceller {
    /// Adaptive FIR coefficients indexed by lag: `weights[0]` multiplies the
    /// newest far-end sample. Invariant: every weight is finite after every
    /// processed sample; the divergence guard zeroes the filter rather than
    /// let a non-finite value persist.
    weights: Vec<f32>,
    /// Ring of the most recent `weights.len()` far-end samples, sanitized
    /// (finite) by construction. `pos` indexes the newest sample.
    history: Vec<f32>,
    /// Index in `history` of the most recent far-end sample.
    pos: usize,
    /// Remaining samples of the double-talk adaptation freeze. Non-zero is
    /// the `double_talk` metric.
    hangover_remaining: u32,
    /// The freeze length in samples that a Geigel trigger (re-)arms, derived
    /// from [`DTD_HANGOVER_MS`] at construction.
    hangover_samples: u32,
    /// Smoothed near-end (microphone) power for the ERLE estimate.
    near_power: f64,
    /// Smoothed residual (output) power for the ERLE estimate.
    residual_power: f64,
    /// Times the divergence guard has zeroed the filter since construction.
    /// Deliberately survives [`reset`](EchoCanceller::reset): the metric is
    /// documented as a since-construction count.
    divergence_resets: u64,
}

impl RhoCanceller {
    /// Constructs the reference canceller for a validated geometry: an
    /// adaptive FIR of `tail_ms` worth of taps at `sample_rate`.
    ///
    /// The caller passes fields from an already validated
    /// [`AecConfig`](crate::AecConfig), so the derived tap count is well above
    /// zero in practice; a degenerate geometry still gets one tap rather than
    /// an empty filter.
    pub(crate) fn new(sample_rate: u32, tail_ms: u16) -> RhoCanceller {
        let taps = ((tail_ms as u64 * sample_rate as u64) / 1000).max(1) as usize;
        let hangover_samples = ((DTD_HANGOVER_MS as u64 * sample_rate as u64) / 1000) as u32;
        RhoCanceller {
            weights: vec![0.0; taps],
            history: vec![0.0; taps],
            // The first push advances to slot zero, so a fresh and a reset
            // instance start from the identical state.
            pos: taps - 1,
            hangover_remaining: 0,
            hangover_samples,
            near_power: 0.0,
            residual_power: 0.0,
            divergence_resets: 0,
        }
    }

    /// Processes one aligned sample pair: pushes the far-end sample into the
    /// history, subtracts the current echo estimate from the near-end sample,
    /// and adapts the filter unless the Geigel detector holds it frozen.
    ///
    /// Both inputs are already finite (the caller sanitizes). The output is
    /// always finite and every weight is finite on return: a non-finite echo
    /// estimate (possible only from pathological, huge-but-finite input
    /// overflowing the arithmetic) trips the divergence guard, which zeroes
    /// the filter, counts the reset, and emits silence for the sample.
    fn step(&mut self, near: f32, far: f32) -> f32 {
        let taps = self.history.len();
        self.pos = if self.pos + 1 == taps {
            0
        } else {
            self.pos + 1
        };
        self.history[self.pos] = far;

        // One pass over the history in fixed lag order (newest to oldest):
        // the echo estimate, the window energy for the normalized step, and
        // the peak magnitude for the Geigel detector. `f64` accumulators for
        // numerical depth; the order is fixed, so the sums are bit-stable.
        let (head, tail) = self.history.split_at(self.pos + 1);
        let mut estimate = 0.0_f64;
        let mut energy = 0.0_f64;
        let mut peak = 0.0_f32;
        for (&weight, &x) in self
            .weights
            .iter()
            .zip(head.iter().rev().chain(tail.iter().rev()))
        {
            estimate += weight as f64 * x as f64;
            energy += x as f64 * x as f64;
            peak = peak.max(x.abs());
        }

        let error = near as f64 - estimate;
        let output = error as f32;
        let diverged = !output.is_finite();
        let output = if diverged {
            // Divergence guard: never let a non-finite value persist. The
            // history is finite by construction, so zeroing the weights
            // restores the invariant; the sample is rendered silent.
            self.weights.fill(0.0);
            self.divergence_resets += 1;
            0.0
        } else {
            output
        };

        // Geigel double-talk detection: a near-end magnitude above half the
        // recent far-end peak cannot be explained by the echo path alone, so
        // it (re-)arms the adaptation freeze for the hangover span.
        if near.abs() > GEIGEL_THRESHOLD * peak {
            self.hangover_remaining = self.hangover_samples;
        } else if self.hangover_remaining > 0 {
            self.hangover_remaining -= 1;
        }

        // NLMS update, skipped while frozen or after a divergence reset (the
        // error that produced the reset is not a usable gradient). The updated
        // weights are checked in the same pass: any non-finite result zeroes
        // the filter, so the finite-weights invariant holds unconditionally.
        if self.hangover_remaining == 0 && !diverged {
            let scale = STEP_SIZE * error / (REGULARIZATION + energy);
            let (head, tail) = self.history.split_at(self.pos + 1);
            let mut all_finite = true;
            for (weight, &x) in self
                .weights
                .iter_mut()
                .zip(head.iter().rev().chain(tail.iter().rev()))
            {
                let updated = *weight + (scale * x as f64) as f32;
                *weight = updated;
                all_finite &= updated.is_finite();
            }
            if !all_finite {
                self.weights.fill(0.0);
                self.divergence_resets += 1;
            }
        }

        // ERLE bookkeeping: smoothed powers only; the dB conversion happens in
        // `metrics` on read, keeping the streaming state transcendental-free.
        self.near_power =
            ERLE_SMOOTHING * self.near_power + (1.0 - ERLE_SMOOTHING) * (near as f64 * near as f64);
        self.residual_power = ERLE_SMOOTHING * self.residual_power
            + (1.0 - ERLE_SMOOTHING) * (output as f64 * output as f64);

        output
    }
}

impl EchoCanceller for RhoCanceller {
    /// Cancels one aligned block sample by sample. Never returns an error
    /// after construction.
    ///
    /// The engine sanitizes upstream, but Rho re-sanitizes defensively: a
    /// non-finite input sample is treated as `0.0` before it can touch the
    /// history or an update, so even a consumer driving the trait directly
    /// with pathological data cannot poison the coefficient state, and the
    /// damage is bounded to the offending sample.
    fn process(&mut self, near: &[f32], far: &[f32], out: &mut Vec<f32>) -> Result<(), AecError> {
        debug_assert_eq!(
            near.len(),
            far.len(),
            "process requires equal-length aligned near and far blocks"
        );
        out.reserve(near.len());
        for (&near_raw, &far_raw) in near.iter().zip(far) {
            let near_sample = if near_raw.is_finite() { near_raw } else { 0.0 };
            let far_sample = if far_raw.is_finite() { far_raw } else { 0.0 };
            out.push(self.step(near_sample, far_sample));
        }
        Ok(())
    }

    /// Appends nothing: Rho is sample-in, sample-out, with no partial-block
    /// carry. Never returns an error after construction.
    fn flush(&mut self, _out: &mut Vec<f32>) -> Result<(), AecError> {
        Ok(())
    }

    /// Zero: a time-domain per-sample canceller introduces no algorithmic
    /// delay.
    fn latency_samples(&self) -> usize {
        0
    }

    /// Clears the filter, the history, the detector, and the ERLE state
    /// without reallocation, restoring the just-constructed state exactly.
    /// The divergence-reset count survives, as its metric is documented
    /// since construction.
    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.history.fill(0.0);
        self.pos = self.history.len() - 1;
        self.hangover_remaining = 0;
        self.near_power = 0.0;
        self.residual_power = 0.0;
    }

    fn metrics(&self) -> CancellerMetrics {
        let erle_db = if self.near_power > ERLE_POWER_FLOOR {
            let ratio = self.near_power / self.residual_power.max(ERLE_POWER_FLOOR);
            (10.0 * ratio.log10()) as f32
        } else {
            0.0
        };
        CancellerMetrics {
            erle_db,
            double_talk: self.hangover_remaining > 0,
            divergence_resets: self.divergence_resets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a whole aligned pair through a fresh instance and returns the
    /// output.
    fn run(rho: &mut RhoCanceller, near: &[f32], far: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        rho.process(near, far, &mut out)
            .expect("rho never fails after construction");
        rho.flush(&mut out).expect("rho flush never fails");
        out
    }

    #[test]
    fn construction_derives_the_tap_count_from_the_tail() {
        let rho = RhoCanceller::new(16_000, 200);
        assert_eq!(rho.weights.len(), 3200);
        assert_eq!(rho.history.len(), 3200);
        assert_eq!(rho.latency_samples(), 0);
        assert_eq!(rho.metrics(), CancellerMetrics::default());
    }

    #[test]
    fn silence_in_yields_exact_silence_out() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let out = run(&mut rho, &[0.0; 1024], &[0.0; 1024]);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    /// With a silent far end the echo estimate is exactly zero and the
    /// weights never move, so the near-end passes through bit-identically.
    #[test]
    fn a_silent_far_end_is_an_exact_passthrough() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let near: Vec<f32> = (0..1024)
            .map(|i| (((i % 41) as f32) / 20.0 - 1.0) * 0.3)
            .collect();
        let far = vec![0.0_f32; near.len()];
        let out = run(&mut rho, &near, &far);
        assert_eq!(out.len(), near.len());
        for (o, n) in out.iter().zip(&near) {
            assert_eq!(o.to_bits(), n.to_bits());
        }
    }

    #[test]
    fn near_end_activity_against_a_silent_far_end_reads_as_double_talk() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let _ = run(&mut rho, &[0.5; 64], &[0.0; 64]);
        assert!(rho.metrics().double_talk);
    }

    #[test]
    fn non_finite_input_yields_finite_output_and_finite_weights() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let near = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25, -0.5];
        let far = [0.5, f32::NAN, 0.25, f32::INFINITY, -0.25];
        let out = run(&mut rho, &near, &far);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(rho.weights.iter().all(|w| w.is_finite()));
    }

    /// Pathological huge-but-finite input can push the error past `f32::MAX`;
    /// the divergence guard must convert that into a counted, bounded reset
    /// with finite output and finite weights, never a poisoned filter.
    ///
    /// The sequence is engineered to actually reach the overflow: the filter
    /// first converges on a huge positive far end (near at 0.4 of the far
    /// peak, inside the Geigel single-talk regime, so adaptation runs), then
    /// the far end flips sign while the near end jumps to `f32::MAX`, so the
    /// learned estimate drives the error beyond the `f32` range.
    #[test]
    fn arithmetic_overflow_trips_the_divergence_guard() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let converge = vec![(0.4 * f32::MAX, f32::MAX); 512];
        let overflow = vec![(f32::MAX, -f32::MAX); 300];
        let near: Vec<f32> = converge.iter().chain(&overflow).map(|&(n, _)| n).collect();
        let far: Vec<f32> = converge.iter().chain(&overflow).map(|&(_, f)| f).collect();
        let out = run(&mut rho, &near, &far);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(rho.weights.iter().all(|w| w.is_finite()));
        assert!(rho.metrics().divergence_resets > 0);
    }

    /// After `reset` the canceller must reproduce a fresh instance's output
    /// bit for bit, while the divergence-reset count survives as documented.
    #[test]
    fn reset_restores_the_just_constructed_state() {
        let near: Vec<f32> = (0..512)
            .map(|i| (((i % 37) as f32) / 18.0 - 1.0) * 0.4)
            .collect();
        let far: Vec<f32> = (0..512)
            .map(|i| (((i % 29) as f32) / 14.0 - 1.0) * 0.6)
            .collect();

        let mut fresh = RhoCanceller::new(16_000, 16);
        let fresh_out = run(&mut fresh, &near, &far);

        let mut reused = RhoCanceller::new(16_000, 16);
        let _ = run(&mut reused, &far, &near);
        reused.reset();
        let reused_out = run(&mut reused, &near, &far);

        assert_eq!(fresh_out.len(), reused_out.len());
        for (a, b) in fresh_out.iter().zip(&reused_out) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn process_appends_without_clearing_and_flush_appends_nothing() {
        let mut rho = RhoCanceller::new(16_000, 16);
        let mut out = vec![7.0_f32];
        rho.process(&[0.1; 8], &[0.2; 8], &mut out)
            .expect("rho never fails after construction");
        assert_eq!(out.len(), 9);
        assert_eq!(out[0], 7.0);
        let len_before = out.len();
        rho.flush(&mut out).expect("rho flush never fails");
        assert_eq!(out.len(), len_before);
    }
}
