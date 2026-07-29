//! Tau: the shipped partitioned-block frequency-domain echo canceller.
//!
//! Tau is the crate's production model, the canceller [`AecModel::Tau`] selects.
//! It is a multi-delay block (MDF) adaptive filter: the modelled echo tail is
//! split into fixed-length partitions, each partition holds one block of the
//! filter's frequency response, and the echo estimate is the sum of those
//! partitions against a matching ring of far-end block spectra. The transform
//! underneath is the crate's own [`RealFft`], so every bit of the frequency
//! domain is owned.
//!
//! # Geometry
//!
//! The block length is [`BLOCK`] (256 samples, 16 ms at 16 kHz) and the
//! transform length is twice that, which is the standard overlap-save framing:
//! a length-512 real transform over a window holding the previous and the
//! current far-end block, of which only the second half of each inverse
//! transform is the valid linear convolution. The partition count follows from
//! the configured tail rather than the default: `ceil(taps / BLOCK)`, which is
//! 13 partitions for the default 200 ms tail at 16 kHz (3200 taps) and 2 for a
//! 32 ms tail.
//!
//! # The adaptive update
//!
//! Each block forms the error `e = near - estimate`, transforms the
//! zero-prefixed error block, and adds a normalized gradient to every
//! partition. The gradient for partition `p` is `conj(X_p) * E`, scaled per bin
//! by [`STEP_SIZE`] over the far-end power summed across partitions plus
//! [`REGULARIZATION`]: the frequency-domain analogue of a time-domain
//! normalized least-mean-squares step. The gradient is then *constrained*:
//! inverse transformed, its second half (the circular wrap-around lags, which
//! are not real filter taps) zeroed, and transformed back. The constrained
//! form is the one that makes the partitioned filter a true linear convolution
//! rather than a circular one.
//!
//! # The step size
//!
//! The normalized step is not a constant. Each bin's step is scaled by how its
//! current far-end power compares with its own recent average
//! ([`STEP_POWER_SMOOTHING`]), capped at [`STEP_SIZE`] and floored at
//! [`MIN_STEP_FRACTION`] of it. The factor is one for a stationary reference by
//! construction.
//!
//! # Double-talk
//!
//! The near-end judgement is made against the echo the filter itself predicts,
//! not against the far-end signal, and it is made on correlation rather than
//! on level. The statistic here is the squared normalized correlation between
//! the near-end block and the predicted echo, which is bounded in `[0, 1]` and
//! is unchanged by scaling either signal, so no gain anywhere in the path can
//! move it.
//!
//! What that coherence is compared against is tracked, not assumed. The
//! envelope the stream has been achieving is tracked asymmetrically
//! ([`DTD_BASELINE_ATTACK`], [`DTD_BASELINE_DECAY`]): it rises quickly and
//! decays slowly.
//!
//! How far below that envelope a block must fall to be called a talker is also
//! tracked, not assumed. The margin is [`DTD_MARGIN_DEVIATIONS`] times the
//! stream's own mean absolute shortfall from the envelope
//! ([`DTD_DEVIATION_SMOOTHING`], floored at [`DTD_DEVIATION_FLOOR`]).
//!
//! A drop past that margin enters a *collapse*, and the collapse holds only
//! while the drop stays unambiguously deep: at least [`DTD_HOLD_DEPTH`]
//! deviations below the envelope. The deviation itself is learned only outside
//! collapsed spans.
//!
//! Nothing is judged until a baseline is established, which happens as soon as
//! the filter demonstrably explains the microphone
//! ([`DTD_HANDOVER_COHERENCE`]) or, for a recording that never reaches it,
//! after [`DTD_ESTABLISH_BLOCKS`] blocks the far end has actually driven.
//!
//! # Graded adaptation and output protection
//!
//! The collapse verdict does not freeze the filter. The two jobs a single
//! verdict would otherwise serve are separated:
//!
//! - *Adaptation control* asks whether the current reference-to-microphone
//!   relationship is trustworthy enough for a filter update, and answers with
//!   a rate, not a verdict: the step is scaled by the smoothed coherence
//!   ([`COHERENCE_SMOOTHING`]) against [`FULL_SPEED_COHERENCE`], floored at
//!   [`STEP_SCALE_FLOOR`] and pinned to that floor during a collapse. A
//!   stream the filter explains adapts at full speed; a stream it explains
//!   poorly is learned cautiously; a stream it explains not at all is crawled
//!   toward so slowly that a talker is never audibly captured.
//! - *Output protection* asks whether the proposed output modification would
//!   damage near-end content, and stands the residual suppressor down (gain
//!   target of one) during a collapse or whenever the smoothed coherence sits
//!   under [`PROTECT_COHERENCE`], without inheriting anything from the
//!   adaptation decision.
//!
//! The `double_talk` metric reports the union: a block in a collapse or under
//! output protection is one the canceller treated as plausibly carrying a
//! near-end talker.
//!
//! # The two-path output
//!
//! Adaptation happens on a *shadow* filter that never drives the delivered
//! signal; the listener hears a separate *output* filter that moves only on
//! evidence. An adapting filter's intermediate states are conjectures, and on
//! a stream with no echo those conjectures are noise; the two-path structure
//! never lets an unproven filter state reach the output.
//!
//! Both filters run over the same far-end block ring; their block error
//! powers, smoothed over established far-active blocks
//! ([`PATH_ERROR_SMOOTHING`]), are directly comparable. The output filter
//! moves toward the shadow only when the comparison earns it, and the move
//! is a rate, not a copy:
//!
//! - *Persistence*: no transfer until [`TRANSFER_QUIET_STREAK_BLOCKS`]
//!   consecutive far-active blocks have carried neither a collapse nor an
//!   output-protection verdict.
//! - *Advantage*: the shadow must be removing at least
//!   [`TRANSFER_MIN_ADVANTAGE`] of the output filter's error, and the
//!   transfer rate rises linearly from there to [`TRANSFER_RATE_MAX`] at
//!   [`TRANSFER_FULL_ADVANTAGE`]. On a no-echo stream the shadow never
//!   clears the floor, so the output filter stays zero and the delivered
//!   signal is the microphone untouched.
//!
//! The reverse copy exists too: a shadow sustaining [`RESCUE_RATIO`] times the
//! output filter's error for [`RESCUE_STREAK_BLOCKS`] blocks is rewound to the
//! output filter rather than left to relearn from zero.
//!
//! # Divergence
//!
//! A canceller removes energy. One that adds energy has diverged, whether or
//! not its coefficients are still finite, and finite divergence is the case a
//! finiteness check cannot see: coefficients grow large but valid, the output
//! grows past the input, and nothing reports it. The smoothed output power is
//! therefore held against the smoothed near-end power
//! ([`DIVERGENCE_SMOOTHING`], [`DIVERGENCE_RATIO`]), and an output filter
//! that sustains an output above its input is halved, counted, and re-earns
//! its correction through the transfer, with the block that tripped the
//! guard delivered as the near end it was given. Both powers are of the same
//! signal, so the test is a ratio and holds at any level. The shadow has its
//! own recovery paths: a
//! non-finite state zeroes it (the output filter is unaffected and the
//! listener never hears it), and a sustained rescue rewinds it to the output
//! filter.
//!
//! # Non-finite input
//!
//! The engine sanitizes upstream, and Tau re-sanitizes defensively: a
//! non-finite input sample is treated as `0.0` before it can reach the
//! transform, the filter state, or an update. A non-finite value that arises
//! from pathological but finite input trips the divergence guard, which zeroes
//! the filter, counts the reset, and emits silence for the block, so the
//! coefficient state of both filters is finite after every processed block
//! without exception.
//!
//! # Determinism
//!
//! The same input produces bit-identical output on every platform, every run,
//! and both supported toolchains:
//!
//! - Every loop iterates a fixed range in a fixed order: partitions ascending,
//!   then bins ascending, then samples ascending. No floating-point reduction
//!   ever reorders, and there are no unordered containers, no threads, no time,
//!   and no randomness.
//! - The block framing is independent of how the caller chunks the stream, so
//!   the arithmetic is identical whether a second of audio arrives in one call
//!   or in sixty.
//! - The arithmetic is IEEE-exact `+`, `-`, `*`, `/`, `abs`, and comparisons
//!   only. Complex products are written as independent products per component,
//!   the way [`crate::fft`] writes them, so no fused multiply-add can arise on
//!   any target, and rustc applies no fast-math reassociation at any
//!   optimization level.
//! - The only transcendentals in reach are the transform's twiddle factors,
//!   which [`RealFft`] captures once at construction from its own deterministic
//!   power-series trig. Tau adds none: the one `log10` runs in [`metrics`] on
//!   read and never feeds back into streaming state.
//!
//! [`AecModel::Tau`]: crate::AecModel::Tau
//! [`metrics`]: EchoCanceller::metrics

use crate::canceller::{CancellerMetrics, EchoCanceller};
use crate::config::Suppression;
use crate::error::AecError;
use crate::fft::{Complex, RealFft};

/// The processing block length in samples: the number of new far-end and
/// near-end samples consumed per frequency-domain turn, and the length of one
/// filter partition. 256 samples is 16 ms at 16 kHz.
pub(crate) const BLOCK: usize = 256;

/// The real transform length: twice [`BLOCK`], the overlap-save window holding
/// the previous and the current far-end block.
const FFT_LEN: usize = 2 * BLOCK;

/// The number of independent complex bins in a length-[`FFT_LEN`] real
/// transform: DC through Nyquist.
const BINS: usize = FFT_LEN / 2 + 1;

/// The maximum frequency-domain step size (mu), the value the per-bin step
/// takes wherever the reference is at or above its own recent level.
const STEP_SIZE: f64 = 1.0;

/// One-pole smoothing coefficient for the per-bin far-end power the step size
/// is judged against, applied per adapting block.
const STEP_POWER_SMOOTHING: f64 = 0.98;

/// The floor the per-bin step factor is held at, as a fraction of
/// [`STEP_SIZE`].
const MIN_STEP_FRACTION: f64 = 0.02;

/// How many of the stream's own mean absolute deviations below the coherence
/// envelope a block must fall to enter a collapse.
const DTD_MARGIN_DEVIATIONS: f64 = 3.0;

/// One-pole smoothing coefficient for the mean absolute shortfall of the
/// block coherence from its envelope, learned on far-active blocks outside
/// collapsed spans.
const DTD_DEVIATION_SMOOTHING: f64 = 0.95;

/// The floor under the tracked deviation.
const DTD_DEVIATION_FLOOR: f64 = 0.01;

/// The depth, in tracked deviations below the envelope, a collapse must
/// sustain to stay held.
const DTD_HOLD_DEPTH: f64 = 10.0;

/// One-pole smoothing coefficient for the coherence the adaptation scale and
/// the output-protection verdict read, applied per far-active block and
/// symmetric in both directions.
const COHERENCE_SMOOTHING: f64 = 0.8;

/// The smoothed coherence at which adaptation runs at full speed; below it
/// the step scales down linearly.
const FULL_SPEED_COHERENCE: f64 = 0.5;

/// The floor under the adaptation scale, and the scale a collapse pins.
const STEP_SCALE_FLOOR: f64 = 0.01;

/// The smoothed coherence below which output protection engages regardless of
/// the collapse state.
const PROTECT_COHERENCE: f64 = 0.2;

/// One-pole smoothing coefficient for the block error powers the two-path
/// comparison reads, applied per established far-active block.
const PATH_ERROR_SMOOTHING: f64 = 0.9;

/// The least share of the output filter's smoothed error the shadow must be
/// removing before any transfer happens.
const TRANSFER_MIN_ADVANTAGE: f64 = 0.1;

/// The advantage at which the transfer runs at its full rate.
const TRANSFER_FULL_ADVANTAGE: f64 = 0.5;

/// The most the output filter moves toward the shadow on one block: three
/// quarters of the remaining difference, reached at full advantage.
const TRANSFER_RATE_MAX: f64 = 0.75;

/// Consecutive quiet established far-active blocks required before any
/// transfer: 128 ms with neither a collapse nor output protection.
const TRANSFER_QUIET_STREAK_BLOCKS: u32 = 8;

/// How far above the output filter's smoothed error power the shadow's must
/// sit to qualify a block toward a rescue: four times.
const RESCUE_RATIO: f64 = 4.0;

/// Consecutive rescue-qualifying blocks before the shadow is rewound to the
/// output filter. Mirrors [`TRANSFER_QUIET_STREAK_BLOCKS`].
const RESCUE_STREAK_BLOCKS: u32 = 8;

/// The coherence at which the filter is trusted immediately, without waiting
/// out [`DTD_ESTABLISH_BLOCKS`].
const DTD_HANDOVER_COHERENCE: f64 = 0.81;

/// Far-active blocks after which the detector is trusted with whatever
/// coherence it has managed, for a recording that never reaches
/// [`DTD_HANDOVER_COHERENCE`] at all. Counted in far-active blocks and never in
/// elapsed time.
///
/// `pub(crate)` for the internal suite.
pub(crate) const DTD_ESTABLISH_BLOCKS: u32 = 32;

/// One-pole coefficient for the coherence baseline when the observed coherence
/// is above it.
const DTD_BASELINE_ATTACK: f64 = 0.5;

/// One-pole coefficient for the coherence baseline when the observed coherence
/// is below it.
const DTD_BASELINE_DECAY: f64 = 0.99;

/// The share of the running peak block energy a block's far-end energy must
/// carry to count as one the far end actively drove. Relative to the reference's
/// own history, so it is a measure of activity and not of level.
const FAR_ACTIVITY_FRACTION: f64 = 1e-4;

/// One-pole smoothing coefficient for the divergence guard's power estimates,
/// applied per block.
const DIVERGENCE_SMOOTHING: f64 = 0.98;

/// How far the smoothed output power may sit above the smoothed near-end power
/// before the filter is judged to have diverged. A canceller that sustains more
/// energy out than in is adding to the microphone signal, which no correct
/// filter state does.
const DIVERGENCE_RATIO: f64 = 1.05;

/// Regularization added to the per-bin far-end power in the normalized step, so
/// a near-silent reference cannot blow the update up.
const REGULARIZATION: f64 = 1e-3;

/// One-pole smoothing coefficient for the ERLE power estimates, applied per
/// sample.
const ERLE_SMOOTHING: f64 = 0.999;

/// Power floor for the ERLE estimate: below this smoothed near-end power the
/// estimate reads zero, and the residual power is floored by it so the reported
/// ratio stays finite.
const ERLE_POWER_FLOOR: f64 = 1e-10;

/// One-pole smoothing coefficient for the residual suppressor's per-bin power
/// estimates, applied per block.
const SUPPRESSION_SMOOTHING: f64 = 0.7;

/// The residual echo the suppressor assumes the linear filter leaves, as a
/// fraction of the echo estimate's power.
const RESIDUAL_ECHO_LEAK: f64 = 0.0003;

/// The suppressor's gain floor: the most it may ever attenuate a bin.
const MIN_SUPPRESSION_GAIN: f64 = 0.1;

/// One-pole smoothing coefficient for the suppressor's gain itself, applied per
/// block, so the gain follows the signal rather than jumping block to block.
const GAIN_SMOOTHING: f64 = 0.5;

/// One-pole smoothing coefficient for the suppressor's gain when it is backing
/// off toward unity. Lower than [`GAIN_SMOOTHING`], so the stage releases
/// faster than it engages.
const GAIN_RELEASE_SMOOTHING: f64 = 0.2;

/// Half-width, in bins, of the moving average the suppressor runs across its
/// gain curve before applying it.
const GAIN_BIN_SPAN: usize = 4;

/// The shipped partitioned-block frequency-domain canceller. See the module
/// documentation for the geometry, the update, and the determinism guarantees.
pub(crate) struct TauCanceller {
    /// The owned length-[`FFT_LEN`] real transform, built once at construction.
    fft: RealFft,
    /// The number of filter partitions, `ceil(taps / BLOCK)` from the tail.
    partitions: usize,
    /// The shadow filter: the frequency-domain filter adaptation runs on,
    /// `partitions * BINS` coefficients, partition major. Partition `p` covers
    /// taps `[p * BLOCK, (p + 1) * BLOCK)`. It never drives the delivered
    /// output directly; it reaches the listener only by being promoted onto
    /// [`output_weights`](Self::output_weights) on sustained evidence.
    /// Invariant: every coefficient is finite after every processed block.
    shadow_weights: Vec<Complex>,
    /// The output filter: the filter the delivered signal actually runs
    /// through, same geometry as the shadow. Never adapted in place; it
    /// changes only when the shadow is promoted onto it, so between
    /// promotions it is exactly static and cannot wander on clean audio or
    /// chase a talker through double-talk.
    /// Invariant: every coefficient is finite after every processed block.
    output_weights: Vec<Complex>,
    /// A ring of the last `partitions` far-end block spectra, partition major.
    /// Slot [`head`](Self::head) holds the newest.
    far_spectra: Vec<Complex>,
    /// Index in [`far_spectra`](Self::far_spectra) of the newest block spectrum.
    head: usize,
    /// The overlap-save far-end window: the previous block followed by the
    /// current one.
    far_window: Vec<f32>,
    /// Near-end samples accumulated toward the next full block. Shorter than
    /// [`BLOCK`] between calls.
    near_carry: Vec<f32>,
    /// Far-end samples accumulated toward the next full block, in step with
    /// [`near_carry`](Self::near_carry).
    far_carry: Vec<f32>,
    /// Whether the block just processed was judged to plausibly carry a
    /// near-end talker (a collapse, or output protection engaged): the
    /// `double_talk` metric, which a caller reads between blocks.
    double_talk: bool,
    /// The coherence envelope the filter has been achieving, which the
    /// detector judges a block against. Zero until the first block that
    /// carries a prediction.
    dtd_baseline: f64,
    /// The stream's mean absolute shortfall from that envelope, the unit the
    /// collapse margin and hold depth are measured in. Learned on far-active
    /// blocks outside collapsed spans.
    dtd_deviation: f64,
    /// Whether the stream is currently in a held coherence collapse.
    dtd_collapsed: bool,
    /// The smoothed coherence the adaptation scale and the output-protection
    /// verdict read.
    coherence_smooth: f64,
    /// The adaptation step scale for the block in hand, in
    /// [[`STEP_SCALE_FLOOR`], 1].
    step_scale: f64,
    /// Whether output protection engages for the block in hand.
    protect_output: bool,
    /// Whether the coherence baseline is established, after which the detector
    /// is allowed to judge blocks.
    dtd_established: bool,
    /// Far-active blocks seen while the baseline establishes.
    dtd_blocks: u32,
    /// The block's shadow-filter echo estimate, held so the detector can read
    /// it while the transform scratch it came from is reused by the update.
    estimate_block: Vec<f32>,
    /// The shadow filter's error for the current block, the adaptation
    /// gradient's source. Distinct from
    /// [`error_block`](Self::error_block), which is the output filter's error
    /// and the signal actually delivered.
    shadow_error_block: Vec<f32>,
    /// Smoothed block error power of the shadow filter over established
    /// far-active blocks, one side of the two-path comparison.
    shadow_error_power: f64,
    /// Smoothed block error power of the output filter over the same blocks,
    /// the other side of the comparison.
    output_error_power: f64,
    /// Consecutive quiet (no collapse, no protection) established far-active
    /// blocks, the transfer's persistence gate.
    quiet_streak: u32,
    /// Consecutive qualifying blocks toward a shadow rescue.
    rescue_streak: u32,
    /// Blocks on which the output filter moved toward the shadow, since
    /// construction, for the internal suite.
    transfers: u64,
    /// The largest single-block far-end energy seen, which
    /// [`FAR_ACTIVITY_FRACTION`] is taken against.
    far_energy_peak: f64,
    /// Smoothed per-bin far-end power the step size is judged against.
    far_power_mean: Vec<f64>,
    /// The per-bin step size for the block in hand.
    step: Vec<f64>,
    /// Smoothed near-end block power for the divergence guard.
    divergence_near_power: f64,
    /// Smoothed output block power for the divergence guard.
    divergence_out_power: f64,
    /// Smoothed near-end power for the ERLE estimate.
    near_power: f64,
    /// Smoothed residual power for the ERLE estimate.
    residual_power: f64,
    /// Times the divergence guard has zeroed the filter since construction.
    /// Deliberately survives [`reset`](EchoCanceller::reset), as the metric is
    /// documented as a since-construction count.
    divergence_resets: u64,
    /// Scratch for one transform result, reused every block.
    spectrum_scratch: Vec<Complex>,
    /// The echo estimate spectrum for the current block.
    estimate_spectrum: Vec<Complex>,
    /// The zero-prefixed error block's spectrum for the current block.
    error_spectrum: Vec<Complex>,
    /// One partition's normalized gradient, before and after the constraint.
    gradient: Vec<Complex>,
    /// Per-bin far-end power summed across the partitions, the normalized
    /// step's denominator.
    far_power: Vec<f64>,
    /// Scratch for an inverse transform result, reused every block.
    time_scratch: Vec<f32>,
    /// Scratch for a transform input, reused every block.
    window_scratch: Vec<f32>,
    /// The current block's error samples, the linear filter's output.
    error_block: Vec<f32>,
    /// The current block's delivered output samples.
    output_block: Vec<f32>,
    /// The configured residual suppression. [`Suppression::Off`] delivers the
    /// linear filter's error untouched.
    suppression: Suppression,
    /// Whether any far-end block has carried energy. Until one has, there is no
    /// echo to suppress and the suppressor is bypassed entirely, which is what
    /// makes a never-active far end a bit-exact passthrough.
    far_active: bool,
    /// Smoothed per-bin power of the echo estimate, the suppressor's handle on
    /// how much echo the block contained.
    echo_power: Vec<f64>,
    /// Smoothed per-bin power of the linear filter's error.
    error_power: Vec<f64>,
    /// The suppressor's overlap-save error window: the previous error block
    /// followed by the current one, so the gain is applied as a filter over a
    /// continuous stream rather than to an isolated block.
    error_window: Vec<f32>,
    /// The spectrum of [`error_window`](Self::error_window), the signal the
    /// suppressor's gain multiplies. Distinct from
    /// [`error_spectrum`](Self::error_spectrum), which is zero-prefixed because
    /// the adaptive update needs that framing for its correlation lags.
    error_window_spectrum: Vec<Complex>,
    /// The suppressor's smoothed per-bin gain, one for an untouched bin.
    gain: Vec<f64>,
    /// The gain curve after the across-bin moving average, the curve actually
    /// applied.
    smoothed_gain: Vec<f64>,
    /// The error spectrum after the suppressor's gain, before its inverse
    /// transform.
    suppressed_spectrum: Vec<Complex>,
}

impl TauCanceller {
    /// Constructs the canceller for a validated geometry: a `tail_ms` tail at
    /// `sample_rate`, partitioned into [`BLOCK`]-sample blocks.
    ///
    /// The caller passes fields from an already validated
    /// [`AecConfig`](crate::AecConfig), so the derived tap count is well above
    /// zero in practice; a degenerate geometry still gets one partition rather
    /// than an empty filter.
    pub(crate) fn new(sample_rate: u32, tail_ms: u16, suppression: Suppression) -> TauCanceller {
        let taps = ((tail_ms as u64 * sample_rate as u64) / 1000).max(1) as usize;
        let partitions = taps.div_ceil(BLOCK).max(1);
        TauCanceller {
            fft: RealFft::new(FFT_LEN),
            partitions,
            shadow_weights: vec![Complex::new(0.0, 0.0); partitions * BINS],
            output_weights: vec![Complex::new(0.0, 0.0); partitions * BINS],
            far_spectra: vec![Complex::new(0.0, 0.0); partitions * BINS],
            // The first block advances to slot zero, so a fresh and a reset
            // instance start from the identical state.
            head: partitions - 1,
            far_window: vec![0.0; FFT_LEN],
            near_carry: Vec::with_capacity(BLOCK),
            far_carry: Vec::with_capacity(BLOCK),
            double_talk: false,
            dtd_baseline: 0.0,
            dtd_deviation: 0.0,
            dtd_collapsed: false,
            coherence_smooth: 0.0,
            step_scale: 1.0,
            protect_output: false,
            dtd_established: false,
            dtd_blocks: 0,
            estimate_block: vec![0.0; BLOCK],
            shadow_error_block: vec![0.0; BLOCK],
            shadow_error_power: 0.0,
            output_error_power: 0.0,
            quiet_streak: 0,
            rescue_streak: 0,
            transfers: 0,
            far_energy_peak: 0.0,
            far_power_mean: vec![0.0; BINS],
            step: vec![STEP_SIZE; BINS],
            divergence_near_power: 0.0,
            divergence_out_power: 0.0,
            near_power: 0.0,
            residual_power: 0.0,
            divergence_resets: 0,
            spectrum_scratch: vec![Complex::new(0.0, 0.0); BINS],
            estimate_spectrum: vec![Complex::new(0.0, 0.0); BINS],
            error_spectrum: vec![Complex::new(0.0, 0.0); BINS],
            gradient: vec![Complex::new(0.0, 0.0); BINS],
            far_power: vec![0.0; BINS],
            time_scratch: vec![0.0; FFT_LEN],
            window_scratch: vec![0.0; FFT_LEN],
            error_block: vec![0.0; BLOCK],
            output_block: vec![0.0; BLOCK],
            suppression,
            far_active: false,
            echo_power: vec![0.0; BINS],
            error_power: vec![0.0; BINS],
            error_window: vec![0.0; FFT_LEN],
            error_window_spectrum: vec![Complex::new(0.0, 0.0); BINS],
            gain: vec![1.0; BINS],
            smoothed_gain: vec![1.0; BINS],
            suppressed_spectrum: vec![Complex::new(0.0, 0.0); BINS],
        }
    }

    /// The number of filter partitions the configured tail produced.
    #[cfg(test)]
    pub(crate) fn partitions(&self) -> usize {
        self.partitions
    }

    /// Transfer blocks since construction: how many blocks the output filter
    /// has moved toward the shadow on. Zero on a stream where the shadow
    /// never earned the output, which is the two-path structure's do-no-harm
    /// property in one number.
    #[cfg(test)]
    pub(crate) fn transfers(&self) -> u64 {
        self.transfers
    }

    /// Zeroes the shadow filter and counts a divergence reset: the recovery
    /// path for a shadow that has gone non-finite while the output filter is
    /// still sound. The output path is untouched, so the listener never hears
    /// the reset; the shadow relearns and earns its way back by transfer.
    fn diverge_shadow(&mut self) {
        self.shadow_weights.fill(Complex::new(0.0, 0.0));
        self.shadow_error_power = self.output_error_power;
        self.rescue_streak = 0;
        self.divergence_resets += 1;
    }

    /// Halves the output filter and counts a divergence reset: the recovery
    /// path for an output filter that sustains more energy out than in. The
    /// shadow is deliberately untouched.
    fn retire_output(&mut self) {
        for w in self.output_weights.iter_mut() {
            w.re *= 0.5;
            w.im *= 0.5;
        }
        // The suppressor's power estimates are derived from the output
        // filter, so they restart with it.
        self.echo_power.fill(0.0);
        self.error_power.fill(0.0);
        self.gain.fill(1.0);
        self.smoothed_gain.fill(1.0);
        // A retired output's error is judged afresh; the tracker is seeded at
        // the guard's near power while the smoothing re-tracks.
        self.output_error_power = self.divergence_near_power;
        self.rescue_streak = 0;
        self.divergence_resets += 1;
    }

    /// Zeroes both filters and counts a divergence reset, the recovery path
    /// from a compromised output state: a non-finite delivered error.
    fn diverge(&mut self) {
        self.shadow_weights.fill(Complex::new(0.0, 0.0));
        self.output_weights.fill(Complex::new(0.0, 0.0));
        // The suppressor's power estimates are derived from the filter, so a
        // non-finite filter can have poisoned them too; clearing them restores
        // an untouched gain alongside a cleared filter.
        self.echo_power.fill(0.0);
        self.error_power.fill(0.0);
        self.gain.fill(1.0);
        self.smoothed_gain.fill(1.0);
        self.shadow_error_power = 0.0;
        self.output_error_power = 0.0;
        self.rescue_streak = 0;
        self.divergence_resets += 1;
    }

    /// The per-block coherence judgement: how much of the microphone block the
    /// filter's own echo prediction accounts for, against how much it has been
    /// accounting for. Returns whether the stream is in a held collapse, and
    /// leaves the block's adaptation scale in [`step_scale`](Self::step_scale)
    /// and the output-protection verdict in
    /// [`protect_output`](Self::protect_output).
    ///
    /// The statistic is the squared normalized correlation between the near-end
    /// block and the predicted echo,
    ///
    /// ```text
    /// coherence = <near, estimate>^2 / (E_near * E_estimate)
    /// ```
    ///
    /// which is bounded in `[0, 1]` and, being normalized on both sides, is
    /// unchanged by scaling either signal. The judgement waits for a baseline
    /// to establish, and adaptation runs at full speed until it has. See the
    /// module documentation for the collapse and hold rules.
    fn double_talk_block(
        &mut self,
        near: &[f32],
        near_energy: f64,
        estimate: &[f32],
        estimate_energy: f64,
        far_is_active: bool,
    ) -> bool {
        if near_energy <= 0.0 || estimate_energy <= 0.0 {
            // No coherence is measurable on this block; the scale and the
            // protection verdict carry over from the last measurable one.
            return false;
        }
        let mut dot = 0.0_f64;
        for (&n, &e) in near.iter().zip(estimate) {
            dot += n as f64 * e as f64;
        }
        // A negative correlation means the prediction is out of phase with the
        // microphone, which no amount of echo explains.
        let coherence = if dot > 0.0 {
            (dot * dot) / (near_energy * estimate_energy)
        } else {
            0.0
        };

        if !self.dtd_established {
            if far_is_active {
                self.track_coherence_baseline(coherence);
                // Seed the deviation and the smoothed coherence during
                // establishment, so the margin already reflects this stream's
                // own noisiness the moment the detector is trusted.
                let shortfall = (coherence - self.dtd_baseline).abs();
                self.dtd_deviation = DTD_DEVIATION_SMOOTHING * self.dtd_deviation
                    + (1.0 - DTD_DEVIATION_SMOOTHING) * shortfall;
                self.coherence_smooth = COHERENCE_SMOOTHING * self.coherence_smooth
                    + (1.0 - COHERENCE_SMOOTHING) * coherence;
                self.dtd_blocks += 1;
                if coherence >= DTD_HANDOVER_COHERENCE || self.dtd_blocks >= DTD_ESTABLISH_BLOCKS {
                    self.dtd_established = true;
                }
            }
            self.step_scale = 1.0;
            self.protect_output = self.coherence_smooth < PROTECT_COHERENCE;
            return false;
        }

        // Whether this block falls past the margin, judged against the
        // envelope and deviation as they stood before this block.
        let entered = coherence
            < self.dtd_baseline
                - DTD_MARGIN_DEVIATIONS * self.dtd_deviation.max(DTD_DEVIATION_FLOOR);
        if far_is_active {
            self.track_coherence_baseline(coherence);
        }

        // The collapse state machine: enter on a drop past the margin, hold
        // only while the drop stays unambiguously deep, release otherwise.
        if !self.dtd_collapsed {
            if entered {
                self.dtd_collapsed = true;
            }
        } else {
            let depth =
                (self.dtd_baseline - coherence) / self.dtd_deviation.max(DTD_DEVIATION_FLOOR);
            if depth < DTD_HOLD_DEPTH {
                self.dtd_collapsed = false;
            }
        }

        // The deviation learns the stream's ordinary shortfall only outside
        // collapsed spans.
        if far_is_active && !self.dtd_collapsed {
            let shortfall = (coherence - self.dtd_baseline).abs();
            self.dtd_deviation = DTD_DEVIATION_SMOOTHING * self.dtd_deviation
                + (1.0 - DTD_DEVIATION_SMOOTHING) * shortfall;
        }
        if far_is_active {
            self.coherence_smooth = COHERENCE_SMOOTHING * self.coherence_smooth
                + (1.0 - COHERENCE_SMOOTHING) * coherence;
        }

        // The two decoupled outputs.
        let scale = (self.coherence_smooth / FULL_SPEED_COHERENCE).clamp(STEP_SCALE_FLOOR, 1.0);
        self.step_scale = if self.dtd_collapsed {
            STEP_SCALE_FLOOR
        } else {
            scale
        };
        self.protect_output = self.dtd_collapsed || self.coherence_smooth < PROTECT_COHERENCE;
        self.dtd_collapsed
    }

    /// Moves the coherence baseline toward an observation: quickly upward and
    /// slowly downward.
    fn track_coherence_baseline(&mut self, coherence: f64) {
        if self.dtd_baseline <= 0.0 {
            self.dtd_baseline = coherence;
        } else if coherence > self.dtd_baseline {
            self.dtd_baseline =
                DTD_BASELINE_ATTACK * self.dtd_baseline + (1.0 - DTD_BASELINE_ATTACK) * coherence;
        } else {
            self.dtd_baseline =
                DTD_BASELINE_DECAY * self.dtd_baseline + (1.0 - DTD_BASELINE_DECAY) * coherence;
        }
    }

    /// Processes one full [`BLOCK`]-sample aligned pair, leaving the delivered
    /// samples in [`output_block`](Self::output_block).
    ///
    /// `adapt` is false for the end-of-stream flush block, whose far-end and
    /// near-end tails are zero-padded and therefore not a usable gradient.
    /// `valid` is how many leading samples of the block the caller actually
    /// supplied: [`BLOCK`] for a streaming block, and the shorter remainder for
    /// that flush block. The filter arithmetic always spans the whole block
    /// (the transform is defined on it), but the per-sample detector and ERLE
    /// bookkeeping run only over the supplied samples, so fabricated padding
    /// never reaches a metric or the detector's state.
    fn run_block(&mut self, near: &[f32], far: &[f32], valid: usize, adapt: bool) {
        debug_assert_eq!(near.len(), BLOCK);
        debug_assert_eq!(far.len(), BLOCK);
        let partitions = self.partitions;

        // Slide the overlap-save window: the previous block becomes the first
        // half, the new block the second.
        self.far_window.copy_within(BLOCK.., 0);
        self.far_window[BLOCK..].copy_from_slice(far);

        // Whether the block carried any far-end energy at all.
        for &sample in far {
            if sample != 0.0 {
                self.far_active = true;
            }
        }

        // Advance the block ring and record the current spectrum.
        self.head = if self.head + 1 == partitions {
            0
        } else {
            self.head + 1
        };
        self.fft
            .forward(&self.far_window, &mut self.spectrum_scratch);
        let head_base = self.head * BINS;
        self.far_spectra[head_base..head_base + BINS].copy_from_slice(&self.spectrum_scratch);

        // The shadow filter's echo estimate spectrum: the partitioned filter
        // against the block ring, accumulated in a fixed partition-then-bin
        // order. This estimate feeds the detector and the adaptive update; it
        // never reaches the delivered output.
        self.spectrum_scratch.fill(Complex::new(0.0, 0.0));
        for partition in 0..partitions {
            let slot = (self.head + partitions - partition) % partitions;
            let spectra_base = slot * BINS;
            let weight_base = partition * BINS;
            for bin in 0..BINS {
                let x = self.far_spectra[spectra_base + bin];
                let w = self.shadow_weights[weight_base + bin];
                let acc = &mut self.spectrum_scratch[bin];
                acc.re += w.re * x.re - w.im * x.im;
                acc.im += w.re * x.im + w.im * x.re;
            }
        }

        // Overlap-save: only the second half of the inverse transform is the
        // valid linear convolution for this block's new samples.
        self.fft
            .inverse(&self.spectrum_scratch, &mut self.time_scratch);
        self.estimate_block
            .copy_from_slice(&self.time_scratch[BLOCK..]);

        let mut shadow_diverged = false;
        for ((slot, &near_sample), &estimate) in self
            .shadow_error_block
            .iter_mut()
            .zip(near.iter())
            .zip(self.estimate_block.iter())
        {
            let error = near_sample - estimate;
            if !error.is_finite() {
                shadow_diverged = true;
            }
            *slot = error;
        }
        if shadow_diverged {
            // Shadow divergence guard: never let a non-finite value persist.
            // The far-end window is finite by construction, so zeroing the
            // shadow restores the invariant; the output filter is unaffected
            // and the delivered block is untouched by the reset.
            self.diverge_shadow();
            self.estimate_block.fill(0.0);
            self.shadow_error_block.copy_from_slice(near);
        }

        // The output filter's echo estimate spectrum, over the same block
        // ring. This is the estimate the listener's signal is corrected by,
        // and the one the residual suppressor sizes the leftover echo from.
        self.estimate_spectrum.fill(Complex::new(0.0, 0.0));
        for partition in 0..partitions {
            let slot = (self.head + partitions - partition) % partitions;
            let spectra_base = slot * BINS;
            let weight_base = partition * BINS;
            for bin in 0..BINS {
                let x = self.far_spectra[spectra_base + bin];
                let w = self.output_weights[weight_base + bin];
                let acc = &mut self.estimate_spectrum[bin];
                acc.re += w.re * x.re - w.im * x.im;
                acc.im += w.re * x.im + w.im * x.re;
            }
        }
        self.fft
            .inverse(&self.estimate_spectrum, &mut self.time_scratch);

        let mut diverged = false;
        for ((slot, &near_sample), &estimate) in self
            .error_block
            .iter_mut()
            .zip(near.iter())
            .zip(self.time_scratch[BLOCK..].iter())
        {
            let error = near_sample - estimate;
            if !error.is_finite() {
                diverged = true;
            }
            *slot = error;
        }
        if diverged {
            // Output divergence guard: a non-finite delivered error means the
            // output path itself is compromised, and both filters restart;
            // the block is rendered silent.
            self.diverge();
            self.error_block.fill(0.0);
        }

        // Whether the far end actually drove this block, measured against its
        // own running peak so the gate is a measure of activity rather than of
        // level. This is what the detector's bootstrap and the coupling tracker
        // are counted and gated on.
        let mut block_energy = 0.0_f64;
        for &sample in far {
            block_energy += sample as f64 * sample as f64;
        }
        if block_energy > self.far_energy_peak {
            self.far_energy_peak = block_energy;
        }
        let far_is_active = self.far_energy_peak > 0.0
            && block_energy > FAR_ACTIVITY_FRACTION * self.far_energy_peak;

        // The double-talk decision, on how much of this block the filter's own
        // prediction accounts for.
        let mut near_energy = 0.0_f64;
        for &sample in &near[..valid] {
            near_energy += sample as f64 * sample as f64;
        }
        // The detector judges the shadow filter's prediction: adaptation
        // control and output protection both ask how much of the microphone
        // the learning filter explains, and the shadow is the filter that
        // learns.
        let mut estimate_energy = 0.0_f64;
        for &sample in &self.estimate_block[..valid] {
            estimate_energy += sample as f64 * sample as f64;
        }
        let estimate = std::mem::take(&mut self.estimate_block);
        let triggered = self.double_talk_block(
            &near[..valid],
            near_energy,
            &estimate[..valid],
            estimate_energy,
            far_is_active,
        );
        self.estimate_block = estimate;

        // A held collapse is its own persistence: it lasts exactly as long as
        // the coherence stays unambiguously collapsed, so no hangover count is
        // layered on top of it. The metric reports the union of the collapse
        // and the output-protection verdict, which is every block the
        // canceller treated as plausibly carrying a near-end talker.
        let collapsed = triggered;
        self.double_talk = collapsed || self.protect_output;

        // The two-path comparison: whether the shadow has earned the output.
        // Both filters saw the same block, so their error powers are directly
        // comparable, and each is smoothed over established far-active blocks
        // only. The transfer needs both persistence (a quiet streak) and
        // advantage (the shadow demonstrably cancelling what the output
        // filter does not), and it moves at a rate graded by that advantage.
        // A rescue is the reverse copy: a shadow persistently far behind the
        // output filter restarts from the best known state.
        let two_path = adapt && !diverged && far_is_active && self.dtd_established;
        let mut rescued = false;
        if two_path {
            let mut shadow_error_energy = 0.0_f64;
            for &sample in &self.shadow_error_block[..valid] {
                shadow_error_energy += sample as f64 * sample as f64;
            }
            let mut output_error_energy = 0.0_f64;
            for &sample in &self.error_block[..valid] {
                output_error_energy += sample as f64 * sample as f64;
            }
            self.shadow_error_power = PATH_ERROR_SMOOTHING * self.shadow_error_power
                + (1.0 - PATH_ERROR_SMOOTHING) * shadow_error_energy;
            self.output_error_power = PATH_ERROR_SMOOTHING * self.output_error_power
                + (1.0 - PATH_ERROR_SMOOTHING) * output_error_energy;

            let quiet = !self.dtd_collapsed && !self.protect_output;
            self.quiet_streak = if quiet { self.quiet_streak + 1 } else { 0 };
            let rescue_q = self.shadow_error_power > RESCUE_RATIO * self.output_error_power;
            self.rescue_streak = if rescue_q { self.rescue_streak + 1 } else { 0 };

            if self.rescue_streak >= RESCUE_STREAK_BLOCKS {
                self.shadow_weights.copy_from_slice(&self.output_weights);
                self.shadow_error_power = self.output_error_power;
                self.rescue_streak = 0;
                rescued = true;
            } else if self.quiet_streak >= TRANSFER_QUIET_STREAK_BLOCKS
                && self.output_error_power > 0.0
            {
                // The graded transfer: the output filter leaks toward the
                // shadow at a rate set by the measured advantage, in a fixed
                // coefficient order. Advantage is the share of the output
                // filter's error the shadow removes; below the transfer
                // margin nothing moves, and the rate rises linearly to its
                // maximum at the full-evidence advantage.
                let advantage = 1.0 - self.shadow_error_power / self.output_error_power;
                if advantage >= TRANSFER_MIN_ADVANTAGE {
                    let leak = TRANSFER_RATE_MAX * (advantage / TRANSFER_FULL_ADVANTAGE).min(1.0);
                    for (w, s) in self
                        .output_weights
                        .iter_mut()
                        .zip(self.shadow_weights.iter())
                    {
                        w.re += leak * (s.re - w.re);
                        w.im += leak * (s.im - w.im);
                    }
                    self.transfers += 1;
                }
            }
        }

        // The residual suppressor runs only once the far end has carried energy:
        // with nothing to suppress it is bypassed entirely, so a never-active
        // far end passes the near-end through bit for bit.
        let suppress =
            self.suppression == Suppression::Conservative && self.far_active && !diverged;
        // Adaptation is never frozen outright: the graded step scale below is
        // how a collapse or a poorly explained stream slows the filter. The
        // update is skipped only on the flush block, after a divergence, and
        // on a rescue, whose error belongs to the state just discarded.
        let adapting = adapt && !diverged && !shadow_diverged && !rescued;

        // The normalized frequency-domain update, skipped on the flush block
        // and after a divergence reset (the error that produced the reset is
        // not a usable gradient).
        if adapting {
            // The zero-prefixed error spectrum, of the shadow filter's own
            // error: the gradient must descend the error of the filter it
            // updates. The prefix is what puts the correlation lags where the
            // gradient needs them, which is why the suppressor below cannot
            // share this transform: it filters a continuous stream and needs
            // the overlap-save framing instead.
            self.window_scratch[..BLOCK].fill(0.0);
            self.window_scratch[BLOCK..].copy_from_slice(&self.shadow_error_block);
            self.fft
                .forward(&self.window_scratch, &mut self.error_spectrum);

            // Per-bin far-end power across the partitions: the normalized step's
            // denominator.
            self.far_power.fill(0.0);
            for partition in 0..partitions {
                let base = partition * BINS;
                for bin in 0..BINS {
                    let x = self.far_spectra[base + bin];
                    self.far_power[bin] += x.re * x.re + x.im * x.im;
                }
            }

            // The per-bin step. A bin carrying its usual power adapts at the
            // full step; one that has fallen below its own recent average is
            // stepped down in proportion. A stationary reference has each bin
            // at its own average, so this factor is one and the step is
            // unchanged. The whole step is then multiplied by the block's
            // graded scale, which is how a collapse or a poorly explained
            // stream slows the filter without ever freezing it.
            for bin in 0..BINS {
                let mean = self.far_power_mean[bin];
                let factor = if mean > 0.0 {
                    let share = (self.far_power[bin] / mean).min(1.0);
                    (share * share).max(MIN_STEP_FRACTION)
                } else {
                    1.0
                };
                self.step[bin] = STEP_SIZE * factor * self.step_scale;
                self.far_power_mean[bin] = STEP_POWER_SMOOTHING * mean
                    + (1.0 - STEP_POWER_SMOOTHING) * self.far_power[bin];
            }

            let mut all_finite = true;
            for partition in 0..partitions {
                let slot = (self.head + partitions - partition) % partitions;
                let spectra_base = slot * BINS;
                for bin in 0..BINS {
                    let x = self.far_spectra[spectra_base + bin];
                    let e = self.error_spectrum[bin];
                    let factor = self.step[bin] / (self.far_power[bin] + REGULARIZATION);
                    // conj(x) * e, written as independent products per component.
                    let re = x.re * e.re + x.im * e.im;
                    let im = x.re * e.im - x.im * e.re;
                    self.gradient[bin] = Complex::new(re * factor, im * factor);
                }

                // The gradient constraint: the lags beyond one block are the
                // circular wrap-around, not real filter taps, so they are zeroed
                // before the gradient is added back to the partition.
                self.fft.inverse(&self.gradient, &mut self.time_scratch);
                self.time_scratch[BLOCK..].fill(0.0);
                self.fft.forward(&self.time_scratch, &mut self.gradient);

                let weight_base = partition * BINS;
                for bin in 0..BINS {
                    let w = &mut self.shadow_weights[weight_base + bin];
                    w.re += self.gradient[bin].re;
                    w.im += self.gradient[bin].im;
                    all_finite &= w.re.is_finite() && w.im.is_finite();
                }
            }
            if !all_finite {
                self.diverge_shadow();
            }
        }

        // The residual suppressor: a conservative Wiener-style post-filter on
        // the linear filter's error, applied per bin and bounded by
        // [`MIN_SUPPRESSION_GAIN`]. It models the residual echo as a fixed
        // fraction of the echo estimate's power, so a bin the linear filter has
        // already cleaned keeps its gain while a bin still dominated by echo is
        // attenuated. Near-end speech raises the error power in exactly the bins
        // it occupies, which is what pulls the gain back toward one and keeps the
        // stage from eating the talker.
        // Slide the suppressor's own overlap-save window over the error stream,
        // regardless of whether the stage is enabled, so enabling it mid-stream
        // never sees a stale previous block.
        self.error_window.copy_within(BLOCK.., 0);
        self.error_window[BLOCK..].copy_from_slice(&self.error_block);

        if suppress {
            self.fft
                .forward(&self.error_window, &mut self.error_window_spectrum);
            for bin in 0..BINS {
                let estimate = self.estimate_spectrum[bin];
                let error = self.error_window_spectrum[bin];
                let estimate_power = estimate.re * estimate.re + estimate.im * estimate.im;
                let error_power = error.re * error.re + error.im * error.im;
                self.echo_power[bin] = SUPPRESSION_SMOOTHING * self.echo_power[bin]
                    + (1.0 - SUPPRESSION_SMOOTHING) * estimate_power;
                self.error_power[bin] = SUPPRESSION_SMOOTHING * self.error_power[bin]
                    + (1.0 - SUPPRESSION_SMOOTHING) * error_power;

                // While output protection is engaged the suppressor stands
                // down: its target returns to unity and the smoothing ramps
                // the applied gain back, so the stage only ever acts on spans
                // believed to be echo alone. This is what "conservative"
                // buys. The verdict is deliberately NOT the adaptation
                // decision: protection answers whether shaping this output
                // could damage near-end content, adaptation answers whether
                // the reference-to-microphone relationship supports a filter
                // update.
                let residual = RESIDUAL_ECHO_LEAK * self.echo_power[bin];
                let denominator = self.error_power[bin] + residual;
                let target = if self.protect_output || denominator <= 0.0 {
                    1.0
                } else {
                    (self.error_power[bin] / denominator).clamp(MIN_SUPPRESSION_GAIN, 1.0)
                };
                // Asymmetric smoothing: the stage backs off toward unity faster
                // than it engages, so a talker's onset is met by a gain already
                // on its way out rather than one still winding down.
                let smoothing = if target > self.gain[bin] {
                    GAIN_RELEASE_SMOOTHING
                } else {
                    GAIN_SMOOTHING
                };
                self.gain[bin] = smoothing * self.gain[bin] + (1.0 - smoothing) * target;
            }

            // The across-bin moving average, in a fixed ascending order over a
            // window clamped to the spectrum's ends.
            for bin in 0..BINS {
                let lo = bin.saturating_sub(GAIN_BIN_SPAN);
                let hi = (bin + GAIN_BIN_SPAN).min(BINS - 1);
                let mut total = 0.0_f64;
                for neighbour in lo..=hi {
                    total += self.gain[neighbour];
                }
                self.smoothed_gain[bin] = total / ((hi - lo + 1) as f64);
            }

            for bin in 0..BINS {
                let error = self.error_window_spectrum[bin];
                let gain = self.smoothed_gain[bin];
                self.suppressed_spectrum[bin] = Complex::new(error.re * gain, error.im * gain);
            }
            self.fft
                .inverse(&self.suppressed_spectrum, &mut self.time_scratch);
            self.output_block
                .copy_from_slice(&self.time_scratch[BLOCK..]);
        } else {
            self.output_block.copy_from_slice(&self.error_block);
        }

        // The divergence guard. A canceller removes energy, so an output that
        // sustains above the near end it was given is an output filter whose
        // correction has gone wrong in a way the finiteness checks above
        // cannot see. The correction is halved rather than zeroed (see
        // `retire_output`), the shadow keeps what it has learned, and this
        // block is delivered as the near end it was given.
        if valid > 0 {
            let mut out_energy = 0.0_f64;
            for &sample in &self.output_block[..valid] {
                out_energy += sample as f64 * sample as f64;
            }
            self.divergence_near_power = DIVERGENCE_SMOOTHING * self.divergence_near_power
                + (1.0 - DIVERGENCE_SMOOTHING) * near_energy;
            self.divergence_out_power = DIVERGENCE_SMOOTHING * self.divergence_out_power
                + (1.0 - DIVERGENCE_SMOOTHING) * out_energy;
            if self.divergence_near_power > 0.0
                && self.divergence_out_power > DIVERGENCE_RATIO * self.divergence_near_power
            {
                self.retire_output();
                self.output_block[..valid].copy_from_slice(&near[..valid]);
                // Start the guard's estimates from the state the reset leaves,
                // so the next block is judged on what the cleared filter does
                // rather than on what the diverged one did.
                self.divergence_out_power = self.divergence_near_power;
            }
        }

        // ERLE bookkeeping: smoothed powers only; the decibel conversion happens
        // in `metrics` on read, keeping the streaming state transcendental-free.
        for (&near_raw, &out_raw) in near[..valid].iter().zip(&self.output_block[..valid]) {
            let near_sample = near_raw as f64;
            let out_sample = out_raw as f64;
            self.near_power = ERLE_SMOOTHING * self.near_power
                + (1.0 - ERLE_SMOOTHING) * (near_sample * near_sample);
            self.residual_power = ERLE_SMOOTHING * self.residual_power
                + (1.0 - ERLE_SMOOTHING) * (out_sample * out_sample);
        }
    }
}

impl EchoCanceller for TauCanceller {
    /// Cancels one aligned block, re-blocking internally to [`BLOCK`] samples.
    /// Never returns an error after construction.
    ///
    /// Samples are accumulated until a full block is available, so a call may
    /// append nothing, one block, or several. The engine sanitizes upstream, but
    /// Tau re-sanitizes defensively: a non-finite input sample is treated as
    /// `0.0` before it can reach the transform or an update, so even a consumer
    /// driving the trait directly with pathological data cannot poison the
    /// filter, and the damage is bounded to the offending sample.
    fn process(&mut self, near: &[f32], far: &[f32], out: &mut Vec<f32>) -> Result<(), AecError> {
        debug_assert_eq!(
            near.len(),
            far.len(),
            "process requires equal-length aligned near and far blocks"
        );
        out.reserve(near.len());
        for (&near_raw, &far_raw) in near.iter().zip(far) {
            self.near_carry
                .push(if near_raw.is_finite() { near_raw } else { 0.0 });
            self.far_carry
                .push(if far_raw.is_finite() { far_raw } else { 0.0 });
            if self.near_carry.len() == BLOCK {
                let near_block = std::mem::take(&mut self.near_carry);
                let far_block = std::mem::take(&mut self.far_carry);
                self.run_block(&near_block, &far_block, BLOCK, true);
                out.extend_from_slice(&self.output_block);
                self.near_carry = near_block;
                self.far_carry = far_block;
                self.near_carry.clear();
                self.far_carry.clear();
            }
        }
        Ok(())
    }

    /// Drains the end-of-stream partial block, zero-padded to a full block, and
    /// appends exactly the samples the caller supplied. The padded block is not
    /// adapted on, because its tail is fabricated silence rather than observed
    /// signal. Never returns an error after construction.
    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AecError> {
        let remainder = self.near_carry.len();
        if remainder == 0 {
            return Ok(());
        }
        let mut near_block = std::mem::take(&mut self.near_carry);
        let mut far_block = std::mem::take(&mut self.far_carry);
        near_block.resize(BLOCK, 0.0);
        far_block.resize(BLOCK, 0.0);
        self.run_block(&near_block, &far_block, remainder, false);
        out.extend_from_slice(&self.output_block[..remainder]);
        near_block.clear();
        far_block.clear();
        self.near_carry = near_block;
        self.far_carry = far_block;
        Ok(())
    }

    /// One block: the framing granularity of the partitioned filter.
    ///
    /// This is a buffering budget, not an index shift. Tau delivers output
    /// samples in step with the near-end samples that produced them, so the
    /// `n`-th appended output sample is the cancelled `n`-th near-end sample
    /// and a consumer must not re-align by this value. What the value reports is
    /// that a near-end sample is not delivered until the block containing it is
    /// complete, so a real-time chain budgets up to one block (16 ms at 16 kHz)
    /// of transport delay for the canceller.
    fn latency_samples(&self) -> usize {
        BLOCK
    }

    /// Clears both filters, the block ring, the carry, the detector, the
    /// two-path comparison, and the ERLE state without reallocation,
    /// restoring the just-constructed state exactly. The divergence-reset
    /// count survives, as its metric is documented since construction, and
    /// the transfer count survives with it.
    fn reset(&mut self) {
        self.shadow_weights.fill(Complex::new(0.0, 0.0));
        self.output_weights.fill(Complex::new(0.0, 0.0));
        self.shadow_error_power = 0.0;
        self.output_error_power = 0.0;
        self.quiet_streak = 0;
        self.rescue_streak = 0;
        self.far_spectra.fill(Complex::new(0.0, 0.0));
        self.head = self.partitions - 1;
        self.far_window.fill(0.0);
        self.near_carry.clear();
        self.far_carry.clear();
        self.double_talk = false;
        self.dtd_baseline = 0.0;
        self.dtd_deviation = 0.0;
        self.dtd_collapsed = false;
        self.coherence_smooth = 0.0;
        self.step_scale = 1.0;
        self.protect_output = false;
        self.dtd_established = false;
        self.dtd_blocks = 0;
        self.far_energy_peak = 0.0;
        self.far_power_mean.fill(0.0);
        self.step.fill(STEP_SIZE);
        self.divergence_near_power = 0.0;
        self.divergence_out_power = 0.0;
        self.near_power = 0.0;
        self.residual_power = 0.0;
        self.far_active = false;
        self.error_window.fill(0.0);
        self.echo_power.fill(0.0);
        self.error_power.fill(0.0);
        self.gain.fill(1.0);
        self.smoothed_gain.fill(1.0);
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
            double_talk: self.double_talk,
            divergence_resets: self.divergence_resets,
        }
    }
}
