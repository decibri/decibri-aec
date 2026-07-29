//! The automatic echo-delay estimator: generalized cross-correlation with phase
//! transform (GCC-PHAT), on the crate's own transform.
//!
//! The [`Aec`](crate::Aec) engine aligns the near-end capture to the far-end
//! reference by an absolute-index offset. A caller who has measured their
//! platform's latency supplies it through
//! [`AecConfig::delay_hint_ms`](crate::AecConfig::delay_hint_ms); a caller who
//! has not needs the offset found for them, because with no offset the near-end
//! reads sit at the reference frontier and the canceller has nothing to cancel.
//! This module is that estimator: the sample-accurate FINE stage of the
//! coarse-to-fine acquisition described in [`crate::acquire`].
//!
//! # The method
//!
//! Each analysis frame transforms a block of near-end capture and the span of
//! far-end reference that could have produced it, forms the cross-spectrum, and
//! divides out its magnitude. That division is the phase transform: it discards
//! how much energy each frequency carries and keeps only the phase relationship,
//! which is what turns a broad correlation hump into a sharp spike and is why
//! the method survives a coloured signal and a reverberant path. The inverse
//! transform of the accumulated cross-spectrum is scanned across the configured
//! search window, and the position of its peak is the delay.
//!
//! # The search origin
//!
//! The window this stage scans is [`DelayEstimator::max_delay`] samples wide,
//! and it starts at a movable [`search_origin`](DelayEstimator::relocate). At
//! origin zero the stage searches `0..=max_delay`, which is what it has always
//! done. The coarse global scan can move that origin so the same width searches
//! a deeper region, which is how a transport delay well past the fine window is
//! reached without widening, or slowing, the fine correlation itself. The span,
//! the frame length, and the transform are identical at every origin, so the
//! confidence gate's peak-over-mean statistics are identical too.
//!
//! # Reporting, not locking
//!
//! This stage reports a [`FineLock`] on every frame whose confidence gate
//! passes. It does not decide whether that candidate becomes the stream's
//! alignment: [`DelayAcquirer`](crate::acquire::DelayAcquirer) owns that
//! decision. Refusing to promote is a supported outcome.
//!
//! # Determinism
//!
//! Frames are cut on the near-end sample count, not on how the caller chunks
//! the stream, so the same audio locks at the same sample on every run. The
//! arithmetic is the transform's plus `+`, `-`, `*`, `/`, comparisons, and
//! `sqrt`, which IEEE 754 requires to be correctly rounded and is therefore
//! bit-identical across platforms; there are no transcendentals, no unordered
//! containers, and no randomness.

use crate::fft::{Complex, RealFft};

/// The smallest analysis transform the estimator will build, so a short search
/// window still correlates over enough signal to mean something.
const MIN_FFT_LEN: usize = 256;

/// The largest analysis transform the estimator will build.
const MAX_FFT_LEN: usize = 65_536;

/// Frames that must be accumulated before a lock is even considered, so a single
/// lucky frame cannot decide the alignment.
const MIN_FRAMES: u32 = 2;

/// How far above the mean of the correlation the peak must stand for the
/// estimate to be trusted, in multiples of that mean.
///
/// The phase transform flattens the spectrum, so a genuine delay produces a
/// spike far above the surrounding floor while an absent one leaves a
/// correlation with no clear winner. The gate is deliberately asymmetric:
/// failing to lock costs a caller nothing they did not already have, while
/// locking onto noise breaks a canceller that would otherwise work.
const CONFIDENCE_THRESHOLD: f64 = 10.0;

/// Regularization for the phase transform's division, so a bin with no energy
/// contributes nothing instead of a zero-over-zero.
const PHAT_EPSILON: f64 = 1e-12;

/// The energy a frame must carry on both sides before it is accumulated at all.
/// Silence carries no delay information, and correlating it would only dilute
/// the frames that do.
const FRAME_ENERGY_FLOOR: f64 = 1e-8;

/// The largest trailing zero-fill a frame may carry and still be accumulated,
/// as a reciprocal fraction of the frame length.
///
/// A far window assembled at the reference frontier is normally short by
/// however far the caller has fed ahead, which under the usual one-block-ahead
/// cadence is a small fraction of a frame. A trailing deficit removes product
/// terms from the highest correlation indices only, so it biases the peak the
/// same way a head deficit does, just far more weakly. Refusing a trailing
/// deficit outright is not an option: no cadence that does not run far ahead of
/// the capture ever produces a frame with none.
const TAIL_DEFICIT_LIMIT_DEN: usize = 4;

/// Samples the reported delay is backed off from the correlation peak.
///
/// The peak of the correlation is where most of the echo path's energy arrives,
/// but it is not where the path begins: a real path rises into its peak over
/// several samples, and a reverberant one can carry a large share of its energy
/// before the peak entirely.
///
/// The estimate sits close to the peak, so left alone it has no margin at all.
/// This is the margin: it recovers the path onset where the peak is not the
/// onset.
///
/// It is applied exactly once, in [`DelayEstimator::observe`], to the
/// origin-relative mapping. Nothing else in the acquisition applies a second
/// one to a fine value.
pub(crate) const LOCK_MARGIN_SAMPLES: usize = 16;

/// Where the automatic delay acquisition has got to.
///
/// Reported through [`AecMetrics::delay`](crate::AecMetrics::delay). A caller
/// who supplied [`AecConfig::delay_hint_ms`](crate::AecConfig::delay_hint_ms) is
/// always [`DelayStatus::Locked`] with [`DelayLockSource::Hint`], and no search
/// of either stage is constructed or run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelayStatus {
    /// Both searches are running and nothing has met the promotion conditions.
    /// The engine's alignment offset is still zero, so the near-end reads sit at
    /// the reference frontier and the canceller has nothing to cancel yet.
    Searching,
    /// The coarse global scan found a region the fine search was not covering,
    /// and the fine search has been re-centred on it. Still no trusted offset.
    Relocated,
    /// A trusted alignment has been promoted and adopted by the engine, and the
    /// local tracker is following it. Left by [`Aec::reset`](crate::Aec::reset)
    /// or by a reacquisition trigger (see [`ReacquireTrigger`]).
    Locked(DelayLockSource),
    /// A reacquisition trigger fired: the previously trusted alignment went
    /// stale, both searches are running again, and the engine keeps the old
    /// offset until a new lock is promoted.
    /// [`AecMetrics::delay_samples`](crate::AecMetrics::delay_samples) still
    /// reports that old offset while this state stands.
    Reacquiring,
}

/// What evidence took a trusted lock back to a global search.
///
/// Reported through
/// [`DelayEstimate::last_reacquire_trigger`]. Every trigger is evaluated on
/// the local tracker's own decision cycles, so a stream with no far-end
/// energy can never fire one: silence teaches nothing and is not evidence
/// against a lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReacquireTrigger {
    /// The tracker's confident correlation peak sat pinned against an edge of
    /// its local window on consecutive decision cycles: the delay has moved
    /// beyond local reach.
    TrackingEdge,
    /// The tracker scored far-end-active cycles for a sustained period without
    /// once reaching its confidence gate: the correlation the lock was built
    /// on is gone.
    ConfidenceLost,
    /// Consecutive confident estimates jumped between unrelated delays: the
    /// local evidence contradicts itself and cannot be followed.
    EstimatorJumping,
    /// A stream discontinuity (a capture stall forcing the engine to re-anchor)
    /// put the lock under suspicion, and the tracker could not re-confirm the
    /// alignment afterwards.
    Discontinuity,
}

/// What evidence promoted the trusted lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DelayLockSource {
    /// The caller's measured `delay_hint_ms`, taken at their word.
    Hint,
    /// The fine peak and the coarse global region agreed, so each corroborates
    /// the other. The adopted value is the fine one, at full sample accuracy.
    GlobalAgreement,
    /// The coarse scan offered no region, and the fine peak stood clear of both
    /// search edges and cleared the raised uncorroborated confidence bar.
    LocalEvidence,
    /// The fine search never produced an agreeing peak inside the relocated
    /// region, so the coarse region itself was adopted, backed off so that the
    /// residual error falls on the early side the filter can absorb.
    CoarseRegion,
}

/// A snapshot of the delay acquisition: what it is searching, and what it found.
///
/// Metadata only, never sample data. Every field is in samples at the configured
/// rate unless its name says otherwise.
///
/// # Stability
///
/// [`status`](DelayEstimate::status) and
/// [`delay_samples`](DelayEstimate::delay_samples) are the stable contract: the
/// acquisition state and the active alignment offset, the latter mirroring
/// [`AecMetrics::delay_samples`](crate::AecMetrics::delay_samples). The remaining
/// fields are diagnostic telemetry of the estimator's internal progress, exposed
/// for observability and tuning. They may change meaning, gain companions, or be
/// removed as the estimator evolves, so treat them as diagnostics to read, not a
/// contract to build durable behavior on.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct DelayEstimate {
    /// Acquisition state.
    pub status: DelayStatus,
    /// The adopted alignment offset once trusted, [`None`] while searching.
    /// Always equal to
    /// [`AecMetrics::delay_samples`](crate::AecMetrics::delay_samples): a
    /// provisional or coarse-only candidate never appears here.
    pub delay_samples: Option<usize>,
    /// Inclusive lower bound of the sample-accurate fine search range.
    pub fine_search_start_samples: usize,
    /// Inclusive upper bound of the sample-accurate fine search range. Always
    /// the start plus `max_echo_delay_ms` in samples: relocation moves the
    /// range, it never resizes it.
    pub fine_search_end_samples: usize,
    /// Inclusive upper bound of the coarse global scan, from
    /// [`AecConfig::max_search_delay_ms`](crate::AecConfig::max_search_delay_ms).
    pub coarse_ceiling_samples: usize,
    /// The coarse global scan's region once it is confident and stable, [`None`]
    /// otherwise. Region-accurate only: its resolution is one coarse bin.
    pub coarse_region_samples: Option<usize>,
    /// One coarse bin in samples: the coarse scan's resolution.
    pub coarse_bin_samples: usize,
    /// The most recent coarse frame's peak normalized correlation, in
    /// `0.0..=1.0`. Zero before the first coarse frame completes.
    pub coarse_correlation: f32,
    /// Whether the coarse peak stood at the ceiling of its range, meaning the
    /// true delay is at or beyond [`Self::coarse_ceiling_samples`] and this
    /// engine cannot align it. A coverage report, not a region.
    pub beyond_ceiling: bool,
    /// Coarse frames completed since construction or the last reset.
    pub coarse_frames: u32,
    /// Fine frames accumulated against the current search origin.
    pub fine_frames: u32,
    /// Fine frames refused because their far window was not fully supported by
    /// fed reference. A frame short at the head gives each candidate lag a
    /// different number of real product terms, which biases the peak toward the
    /// search edge; refusing it is cheaper than correcting for it.
    pub fine_frames_skipped: u32,
    /// Whether the fine search has been re-centred on a coarse region.
    pub relocated: bool,
    /// Total fine correlation scans since construction or the last engine
    /// reset. Monotonic across relocations and tracking cycles, so a poller
    /// can detect each new scan and read its evidence below exactly once.
    pub fine_scans: u64,
    /// The most recent fine scan's peak-over-mean ratio, gate outcome aside.
    /// Zero until the first scan. This is the raw evidence the uncorroborated
    /// confidence gate is calibrated against.
    pub fine_last_ratio: f64,
    /// The delay the most recent fine scan's peak mapped to, margin applied,
    /// [`None`] until the first scan. A report of where the correlation
    /// peaked, not a candidate: the gates may have refused it.
    pub fine_last_delay_samples: Option<usize>,
    /// The search origin the most recent fine scan ran against. The CURRENT
    /// range above can differ: a promotion or relocation in the same block
    /// moves the range after the scan that caused it.
    pub fine_last_origin_samples: usize,
    /// Whether the most recent fine scan's peak stood clear of both ends of
    /// the range it scanned. An edge-pinned peak is a peak whose true maximum
    /// may lie outside the range entirely.
    pub fine_last_peak_interior: bool,
    /// Alignment updates applied by the local tracker after the initial lock.
    pub tracking_moves: u32,
    /// Times a reacquisition trigger returned the acquisition to a global
    /// search after a trusted lock.
    pub reacquisitions: u32,
    /// The trigger behind the most recent reacquisition, [`None`] until one
    /// fires.
    pub last_reacquire_trigger: Option<ReacquireTrigger>,
    /// Times the coarse scan was re-armed after giving up, because sustained
    /// far-end excitation kept arriving on a stream that never locked.
    pub coarse_rearms: u32,
    /// Coarse regions that failed re-verification on the last-resort adoption
    /// path: the scan was asked to find the region a second time and found a
    /// different one, so nothing was adopted. A non-zero count on a stream
    /// that never locked is the signature of a spurious or aliased region.
    ///
    /// It climbs to a fixed stand-down bound and holds there for the rest of
    /// the stream. After eight failed coarse-only re-verifications the
    /// coarse-only last-resort path stands down until the AEC is reset
    /// ([`Aec::reset`](crate::Aec::reset)), and this counter stops advancing at
    /// that bound. Normal corroborated acquisition (a fine lock that agrees
    /// with a coarse region) remains available throughout, so a stream that
    /// stood down can still lock the moment a real, reproducible path appears.
    /// See also [`Self::coarse_last_resort_exhausted`], the derived flag a
    /// caller can read instead of comparing this count against the bound.
    pub coarse_regions_rejected: u32,
    /// Whether the coarse-only last-resort adoption path has stood down for the
    /// rest of this stream (until [`Aec::reset`](crate::Aec::reset)).
    ///
    /// `true` once [`Self::coarse_regions_rejected`] has reached its stand-down
    /// bound: the scan repeatedly found confident regions that never
    /// reproduced, judged them wander rather than a path, and stopped paying to
    /// re-scan for a coarse-only adoption. It is a derived, read-only view of
    /// the counter reaching that bound, exposed so a caller need not know the
    /// bound's value. `false` on every healthy stream and on the hinted path.
    ///
    /// It does NOT mean acquisition has given up: normal corroborated
    /// acquisition stays available, so this can be `true` while the engine goes
    /// on to lock a genuine path the ordinary way.
    pub coarse_last_resort_exhausted: bool,
    /// Consecutive tracking cycles on which the estimator has contradicted
    /// ITSELF about where the delay is, as the run currently stands.
    ///
    /// A held lock re-confirms its alignment every cycle and re-confirmation
    /// clears the run, so this is zero on a healthy stream and stays zero
    /// however long the stream runs. A run that climbs is the estimator
    /// offering the alignment incompatible answers; once it is long enough it
    /// fires [`ReacquireTrigger::EstimatorJumping`] and resets.
    ///
    /// Diagnostic only: nothing in the engine reads it.
    pub tracking_contradiction_run: u32,
    /// The longest [`tracking_contradiction_run`](DelayEstimate::tracking_contradiction_run)
    /// seen since the engine was constructed or reset.
    ///
    /// The run itself is cleared by the very re-confirmation that ends an
    /// episode, and a reacquisition replaces the tracker outright, so a
    /// contradiction that climbed and then subsided leaves no trace in the live
    /// value. This high-water mark is what makes such an episode reportable
    /// after the fact. A non-zero value here with a zero live run means the
    /// stream contradicted itself and recovered without ever escalating.
    ///
    /// Diagnostic only: nothing in the engine reads it.
    pub tracking_contradiction_run_max: u32,
}

/// How much fed reference actually backed a fine frame's far window.
///
/// The engine assembles that window from the reference ring and renders any
/// index it cannot supply as silence. Those zeros are not signal, and where they
/// fall decides whether the frame is usable: see [`TAIL_DEFICIT_LIMIT_DEN`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WindowSupport {
    /// Leading window samples that no fed reference backed, because the window
    /// reached back past the start of the stream.
    pub(crate) missing_head: usize,
    /// Trailing window samples that no fed reference backed, because the window
    /// reached forward past the reference frontier.
    pub(crate) missing_tail: usize,
}

/// What one completed fine frame contributed, whether or not it produced a
/// candidate.
///
/// The acquirer needs more than the candidate stream: tracking counts SCORED
/// frames (frames that carried energy and support and were accumulated) to
/// pace its decision cycles, and the confidence calibration needs the peak
/// ratio of every scored frame, not only of the frames that cleared the gate.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FineObservation {
    /// Whether the frame was accumulated: it carried energy on both sides and
    /// its far window was adequately supported. A frame refused for support or
    /// silence is not scored and teaches nothing.
    pub(crate) scored: bool,
    /// The candidate that cleared the confidence gate this frame, if any.
    pub(crate) candidate: Option<FineLock>,
}

/// One fine-frame correlation that cleared the confidence gate.
///
/// Purely a report. The estimator does not decide whether to adopt it: the
/// false-lock signatures the acquirer screens for are diagnosable from
/// [`peak_at`](FineLock::peak_at), not from the delay alone.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FineLock {
    /// The candidate delay in samples: the origin-relative mapping applied and
    /// backed off by [`LOCK_MARGIN_SAMPLES`].
    pub(crate) delay: usize,
    /// The raw correlation index of the peak, in `0..=max_delay`. Index zero
    /// maps to the deepest delay in the range and index `max_delay` to the
    /// shallowest, so a peak pinned at either end is a peak whose true maximum
    /// may lie outside the range entirely.
    pub(crate) peak_at: usize,
    /// The peak's height in multiples of the mean over the scanned range.
    pub(crate) ratio: f64,
}

/// The sample-accurate fine delay estimator. See the module documentation.
pub(crate) struct DelayEstimator {
    /// The analysis transform, sized at construction.
    fft: RealFft,
    /// The search window's width in samples: the span of delays the estimator
    /// scans, starting at [`DelayEstimator::search_origin`].
    max_delay: usize,
    /// The near-end analysis frame length.
    frame_len: usize,
    /// The lower bound of the search range in samples: the delay that
    /// correlation index `max_delay` maps to. Zero until the acquirer re-centres
    /// the search on a coarse region.
    search_origin: usize,
    /// The near-end frame under construction. Cut on sample count, so framing is
    /// independent of the caller's chunking.
    near_frame: Vec<f32>,
    /// The accumulated phase-transformed cross-spectrum across frames.
    accumulated: Vec<Complex>,
    /// Frames accumulated against the current search origin.
    frames: u32,
    /// Frames refused for want of window support since the last reset.
    skipped: u32,
    /// The most recent scanned frame's peak-over-mean ratio, whether or not it
    /// cleared the gate. Diagnostics: this is the raw evidence the confidence
    /// calibration measures. Persists across restarts, because a tracking
    /// cycle restarts the accumulator immediately after its concluding scan.
    last_scan_ratio: f64,
    /// The most recent scanned frame's raw peak index, [`None`] until a frame
    /// has been scanned since construction or the last reset.
    last_scan_peak_at: Option<usize>,
    /// The search origin the most recent scan ran against, because the peak
    /// index is only meaningful against the origin of its own scan.
    last_scan_origin: usize,
    /// Total scans since construction or the last reset. Monotonic across
    /// restarts and relocations, so a poller can detect each new scan.
    scans: u64,
    /// Scratch for a zero-padded transform input.
    input_scratch: Vec<f32>,
    /// Scratch for the near-end frame's spectrum.
    near_spectrum: Vec<Complex>,
    /// Scratch for the far-end window's spectrum.
    far_spectrum: Vec<Complex>,
    /// Scratch for the correlation the inverse transform produces.
    correlation: Vec<f32>,
    /// The near-end frame the most recent scored
    /// [`observe`](DelayEstimator::observe) consumed, retained so a relocation
    /// that immediately follows can re-correlate it against the new range
    /// instead of discarding it. Its spectrum is independent of the search
    /// origin, so only the far window is re-read on the rescan.
    retained_near: Vec<f32>,
    /// Whether [`retained_near`](DelayEstimator::retained_near) holds a frame the
    /// current position may rescan: true only after a scored, energetic frame,
    /// false after one refused for want of support or for silence.
    retained_near_valid: bool,
    /// Scratch for the single-frame phase-transformed cross-spectrum the rescan
    /// forms, kept separate from [`accumulated`](DelayEstimator::accumulated) so
    /// a rescan that does not promote leaves the origin's own accumulation
    /// untouched.
    rescan_cross: Vec<Complex>,
}

impl DelayEstimator {
    /// Constructs the estimator for a search window of `max_delay_ms` at
    /// `sample_rate`.
    ///
    /// The analysis frame and the transform are sized from that window: the
    /// far-end span must cover the frame plus the whole window, so that every
    /// delay in the window is correlated over the same amount of signal and the
    /// estimate is not biased toward the short delays that would otherwise
    /// overlap more.
    pub(crate) fn new(sample_rate: u32, max_delay_ms: u16) -> DelayEstimator {
        let max_delay = ((max_delay_ms as u64 * sample_rate as u64) / 1000).max(1) as usize;
        let fft_len = (2 * max_delay)
            .next_power_of_two()
            .clamp(MIN_FFT_LEN, MAX_FFT_LEN);
        // The far-end window is the frame plus the search span, and it must fit
        // the transform without wrapping.
        let frame_len = fft_len.saturating_sub(max_delay).max(1);
        let bins = fft_len / 2 + 1;
        DelayEstimator {
            fft: RealFft::new(fft_len),
            max_delay,
            frame_len,
            search_origin: 0,
            near_frame: Vec::with_capacity(frame_len),
            accumulated: vec![Complex::new(0.0, 0.0); bins],
            frames: 0,
            skipped: 0,
            last_scan_ratio: 0.0,
            last_scan_peak_at: None,
            last_scan_origin: 0,
            scans: 0,
            input_scratch: vec![0.0; fft_len],
            near_spectrum: vec![Complex::new(0.0, 0.0); bins],
            far_spectrum: vec![Complex::new(0.0, 0.0); bins],
            correlation: vec![0.0; fft_len],
            retained_near: Vec::with_capacity(frame_len),
            retained_near_valid: false,
            rescan_cross: vec![Complex::new(0.0, 0.0); bins],
        }
    }

    /// The near-end analysis frame length in samples.
    pub(crate) fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// The number of far-end samples one frame needs: the frame plus the whole
    /// search window. Constant for the life of the estimator, because relocation
    /// moves the search range without resizing it.
    pub(crate) fn window_len(&self) -> usize {
        self.frame_len + self.max_delay
    }

    /// The search window's width in samples.
    pub(crate) fn max_delay(&self) -> usize {
        self.max_delay
    }

    /// How far back from the near frame's end the far window starts: the search
    /// origin plus the whole search span.
    pub(crate) fn window_back_off(&self) -> usize {
        self.search_origin + self.max_delay
    }

    /// The inclusive search range in samples, `(origin, origin + span)`.
    pub(crate) fn search_range(&self) -> (usize, usize) {
        (self.search_origin, self.search_origin + self.max_delay)
    }

    /// Frames accumulated against the current search origin.
    pub(crate) fn frames(&self) -> u32 {
        self.frames
    }

    /// Frames refused for want of window support since the last reset.
    pub(crate) fn skipped(&self) -> u32 {
        self.skipped
    }

    /// The most recent scanned frame's peak-over-mean ratio, gate outcome
    /// aside. Zero until a frame has been scanned.
    pub(crate) fn last_scan_ratio(&self) -> f64 {
        self.last_scan_ratio
    }

    /// The most recent scanned frame's raw peak index, [`None`] until a frame
    /// has been scanned.
    pub(crate) fn last_scan_peak_at(&self) -> Option<usize> {
        self.last_scan_peak_at
    }

    /// The search origin the most recent scan ran against.
    pub(crate) fn last_scan_origin(&self) -> usize {
        self.last_scan_origin
    }

    /// Total scans since construction or the last reset.
    pub(crate) fn scans(&self) -> u64 {
        self.scans
    }

    /// Re-centres the search on `origin`, discarding everything accumulated
    /// against the previous one.
    ///
    /// The frame length, the search span, and the transform are unchanged, so
    /// the correlation stays exactly as wrap-free as it was: `frame_len +
    /// max_delay == fft_len` still holds, every scanned index still gets exactly
    /// `frame_len` product terms, and the peak-over-mean statistics the
    /// confidence gate is calibrated against are unchanged. Only where the far
    /// window is read from moves.
    pub(crate) fn relocate(&mut self, origin: usize) {
        self.search_origin = origin;
        self.restart();
    }

    /// Discards the accumulation, keeping the search origin.
    ///
    /// The accumulator has no forgetting factor, so this is the only way to
    /// un-bank a frame that should not have counted, such as one straddling a
    /// re-anchor seam.
    pub(crate) fn restart(&mut self) {
        self.near_frame.clear();
        self.accumulated.fill(Complex::new(0.0, 0.0));
        self.frames = 0;
    }

    /// Appends one near-end sample, returning whether that completed a frame.
    ///
    /// A completed frame is consumed by the next
    /// [`observe`](DelayEstimator::observe) call.
    pub(crate) fn push_near(&mut self, sample: f32) -> bool {
        self.near_frame.push(sample);
        self.near_frame.len() == self.frame_len
    }

    /// Correlates the completed near-end frame against `far_window` and reports
    /// what the frame contributed: whether it was scored, and the candidate
    /// delay if it cleared the confidence gate.
    ///
    /// `far_window` holds exactly [`window_len`](DelayEstimator::window_len)
    /// far-end samples, ending where the near-end frame ends and reaching
    /// [`window_back_off`](DelayEstimator::window_back_off) samples further
    /// back, so that every candidate delay in the search range is covered.
    /// `support` says how much of that window fed reference actually backed. The
    /// frame is consumed either way.
    pub(crate) fn observe(
        &mut self,
        far_window: &[f32],
        support: WindowSupport,
    ) -> FineObservation {
        debug_assert_eq!(self.near_frame.len(), self.frame_len);
        debug_assert_eq!(far_window.len(), self.window_len());

        // A window short at the head does not reach back far enough to cover the
        // deepest candidate lag with real signal, so the shallow lags correlate
        // over more of the frame than the deep ones and the peak is drawn toward
        // the search edge whatever the audio says. A window short at the tail
        // has the same defect in the other direction, but weakly, and every
        // cadence that does not run far ahead produces one, so it is bounded
        // rather than refused.
        if support.missing_head > 0
            || support.missing_tail > self.frame_len / TAIL_DEFICIT_LIMIT_DEN
        {
            self.near_frame.clear();
            self.retained_near_valid = false;
            self.skipped = self.skipped.saturating_add(1);
            return FineObservation::default();
        }

        let near_energy: f64 = self
            .near_frame
            .iter()
            .map(|&s| s as f64 * s as f64)
            .sum::<f64>();
        let far_energy: f64 = far_window.iter().map(|&s| s as f64 * s as f64).sum::<f64>();

        if near_energy < FRAME_ENERGY_FLOOR || far_energy < FRAME_ENERGY_FLOOR {
            // Nothing to learn from a silent frame.
            self.near_frame.clear();
            self.retained_near_valid = false;
            return FineObservation::default();
        }

        // The near-end frame, zero-padded to the transform length.
        self.input_scratch.fill(0.0);
        self.input_scratch[..self.frame_len].copy_from_slice(&self.near_frame);
        // Retain the raw frame so a relocation that follows this observe can
        // re-correlate it against the new range instead of discarding it. The
        // spectrum is recomputed on the rescan from these samples, because the
        // near frame does not depend on the search origin.
        self.retained_near.clear();
        self.retained_near.extend_from_slice(&self.near_frame);
        self.retained_near_valid = true;
        self.near_frame.clear();
        self.fft
            .forward(&self.input_scratch, &mut self.near_spectrum);

        // The far-end window, zero-padded to the transform length.
        self.input_scratch.fill(0.0);
        self.input_scratch[..far_window.len()].copy_from_slice(far_window);
        self.fft
            .forward(&self.input_scratch, &mut self.far_spectrum);

        // The phase-transformed cross-spectrum conj(near) * far, accumulated in
        // fixed ascending bin order.
        for bin in 0..self.accumulated.len() {
            let near = self.near_spectrum[bin];
            let far = self.far_spectrum[bin];
            let re = near.re * far.re + near.im * far.im;
            let im = near.re * far.im - near.im * far.re;
            let magnitude = (re * re + im * im).sqrt();
            let scale = 1.0 / (magnitude + PHAT_EPSILON);
            self.accumulated[bin].re += re * scale;
            self.accumulated[bin].im += im * scale;
        }
        self.frames += 1;
        if self.frames < MIN_FRAMES {
            return FineObservation {
                scored: true,
                candidate: None,
            };
        }

        self.fft.inverse(&self.accumulated, &mut self.correlation);

        // The correlation at offset k corresponds to a delay of
        // `search_origin + max_delay - k`, so the whole search range lives in
        // `0..=max_delay`.
        let mut peak = 0.0_f64;
        let mut peak_at = 0_usize;
        let mut total = 0.0_f64;
        for k in 0..=self.max_delay {
            let value = (self.correlation[k] as f64).abs();
            total += value;
            if value > peak {
                peak = value;
                peak_at = k;
            }
        }
        let mean = total / ((self.max_delay + 1) as f64);
        let ratio = if mean > 0.0 { peak / mean } else { 0.0 };
        self.last_scan_ratio = ratio;
        self.last_scan_peak_at = Some(peak_at);
        self.last_scan_origin = self.search_origin;
        self.scans += 1;
        if mean <= 0.0 || peak < CONFIDENCE_THRESHOLD * mean {
            return FineObservation {
                scored: true,
                candidate: None,
            };
        }

        // Back the candidate delay off the peak by the safety margin, which the
        // filter downstream needs because it can only model lags at or after the
        // offset it is given. Saturating, so a delay shorter than the margin
        // reports zero rather than wrapping.
        let delay =
            (self.search_origin + self.max_delay - peak_at).saturating_sub(LOCK_MARGIN_SAMPLES);
        FineObservation {
            scored: true,
            candidate: Some(FineLock {
                delay,
                peak_at,
                ratio,
            }),
        }
    }

    /// Re-correlates the retained near frame against `far_window` at the CURRENT
    /// (post-[`relocate`](DelayEstimator::relocate)) origin, as a single
    /// corroborated frame, and reports the candidate if it clears the confidence
    /// gate.
    ///
    /// Called once immediately after a relocation to recover an alignment from
    /// the frame that completed at the OLD origin instead of discarding it. The
    /// retained near frame is the one the most recent
    /// [`observe`](DelayEstimator::observe) scanned; its spectrum does not depend
    /// on the search origin, so only the far window is re-read, and `far_window`
    /// must be the window for the NEW origin, exactly as
    /// [`window_back_off`](DelayEstimator::window_back_off) now reports it.
    ///
    /// The single frame is scanned WITHOUT the [`MIN_FRAMES`] wait the ordinary
    /// path enforces, because the promotion it feeds is gated by the caller on
    /// the coarse region's independent corroboration and an interior-peak test
    /// rather than on accumulation depth. It deliberately does NOT touch the
    /// accumulator: the cross-spectrum goes to its own scratch, so a rescan that
    /// does not promote leaves the estimator exactly as the relocate left it and
    /// the ordinary path resumes byte for byte unchanged.
    pub(crate) fn rescan(&mut self, far_window: &[f32], support: WindowSupport) -> FineObservation {
        debug_assert_eq!(far_window.len(), self.window_len());

        // A frame refused for support or silence retains nothing to rescan, and
        // a relocation triggered by anything other than an agreeing candidate
        // never reaches here anyway.
        if !self.retained_near_valid {
            return FineObservation::default();
        }
        // The same support and energy discipline as `observe`: a head-short
        // window biases the peak toward the search edge, and a silent window
        // teaches nothing.
        if support.missing_head > 0
            || support.missing_tail > self.frame_len / TAIL_DEFICIT_LIMIT_DEN
        {
            return FineObservation::default();
        }
        let near_energy: f64 = self
            .retained_near
            .iter()
            .map(|&s| s as f64 * s as f64)
            .sum::<f64>();
        let far_energy: f64 = far_window.iter().map(|&s| s as f64 * s as f64).sum::<f64>();
        if near_energy < FRAME_ENERGY_FLOOR || far_energy < FRAME_ENERGY_FLOOR {
            return FineObservation::default();
        }

        // The retained near-end frame, zero-padded to the transform length.
        self.input_scratch.fill(0.0);
        self.input_scratch[..self.frame_len].copy_from_slice(&self.retained_near);
        self.fft
            .forward(&self.input_scratch, &mut self.near_spectrum);

        // The far-end window at the new origin, zero-padded to the transform
        // length.
        self.input_scratch.fill(0.0);
        self.input_scratch[..far_window.len()].copy_from_slice(far_window);
        self.fft
            .forward(&self.input_scratch, &mut self.far_spectrum);

        // The phase-transformed cross-spectrum of this one frame, into dedicated
        // scratch so the origin's own (empty) accumulation stays empty.
        for bin in 0..self.rescan_cross.len() {
            let near = self.near_spectrum[bin];
            let far = self.far_spectrum[bin];
            let re = near.re * far.re + near.im * far.im;
            let im = near.re * far.im - near.im * far.re;
            let magnitude = (re * re + im * im).sqrt();
            let scale = 1.0 / (magnitude + PHAT_EPSILON);
            self.rescan_cross[bin] = Complex::new(re * scale, im * scale);
        }
        self.fft.inverse(&self.rescan_cross, &mut self.correlation);

        let mut peak = 0.0_f64;
        let mut peak_at = 0_usize;
        let mut total = 0.0_f64;
        for k in 0..=self.max_delay {
            let value = (self.correlation[k] as f64).abs();
            total += value;
            if value > peak {
                peak = value;
                peak_at = k;
            }
        }
        let mean = total / ((self.max_delay + 1) as f64);
        let ratio = if mean > 0.0 { peak / mean } else { 0.0 };
        // A rescan is a real scan, recorded like any other so the calibration
        // instrument sees its evidence.
        self.last_scan_ratio = ratio;
        self.last_scan_peak_at = Some(peak_at);
        self.last_scan_origin = self.search_origin;
        self.scans += 1;
        if mean <= 0.0 || peak < CONFIDENCE_THRESHOLD * mean {
            return FineObservation {
                scored: true,
                candidate: None,
            };
        }
        let delay =
            (self.search_origin + self.max_delay - peak_at).saturating_sub(LOCK_MARGIN_SAMPLES);
        FineObservation {
            scored: true,
            candidate: Some(FineLock {
                delay,
                peak_at,
                ratio,
            }),
        }
    }

    /// Clears the estimator to its just-constructed state, discarding the search
    /// origin so the next stream estimates afresh.
    pub(crate) fn reset(&mut self) {
        self.search_origin = 0;
        self.skipped = 0;
        self.last_scan_ratio = 0.0;
        self.last_scan_peak_at = None;
        self.last_scan_origin = 0;
        self.scans = 0;
        self.retained_near_valid = false;
        self.restart();
    }
}
