//! The coarse global delay scan: a cheap, heavily downsampled search for the
//! REGION the echo lives in, across the whole configured ceiling.
//!
//! # Why a second scan exists
//!
//! The fine estimator ([`crate::delay`]) is sample-accurate but narrow, and
//! widening it to cover a Bluetooth or speakerphone transport delay is costly.
//! This scan instead answers a strictly easier question, "roughly where", at a
//! resolution of one millisecond, and hands the answer to the fine estimator as
//! a place to look.
//!
//! # The signal chain
//!
//! Both streams are reduced to a magnitude envelope at [`COARSE_RATE_HZ`]: the
//! mean of the absolute sample value over each block of `D` native samples. That
//! is a length-`D` box lowpass on the rectified signal.
//!
//! Both chains are cut on the FAR-ABSOLUTE sample index, so envelope block `j`
//! covers far-absolute natives `[j*D, (j+1)*D)` on both sides, sharing one grid.
//!
//! # The correlation
//!
//! Per lag, a mean-removed normalized (Pearson) cross-correlation, where each
//! lag is normalized by the energy in its OWN far segment. The per-lag energies
//! come from prefix sums.
//!
//! The correlation is signed, not magnitude.
//!
//! # Deciding
//!
//! Per-lag correlations are averaged across [`COARSE_MIN_FRAMES`] frames before
//! anything is decided. The averaged peak must then clear an absolute
//! correlation floor, be a clear winner against the best competitor outside a
//! guard band, and stand clear of the ceiling. A peak pinned at the ceiling is
//! not a region: it means the true delay is at or past the ceiling, and that is
//! reported as a coverage failure rather than acted on.
//!
//! No frame is evaluated at all until the far history covers the whole window,
//! so no decision is ever taken on a partially zero-filled scan.
//!
//! # Determinism
//!
//! Blocks are cut on absolute sample index, not on caller chunking. The
//! arithmetic is `+`, `-`, `*`, `/`, comparison, `abs`, and `sqrt` only, all
//! correctly rounded by IEEE 754, with no transcendentals anywhere. Every
//! reduction runs in one fixed order over an indexed slice, the lag scan runs
//! descending with a strict comparison so ties resolve to the shorter delay, and
//! every buffer is allocated once at construction. There are no unordered
//! containers and no randomness.

/// The rate the coarse global scan runs at. Sets both the decimation factor and
/// the coarse bin size.
pub(crate) const COARSE_RATE_HZ: u32 = 1000;

/// Length of one coarse analysis frame, in milliseconds.
const COARSE_FRAME_MS: usize = 256;

/// Frames averaged before any region is declared.
const COARSE_MIN_FRAMES: u32 = 4;

/// The averaged peak correlation a region must reach.
const COARSE_CONFIDENT_CORR: f64 = 0.25;

/// How far the peak must stand above the best competitor outside the guard band.
const COARSE_SIDELOBE_RATIO: f64 = 1.5;

/// Half-width of the exclusion band around the peak for the competitor test, in
/// milliseconds.
const COARSE_PEAK_GUARD_MS: usize = 30;

/// How close to the ceiling a peak may sit and still be called a region, in
/// milliseconds. A peak inside this band of the ceiling is reported as
/// [`CoarseScan::beyond_ceiling`] instead.
const COARSE_CEILING_GUARD_MS: usize = 8;

/// The mean envelope level the far window must carry before a frame is scored,
/// in linear amplitude.
const COARSE_FAR_ACTIVE_FLOOR: f64 = 3.162_277_7e-4;

/// Division guard on the per-frame and per-lag energies, so a flat segment
/// yields a correlation of exactly zero rather than a zero-over-zero. Not a
/// level gate: [`COARSE_FAR_ACTIVE_FLOOR`] is that.
const COARSE_VARIANCE_FLOOR: f64 = 1e-12;

/// Active frames after which a scan that has never found a region stops working.
///
/// Bounds the cost of a stream whose delay is simply not findable.
const COARSE_GIVE_UP_FRAMES: u32 = 64;

/// Slack retained in the decimated far history beyond the scanned window, in
/// bins. Mirrors the native reference ring's own slack so the two histories
/// expire together.
const COARSE_RING_SLACK_BINS: usize = 4000;

/// The largest trailing zero-fill a coarse window may carry and still be scored,
/// as a reciprocal fraction of the frame.
const COARSE_TAIL_DEFICIT_DEN: usize = 4;

/// The coarse global scan's current best region.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoarseRegion {
    /// Delay in NATIVE samples, quantized to one coarse bin.
    pub(crate) delay: usize,
    /// Averaged normalized correlation at the peak, in `0.0..=1.0`.
    pub(crate) corr: f64,
}

/// The cheap, heavily downsampled global delay scan. See the module
/// documentation.
pub(crate) struct CoarseScan {
    /// Native samples per envelope bin.
    decimation: usize,
    /// The scanned span in bins: the ceiling.
    span: usize,
    /// One analysis frame in bins.
    frame: usize,
    /// Ceiling guard in bins.
    ceiling_guard: usize,
    /// Competitor exclusion half-width in bins.
    peak_guard: usize,

    /// The far envelope, addressed by decimated absolute index.
    far_env: Vec<f32>,
    /// The decimated absolute index of the next block to be written.
    far_env_next_abs: u64,
    /// Running sum of absolute values in the far block under construction.
    far_acc: f64,
    /// Native far samples accumulated into that block.
    far_count: usize,
    /// Native far samples pushed in total; mirrors the reference ring's own
    /// absolute counter so the two grids cannot drift.
    far_pushed: u64,

    /// The near envelope frame under construction.
    near_env: Vec<f32>,
    /// Running sum of absolute values in the near block under construction.
    near_acc: f64,
    /// Native near samples accumulated into that block.
    near_count: usize,
    /// The decimated absolute index the near frame's first bin sits at.
    near_frame_start_abs: u64,

    /// Scratch: the far window one frame scans.
    window: Vec<f32>,
    /// Scratch: prefix sums of the window.
    prefix_sum: Vec<f64>,
    /// Scratch: prefix sums of the window's squares.
    prefix_sq: Vec<f64>,
    /// Scratch: this frame's per-lag correlation.
    corr: Vec<f64>,
    /// The running per-lag average across accumulated frames.
    corr_avg: Vec<f64>,

    /// Frames accumulated into `corr_avg` since the last decision cycle.
    accumulated: u32,
    /// Frames scored in total since construction or reset.
    frames: u32,
    /// The most recent frame's peak correlation, for diagnostics.
    last_corr: f32,
    /// The confirmed region, once one exists.
    best: Option<CoarseRegion>,
    /// Whether the peak stood at the ceiling, meaning the delay is out of reach.
    beyond_ceiling: bool,
    /// Whether the scan has given up.
    exhausted: bool,
}

impl CoarseScan {
    /// Constructs the scan for a ceiling of `ceiling_ms` at `sample_rate`.
    pub(crate) fn new(sample_rate: u32, ceiling_ms: u16) -> CoarseScan {
        let decimation = (((sample_rate + COARSE_RATE_HZ / 2) / COARSE_RATE_HZ) as usize).max(1);
        // One bin is one native block, so the span in bins is the ceiling in
        // milliseconds scaled by the achieved bin rate rather than the nominal
        // one, which matters only at rates that are not a multiple of 1 kHz.
        let bin_rate = sample_rate as usize / decimation;
        let span = ((ceiling_ms as usize * bin_rate) / 1000).max(1);
        let frame = ((COARSE_FRAME_MS * bin_rate) / 1000).max(1);
        let ceiling_guard = ((COARSE_CEILING_GUARD_MS * bin_rate) / 1000).max(1);
        let peak_guard = ((COARSE_PEAK_GUARD_MS * bin_rate) / 1000).max(1);
        let window_len = frame + span;
        CoarseScan {
            decimation,
            span,
            frame,
            ceiling_guard,
            peak_guard,
            far_env: vec![0.0; window_len + COARSE_RING_SLACK_BINS],
            far_env_next_abs: 0,
            far_acc: 0.0,
            far_count: 0,
            far_pushed: 0,
            near_env: Vec::with_capacity(frame),
            near_acc: 0.0,
            near_count: 0,
            near_frame_start_abs: 0,
            window: vec![0.0; window_len],
            prefix_sum: vec![0.0; window_len + 1],
            prefix_sq: vec![0.0; window_len + 1],
            corr: vec![0.0; span + 1],
            corr_avg: vec![0.0; span + 1],
            accumulated: 0,
            frames: 0,
            last_corr: 0.0,
            best: None,
            beyond_ceiling: false,
            exhausted: false,
        }
    }

    /// One coarse bin in native samples: the scan's resolution.
    pub(crate) fn bin_samples(&self) -> usize {
        self.decimation
    }

    /// The scanned ceiling in native samples.
    pub(crate) fn ceiling_samples(&self) -> usize {
        self.span * self.decimation
    }

    /// The confirmed region, if the scan has found one.
    pub(crate) fn region(&self) -> Option<CoarseRegion> {
        self.best
    }

    /// Whether the peak stood at the ceiling, meaning the true delay is at or
    /// beyond the ceiling and this engine cannot align it.
    pub(crate) fn beyond_ceiling(&self) -> bool {
        self.beyond_ceiling
    }

    /// Frames scored since construction or the last reset.
    pub(crate) fn frames(&self) -> u32 {
        self.frames
    }

    /// The scan's current peak correlation: the confirmed region's averaged
    /// value once one exists, and the most recent frame's peak before that.
    pub(crate) fn correlation(&self) -> f32 {
        match self.best {
            Some(region) => region.corr as f32,
            None => self.last_corr,
        }
    }

    /// Whether the scan has stopped working, either because it found a region or
    /// because it gave up.
    pub(crate) fn finished(&self) -> bool {
        self.best.is_some() || self.exhausted
    }

    /// Whether the scan gave up without confirming a region.
    pub(crate) fn exhausted(&self) -> bool {
        self.exhausted
    }

    /// Clears the decision state and resumes scanning on the live far history.
    ///
    /// The far envelope chain is preserved: it has kept building through
    /// [`push_far`](CoarseScan::push_far) the whole time, so the re-armed scan
    /// scores its first frame as soon as a near frame completes, against real
    /// history rather than a zero-filled window. Used by the acquirer to
    /// re-verify a region before last-resort adoption, to retry after the scan
    /// gave up, and to re-enter the global search on a reacquisition trigger.
    /// [`reset`](CoarseScan::reset) remains the only full clear, because
    /// zeroing the absolute far counters mid-stream would misalign the two
    /// grids permanently.
    pub(crate) fn rearm(&mut self) {
        self.near_env.clear();
        self.near_acc = 0.0;
        self.near_count = 0;
        self.near_frame_start_abs = 0;
        for value in self.corr_avg.iter_mut() {
            *value = 0.0;
        }
        self.accumulated = 0;
        self.frames = 0;
        self.last_corr = 0.0;
        self.best = None;
        self.beyond_ceiling = false;
        self.exhausted = false;
    }

    /// Appends far-end reference samples, in played order.
    ///
    /// Called with exactly the samples the reference ring received, immediately
    /// after it received them, so the two absolute grids stay identical.
    ///
    /// The envelope chain runs even when the scan is finished. It costs one
    /// absolute value and one add per native sample, and it is what keeps the
    /// decimated grid contiguous so a later [`rearm`](CoarseScan::rearm), for a
    /// reacquisition or a retry, resumes on live history instead of a gap that
    /// would silently misalign the two grids.
    pub(crate) fn push_far(&mut self, reference: &[f32]) {
        for &sample in reference {
            self.far_acc += (sample as f64).abs();
            self.far_count += 1;
            self.far_pushed += 1;
            if self.far_count == self.decimation {
                let value = (self.far_acc / self.decimation as f64) as f32;
                let slot = (self.far_env_next_abs % self.far_env.len() as u64) as usize;
                self.far_env[slot] = value;
                self.far_env_next_abs += 1;
                self.far_acc = 0.0;
                self.far_count = 0;
            }
        }
    }

    /// Appends one near-end sample carrying far-absolute index `abs`, returning
    /// whether that completed a coarse frame ready to be scored.
    ///
    /// Blocks are cut where `abs` crosses a multiple of the decimation, so the
    /// near chain lands on the same grid the far chain does regardless of where
    /// the engine's anchor fell.
    pub(crate) fn push_near(&mut self, sample: f32, abs: u64) -> bool {
        if self.finished() {
            return false;
        }
        if self.near_count == 0 && self.near_env.is_empty() {
            // Start the frame on a block boundary; anything before one is
            // discarded rather than padded, so the grids stay aligned.
            if !abs.is_multiple_of(self.decimation as u64) {
                return false;
            }
            self.near_frame_start_abs = abs / self.decimation as u64;
        }
        self.near_acc += (sample as f64).abs();
        self.near_count += 1;
        if self.near_count == self.decimation {
            let value = (self.near_acc / self.decimation as f64) as f32;
            self.near_env.push(value);
            self.near_acc = 0.0;
            self.near_count = 0;
        }
        self.near_env.len() == self.frame
    }

    /// Scores the completed near frame against the far history, folding the
    /// result into the running average and confirming a region when the evidence
    /// is sufficient. The frame is consumed either way.
    pub(crate) fn observe(&mut self) {
        debug_assert_eq!(self.near_env.len(), self.frame);
        let window_len = self.frame + self.span;

        // The frame's last bin is `near_frame_start_abs + frame - 1`; the window
        // ends one past it and reaches the whole span further back.
        let window_end = self.near_frame_start_abs + self.frame as u64;
        let window_start = window_end as i64 - window_len as i64;

        // No decision is ever taken on a partially covered scan: if the far
        // history does not reach back over the whole window, or does not yet
        // reach forward to its end, the frame is discarded outright.
        let oldest_available = self
            .far_env_next_abs
            .saturating_sub(self.far_env.len() as u64) as i64;
        if window_start < 0 || window_start < oldest_available {
            self.near_env.clear();
            return;
        }

        // A tail deficit is expected: the near frame ends a little beyond the
        // newest reference sample fed, so the window's last few bins have no
        // history behind them and are zero-filled.
        let deficit = window_end.saturating_sub(self.far_env_next_abs) as usize;
        if deficit > self.frame / COARSE_TAIL_DEFICIT_DEN {
            self.near_env.clear();
            return;
        }

        let supported = window_len - deficit;
        for offset in 0..supported {
            let index =
                ((window_start as u64 + offset as u64) % self.far_env.len() as u64) as usize;
            self.window[offset] = self.far_env[index];
        }
        for slot in self.window[supported..window_len].iter_mut() {
            *slot = 0.0;
        }

        // The far window must carry real level before it is scored.
        let mut far_sum = 0.0_f64;
        for &value in self.window[..window_len].iter() {
            far_sum += value as f64;
        }
        if far_sum / window_len as f64 <= COARSE_FAR_ACTIVE_FLOOR {
            self.near_env.clear();
            return;
        }

        // The near frame, mean removed once.
        let frame = self.frame;
        let mut near_mean = 0.0_f64;
        for &value in self.near_env.iter() {
            near_mean += value as f64;
        }
        near_mean /= frame as f64;
        let mut near_energy = 0.0_f64;
        for &value in self.near_env.iter() {
            let centred = value as f64 - near_mean;
            near_energy += centred * centred;
        }
        if near_energy <= COARSE_VARIANCE_FLOOR {
            self.near_env.clear();
            return;
        }

        // Prefix sums over the far window, strictly ascending.
        self.prefix_sum[0] = 0.0;
        self.prefix_sq[0] = 0.0;
        for index in 0..window_len {
            let value = self.window[index] as f64;
            self.prefix_sum[index + 1] = self.prefix_sum[index] + value;
            self.prefix_sq[index + 1] = self.prefix_sq[index] + value * value;
        }

        self.frames = self.frames.saturating_add(1);

        // Lag k in the window maps to a delay of `span - k` bins. Scanned
        // descending so that delay ascends and an exact tie resolves to the
        // shorter delay.
        let mut peak = 0.0_f64;
        for k in (0..=self.span).rev() {
            let s1 = self.prefix_sum[k + frame] - self.prefix_sum[k];
            let s2 = self.prefix_sq[k + frame] - self.prefix_sq[k];
            let far_energy = s2 - s1 * s1 / frame as f64;
            let value = if far_energy <= COARSE_VARIANCE_FLOOR {
                0.0
            } else {
                let mut num = 0.0_f64;
                for offset in 0..frame {
                    let near = self.near_env[offset] as f64 - near_mean;
                    num += near * self.window[k + offset] as f64;
                }
                num / (near_energy * far_energy).sqrt()
            };
            self.corr[k] = value;
            if value > peak {
                peak = value;
            }
        }
        self.near_env.clear();
        self.last_corr = peak as f32;

        // Fold into the running per-lag average.
        for k in 0..=self.span {
            self.corr_avg[k] += self.corr[k];
        }
        self.accumulated += 1;

        if self.accumulated < COARSE_MIN_FRAMES {
            if self.frames >= COARSE_GIVE_UP_FRAMES {
                self.exhausted = true;
            }
            return;
        }

        let scale = 1.0 / self.accumulated as f64;
        let mut avg_peak = 0.0_f64;
        let mut avg_peak_at = 0_usize;
        for k in (0..=self.span).rev() {
            let value = self.corr_avg[k] * scale;
            self.corr_avg[k] = value;
            if value > avg_peak {
                avg_peak = value;
                avg_peak_at = k;
            }
        }
        let delay_bins = self.span - avg_peak_at;
        self.decide(delay_bins, avg_peak);

        // Start a fresh averaging cycle either way.
        for value in self.corr_avg.iter_mut() {
            *value = 0.0;
        }
        self.accumulated = 0;
        if self.best.is_none() && self.frames >= COARSE_GIVE_UP_FRAMES {
            self.exhausted = true;
        }
    }

    /// Applies the confidence, competitor and ceiling tests to an averaged peak.
    fn decide(&mut self, delay_bins: usize, avg_peak: f64) {
        if avg_peak < COARSE_CONFIDENT_CORR {
            return;
        }

        // A peak pinned at the ceiling means the true delay is at or beyond it.
        // That is a coverage report, not a region.
        if delay_bins + self.ceiling_guard > self.span {
            self.beyond_ceiling = true;
            return;
        }

        // The peak must be the clear winner among regions; competitors inside
        // the guard band are excluded.
        let peak_at = self.span - delay_bins;
        let mut second = 0.0_f64;
        for k in 0..=self.span {
            if k.abs_diff(peak_at) <= self.peak_guard {
                continue;
            }
            if self.corr_avg[k] > second {
                second = self.corr_avg[k];
            }
        }
        if second > 0.0 && avg_peak < COARSE_SIDELOBE_RATIO * second {
            return;
        }

        self.beyond_ceiling = false;
        self.best = Some(CoarseRegion {
            delay: delay_bins * self.decimation,
            corr: avg_peak,
        });
    }

    /// Clears the scan to its just-constructed state.
    pub(crate) fn reset(&mut self) {
        self.far_env.fill(0.0);
        self.far_env_next_abs = 0;
        self.far_acc = 0.0;
        self.far_count = 0;
        self.far_pushed = 0;
        self.near_env.clear();
        self.near_acc = 0.0;
        self.near_count = 0;
        self.near_frame_start_abs = 0;
        for value in self.corr_avg.iter_mut() {
            *value = 0.0;
        }
        self.accumulated = 0;
        self.frames = 0;
        self.last_corr = 0.0;
        self.best = None;
        self.beyond_ceiling = false;
        self.exhausted = false;
    }

    /// Discards the near-side evidence gathered before a seam, keeping the far
    /// history.
    ///
    /// Called when the engine re-anchors: the partial frame straddles a seam in
    /// the near stream and correlating across it would be meaningless. The
    /// running per-lag average goes with it, because a decision is taken only
    /// every [`COARSE_MIN_FRAMES`] frames and the frames already banked were
    /// measured against the near stream the seam ended. Keeping them would let
    /// up to three pre-seam frames average with post-seam ones.
    ///
    /// This is deliberately narrower than [`rearm`](CoarseScan::rearm): the
    /// give-up budget, the confirmed region and the exhausted flag all survive,
    /// so a re-anchor cannot re-open the search budget each time it fires.
    pub(crate) fn discard_near(&mut self) {
        self.near_env.clear();
        self.near_acc = 0.0;
        self.near_count = 0;
        for value in self.corr_avg.iter_mut() {
            *value = 0.0;
        }
        self.accumulated = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const CEILING_MS: u16 = 250;

    /// Deterministic zero-mean noise, so every run scores the same frames.
    fn noise(count: usize) -> Vec<f32> {
        let mut state = 0x1234_5678_9abc_def0_u64;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1_u32 << 31) as f32) - 0.5
            })
            .collect()
    }

    /// The seam clear must be symmetric with what the frame count means.
    ///
    /// Per-lag correlations are averaged across [`COARSE_MIN_FRAMES`] frames
    /// before a decision, so up to three scored frames are still banked when a
    /// re-anchor arrives. Clearing only the partial near frame leaves those
    /// pre-seam frames to average with post-seam ones, across a discontinuity
    /// that makes the two incomparable.
    #[test]
    fn discarding_the_near_frame_at_a_seam_also_clears_the_running_average() {
        let mut scan = CoarseScan::new(RATE, CEILING_MS);
        let delay = 2 * scan.decimation;
        let far = noise(64_000);
        scan.push_far(&far);

        // Score three frames: one short of a decision, so they are still in the
        // running average when the seam arrives.
        let mut abs = 8_000_u64;
        while scan.accumulated < COARSE_MIN_FRAMES - 1 {
            let near = far[abs as usize - delay];
            if scan.push_near(near, abs) {
                scan.observe();
            }
            abs += 1;
            assert!(
                abs < 60_000,
                "the frames must score well inside the history"
            );
        }
        assert!(
            scan.corr_avg.iter().any(|&value| value != 0.0),
            "the banked frames left real evidence in the average"
        );

        scan.discard_near();

        assert_eq!(
            scan.accumulated, 0,
            "the seam clears the banked frame count"
        );
        assert!(
            scan.corr_avg.iter().all(|&value| value == 0.0),
            "pre-seam frames must not average with post-seam ones"
        );
    }

    /// The decimated write cursor is a `u64`, and the slot it maps to must be
    /// reduced BEFORE it is narrowed to a `usize`. Narrowing first truncates on
    /// a 32-bit target, so after 2^32 decimated blocks (about 49.7 days of
    /// streaming at 1 kHz bins) the mapping jumps instead of advancing by one
    /// and the far grid silently misaligns against the near one.
    ///
    /// On a 64-bit host both orders agree and this test passes either way; it
    /// bites when the crate is built for a 32-bit target.
    #[test]
    fn the_envelope_slot_mapping_stays_contiguous_across_a_32_bit_index_wrap() {
        let mut scan = CoarseScan::new(RATE, CEILING_MS);
        let len = scan.far_env.len() as u64;
        let wrap = 1_u64 << 32;

        // Park the cursor one block short of the wrap, then write two blocks
        // that straddle it, each at its own level so the slot each landed in is
        // identifiable.
        scan.far_env_next_abs = wrap - 1;
        scan.push_far(&vec![0.25_f32; scan.decimation]);
        scan.push_far(&vec![0.75_f32; scan.decimation]);

        let before = ((wrap - 1) % len) as usize;
        let after = (wrap % len) as usize;
        assert_ne!(before, after, "the two blocks occupy different slots");
        assert_eq!(scan.far_env[before], 0.25, "the pre-wrap block");
        assert_eq!(
            scan.far_env[after], 0.75,
            "the block after the wrap must land in the NEXT slot, not in the \
             one a truncated index would name"
        );
    }
}
