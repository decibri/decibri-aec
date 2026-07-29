//! Windowed signal metrics for the benchmark harness: activity
//! classification, near-end projection gain, masked energy reduction, and an
//! engine-independent echo-delay estimate.
//!
//! Everything here consumes the engine-rate signals the harness fed and
//! received (near-end input, the engine output it is time-aligned with, and
//! the far-end reference); nothing here touches the engine.

/// Analysis window length in milliseconds.
pub const WINDOW_MS: usize = 200;

/// Analysis hop in milliseconds.
pub const HOP_MS: usize = 100;

/// A window is active when its RMS sits within this many dB of the signal's
/// loud level.
pub const ACTIVE_REL_DB: f64 = 20.0;

/// Absolute activity floor in dBFS. A window below this is never active.
pub const ACTIVE_FLOOR_DBFS: f64 = -70.0;

/// A mic window counts as near-end active when its mic-over-far level
/// exceeds the clip's echo-coupling estimate by at least this margin.
pub const NEAR_ACTIVE_MARGIN_DB: f64 = 6.0;

/// A far-active window counts as echo dominant when its mic-over-far level
/// sits within this margin of the echo-coupling estimate.
pub const ECHO_DOMINANT_MARGIN_DB: f64 = 3.0;

/// The echo-coupling estimate is this quantile of the mic-over-far level
/// across far-active windows.
pub const ECHO_COUPLING_Q: f64 = 0.10;

/// Minimum far-active window count before the coupling estimate is trusted.
pub const MIN_COUPLING_WINDOWS: usize = 5;

/// Echo-reduction windows must start at or after this many seconds.
pub const CONVERGED_START_S: f64 = 4.0;

/// The envelope cross-correlation peak must reach this coefficient before
/// the harness trusts its own delay estimate enough to raise flags with it.
pub const DELAY_CONFIDENT_CORR: f64 = 0.25;

/// Upper bound of the harness delay search, in milliseconds.
pub const DELAY_SEARCH_MAX_MS: usize = 1000;

/// Decimation factor from the engine rate to the 1 kHz envelope rate the
/// delay search runs at.
const ENV_DECIM: usize = 16;

/// RMS of a span in dBFS; -120 for an empty or silent span, standing in for
/// minus infinity while staying finite for sorting and printing.
pub fn rms_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return -120.0;
    }
    let energy: f64 = samples.iter().map(|&s| s as f64 * s as f64).sum();
    let rms = (energy / samples.len() as f64).sqrt();
    if rms <= 1e-9 {
        return -120.0;
    }
    20.0 * rms.log10()
}

/// The `q` quantile of an already sorted, non-empty slice, by nearest rank.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Per-window activity classification for one processed clip.
///
/// Windows are [`WINDOW_MS`] long every [`HOP_MS`], full windows only, and
/// `starts[i]` is window `i`'s first sample index in the near-end timeline.
/// The far end is read through the supplied alignment delay: the far content
/// paired with near window `[s, s+len)` is `far[s-delay .. s+len-delay)`,
/// clamped to the signal.
pub struct WindowAnalysis {
    pub window_len: usize,
    pub starts: Vec<usize>,
    pub far_active: Vec<bool>,
    pub near_active: Vec<bool>,
    pub echo_dominant: Vec<bool>,
    /// The clip's mic-over-far echo coupling estimate in dB, when enough
    /// far-active windows existed to form one.
    pub echo_coupling_db: Option<f64>,
}

/// Classifies every full window of the clip.
pub fn analyze(near: &[f32], far: &[f32], delay_samples: usize, rate: u32) -> WindowAnalysis {
    let window_len = WINDOW_MS * rate as usize / 1000;
    let hop = HOP_MS * rate as usize / 1000;
    let mut starts = Vec::new();
    let mut s = 0usize;
    while s + window_len <= near.len() {
        starts.push(s);
        s += hop;
    }

    let mic_db: Vec<f64> = starts
        .iter()
        .map(|&s| rms_dbfs(&near[s..s + window_len]))
        .collect();
    let far_db: Vec<f64> = starts
        .iter()
        .map(|&s| {
            let lo = s.saturating_sub(delay_samples).min(far.len());
            let hi = (s + window_len)
                .saturating_sub(delay_samples)
                .min(far.len());
            rms_dbfs(&far[lo..hi])
        })
        .collect();

    let active_flags = |levels: &[f64]| -> Vec<bool> {
        let mut sorted = levels.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("window levels are finite"));
        if sorted.is_empty() {
            return Vec::new();
        }
        let loud = percentile(&sorted, 0.95);
        if loud <= ACTIVE_FLOOR_DBFS {
            return vec![false; levels.len()];
        }
        levels
            .iter()
            .map(|&db| db > loud - ACTIVE_REL_DB && db > ACTIVE_FLOOR_DBFS)
            .collect()
    };
    let mic_active = active_flags(&mic_db);
    let far_active = active_flags(&far_db);

    // Mic-over-far level of the far-active windows, low tail first.
    let mut coupling: Vec<f64> = starts
        .iter()
        .enumerate()
        .filter(|&(i, _)| far_active[i])
        .map(|(i, _)| mic_db[i] - far_db[i])
        .collect();
    coupling.sort_by(|a, b| a.partial_cmp(b).expect("window levels are finite"));
    let echo_coupling_db = if coupling.len() >= MIN_COUPLING_WINDOWS {
        Some(percentile(&coupling, ECHO_COUPLING_Q))
    } else {
        None
    };

    let mut near_active = vec![false; starts.len()];
    let mut echo_dominant = vec![false; starts.len()];
    for i in 0..starts.len() {
        let over_far = mic_db[i] - far_db[i];
        near_active[i] = mic_active[i]
            && (!far_active[i]
                || match echo_coupling_db {
                    Some(c) => over_far >= c + NEAR_ACTIVE_MARGIN_DB,
                    // No coupling estimate: mic activity is attributed to the near end.
                    None => true,
                });
        echo_dominant[i] = far_active[i]
            && match echo_coupling_db {
                Some(c) => over_far <= c + ECHO_DOMINANT_MARGIN_DB,
                None => false,
            };
    }

    WindowAnalysis {
        window_len,
        starts,
        far_active,
        near_active,
        echo_dominant,
        echo_coupling_db,
    }
}

/// Near-end projection gain statistics over the clip's near-end-active
/// windows.
pub struct ProjectionStats {
    /// Near-end-active windows measured.
    pub n: usize,
    /// Windows whose gain fell below -3 dB and below -6 dB.
    pub below_3db: usize,
    pub below_6db: usize,
    pub median_db: f64,
    pub p5_db: f64,
    pub min_db: f64,
    /// Start time of the minimum-gain window, seconds into the clip.
    pub min_at_s: f64,
}

impl ProjectionStats {
    pub fn below_3db_pct(&self) -> f64 {
        self.below_3db as f64 * 100.0 / self.n.max(1) as f64
    }
    pub fn below_6db_pct(&self) -> f64 {
        self.below_6db as f64 * 100.0 / self.n.max(1) as f64
    }
}

/// Per-window projection gain of the output onto the near-end input:
/// `g = sum(out * near) / sum(near^2)` over the window, in dB.
///
/// Returns `None` when the clip has no near-end-active windows.
pub fn projection_stats(
    near: &[f32],
    out: &[f32],
    wa: &WindowAnalysis,
    rate: u32,
) -> Option<ProjectionStats> {
    let mut gains: Vec<(f64, usize)> = Vec::new();
    for (i, &s) in wa.starts.iter().enumerate() {
        if !wa.near_active[i] || s + wa.window_len > out.len() {
            continue;
        }
        let mut cross = 0.0f64;
        let mut denom = 0.0f64;
        for k in s..s + wa.window_len {
            cross += out[k] as f64 * near[k] as f64;
            denom += near[k] as f64 * near[k] as f64;
        }
        let g = if denom > 0.0 { cross / denom } else { 0.0 };
        let db = if g <= 1e-6 { -120.0 } else { 20.0 * g.log10() };
        gains.push((db, s));
    }
    if gains.is_empty() {
        return None;
    }

    let mut sorted: Vec<f64> = gains.iter().map(|&(db, _)| db).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("gains are finite"));
    let &(min_db, min_start) = gains
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).expect("gains are finite"))
        .expect("gains is non-empty");
    Some(ProjectionStats {
        n: gains.len(),
        below_3db: sorted.iter().filter(|&&db| db < -3.0).count(),
        below_6db: sorted.iter().filter(|&&db| db < -6.0).count(),
        median_db: percentile(&sorted, 0.50),
        p5_db: percentile(&sorted, 0.05),
        min_db,
        min_at_s: min_start as f64 / f64::from(rate),
    })
}

/// Energy reduction `10 * log10(E_near / E_out)` over the union of the
/// selected windows' samples (a mask, so overlapping windows do not double
/// count), plus the selected-window count. `None` when no window is selected
/// or the masked near end is silent.
pub fn masked_reduction_db<F: Fn(usize) -> bool>(
    near: &[f32],
    out: &[f32],
    wa: &WindowAnalysis,
    select: F,
) -> (Option<f64>, usize) {
    let span = near.len().min(out.len());
    let mut mask = vec![false; span];
    let mut count = 0usize;
    for (i, &s) in wa.starts.iter().enumerate() {
        if !select(i) {
            continue;
        }
        count += 1;
        for flag in mask
            .iter_mut()
            .skip(s)
            .take(wa.window_len.min(span - s.min(span)))
        {
            *flag = true;
        }
    }
    if count == 0 {
        return (None, 0);
    }
    let mut e_near = 0.0f64;
    let mut e_out = 0.0f64;
    for k in 0..span {
        if mask[k] {
            e_near += near[k] as f64 * near[k] as f64;
            e_out += out[k] as f64 * out[k] as f64;
        }
    }
    if e_near <= 0.0 {
        return (None, count);
    }
    let db = if e_out <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (e_near / e_out).log10()
    };
    (Some(db), count)
}

/// The harness's own echo-delay estimate, independent of the engine.
pub struct DelayEstimate {
    /// Lag of the envelope correlation peak, milliseconds (1 ms resolution).
    pub lag_ms: usize,
    /// Normalized correlation coefficient at the peak.
    pub corr: f64,
}

/// Estimates the far-to-near echo delay by normalized cross-correlation of
/// 1 kHz magnitude envelopes, searching lags 0 to [`DELAY_SEARCH_MAX_MS`].
///
/// Returns `None` when the far end carries no energy above
/// [`ACTIVE_FLOOR_DBFS`] (nothing to correlate, the nearend-singletalk case).
pub fn estimate_delay(near: &[f32], far: &[f32], rate: u32) -> Option<DelayEstimate> {
    debug_assert_eq!(rate as usize % (ENV_DECIM * 1000), 0);
    let envelope = |x: &[f32]| -> Vec<f64> {
        x.chunks(ENV_DECIM)
            .map(|c| c.iter().map(|&s| (s as f64).abs()).sum::<f64>() / c.len() as f64)
            .collect()
    };
    let mut env_far = envelope(far);
    let mut env_near = envelope(near);
    let floor = 10.0f64.powf(ACTIVE_FLOOR_DBFS / 20.0);
    if !env_far.iter().any(|&v| v > floor) {
        return None;
    }

    let len = env_far.len().min(env_near.len());
    env_far.truncate(len);
    env_near.truncate(len);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let m_far = mean(&env_far);
    let m_near = mean(&env_near);
    for v in env_far.iter_mut() {
        *v -= m_far;
    }
    for v in env_near.iter_mut() {
        *v -= m_near;
    }

    // Prefix sums of squares give each lag's overlap norms in O(1), so the
    // whole search is one pass per lag over the overlap.
    let prefix_sq = |v: &[f64]| -> Vec<f64> {
        let mut acc = 0.0;
        let mut out = Vec::with_capacity(v.len() + 1);
        out.push(0.0);
        for &x in v {
            acc += x * x;
            out.push(acc);
        }
        out
    };
    let far_sq = prefix_sq(&env_far);
    let near_sq = prefix_sq(&env_near);

    let max_lag = DELAY_SEARCH_MAX_MS.min(len / 2);
    let mut best = DelayEstimate {
        lag_ms: 0,
        corr: 0.0,
    };
    for lag in 0..=max_lag {
        let n = len - lag;
        if n == 0 {
            break;
        }
        let mut cross = 0.0f64;
        for i in 0..n {
            cross += env_far[i] * env_near[i + lag];
        }
        let norm_far = far_sq[n] - far_sq[0];
        let norm_near = near_sq[lag + n] - near_sq[lag];
        let denom = (norm_far * norm_near).sqrt();
        if denom <= 0.0 {
            continue;
        }
        let corr = cross / denom;
        if corr > best.corr {
            best = DelayEstimate { lag_ms: lag, corr };
        }
    }
    Some(best)
}
