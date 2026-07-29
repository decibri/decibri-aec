//! Coherence census: an engine-free survey of how linearly predictable the far
//! reference is across a clip pool.
//!
//! For every `<stem>_mic.wav` and `<stem>_lpb.wav` pair under a scenario folder
//! it computes, from the raw signals alone, quantities that describe the
//! recording and nothing about any canceller: the best near-to-far
//! magnitude-squared coherence over a delay scan and the lag it occurs at, the
//! linear echo-reduction ceiling that coherence implies, the distribution of
//! one-second-window coherence across three fixed bins, a two-part stationarity
//! measure (how far the windowed coherence and its best lag move across the
//! clip), near and far levels, far duty, and whether the far reference carries
//! meaningful energy at all.
//!
//! It constructs no [`decibri_aec`] type, needs no lock, and measures nothing
//! about the engine: the crate is not imported here. The only non-standard
//! dependencies are WAV decode and input-rate conversion, matching the other
//! bench examples.
//!
//! ```text
//! cargo run --release --example coherence-census -- \
//!     --pool data/test_set_icassp2022 --out data/census
//! ```
//!
//! The output is written to a gitignored path (everything under `data/` is
//! ignored): a JSON with one record per clip plus an aggregate summary, and a
//! readable text summary.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[path = "shared/resample.rs"]
mod resample;

/// The rate the coherence is measured at, matching the shipped engine rate.
const ENGINE_RATE: u32 = 16_000;

/// Measurement protocol version recorded in the census artifact, matching the
/// benchmark kit's: protocol 2 converts input rates through the shared
/// lag-zero, exact-length contract.
const PROTOCOL: u32 = 2;

/// The three scenario folders under the pool root.
const SCENARIOS: [&str; 3] = ["doubletalk", "farend-singletalk", "nearend-singletalk"];

/// Block length for level statistics, in samples.
const LEVEL_BLOCK: usize = 256;

/// How far below the loud level a block may sit and still count as active, in
/// decibels, for the far duty measure.
const FAR_ACTIVE_REL_DB: f64 = 20.0;

/// Welch segment length for the coherence estimate, in samples: 256 ms at the
/// engine rate.
const COH_SEG: usize = 4096;

/// Welch hop for the accurate coherence at a fixed delay, in samples: 75 percent
/// overlap.
const COH_HOP: usize = 1024;

/// Welch hop used while scanning delays, in samples: 50 percent overlap, half
/// the segments, to locate the best delay cheaply.
const COH_HOP_SCAN: usize = 2048;

/// One-second window length for the coherence and lag time courses, in samples.
const COH_WINDOW: usize = ENGINE_RATE as usize;

/// Upper bound of the delay scan, in milliseconds.
const DELAY_SCAN_MAX_MS: usize = 1000;

/// Delay scan step, in milliseconds.
const DELAY_STEP_MS: usize = 10;

/// Decimation from the engine rate to the 1 kHz envelope rate the lag search
/// runs at.
const ENV_DECIM: usize = 16;

/// A near or far window is scored only when its RMS sits above this absolute
/// floor in dBFS.
const COH_ACTIVE_FLOOR_DBFS: f64 = -60.0;

/// Number of equal time chunks the clip is split into for the robust coherence
/// stationarity measure.
const COH_CHUNKS: usize = 4;

/// A per-window lag vote is kept only when its best normalized envelope
/// correlation reaches this level, so windows with no genuine alignment peak do
/// not contribute noise to the lag movement.
const LAG_CORR_GATE: f64 = 0.5;

/// A clip's far reference is treated as carrying no meaningful energy, and the
/// clip is reported not-applicable, when its loud (95th percentile) block level
/// sits below this floor in dBFS.
const FAR_ENERGY_FLOOR_DBFS: f64 = -60.0;

/// Lower coherence bin edge: the shipped detector's `PROTECT_COHERENCE`, read
/// from `src/tau.rs`. Coherence below this is the low-predictability bin.
const BIN_LOW_EDGE: f64 = 0.2;

/// Upper coherence bin edge: the shipped detector's `FULL_SPEED_COHERENCE`, read
/// from `src/tau.rs`. Coherence at or above this is the high-predictability bin.
const BIN_HIGH_EDGE: f64 = 0.5;

/// Half-width of the per-window lag search around the clip's bulk delay, in
/// milliseconds.
const LAG_SEARCH_HALF_MS: usize = 64;

/// A clip is flagged nonstationary on the lag axis when its best per-window lag
/// moves by more than this many milliseconds across the clip.
const NONSTATIONARY_LAG_MS: f64 = 10.0;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Args::parse(&args).and_then(|args| run(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: cargo run --release --example coherence-census -- \
                 --pool <clip-root> [--out <dir>]"
            );
            ExitCode::FAILURE
        }
    }
}

/// One parsed invocation.
struct Args {
    pool: PathBuf,
    out: PathBuf,
}

impl Args {
    fn parse(args: &[String]) -> Result<Args, String> {
        let mut pool = None;
        let mut out = None;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--pool" => {
                    pool = Some(PathBuf::from(
                        iter.next().cloned().ok_or("--pool needs a value")?,
                    ))
                }
                "--out" => {
                    out = Some(PathBuf::from(
                        iter.next().cloned().ok_or("--out needs a value")?,
                    ))
                }
                other => return Err(format!("unknown argument '{other}'")),
            }
        }
        let pool = pool.ok_or("--pool is required")?;
        let out = out.unwrap_or_else(|| PathBuf::from("data/census"));
        Ok(Args { pool, out })
    }
}

/// One clip to process.
struct Task {
    scenario: String,
    stem: String,
    mic: PathBuf,
    lpb: PathBuf,
}

fn run(args: &Args) -> Result<(), String> {
    let mut tasks: Vec<Task> = Vec::new();
    for scenario in SCENARIOS {
        let dir = args.pool.join(scenario);
        let entries =
            fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        let mut stems: Vec<String> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot read entry: {e}"))?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix("_mic.wav") {
                stems.push(stem.to_string());
            }
        }
        stems.sort();
        for stem in stems {
            tasks.push(Task {
                scenario: scenario.to_string(),
                stem: stem.clone(),
                mic: dir.join(format!("{stem}_mic.wav")),
                lpb: dir.join(format!("{stem}_lpb.wav")),
            });
        }
    }
    if tasks.is_empty() {
        return Err(format!("no clip pairs found under {}", args.pool.display()));
    }
    eprintln!("census: {} clip pairs found", tasks.len());

    let started = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let chunk = tasks.len().div_ceil(workers);

    let mut records: Vec<Record> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for group in tasks.chunks(chunk) {
            handles.push(scope.spawn(move || {
                let mut local: Vec<Record> = Vec::new();
                for task in group {
                    match census_clip(task) {
                        Ok(record) => local.push(record),
                        Err(message) => {
                            eprintln!("  {}: FAILED: {message}", task.stem)
                        }
                    }
                }
                local
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    let elapsed = started.elapsed().as_secs_f64();

    records.sort_by(|a, b| {
        (a.scenario.as_str(), a.stem.as_str()).cmp(&(b.scenario.as_str(), b.stem.as_str()))
    });

    fs::create_dir_all(&args.out)
        .map_err(|e| format!("cannot create {}: {e}", args.out.display()))?;
    let json = build_json(&records, elapsed, tasks.len());
    let text = build_text(&records, elapsed);
    let json_path = args.out.join("coherence-census.json");
    let text_path = args.out.join("coherence-census.txt");
    fs::write(&json_path, json)
        .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
    fs::write(&text_path, &text)
        .map_err(|e| format!("cannot write {}: {e}", text_path.display()))?;

    print!("{text}");
    eprintln!(
        "census: wrote {} and {} in {elapsed:.1}s",
        json_path.display(),
        text_path.display()
    );
    Ok(())
}

/// The predictability bin of a coherence value, using the frozen edges.
fn bin_of(coherence: f64) -> &'static str {
    if coherence < BIN_LOW_EDGE {
        "low"
    } else if coherence < BIN_HIGH_EDGE {
        "marginal"
    } else {
        "high"
    }
}

/// One clip's census record: only clip-intrinsic quantities.
struct Record {
    scenario: String,
    stem: String,
    with_movement: bool,
    duration_s: f64,
    near_len: usize,
    far_len: usize,
    near_med_dbfs: f64,
    far_med_dbfs: f64,
    far_loud_dbfs: f64,
    far_duty_pct: f64,
    far_has_energy: bool,
    not_applicable: bool,
    best_coh: f64,
    best_coh_delay_ms: f64,
    max_linear_erle_db: f64,
    coh_window_count: usize,
    coh_win_below_low_pct: f64,
    coh_win_mid_pct: f64,
    coh_win_above_high_pct: f64,
    coh_win_mean: f64,
    coh_chunk_count: usize,
    coh_chunk_min: f64,
    coh_chunk_max: f64,
    coh_chunk_first: f64,
    coh_chunk_last: f64,
    coh_chunk_span: f64,
    coherence_nonstationary: bool,
    lag_window_count: usize,
    lag_min_ms: f64,
    lag_max_ms: f64,
    lag_span_ms: f64,
    lag_std_ms: f64,
    lag_nonstationary: bool,
    nonstationary: bool,
    predictability_bin: String,
}

/// Computes one clip's census record from its raw signal pair.
fn census_clip(task: &Task) -> Result<Record, String> {
    let near = to_engine_rate(wav::read_mono(&task.mic)?)?;
    let far = to_engine_rate(wav::read_mono(&task.lpb)?)?;
    let duration_s = near.len() as f64 / f64::from(ENGINE_RATE);

    // Far-reference character and the not-applicable test.
    let near_med = median_block_dbfs(&near, LEVEL_BLOCK);
    let far_med = median_block_dbfs(&far, LEVEL_BLOCK);
    let far_loud = block_dbfs_pct(&far, LEVEL_BLOCK, 0.95);
    let far_duty = far_duty_cycle(&far, LEVEL_BLOCK);
    let far_has_energy = far_loud > FAR_ENERGY_FLOOR_DBFS;
    let not_applicable = !far_has_energy;

    // The best coherence over the delay scan, the lag it occurs at, and the
    // linear echo-reduction ceiling it implies. Skipped for not-applicable
    // clips, whose coherence carries no meaning.
    let (best_coh, best_delay) = if not_applicable {
        (0.0, 0usize)
    } else {
        best_coherence(&near, &far)
    };
    let best_coh_delay_ms = best_delay as f64 * 1000.0 / f64::from(ENGINE_RATE);
    let max_linear_erle = if best_coh < 1.0 {
        -10.0 * (1.0 - best_coh).log10()
    } else {
        f64::INFINITY
    };

    // The one-second-window coherence distribution across the three bins, at the
    // best delay.
    let (windows, win_mean, below, mid, above) = if not_applicable {
        (0usize, 0.0, 0.0, 0.0, 0.0)
    } else {
        window_coherence(&near, &far, best_delay)
    };

    // The coherence stationarity, measured over a few robust time chunks where
    // both near and far are active: a clip whose predictability crosses a bin
    // between chunks is flagged coherence-nonstationary.
    let (chunks, chunk_min, chunk_max, chunk_first, chunk_last) = if not_applicable {
        (0usize, 0.0, 0.0, 0.0, 0.0)
    } else {
        chunk_coherence(&near, &far, best_delay)
    };
    let chunk_span = if chunks > 0 {
        chunk_max - chunk_min
    } else {
        0.0
    };
    let coherence_nonstationary = chunks >= 2 && bin_of(chunk_min) != bin_of(chunk_max);

    // The lag stationarity: how far the best per-window lag moves, counting only
    // windows whose envelope correlation clears the gate.
    let (lag_windows, lag_min, lag_max, lag_std) = if not_applicable {
        (0usize, 0.0, 0.0, 0.0)
    } else {
        window_lag_course(&near, &far, best_delay)
    };
    let lag_span = if lag_windows > 0 {
        lag_max - lag_min
    } else {
        0.0
    };
    let lag_nonstationary = lag_windows >= 2 && lag_span > NONSTATIONARY_LAG_MS;

    let nonstationary = !not_applicable && (coherence_nonstationary || lag_nonstationary);

    let predictability_bin = if not_applicable {
        "na".to_string()
    } else {
        bin_of(best_coh).to_string()
    };

    Ok(Record {
        scenario: task.scenario.clone(),
        stem: task.stem.clone(),
        with_movement: task.stem.contains("with-movement"),
        duration_s,
        near_len: near.len(),
        far_len: far.len(),
        near_med_dbfs: near_med,
        far_med_dbfs: far_med,
        far_loud_dbfs: far_loud,
        far_duty_pct: far_duty,
        far_has_energy,
        not_applicable,
        best_coh,
        best_coh_delay_ms,
        max_linear_erle_db: max_linear_erle,
        coh_window_count: windows,
        coh_win_below_low_pct: below,
        coh_win_mid_pct: mid,
        coh_win_above_high_pct: above,
        coh_win_mean: win_mean,
        coh_chunk_count: chunks,
        coh_chunk_min: chunk_min,
        coh_chunk_max: chunk_max,
        coh_chunk_first: chunk_first,
        coh_chunk_last: chunk_last,
        coh_chunk_span: chunk_span,
        coherence_nonstationary,
        lag_window_count: lag_windows,
        lag_min_ms: lag_min,
        lag_max_ms: lag_max,
        lag_span_ms: lag_span,
        lag_std_ms: lag_std,
        lag_nonstationary,
        nonstationary,
        predictability_bin,
    })
}

/// Brings a decoded clip to [`ENGINE_RATE`] through the shared
/// [`resample::resample_aligned`] contract, so the census measures the same
/// timeline the benchmark and `cancel` do.
fn to_engine_rate(clip: wav::MonoClip) -> Result<Vec<f32>, String> {
    resample::resample_aligned(&clip.samples, clip.sample_rate, ENGINE_RATE)
}

/// RMS of a span in dBFS; -120 for an empty or silent span.
fn rms_dbfs(samples: &[f32]) -> f64 {
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

/// The `q` quantile of an already sorted non-empty slice, by nearest rank.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Median block-RMS level of a signal in dBFS.
fn median_block_dbfs(x: &[f32], block: usize) -> f64 {
    let mut levels: Vec<f64> = x.chunks(block).map(rms_dbfs).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    percentile(&levels, 0.50)
}

/// The `q` quantile block-RMS level of a signal in dBFS.
fn block_dbfs_pct(x: &[f32], block: usize, q: f64) -> f64 {
    let mut levels: Vec<f64> = x.chunks(block).map(rms_dbfs).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    percentile(&levels, q)
}

/// The fraction of a signal's blocks, in percent, whose RMS sits within
/// [`FAR_ACTIVE_REL_DB`] of the signal's loud (95th percentile) block level.
fn far_duty_cycle(x: &[f32], block: usize) -> f64 {
    let levels: Vec<f64> = x.chunks(block).map(rms_dbfs).collect();
    if levels.is_empty() {
        return 0.0;
    }
    let mut sorted = levels.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let loud = percentile(&sorted, 0.95);
    if loud <= -119.0 {
        return 0.0;
    }
    let floor = loud - FAR_ACTIVE_REL_DB;
    levels.iter().filter(|&&db| db > floor).count() as f64 * 100.0 / levels.len() as f64
}

/// A Hann window of length `n`.
fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = std::f64::consts::PI * i as f64 / (n - 1) as f64;
            let s = x.sin();
            s * s
        })
        .collect()
}

/// The number of Welch segments of length [`COH_SEG`] at a given hop over a
/// signal of length `len`.
fn segment_count(len: usize, hop: usize) -> usize {
    if len < COH_SEG {
        0
    } else {
        (len - COH_SEG) / hop + 1
    }
}

/// Precomputed windowed near-segment spectra plus the per-bin near power, held
/// constant while the far delay is scanned.
struct NearSpectra {
    seg_re: Vec<Vec<f64>>,
    seg_im: Vec<Vec<f64>>,
    syy: Vec<f64>,
    bins: usize,
}

impl NearSpectra {
    /// Windows and transforms every near segment at the given hop once.
    fn build(near: &[f32], window: &[f64], hop: usize) -> NearSpectra {
        let bins = COH_SEG / 2 + 1;
        let count = segment_count(near.len(), hop);
        let mut seg_re = Vec::with_capacity(count);
        let mut seg_im = Vec::with_capacity(count);
        let mut syy = vec![0.0f64; bins];
        for s in 0..count {
            let start = s * hop;
            let mut yr = vec![0.0f64; COH_SEG];
            let mut yi = vec![0.0f64; COH_SEG];
            for k in 0..COH_SEG {
                yr[k] = near[start + k] as f64 * window[k];
            }
            fft(&mut yr, &mut yi);
            for b in 0..bins {
                syy[b] += yr[b] * yr[b] + yi[b] * yi[b];
            }
            seg_re.push(yr);
            seg_im.push(yi);
        }
        NearSpectra {
            seg_re,
            seg_im,
            syy,
            bins,
        }
    }

    /// The near-power-weighted coherent fraction of far against the cached near
    /// spectra, with far read delayed by `delay` samples. Reuses the near FFTs;
    /// only the far segment FFTs are formed here.
    fn coherent_fraction(&self, far: &[f32], window: &[f64], hop: usize, delay: usize) -> f64 {
        if self.seg_re.is_empty() {
            return 0.0;
        }
        let bins = self.bins;
        let mut sxx = vec![0.0f64; bins];
        let mut sxy_re = vec![0.0f64; bins];
        let mut sxy_im = vec![0.0f64; bins];
        let mut xr = vec![0.0f64; COH_SEG];
        let mut xi = vec![0.0f64; COH_SEG];
        for (s, (nr, ni)) in self.seg_re.iter().zip(self.seg_im.iter()).enumerate() {
            let start = s * hop;
            for k in 0..COH_SEG {
                let idx = start as isize + k as isize - delay as isize;
                xr[k] = if idx >= 0 && (idx as usize) < far.len() {
                    far[idx as usize] as f64 * window[k]
                } else {
                    0.0
                };
                xi[k] = 0.0;
            }
            fft(&mut xr, &mut xi);
            for b in 0..bins {
                sxx[b] += xr[b] * xr[b] + xi[b] * xi[b];
                sxy_re[b] += xr[b] * nr[b] + xi[b] * ni[b];
                sxy_im[b] += xr[b] * ni[b] - xi[b] * nr[b];
            }
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for b in 0..bins {
            let cross = sxy_re[b] * sxy_re[b] + sxy_im[b] * sxy_im[b];
            let denom = sxx[b] * self.syy[b];
            let msc = if denom > 0.0 { cross / denom } else { 0.0 };
            num += msc * self.syy[b];
            den += self.syy[b];
        }
        if den > 0.0 {
            (num / den).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// The best broadband coherent fraction over the delay scan and the delay in
/// samples that achieves it. The scan runs at a coarse hop over a cached near
/// spectrum to locate the delay cheaply, then the coherence at that delay and
/// its two neighbours is recomputed at the accurate hop, and the largest is
/// returned.
fn best_coherence(near: &[f32], far: &[f32]) -> (f64, usize) {
    let window = hann(COH_SEG);
    if segment_count(near.len(), COH_HOP_SCAN) == 0 {
        return (0.0, 0);
    }
    let scan = NearSpectra::build(near, &window, COH_HOP_SCAN);
    let step = DELAY_STEP_MS * ENGINE_RATE as usize / 1000;
    let max_delay = DELAY_SCAN_MAX_MS * ENGINE_RATE as usize / 1000;
    let mut best_scan = (0.0f64, 0usize);
    let mut delay = 0usize;
    while delay <= max_delay {
        let frac = scan.coherent_fraction(far, &window, COH_HOP_SCAN, delay);
        if frac > best_scan.0 {
            best_scan = (frac, delay);
        }
        delay += step;
    }

    // Refine the reported ceiling at the accurate hop around the scan's argmax.
    let accurate = NearSpectra::build(near, &window, COH_HOP);
    let mut best = (0.0f64, best_scan.1);
    for d in [
        best_scan.1.wrapping_sub(step),
        best_scan.1,
        best_scan.1 + step,
    ] {
        if d > max_delay {
            continue;
        }
        let frac = accurate.coherent_fraction(far, &window, COH_HOP, d);
        if frac > best.0 {
            best = (frac, d);
        }
    }
    best
}

/// The near-to-far coherent fraction over successive one-second windows,
/// near-active windows only, at a fixed delay. Returns the window count, the
/// mean coherent fraction, and the percentage of windows in each of the three
/// fixed bins.
fn window_coherence(near: &[f32], far: &[f32], delay: usize) -> (usize, f64, f64, f64, f64) {
    let window = hann(COH_SEG);
    let far_aligned = align_far(far, near.len(), delay);
    let mut values: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start + COH_WINDOW <= near.len() {
        let nw = &near[start..start + COH_WINDOW];
        if rms_dbfs(nw) > COH_ACTIVE_FLOOR_DBFS {
            let fw = &far_aligned[start..start + COH_WINDOW];
            let near_spec = NearSpectra::build(nw, &window, COH_HOP);
            values.push(near_spec.coherent_fraction(fw, &window, COH_HOP, 0));
        }
        start += COH_WINDOW;
    }
    if values.is_empty() {
        return (0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let below = values.iter().filter(|&&v| v < BIN_LOW_EDGE).count() as f64 * 100.0 / n as f64;
    let above = values.iter().filter(|&&v| v >= BIN_HIGH_EDGE).count() as f64 * 100.0 / n as f64;
    let mid = 100.0 - below - above;
    (n, mean, below, mid, above)
}

/// The near-to-far coherent fraction over [`COH_CHUNKS`] equal time chunks at a
/// fixed delay, counting only chunks where both near and the aligned far region
/// are active and long enough to estimate. Returns the scored-chunk count and
/// the min, max, first and last chunk coherent fraction. Each chunk spans many
/// segments, so a chunk estimate is far less noisy than a one-second window.
fn chunk_coherence(near: &[f32], far: &[f32], delay: usize) -> (usize, f64, f64, f64, f64) {
    let window = hann(COH_SEG);
    let far_aligned = align_far(far, near.len(), delay);
    let chunk = near.len() / COH_CHUNKS;
    if chunk < COH_SEG {
        return (0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut values: Vec<f64> = Vec::new();
    for c in 0..COH_CHUNKS {
        let start = c * chunk;
        let end = start + chunk;
        let nw = &near[start..end];
        let fw = &far_aligned[start..end];
        if rms_dbfs(nw) > COH_ACTIVE_FLOOR_DBFS && rms_dbfs(fw) > COH_ACTIVE_FLOOR_DBFS {
            let near_spec = NearSpectra::build(nw, &window, COH_HOP);
            values.push(near_spec.coherent_fraction(fw, &window, COH_HOP, 0));
        }
    }
    if values.is_empty() {
        return (0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (n, min, max, values[0], values[n - 1])
}

/// Builds a far array time-aligned to the near timeline: `aligned[t]` is
/// `far[t - delay]`, or zero where that index is out of range.
fn align_far(far: &[f32], len: usize, delay: usize) -> Vec<f32> {
    let mut aligned = vec![0.0f32; len];
    for (t, slot) in aligned.iter_mut().enumerate() {
        if t >= delay {
            let idx = t - delay;
            if idx < far.len() {
                *slot = far[idx];
            }
        }
    }
    aligned
}

/// The best per-window lag around the clip's bulk delay, over one-second
/// windows where both near and the aligned far region are active. Returns the
/// window count and the min, max and standard deviation of the absolute lag in
/// milliseconds.
fn window_lag_course(near: &[f32], far: &[f32], bulk_delay: usize) -> (usize, f64, f64, f64) {
    let env_near = envelope(near, ENV_DECIM);
    let env_far = envelope(far, ENV_DECIM);
    let bulk_env = bulk_delay / ENV_DECIM;
    let win_env = COH_WINDOW / ENV_DECIM;
    let half = LAG_SEARCH_HALF_MS;
    let bulk_ms = bulk_delay as f64 * 1000.0 / f64::from(ENGINE_RATE);

    let mut lags: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start + COH_WINDOW <= near.len() {
        let nw = &near[start..start + COH_WINDOW];
        if rms_dbfs(nw) > COH_ACTIVE_FLOOR_DBFS {
            let ws = start / ENV_DECIM;
            if let Some(offset) = best_lag_offset(&env_near, &env_far, ws, win_env, bulk_env, half)
            {
                lags.push(bulk_ms + offset);
            }
        }
        start += COH_WINDOW;
    }
    if lags.is_empty() {
        return (0, 0.0, 0.0, 0.0);
    }
    let n = lags.len();
    let min = lags.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = lags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = lags.iter().sum::<f64>() / n as f64;
    let var = lags.iter().map(|&l| (l - mean) * (l - mean)).sum::<f64>() / n as f64;
    (n, min, max, var.sqrt())
}

/// The offset in milliseconds, within +/- `half` of the bulk delay, at which a
/// one-second near envelope window best aligns to the far envelope. Returns
/// `None` when the far region carries no energy over the search span, or when
/// the best normalized correlation does not reach [`LAG_CORR_GATE`], so windows
/// without a genuine alignment peak contribute no lag vote.
fn best_lag_offset(
    env_near: &[f64],
    env_far: &[f64],
    win_start: usize,
    win_len: usize,
    bulk_env: usize,
    half: usize,
) -> Option<f64> {
    let near_end = (win_start + win_len).min(env_near.len());
    if near_end <= win_start {
        return None;
    }
    let nw = &env_near[win_start..near_end];
    let m_near = nw.iter().sum::<f64>() / nw.len() as f64;
    let near_centered: Vec<f64> = nw.iter().map(|&v| v - m_near).collect();
    let near_norm: f64 = near_centered.iter().map(|&v| v * v).sum();
    if near_norm <= 0.0 {
        return None;
    }
    let mut best: Option<(f64, f64)> = None;
    let mut any_far = false;
    for off in -(half as isize)..=(half as isize) {
        let base = win_start as isize - bulk_env as isize - off;
        let mut cross = 0.0f64;
        let mut far_norm = 0.0f64;
        for (i, &nc) in near_centered.iter().enumerate() {
            let fi = base + i as isize;
            let fv = if fi >= 0 && (fi as usize) < env_far.len() {
                env_far[fi as usize]
            } else {
                0.0
            };
            cross += nc * fv;
            far_norm += fv * fv;
        }
        if far_norm > 0.0 {
            any_far = true;
            let corr = cross / (near_norm * far_norm).sqrt();
            if best.map(|(c, _)| corr > c).unwrap_or(true) {
                best = Some((corr, off as f64));
            }
        }
    }
    if any_far {
        best.filter(|(corr, _)| *corr >= LAG_CORR_GATE)
            .map(|(_, off)| off)
    } else {
        None
    }
}

/// The 1 kHz magnitude envelope of a signal, decimated by `decim`.
fn envelope(x: &[f32], decim: usize) -> Vec<f64> {
    x.chunks(decim)
        .map(|c| c.iter().map(|&s| (s as f64).abs()).sum::<f64>() / c.len() as f64)
        .collect()
}

/// In-place iterative radix-2 Cooley-Tukey FFT over separate real and imaginary
/// buffers, whose length must be a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();
        let mut i = 0usize;
        while i < n {
            let mut w_re = 1.0f64;
            let mut w_im = 0.0f64;
            for k in 0..len / 2 {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + len / 2] * w_re - im[i + k + len / 2] * w_im;
                let v_im = re[i + k + len / 2] * w_im + im[i + k + len / 2] * w_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                let nw_re = w_re * wlen_re - w_im * wlen_im;
                let nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Formats a float for JSON, emitting `"inf"` for a non-finite value.
fn jf(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "\"inf\"".to_string()
    }
}

/// The JSON object for one record.
fn record_json(r: &Record) -> String {
    format!(
        "{{\"stem\":\"{}\",\"scenario\":\"{}\",\"with_movement\":{},\
         \"duration_s\":{},\"near_len\":{},\"far_len\":{},\
         \"near_med_dbfs\":{},\"far_med_dbfs\":{},\"far_loud_dbfs\":{},\
         \"far_duty_pct\":{},\"far_has_energy\":{},\"not_applicable\":{},\
         \"predictability_bin\":\"{}\",\
         \"best_coh\":{},\"best_coh_delay_ms\":{},\"max_linear_erle_db\":{},\
         \"coh_window_count\":{},\"coh_win_below_low_pct\":{},\"coh_win_mid_pct\":{},\
         \"coh_win_above_high_pct\":{},\"coh_win_mean\":{},\
         \"coh_chunk_count\":{},\"coh_chunk_min\":{},\"coh_chunk_max\":{},\
         \"coh_chunk_first\":{},\"coh_chunk_last\":{},\"coh_chunk_span\":{},\
         \"coherence_nonstationary\":{},\"lag_window_count\":{},\"lag_min_ms\":{},\
         \"lag_max_ms\":{},\"lag_span_ms\":{},\"lag_std_ms\":{},\
         \"lag_nonstationary\":{},\"nonstationary\":{}}}",
        r.stem,
        r.scenario,
        r.with_movement,
        jf(r.duration_s),
        r.near_len,
        r.far_len,
        jf(r.near_med_dbfs),
        jf(r.far_med_dbfs),
        jf(r.far_loud_dbfs),
        jf(r.far_duty_pct),
        r.far_has_energy,
        r.not_applicable,
        r.predictability_bin,
        jf(r.best_coh),
        jf(r.best_coh_delay_ms),
        jf(r.max_linear_erle_db),
        r.coh_window_count,
        jf(r.coh_win_below_low_pct),
        jf(r.coh_win_mid_pct),
        jf(r.coh_win_above_high_pct),
        jf(r.coh_win_mean),
        r.coh_chunk_count,
        jf(r.coh_chunk_min),
        jf(r.coh_chunk_max),
        jf(r.coh_chunk_first),
        jf(r.coh_chunk_last),
        jf(r.coh_chunk_span),
        r.coherence_nonstationary,
        r.lag_window_count,
        jf(r.lag_min_ms),
        jf(r.lag_max_ms),
        jf(r.lag_span_ms),
        jf(r.lag_std_ms),
        r.lag_nonstationary,
        r.nonstationary,
    )
}

/// Per-group tallies over a set of records.
struct Tally {
    total: usize,
    na: usize,
    low: usize,
    marginal: usize,
    high: usize,
    nonstationary: usize,
    ns_coherence: usize,
    ns_lag: usize,
    with_movement: usize,
    ns_and_movement: usize,
    ns_no_movement: usize,
}

impl Tally {
    fn of(records: &[&Record]) -> Tally {
        let mut t = Tally {
            total: records.len(),
            na: 0,
            low: 0,
            marginal: 0,
            high: 0,
            nonstationary: 0,
            ns_coherence: 0,
            ns_lag: 0,
            with_movement: 0,
            ns_and_movement: 0,
            ns_no_movement: 0,
        };
        for r in records {
            if r.with_movement {
                t.with_movement += 1;
            }
            if r.not_applicable {
                t.na += 1;
                continue;
            }
            match r.predictability_bin.as_str() {
                "low" => t.low += 1,
                "marginal" => t.marginal += 1,
                "high" => t.high += 1,
                _ => {}
            }
            if r.coherence_nonstationary {
                t.ns_coherence += 1;
            }
            if r.lag_nonstationary {
                t.ns_lag += 1;
            }
            if r.nonstationary {
                t.nonstationary += 1;
                if r.with_movement {
                    t.ns_and_movement += 1;
                } else {
                    t.ns_no_movement += 1;
                }
            }
        }
        t
    }

    fn applicable(&self) -> usize {
        self.total - self.na
    }

    fn json(&self) -> String {
        format!(
            "{{\"total\":{},\"not_applicable\":{},\"applicable\":{},\
             \"low\":{},\"marginal\":{},\"high\":{},\"nonstationary\":{},\
             \"nonstationary_coherence\":{},\"nonstationary_lag\":{},\
             \"with_movement\":{},\"nonstationary_and_movement\":{},\
             \"nonstationary_no_movement\":{}}}",
            self.total,
            self.na,
            self.applicable(),
            self.low,
            self.marginal,
            self.high,
            self.nonstationary,
            self.ns_coherence,
            self.ns_lag,
            self.with_movement,
            self.ns_and_movement,
            self.ns_no_movement,
        )
    }
}

/// The full census JSON: metadata, aggregate, and every record.
fn build_json(records: &[Record], elapsed: f64, pairs: usize) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"schema\": \"decibri-aec-census/v1\",\n");
    s.push_str(&format!("  \"protocol\": {PROTOCOL},\n"));
    s.push_str("  \"engine_free\": true,\n");
    s.push_str(&format!("  \"coherence_rate_hz\": {ENGINE_RATE},\n"));
    s.push_str(&format!(
        "  \"bin_low_edge\": {BIN_LOW_EDGE}, \"bin_high_edge\": {BIN_HIGH_EDGE},\n"
    ));
    s.push_str(&format!(
        "  \"far_energy_floor_dbfs\": {FAR_ENERGY_FLOOR_DBFS}, \
         \"nonstationary_lag_ms\": {NONSTATIONARY_LAG_MS},\n"
    ));
    s.push_str(&format!(
        "  \"delay_scan_max_ms\": {DELAY_SCAN_MAX_MS}, \"delay_step_ms\": {DELAY_STEP_MS},\n"
    ));
    s.push_str(&format!("  \"pairs_found\": {pairs},\n"));
    s.push_str(&format!("  \"runtime_s\": {elapsed:.2},\n"));

    s.push_str("  \"aggregate\": {\n");
    let overall: Vec<&Record> = records.iter().collect();
    s.push_str(&format!(
        "    \"overall\": {},\n",
        Tally::of(&overall).json()
    ));
    for (i, scenario) in SCENARIOS.iter().enumerate() {
        let group: Vec<&Record> = records.iter().filter(|r| r.scenario == *scenario).collect();
        let comma = if i + 1 < SCENARIOS.len() { "," } else { "" };
        s.push_str(&format!(
            "    \"{scenario}\": {}{comma}\n",
            Tally::of(&group).json()
        ));
    }
    s.push_str("  },\n");

    s.push_str("  \"records\": [\n");
    for (i, r) in records.iter().enumerate() {
        let comma = if i + 1 < records.len() { "," } else { "" };
        s.push_str(&format!("    {}{comma}\n", record_json(r)));
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

/// A readable text summary: the per-scenario and overall distribution.
fn build_text(records: &[Record], elapsed: f64) -> String {
    let mut s = String::new();
    s.push_str("Coherence census (engine-free)\n");
    s.push_str(&format!(
        "bins: low < {BIN_LOW_EDGE}, marginal < {BIN_HIGH_EDGE}, high >= {BIN_HIGH_EDGE}  \
         (read from src/tau.rs)\n"
    ));
    s.push_str(&format!(
        "not-applicable: far loud level < {FAR_ENERGY_FLOOR_DBFS} dBFS   \
         nonstationary lag: > {NONSTATIONARY_LAG_MS} ms\n"
    ));
    s.push_str(&format!("runtime: {elapsed:.1}s\n\n"));
    s.push_str(&format!(
        "{:<20} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5}\n",
        "group", "clips", "na", "low", "marg", "high", "nonst", "ns-co", "ns-lg", "movmt", "ns+mv"
    ));
    let row = |s: &mut String, name: &str, t: &Tally| {
        s.push_str(&format!(
            "{:<20} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5}\n",
            name,
            t.total,
            t.na,
            t.low,
            t.marginal,
            t.high,
            t.nonstationary,
            t.ns_coherence,
            t.ns_lag,
            t.with_movement,
            t.ns_and_movement,
        ));
    };
    let overall: Vec<&Record> = records.iter().collect();
    row(&mut s, "overall", &Tally::of(&overall));
    for scenario in SCENARIOS {
        let group: Vec<&Record> = records.iter().filter(|r| r.scenario == scenario).collect();
        row(&mut s, scenario, &Tally::of(&group));
    }
    s
}

/// WAV decode, the same implementation the other bench examples use, duplicated
/// rather than shared.
mod wav {
    use std::path::Path;

    /// A decoded mono clip.
    pub struct MonoClip {
        pub samples: Vec<f32>,
        pub sample_rate: u32,
    }

    /// Reads a WAV as mono `f32` in `[-1.0, 1.0]`.
    pub fn read_mono(path: &Path) -> Result<MonoClip, String> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let spec = reader.spec();
        let supported = match spec.sample_format {
            hound::SampleFormat::Float => spec.bits_per_sample == 32,
            hound::SampleFormat::Int => (1..=32).contains(&spec.bits_per_sample),
        };
        if !supported {
            return Err(format!(
                "unsupported bit depth in {}: {} bits {:?}",
                path.display(),
                spec.bits_per_sample,
                spec.sample_format
            ));
        }
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .map_err(|e| format!("cannot decode {}: {e}", path.display()))?,
            hound::SampleFormat::Int => {
                let full_scale = (1_i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / full_scale))
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("cannot decode {}: {e}", path.display()))?
            }
        };
        let channels = spec.channels.max(1) as usize;
        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                .collect()
        };
        Ok(MonoClip {
            samples,
            sample_rate: spec.sample_rate,
        })
    }
}
