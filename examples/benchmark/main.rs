//! One-command benchmark harness for the [`Aec`] engine over a folder of
//! AEC-Challenge-style clip pairs.
//!
//! ```text
//! cargo run --release --example benchmark -- <clip-root> [--set-name NAME]
//!     [--run-dir DIR | --out-root DIR] [--scenario NAME] [--limit N]
//! cargo run --release --example benchmark -- --split <manifest.json>
//!     [--set train|test] [--run LABEL] [--run-dir DIR | --out-root DIR] [--limit N]
//! ```
//!
//! `--run-dir DIR` puts one run's whole output in one self-contained folder:
//! `DIR/bench.json`, `DIR/bench.txt`, `DIR/enhanced/`, `DIR/aligned/`, and the
//! scoring and join steps write their own artifacts beside them. `--out-root
//! DIR` is the older layout, which fans a run out across `DIR/results/`,
//! `DIR/enhanced/<set>/` and `DIR/aligned/<set>/` with the run stamp in each
//! file name. The two are mutually exclusive; neither may name the crate root.
//!
//! `--limit N` caps the run at N pairs total, drawn stratified across the
//! scenarios in proportion to each scenario's share (largest-remainder
//! rounding) and taking the sorted-stem prefix within each, so a small limit
//! stays representative and selects the same pairs every time. It applies to
//! both input modes.
//!
//! Two input modes select which pairs run. The FOLDER mode takes a
//! `<clip-root>` and scores every pair it finds. The SPLIT mode takes a
//! `--split <manifest.json>` written by `examples/make-split.rs`, scores only
//! the pairs that manifest names for the chosen set (default `train`), and
//! APPENDS one result entry (per-scenario internal metrics) to the manifest's
//! `results` array, leaving every frozen split field byte-identical. The
//! optional AECMOS step then fills that same entry's `aecmos` field.
//!
//! `<clip-root>` holds up to three scenario folders (`doubletalk/`,
//! `farend-singletalk/`, `nearend-singletalk/`), each containing
//! `<stem>_mic.wav` / `<stem>_lpb.wav` pairs, the naming the Microsoft
//! AEC-Challenge test sets use. Every pair is resampled to the engine rate
//! with decibri-resampler, compensated for the resampler's reported latency
//! and pinned to the theoretical output length, run through the shipped
//! canceller (Tau, default configuration, estimator path, no delay hint),
//! measured, and written out. Each clip also yields a canonical aligned
//! triplet (`<stem>_mic-16k-aligned.wav`, `<stem>_lpb-16k-aligned.wav`,
//! `<stem>_enhanced-16k-aligned.wav`) under the gitignored output root: the
//! exact engine-rate samples the canceller consumed and produced, one shared
//! length and timeline, so the scoring step needs no resampling of its own.
//!
//! The harness prints a per-scenario summary table and preserves a
//! machine-comparable result: a versioned JSON file plus the same table as
//! text, stamped with the UTC time, the SHA-256 of every file under `src/`
//! (so a result is tied to the exact frozen source that produced it), the
//! set name, the input sample rate, and the scored clip portions. Processed
//! output WAVs land under the run folder, or under the gitignored
//! `data/bench-output/` folder when neither path flag is given, where the
//! optional AECMOS step (`benchmarks/run_aecmos.py`) can read them.
//!
//! Metric definitions and thresholds are the documented constants in this file
//! and `metrics.rs`; the result schema is the field set the JSON renderers
//! below emit, and every result carries the measurement protocol version.
//! The engine, turn cadence, and WAV handling mirror the `cancel` example.
//! Nothing in this file is shipped and nothing here alters the engine: it is a
//! measurement instrument over the public API only.

#![forbid(unsafe_code)]

mod manifest;
mod metrics;
mod provenance;
#[path = "../shared/resample.rs"]
mod resample;
mod wav;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use decibri_aec::{Aec, AecConfig, OutputTransitionPolicy, Suppression};

use metrics::{DelayEstimate, ProjectionStats};

/// The rate the engine runs at, matching `cancel`'s `ENGINE_RATE`.
const ENGINE_RATE: u32 = 16_000;

/// The per-turn chunk size, matching `cancel`'s `TURN`: 256 samples, one Tau
/// block, the cadence the crate's quality suite and the decibri capture
/// chain drive the engine with.
const TURN: usize = 256;

/// The unavailable-reference ceiling above which a clip is marked
/// NOT-MEASURED.
const STARVED_MEASURED_LIMIT_PCT: f64 = 10.0;

/// How far below the clip's loud level a far-end block may sit and still
/// count as far-active for the protection-rate metric.
const FAR_ACTIVE_REL_DB: f64 = 20.0;

/// When the engine's locked delay and the harness's own estimate disagree by
/// more than this many milliseconds (with a confident estimate), the clip is
/// flagged.
const DELAY_MISMATCH_MS: f64 = 20.0;

/// The AECMOS model location the optional scoring step reads, relative to
/// the crate root. The harness only checks whether the file exists, to print
/// the right note; it never reads or requires it.
const AECMOS_MODEL_REL: &str = "models/Run_1663915512_Stage_0.onnx";

/// Result schema identifier written into every JSON result.
const SCHEMA: &str = "decibri-aec-bench/internal/v3";

/// Measurement protocol version recorded in every result this kit writes.
const PROTOCOL: u32 = 2;

/// The run label recorded in a split result entry when `--run` is omitted.
/// The entry's UTC date already makes each run distinct; this only names it.
const DEFAULT_RUN_LABEL: &str = "dev";

/// Which side of a split the `--split` mode scores. Train is the default so
/// the held-back test set is never run by accident; `--set test` is the
/// deliberate end-of-cycle check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetKind {
    Train,
    Test,
}

impl SetKind {
    fn as_str(self) -> &'static str {
        match self {
            SetKind::Train => "train",
            SetKind::Test => "test",
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Cli::parse(&args).and_then(|cli| run(&cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: cargo run --release --example benchmark -- <clip-root> \
                 [--set-name NAME] [--run-dir DIR | --out-root DIR]\n\
                 \x20      [--scenario doubletalk|farend-singletalk|nearend-singletalk] \
                 [--limit N]\n\
                 \x20  or:  cargo run --release --example benchmark -- \
                 --split <manifest.json> [--set train|test] [--run LABEL] \
                 [--run-dir DIR | --out-root DIR] [--limit N]"
            );
            ExitCode::FAILURE
        }
    }
}

/// The three AEC-Challenge scenarios, in the order they are reported:
/// clean cancellation first, the hard case second, do-no-harm third.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    FarendSingletalk,
    Doubletalk,
    NearendSingletalk,
}

impl Scenario {
    const ALL: [Scenario; 3] = [
        Scenario::FarendSingletalk,
        Scenario::Doubletalk,
        Scenario::NearendSingletalk,
    ];

    /// The folder name under the clip root, which is also the reported name.
    fn dir_name(self) -> &'static str {
        match self {
            Scenario::FarendSingletalk => "farend-singletalk",
            Scenario::Doubletalk => "doubletalk",
            Scenario::NearendSingletalk => "nearend-singletalk",
        }
    }

    /// The AECMOS talk-type marker for this scenario, recorded per clip so
    /// the optional scoring step needs no mapping of its own.
    fn talk_type(self) -> &'static str {
        match self {
            Scenario::FarendSingletalk => "st",
            Scenario::Doubletalk => "dt",
            Scenario::NearendSingletalk => "nst",
        }
    }

    /// What the scenario proves, printed under each table heading.
    fn tagline(self) -> &'static str {
        match self {
            Scenario::FarendSingletalk => "pure echo; ERLE is meaningful here",
            Scenario::Doubletalk => "echo removal while the near end talks",
            Scenario::NearendSingletalk => "no echo; the output must not damage speech",
        }
    }
}

/// One parsed invocation. `root` (folder mode) and `split` (split mode) are
/// mutually exclusive; exactly one is required, checked in `run`.
struct Cli {
    root: Option<PathBuf>,
    set_name: Option<String>,
    out_root: Option<PathBuf>,
    /// One self-contained folder for this run's whole output. Mutually
    /// exclusive with `out_root`, checked in `resolve_output`.
    run_dir: Option<PathBuf>,
    scenario: Option<Scenario>,
    limit: Option<usize>,
    split: Option<PathBuf>,
    set: SetKind,
    run: Option<String>,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut root = None;
        let mut set_name = None;
        let mut out_root = None;
        let mut run_dir = None;
        let mut scenario = None;
        let mut limit = None;
        let mut split = None;
        let mut set = SetKind::Train;
        let mut run = None;

        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            let mut value = |name: &str| -> Result<String, String> {
                iter.next()
                    .cloned()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match arg.as_str() {
                "--set-name" => set_name = Some(value("--set-name")?),
                "--out-root" => out_root = Some(PathBuf::from(value("--out-root")?)),
                "--run-dir" => run_dir = Some(PathBuf::from(value("--run-dir")?)),
                "--scenario" => {
                    let raw = value("--scenario")?;
                    scenario = Some(
                        Scenario::ALL
                            .into_iter()
                            .find(|s| s.dir_name() == raw)
                            .ok_or_else(|| format!("unknown scenario '{raw}'"))?,
                    );
                }
                "--limit" => {
                    let raw = value("--limit")?;
                    limit = Some(
                        raw.parse::<usize>()
                            .map_err(|_| format!("--limit value '{raw}' is not a valid count"))?,
                    );
                }
                "--split" => split = Some(PathBuf::from(value("--split")?)),
                "--set" => {
                    let raw = value("--set")?;
                    set = match raw.as_str() {
                        "train" => SetKind::Train,
                        "test" => SetKind::Test,
                        other => {
                            return Err(format!("--set must be 'train' or 'test', got '{other}'"))
                        }
                    };
                }
                "--run" => run = Some(value("--run")?),
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag '{other}'"));
                }
                other => {
                    if root.is_some() {
                        return Err("expected exactly one <clip-root>".to_string());
                    }
                    root = Some(PathBuf::from(other));
                }
            }
        }

        Ok(Cli {
            root,
            set_name,
            out_root,
            run_dir,
            scenario,
            limit,
            split,
            set,
            run,
        })
    }
}

/// Builds the run configuration: the shipped `AecConfig` default at the engine
/// rate.
fn build_config() -> AecConfig {
    let mut config = AecConfig::default();
    config.sample_rate = ENGINE_RATE;
    config
}

/// One discovered `_mic` / `_lpb` pair.
struct Pair {
    stem: String,
    mic: PathBuf,
    lpb: PathBuf,
}

/// A clip brought to the engine rate, remembering its on-disk duration.
struct EngineClip {
    samples: Vec<f32>,
    input_seconds: f64,
}

/// Brings a decoded clip to [`ENGINE_RATE`] through the shared
/// [`resample::resample_aligned`] contract, remembering the clip's duration at
/// its on-disk rate.
fn to_engine_rate(clip: wav::MonoClip) -> Result<EngineClip, String> {
    let input_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
    let samples = resample::resample_aligned(&clip.samples, clip.sample_rate, ENGINE_RATE)?;
    Ok(EngineClip {
        samples,
        input_seconds,
    })
}

/// The samples cut or zero-padded to exactly `len`, placing a clip onto the
/// shared canonical timeline.
fn fit_len(mut samples: Vec<f32>, len: usize) -> Vec<f32> {
    samples.resize(len, 0.0);
    samples
}

/// Drives `aec` over one aligned pair with the kit's turn cadence: feed one
/// reference chunk, process one capture chunk, flush at the end. `on_turn`
/// sees the engine after each processed turn, for per-turn metrics.
fn run_canceller(
    aec: &mut Aec,
    near: &[f32],
    far: &[f32],
    mut on_turn: impl FnMut(&Aec, u64),
) -> Result<Vec<f32>, String> {
    let mut out: Vec<f32> = Vec::with_capacity(near.len() + TURN);
    let mut turns = 0u64;
    let mut far_chunks = far.chunks(TURN);
    for near_chunk in near.chunks(TURN) {
        if let Some(far_chunk) = far_chunks.next() {
            aec.feed_reference(far_chunk);
        }
        aec.process(near_chunk, &mut out)
            .map_err(|e| format!("processing failed: {e}"))?;
        turns += 1;
        on_turn(aec, turns);
    }
    aec.flush(&mut out)
        .map_err(|e| format!("flush failed: {e}"))?;
    Ok(out)
}

/// Energy reduction from `input` to `output` in decibels, accumulated in
/// `f64`, matching `cancel`'s `reduction_db`.
fn reduction_db(input: &[f32], output: &[f32]) -> f64 {
    let energy = |samples: &[f32]| -> f64 { samples.iter().map(|&s| s as f64 * s as f64).sum() };
    let input_energy = energy(input);
    let output_energy = energy(output);
    if input_energy <= 0.0 {
        return 0.0;
    }
    if output_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (input_energy / output_energy).log10()
}

/// Everything measured about one clip: the engine's own counters plus the
/// harness's windowed metrics. Serialized verbatim into the result JSON.
struct ClipRecord {
    id: String,
    scenario: Scenario,
    mic_rel: String,
    lpb_rel: String,
    enh_rel: String,
    aligned_mic_rel: String,
    aligned_lpb_rel: String,
    aligned_enh_rel: String,
    mic_rate: u32,
    lpb_rate: u32,
    duration_s: f64,
    engine_samples: usize,
    turns: u64,
    measured: bool,
    starved: u64,
    starved_pct: f64,
    parked: u64,
    parked_pct: f64,
    dropped: u64,
    divergence_resets: u64,
    internal_erle_db: f64,
    locked: bool,
    engine_delay_samples: Option<usize>,
    lock_turn: Option<u64>,
    tracking_moves: u32,
    reacquisitions: u32,
    coarse_rearms: u32,
    regions_rejected: u32,
    reacquire_trigger: Option<String>,
    est: Option<DelayEstimate>,
    within_window: Option<bool>,
    delay_mismatch: bool,
    double_talk_rate_pct: f64,
    freeze_far_active_pct: Option<f64>,
    echo_coupling_db: Option<f64>,
    reduction_full_db: f64,
    echo_reduction_conv_db: Option<f64>,
    echo_windows: usize,
    near_end: Option<ProjectionStats>,
    audio_s: f64,
    engine_wall_s: f64,
    x_realtime: f64,
}

impl ClipRecord {
    fn engine_delay_ms(&self) -> Option<f64> {
        self.engine_delay_samples
            .map(|s| s as f64 * 1000.0 / f64::from(ENGINE_RATE))
    }
}

/// Fixed run-wide context handed to every clip.
struct RunContext {
    crate_root: PathBuf,
    enhanced_root: PathBuf,
    aligned_root: PathBuf,
    config: AecConfig,
}

/// Files that could not be processed, kept visible in the report.
#[derive(Default)]
struct Flags {
    unpaired: Vec<String>,
    read_errors: Vec<(String, String)>,
}

/// Where one run's output goes, resolved once and shared by both input modes.
struct OutputPaths {
    enhanced_root: PathBuf,
    aligned_root: PathBuf,
    results_root: PathBuf,
    /// File stem shared by this run's `.json` and `.txt` result files.
    base: String,
    /// What a split manifest records as `internal_result_file`: a bare file
    /// name in the `--out-root` layout, a crate-root-relative path in a run
    /// folder. Readers accept both.
    internal_ref: String,
}

/// Resolves the output layout from `--run-dir`, `--out-root`, or neither.
///
/// `--run-dir DIR` gives one self-contained folder: `DIR/bench.json`,
/// `DIR/bench.txt`, `DIR/enhanced/`, `DIR/aligned/`. Otherwise the run fans out
/// under `--out-root DIR` (default `data/bench-output/`) as `results/`,
/// `enhanced/<set>/` and `aligned/<set>/`, with the stamp in each file name.
///
/// The two flags are mutually exclusive, and neither may resolve to the crate
/// root: nothing this harness writes belongs there.
fn resolve_output(
    cli: &Cli,
    crate_root: &Path,
    set_name: &str,
    stamp_compact: &str,
) -> Result<OutputPaths, String> {
    if cli.run_dir.is_some() && cli.out_root.is_some() {
        return Err("give either --run-dir or --out-root, not both".to_string());
    }

    if let Some(run_dir) = &cli.run_dir {
        let run_dir = absolutise(run_dir, crate_root);
        reject_crate_root(&run_dir, crate_root, "--run-dir")?;
        let base = "bench".to_string();
        let internal_ref = rel_display(&run_dir.join(format!("{base}.json")), crate_root);
        return Ok(OutputPaths {
            enhanced_root: run_dir.join("enhanced"),
            aligned_root: run_dir.join("aligned"),
            results_root: run_dir,
            base,
            internal_ref,
        });
    }

    let out_root = cli
        .out_root
        .clone()
        .map(|dir| absolutise(&dir, crate_root))
        .unwrap_or_else(|| crate_root.join("data").join("bench-output"));
    reject_crate_root(&out_root, crate_root, "--out-root")?;
    let base = format!("bench-{stamp_compact}-{set_name}");
    let internal_ref = format!("{base}.json");
    Ok(OutputPaths {
        enhanced_root: out_root.join("enhanced").join(set_name),
        aligned_root: out_root.join("aligned").join(set_name),
        results_root: out_root.join("results"),
        base,
        internal_ref,
    })
}

/// A relative path is taken relative to the crate root, so a run folder names
/// the same place whatever directory cargo was invoked from.
fn absolutise(path: &Path, crate_root: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        crate_root.join(path)
    }
}

/// Rejects an output directory that resolves to the crate root itself. The
/// directory is created first so the comparison is between two real paths.
fn reject_crate_root(dir: &Path, crate_root: &Path, flag: &str) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let resolved = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let root = crate_root
        .canonicalize()
        .unwrap_or_else(|_| crate_root.to_path_buf());
    if resolved == root {
        return Err(format!(
            "{flag} must not be the crate root ({}); write under data/ instead",
            root.display()
        ));
    }
    Ok(())
}

fn run(cli: &Cli) -> Result<(), String> {
    provenance::self_check()?;
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Split mode reads its pairs from a manifest and appends a result to it;
    // folder mode scores a clip-root and preserves a standalone result.
    if let Some(split_path) = cli.split.clone() {
        if cli.root.is_some() {
            return Err(
                "give either a <clip-root> or --split <manifest.json>, not both".to_string(),
            );
        }
        return run_split(cli, &crate_root, &split_path);
    }

    let root = cli
        .root
        .clone()
        .ok_or("a <clip-root> folder or --split <manifest.json> is required")?;
    if !root.is_dir() {
        return Err(format!("clip root {} is not a folder", root.display()));
    }
    let set_name = match &cli.set_name {
        Some(name) => sanitize_set_name(name),
        None => derive_set_name(&root, &crate_root),
    };
    // Provenance first: the hashes are computed before any processing, so
    // the stamp reflects the source as it was when the run started.
    let source_files = provenance::source_hashes(&crate_root.join("src"))?;
    let source_combined = provenance::combined_sha256(&source_files);
    let stamp = provenance::utc_now();

    let out = resolve_output(cli, &crate_root, &set_name, &stamp.compact)?;
    let results_root = out.results_root.clone();

    let config = build_config();
    let ctx = RunContext {
        crate_root: crate_root.clone(),
        enhanced_root: out.enhanced_root,
        aligned_root: out.aligned_root,
        config,
    };

    // Discover pairs scenario by scenario.
    let mut flags = Flags::default();
    let mut work: Vec<(Scenario, Vec<Pair>)> = Vec::new();
    for scenario in Scenario::ALL {
        if let Some(only) = cli.scenario {
            if only != scenario {
                continue;
            }
        }
        let dir = root.join(scenario.dir_name());
        if !dir.is_dir() {
            continue;
        }
        let pairs = discover_pairs(&dir, &mut flags)?;
        work.push((scenario, pairs));
    }
    if let Some(limit) = cli.limit {
        stratified_limit(&mut work, limit);
    }
    if work.iter().all(|(_, pairs)| pairs.is_empty()) {
        return Err(format!(
            "no scenario folder with clip pairs under {}; expected doubletalk/, \
             farend-singletalk/, or nearend-singletalk/ holding <stem>_mic.wav and \
             <stem>_lpb.wav files",
            root.display()
        ));
    }

    let total: usize = work.iter().map(|(_, pairs)| pairs.len()).sum();
    eprintln!("benchmark: {total} pairs in set '{set_name}'");

    let mut clips: Vec<ClipRecord> = Vec::new();
    let mut done = 0usize;
    for (scenario, pairs) in &work {
        for pair in pairs {
            done += 1;
            match process_clip(&ctx, *scenario, pair) {
                Ok(record) => {
                    eprintln!(
                        "  [{done}/{total}] {}/{} ok ({:.0}x realtime)",
                        scenario.dir_name(),
                        pair.stem,
                        record.x_realtime
                    );
                    clips.push(record);
                }
                Err(message) => {
                    eprintln!(
                        "  [{done}/{total}] {}/{} FAILED",
                        scenario.dir_name(),
                        pair.stem
                    );
                    flags
                        .read_errors
                        .push((format!("{}/{}", scenario.dir_name(), pair.stem), message));
                }
            }
        }
    }

    let aecmos_model = crate_root.join(AECMOS_MODEL_REL);
    let aecmos_present = aecmos_model.is_file();

    let report = render_report(
        &clips,
        &flags,
        &set_name,
        &root,
        &ctx,
        &source_combined,
        &stamp.iso,
        aecmos_present,
    );
    let json = render_json(
        &clips,
        &flags,
        &set_name,
        &root,
        &ctx,
        &source_files,
        &source_combined,
        &stamp.iso,
        aecmos_present,
    );

    std::fs::create_dir_all(&results_root)
        .map_err(|e| format!("cannot create {}: {e}", results_root.display()))?;
    let json_path = results_root.join(format!("{}.json", out.base));
    let text_path = results_root.join(format!("{}.txt", out.base));
    std::fs::write(&json_path, &json)
        .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
    std::fs::write(&text_path, &report)
        .map_err(|e| format!("cannot write {}: {e}", text_path.display()))?;

    print!("{report}");
    println!("results:");
    println!("  {}", rel_display(&json_path, &crate_root));
    println!("  {}", rel_display(&text_path, &crate_root));
    Ok(())
}

/// Split mode: run the pairs a manifest names for the chosen set, preserve the
/// same standalone internal result folder mode writes, and APPEND one result
/// entry to the manifest's `results` array. The engine path, per-clip metrics,
/// and enhanced-output writing are identical to folder mode; only the source
/// of pairs and the extra manifest append differ.
fn run_split(cli: &Cli, crate_root: &Path, split_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(split_path)
        .map_err(|e| format!("cannot read manifest {}: {e}", split_path.display()))?;
    let manifest = manifest::parse(&text)?;
    let set = cli.set;

    let pool_abs = if Path::new(&manifest.pool).is_absolute() {
        PathBuf::from(&manifest.pool)
    } else {
        crate_root.join(&manifest.pool)
    };
    if !pool_abs.is_dir() {
        return Err(format!(
            "manifest pool {} is not a folder",
            pool_abs.display()
        ));
    }

    // Provenance first, mirroring folder mode, so the stamp reflects the
    // source as it was when the run started.
    let source_files = provenance::source_hashes(&crate_root.join("src"))?;
    let source_combined = provenance::combined_sha256(&source_files);
    let stamp = provenance::utc_now();

    let config = build_config();

    // Enhanced output and the standalone result carry a set name that names
    // the manifest and the side, so a train run and a test run never collide.
    let set_name = format!("{}-{}", sanitize_set_name(&manifest.name), set.as_str());
    let out = resolve_output(cli, crate_root, &set_name, &stamp.compact)?;
    let results_root = out.results_root.clone();

    let ctx = RunContext {
        crate_root: crate_root.to_path_buf(),
        enhanced_root: out.enhanced_root,
        aligned_root: out.aligned_root,
        config,
    };

    // Build the work list from the manifest's stems, in the reported scenario
    // order. Stems the manifest names but the pool lacks are flagged, not fatal.
    let mut flags = Flags::default();
    let mut work: Vec<(Scenario, Vec<Pair>)> = Vec::new();
    for scenario in Scenario::ALL {
        let dir = pool_abs.join(scenario.dir_name());
        let mut pairs = Vec::new();
        for stem in manifest.stems(set.as_str(), scenario.dir_name()) {
            let mic = dir.join(format!("{stem}_mic.wav"));
            let lpb = dir.join(format!("{stem}_lpb.wav"));
            if !mic.is_file() || !lpb.is_file() {
                flags.unpaired.push(format!(
                    "{}/{stem} (named by manifest, missing from pool)",
                    scenario.dir_name()
                ));
                continue;
            }
            pairs.push(Pair {
                stem: stem.clone(),
                mic,
                lpb,
            });
        }
        work.push((scenario, pairs));
    }
    if let Some(limit) = cli.limit {
        stratified_limit(&mut work, limit);
    }
    if work.iter().all(|(_, pairs)| pairs.is_empty()) {
        return Err(format!(
            "manifest {} names no {} clips present under {}",
            split_path.display(),
            set.as_str(),
            pool_abs.display()
        ));
    }

    let total: usize = work.iter().map(|(_, pairs)| pairs.len()).sum();
    eprintln!(
        "benchmark: split '{}', set '{}', {total} pairs",
        manifest.name,
        set.as_str()
    );

    let mut clips: Vec<ClipRecord> = Vec::new();
    let mut done = 0usize;
    for (scenario, pairs) in &work {
        for pair in pairs {
            done += 1;
            match process_clip(&ctx, *scenario, pair) {
                Ok(record) => {
                    eprintln!(
                        "  [{done}/{total}] {}/{} ok ({:.0}x realtime)",
                        scenario.dir_name(),
                        pair.stem,
                        record.x_realtime
                    );
                    clips.push(record);
                }
                Err(message) => {
                    eprintln!(
                        "  [{done}/{total}] {}/{} FAILED",
                        scenario.dir_name(),
                        pair.stem
                    );
                    flags
                        .read_errors
                        .push((format!("{}/{}", scenario.dir_name(), pair.stem), message));
                }
            }
        }
    }

    let aecmos_model = crate_root.join(AECMOS_MODEL_REL);
    let aecmos_present = aecmos_model.is_file();

    // The standalone internal result: the per-clip file with mic/lpb/enhanced
    // paths the AECMOS step reads, identical in shape to folder mode's output.
    let report = render_report(
        &clips,
        &flags,
        &set_name,
        &pool_abs,
        &ctx,
        &source_combined,
        &stamp.iso,
        aecmos_present,
    );
    let json = render_json(
        &clips,
        &flags,
        &set_name,
        &pool_abs,
        &ctx,
        &source_files,
        &source_combined,
        &stamp.iso,
        aecmos_present,
    );
    std::fs::create_dir_all(&results_root)
        .map_err(|e| format!("cannot create {}: {e}", results_root.display()))?;
    let json_path = results_root.join(format!("{}.json", out.base));
    let text_path = results_root.join(format!("{}.txt", out.base));
    std::fs::write(&json_path, &json)
        .map_err(|e| format!("cannot write {}: {e}", json_path.display()))?;
    std::fs::write(&text_path, &report)
        .map_err(|e| format!("cannot write {}: {e}", text_path.display()))?;

    // Build and splice the manifest result entry. The append rewrites only the
    // interior of the `results` array; every frozen field stays byte-identical.
    let run_label = cli
        .run
        .clone()
        .unwrap_or_else(|| DEFAULT_RUN_LABEL.to_string());
    let entry = render_result_entry(
        &run_label,
        &stamp.iso,
        set.as_str(),
        &source_combined,
        &out.internal_ref,
        &clips,
    );
    let updated = manifest::append_result(&text, &entry)?;
    std::fs::write(split_path, &updated)
        .map_err(|e| format!("cannot write manifest {}: {e}", split_path.display()))?;

    print!("{report}");
    println!("split manifest updated:");
    println!("  {}", rel_display(split_path, crate_root));
    println!(
        "  appended run '{run_label}' (set {}); internal metrics recorded, aecmos pending",
        set.as_str()
    );
    println!("  internal detail: {}", rel_display(&json_path, crate_root));
    if aecmos_present {
        println!(
            "  fill aecmos with: python benchmarks{}run_aecmos.py --split {}",
            std::path::MAIN_SEPARATOR,
            rel_display(split_path, crate_root)
        );
    }
    Ok(())
}

/// Renders one manifest result entry as a single line: the run identity, the
/// per-scenario internal aggregates, and a null `aecmos` the Python step later
/// fills. Kept to one line so both the Rust append and the Python fill can
/// splice it without reserializing the surrounding manifest.
fn render_result_entry(
    run: &str,
    date: &str,
    set: &str,
    source_sha: &str,
    internal_file: &str,
    clips: &[ClipRecord],
) -> String {
    // Scenario order matches the manifest's own `split`/`train`/`test` blocks.
    let order = [
        Scenario::Doubletalk,
        Scenario::FarendSingletalk,
        Scenario::NearendSingletalk,
    ];
    let mut internal = String::from("{");
    for (i, scenario) in order.iter().enumerate() {
        let rows: Vec<&ClipRecord> = clips.iter().filter(|c| c.scenario == *scenario).collect();
        let comma = if i + 1 < order.len() { ", " } else { "" };
        internal.push_str(&format!(
            "{}: {}{comma}",
            jstr(scenario.dir_name()),
            scenario_internal_json(&rows)
        ));
    }
    internal.push('}');
    format!(
        "{{\"run\": {}, \"date\": {}, \"set\": {}, \"protocol\": {PROTOCOL}, \
         \"source_sha\": {}, \
         \"internal_result_file\": {}, \"internal\": {internal}, \"aecmos\": null}}",
        jstr(run),
        jstr(date),
        jstr(set),
        jstr(source_sha),
        jstr(internal_file),
    )
}

/// The per-scenario internal aggregate block for a result entry: the same
/// numbers the console summary reports, reduced to medians and pooled rates.
/// Metrics that do not apply to a scenario are `null` (for example near-end
/// projection for farend-singletalk, or converged echo reduction for
/// nearend-singletalk).
fn scenario_internal_json(rows: &[&ClipRecord]) -> String {
    let clips = rows.len();
    let measured = rows.iter().filter(|r| r.measured).count();
    let locked = rows.iter().filter(|r| r.locked).count();
    let meas: Vec<&&ClipRecord> = rows.iter().filter(|r| r.measured).collect();

    let opt_num = |v: Option<f64>| v.map(jnum).unwrap_or_else(|| "null".to_string());

    let erc: Vec<f64> = meas
        .iter()
        .filter_map(|r| r.echo_reduction_conv_db)
        .filter(|v| v.is_finite())
        .collect();
    let erc_json = if erc.is_empty() {
        "{\"median\": null, \"min\": null}".to_string()
    } else {
        format!(
            "{{\"median\": {}, \"min\": {}}}",
            jnum(median(&erc).expect("non-empty")),
            jnum(erc.iter().cloned().fold(f64::INFINITY, f64::min))
        )
    };

    let redf: Vec<f64> = rows
        .iter()
        .map(|r| r.reduction_full_db)
        .filter(|v| v.is_finite())
        .collect();
    let ierle: Vec<f64> = rows
        .iter()
        .map(|r| r.internal_erle_db)
        .filter(|v| v.is_finite())
        .collect();
    let frz: Vec<f64> = meas
        .iter()
        .filter_map(|r| r.freeze_far_active_pct)
        .collect();
    let dt: Vec<f64> = rows.iter().map(|r| r.double_talk_rate_pct).collect();

    let meds: Vec<f64> = meas
        .iter()
        .filter_map(|r| r.near_end.as_ref().map(|p| p.median_db))
        .collect();
    let mut windows = 0usize;
    let mut below3 = 0usize;
    let mut below6 = 0usize;
    for r in &meas {
        if let Some(p) = &r.near_end {
            windows += p.n;
            below3 += p.below_3db;
            below6 += p.below_6db;
        }
    }
    let near_json = if meds.is_empty() && windows == 0 {
        "null".to_string()
    } else {
        format!(
            "{{\"median_of_medians\": {}, \"windows\": {}, \"below_3db_pct\": {}, \
             \"below_6db_pct\": {}}}",
            opt_num(median(&meds)),
            windows,
            jnum(below3 as f64 * 100.0 / windows.max(1) as f64),
            jnum(below6 as f64 * 100.0 / windows.max(1) as f64),
        )
    };

    format!(
        "{{\"clips\": {clips}, \"measured\": {measured}, \"locked\": {locked}, \
         \"echo_reduction_conv_db\": {erc_json}, \
         \"reduction_full_db\": {{\"median\": {}}}, \
         \"internal_erle_db\": {{\"median\": {}}}, \
         \"freeze_far_active_pct\": {{\"median\": {}}}, \
         \"near_end_projection_db\": {near_json}, \
         \"double_talk_rate_pct\": {{\"median\": {}}}}}",
        opt_num(median(&redf)),
        opt_num(median(&ierle)),
        opt_num(median(&frz)),
        opt_num(median(&dt)),
    )
}

/// Lists `<stem>_mic.wav` / `<stem>_lpb.wav` pairs in one scenario folder,
/// sorted by stem so runs are deterministic. Files with one side missing are
/// recorded as unpaired and skipped.
fn discover_pairs(dir: &Path, flags: &mut Flags) -> Result<Vec<Pair>, String> {
    let mut mics: Vec<String> = Vec::new();
    let mut lpbs: Vec<String> = Vec::new();
    let read = std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    for entry in read {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_suffix("_mic.wav") {
            mics.push(stem.to_string());
        } else if let Some(stem) = name.strip_suffix("_lpb.wav") {
            lpbs.push(stem.to_string());
        }
    }
    mics.sort();
    lpbs.sort();

    let mut pairs = Vec::new();
    for stem in &mics {
        if lpbs.binary_search(stem).is_ok() {
            pairs.push(Pair {
                stem: stem.clone(),
                mic: dir.join(format!("{stem}_mic.wav")),
                lpb: dir.join(format!("{stem}_lpb.wav")),
            });
        } else {
            flags
                .unpaired
                .push(format!("{} (no _lpb.wav)", dir.join(stem).display()));
        }
    }
    for stem in &lpbs {
        if mics.binary_search(stem).is_err() {
            flags
                .unpaired
                .push(format!("{} (no _mic.wav)", dir.join(stem).display()));
        }
    }
    Ok(pairs)
}

/// Splits `target` across the scenarios in `sizes` in proportion to each
/// scenario's share, with largest-remainder rounding, so the parts sum to
/// `min(target, total)` and no scenario is allocated more than it holds. This
/// mirrors the allocation in `examples/make-split.rs`; the two examples cannot
/// share a private function without moving it into the crate, so the logic is
/// restated here.
fn allocate(sizes: &[usize], target: usize) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    if total == 0 || target == 0 {
        return vec![0; sizes.len()];
    }
    let target = target.min(total);
    let mut alloc: Vec<usize> = Vec::with_capacity(sizes.len());
    let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(sizes.len());
    let mut assigned = 0usize;
    for (i, &size) in sizes.iter().enumerate() {
        let quota = target as f64 * size as f64 / total as f64;
        let floor = (quota.floor() as usize).min(size);
        alloc.push(floor);
        assigned += floor;
        remainders.push((quota - quota.floor(), i));
    }
    remainders.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .expect("remainders are finite")
            .then(a.1.cmp(&b.1))
    });
    let mut leftover = target.saturating_sub(assigned);
    let mut progressed = true;
    while leftover > 0 && progressed {
        progressed = false;
        for &(_, i) in &remainders {
            if leftover == 0 {
                break;
            }
            if alloc[i] < sizes[i] {
                alloc[i] += 1;
                leftover -= 1;
                progressed = true;
            }
        }
    }
    alloc
}

/// Reduces `work` to `total` pairs, stratified across scenarios by [`allocate`],
/// keeping each scenario's sorted-stem prefix (pairs enter sorted by stem in
/// both input modes) so the selection is deterministic and repeatable. A
/// `total` at or above the current pair count leaves `work` unchanged.
fn stratified_limit(work: &mut [(Scenario, Vec<Pair>)], total: usize) {
    let sizes: Vec<usize> = work.iter().map(|(_, p)| p.len()).collect();
    let keep = allocate(&sizes, total);
    for ((_, pairs), &k) in work.iter_mut().zip(keep.iter()) {
        pairs.truncate(k);
    }
}

/// Runs the engine over one pair and measures everything the report needs.
fn process_clip(ctx: &RunContext, scenario: Scenario, pair: &Pair) -> Result<ClipRecord, String> {
    let mic = wav::read_mono(&pair.mic)?;
    let lpb = wav::read_mono(&pair.lpb)?;
    let mic_rate = mic.sample_rate;
    let lpb_rate = lpb.sample_rate;
    let near = to_engine_rate(mic)?;
    let far_clip = to_engine_rate(lpb)?;

    // The canonical timeline: the near end sets the length, and the far end
    // is placed onto it, so every signal of the triplet shares one length,
    // one rate, and one sample convention.
    let n = near.samples.len();
    let far = fit_len(far_clip.samples, n);

    let mut aec = Aec::new(ctx.config.clone())
        .map_err(|e| format!("engine rejected the configuration: {e}"))?;

    // One turn: feed the reference chunk, then process the capture chunk.
    // The per-turn metrics snapshot is what the double-talk rate and the
    // lock turn are counted from.
    let mut double_talk_turns = 0u64;
    let mut frozen_turns: Vec<bool> = Vec::with_capacity(n / TURN + 1);
    let mut lock_turn: Option<u64> = None;
    let mut turns = 0u64;

    let started = Instant::now();
    let out = run_canceller(&mut aec, &near.samples, &far, |aec, turn| {
        turns = turn;
        let snapshot = aec.metrics();
        if snapshot.canceller.double_talk {
            double_talk_turns += 1;
        }
        frozen_turns.push(snapshot.canceller.double_talk);
        if lock_turn.is_none() && snapshot.delay_samples.is_some() {
            lock_turn = Some(turn - 1);
        }
    })?;
    let engine_wall_s = started.elapsed().as_secs_f64();

    // The engine emits its output time-aligned one to one with the near-end
    // input, and the flush balances the totals; anything else is an error.
    if out.len() != n {
        return Err(format!(
            "enhanced length {} does not match the canonical length {n}",
            out.len()
        ));
    }

    // The enhanced output, written exactly as produced, for listening and
    // for AECMOS.
    let scenario_dir = ctx.enhanced_root.join(scenario.dir_name());
    std::fs::create_dir_all(&scenario_dir)
        .map_err(|e| format!("cannot create {}: {e}", scenario_dir.display()))?;
    let enh_path = scenario_dir.join(format!("{}_enh.wav", pair.stem));
    wav::write_mono(&enh_path, &out, ENGINE_RATE)?;

    // The canonical aligned triplet: the exact samples the engine consumed
    // and produced, one shared timeline, for scoring without any further
    // resampling.
    let aligned_dir = ctx.aligned_root.join(scenario.dir_name());
    std::fs::create_dir_all(&aligned_dir)
        .map_err(|e| format!("cannot create {}: {e}", aligned_dir.display()))?;
    let aligned_mic = aligned_dir.join(format!("{}_mic-16k-aligned.wav", pair.stem));
    let aligned_lpb = aligned_dir.join(format!("{}_lpb-16k-aligned.wav", pair.stem));
    let aligned_enh = aligned_dir.join(format!("{}_enhanced-16k-aligned.wav", pair.stem));
    wav::write_mono(&aligned_mic, &near.samples, ENGINE_RATE)?;
    wav::write_mono(&aligned_lpb, &far, ENGINE_RATE)?;
    wav::write_mono(&aligned_enh, &out, ENGINE_RATE)?;

    let m = aec.metrics();

    let near_span = &near.samples[..n];
    let out_span = &out[..n];

    // The harness's own delay estimate, independent of the engine.
    let est = metrics::estimate_delay(near_span, &far, ENGINE_RATE);
    let est_confident = est
        .as_ref()
        .map(|e| e.corr >= metrics::DELAY_CONFIDENT_CORR)
        .unwrap_or(false);
    let within_window = match (&est, est_confident) {
        (Some(e), true) => Some(e.lag_ms <= usize::from(ctx.config.max_echo_delay_ms)),
        _ => None,
    };
    let engine_delay_ms = m
        .delay_samples
        .map(|s| s as f64 * 1000.0 / f64::from(ENGINE_RATE));
    let delay_mismatch = match (engine_delay_ms, &est, est_confident) {
        (Some(engine_ms), Some(e), true) => (engine_ms - e.lag_ms as f64).abs() > DELAY_MISMATCH_MS,
        _ => false,
    };

    // Window classification reads the far end through the alignment the
    // engine actually used when it locked, else the harness estimate.
    let delay_used = m.delay_samples.or_else(|| {
        est.as_ref()
            .filter(|e| e.corr >= metrics::DELAY_CONFIDENT_CORR)
            .map(|e| e.lag_ms * ENGINE_RATE as usize / 1000)
    });
    let wa = metrics::analyze(near_span, &far, delay_used.unwrap_or(0), ENGINE_RATE);

    let conv_start = (metrics::CONVERGED_START_S * f64::from(ENGINE_RATE)) as usize;
    let (echo_reduction_conv_db, echo_windows) = match scenario {
        Scenario::FarendSingletalk => metrics::masked_reduction_db(near_span, out_span, &wa, |i| {
            wa.far_active[i] && wa.starts[i] >= conv_start
        }),
        Scenario::Doubletalk => metrics::masked_reduction_db(near_span, out_span, &wa, |i| {
            wa.echo_dominant[i] && wa.starts[i] >= conv_start
        }),
        Scenario::NearendSingletalk => (None, 0),
    };
    let near_end = match scenario {
        Scenario::FarendSingletalk => None,
        _ => metrics::projection_stats(near_span, out_span, &wa, ENGINE_RATE),
    };

    // Protection rate over post-lock far-active turns: the far block judged
    // on turn t spans far indices [TURN * (t + 1) - delay - TURN,
    // TURN * (t + 1) - delay), because the harness feeds one reference turn
    // ahead of each capture turn.
    let freeze_far_active_pct = freeze_rate(&far, &frozen_turns, turns, lock_turn, m.delay_samples);

    let starved_pct = if n > 0 {
        m.reference_starved as f64 * 100.0 / n as f64
    } else {
        0.0
    };
    let parked_pct = if n > 0 {
        m.acquisition_parked as f64 * 100.0 / n as f64
    } else {
        0.0
    };
    // Whether the clip's numbers are trustworthy.
    let locked = m.delay_samples.is_some();
    let measured = starved_pct <= STARVED_MEASURED_LIMIT_PCT
        && (locked || parked_pct <= STARVED_MEASURED_LIMIT_PCT);

    Ok(ClipRecord {
        id: pair.stem.clone(),
        scenario,
        mic_rel: rel_display(&pair.mic, &ctx.crate_root),
        lpb_rel: rel_display(&pair.lpb, &ctx.crate_root),
        enh_rel: rel_display(&enh_path, &ctx.crate_root),
        aligned_mic_rel: rel_display(&aligned_mic, &ctx.crate_root),
        aligned_lpb_rel: rel_display(&aligned_lpb, &ctx.crate_root),
        aligned_enh_rel: rel_display(&aligned_enh, &ctx.crate_root),
        mic_rate,
        lpb_rate,
        duration_s: near.input_seconds,
        engine_samples: n,
        turns,
        measured,
        starved: m.reference_starved,
        starved_pct,
        parked: m.acquisition_parked,
        parked_pct,
        dropped: m.reference_dropped,
        divergence_resets: m.canceller.divergence_resets,
        internal_erle_db: f64::from(m.canceller.erle_db),
        locked,
        engine_delay_samples: m.delay_samples,
        lock_turn,
        tracking_moves: m.delay.tracking_moves,
        reacquisitions: m.delay.reacquisitions,
        coarse_rearms: m.delay.coarse_rearms,
        regions_rejected: m.delay.coarse_regions_rejected,
        reacquire_trigger: m.delay.last_reacquire_trigger.map(|t| format!("{t:?}")),
        est,
        within_window,
        delay_mismatch,
        double_talk_rate_pct: if turns > 0 {
            double_talk_turns as f64 * 100.0 / turns as f64
        } else {
            0.0
        },
        freeze_far_active_pct,
        echo_coupling_db: wa.echo_coupling_db,
        reduction_full_db: reduction_db(near_span, out_span),
        echo_reduction_conv_db,
        echo_windows,
        near_end,
        audio_s: near.input_seconds,
        engine_wall_s,
        x_realtime: near.input_seconds / engine_wall_s.max(1e-9),
    })
}

/// The freeze rate over post-lock far-active turns, `None` when no far-end
/// block carried energy (the nearend-singletalk case).
fn freeze_rate(
    far: &[f32],
    frozen_turns: &[bool],
    turns: u64,
    lock_turn: Option<u64>,
    delay_samples: Option<usize>,
) -> Option<f64> {
    let delay_offset = delay_samples.unwrap_or(0);
    let counted_from = lock_turn.unwrap_or(0) as usize;
    let mut block_rms: Vec<f64> = Vec::new();
    for turn in counted_from..turns as usize {
        let end = (TURN * (turn + 1)).saturating_sub(delay_offset);
        let start = end.saturating_sub(TURN);
        let block = &far[start.min(far.len())..end.min(far.len())];
        let energy: f64 = block.iter().map(|&s| s as f64 * s as f64).sum();
        let rms = if block.is_empty() {
            0.0
        } else {
            (energy / block.len() as f64).sqrt()
        };
        block_rms.push(rms);
    }
    let mut sorted = block_rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("block RMS values are finite"));
    let loud = if sorted.is_empty() {
        0.0
    } else {
        sorted[(sorted.len() - 1).min(sorted.len() * 95 / 100)]
    };
    if loud <= 0.0 {
        return None;
    }
    let active_floor = loud * 10.0f64.powf(-FAR_ACTIVE_REL_DB / 20.0);
    let mut far_active = 0u64;
    let mut frozen = 0u64;
    for (i, &rms) in block_rms.iter().enumerate() {
        if rms > active_floor {
            far_active += 1;
            if frozen_turns[counted_from + i] {
                frozen += 1;
            }
        }
    }
    if far_active == 0 {
        None
    } else {
        Some(frozen as f64 * 100.0 / far_active as f64)
    }
}

// ---------------------------------------------------------------------------
// Reporting: the console/text table and the preserved JSON result.
// ---------------------------------------------------------------------------

/// Median of a slice, `None` when empty.
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("values are finite"));
    Some(metrics::percentile(&sorted, 0.50))
}

/// A display id short enough for the table: the stem with its scenario
/// suffix stripped, plus a `mv` marker for with-movement variants.
fn short_id(record: &ClipRecord) -> String {
    let stem = &record.id;
    let base = stem
        .find(&format!("_{}", record.scenario.dir_name()))
        .or_else(|| stem.find('_'))
        .map(|at| &stem[..at])
        .unwrap_or(stem);
    let marker = if stem.contains("-with-movement") {
        " mv"
    } else {
        ""
    };
    let mut id = base.to_string();
    if id.len() > 24 {
        id.truncate(24);
    }
    format!("{id}{marker}")
}

fn fmt_opt_db(value: Option<f64>, width: usize) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:>width$.1}"),
        Some(_) => format!("{:>width$}", "inf"),
        None => format!("{:>width$}", "-"),
    }
}

/// Whether a clip's confident harness estimate lies past the global search
/// ceiling (`max_search_delay_ms`). This, and only this, is a coverage failure.
/// A confident estimate past the local fine window (`max_echo_delay_ms`) but
/// within the ceiling is reached normally by relocating the fine search onto
/// the coarse region, so it is not a failure.
fn beyond_search_ceiling(record: &ClipRecord, max_search_delay_ms: u16) -> bool {
    record.within_window == Some(false)
        && record
            .est
            .as_ref()
            .map(|e| e.lag_ms > usize::from(max_search_delay_ms))
            .unwrap_or(false)
}

fn flags_column(record: &ClipRecord, max_search_delay_ms: u16) -> String {
    let mut flags = String::new();
    if !record.measured {
        flags.push('S');
    }
    let beyond_fine = record.within_window == Some(false);
    let beyond_ceiling = beyond_search_ceiling(record, max_search_delay_ms);
    if beyond_fine && !beyond_ceiling && record.locked {
        // Estimate past the local fine window but within the global ceiling,
        // reached by relocating the fine search: normal operation, not a
        // failure.
        flags.push('R');
    }
    if beyond_ceiling {
        // Confident estimate past the global search ceiling: coverage failure.
        flags.push('C');
    }
    if !record.locked
        && record.est.as_ref().map(|e| e.corr).unwrap_or(0.0) >= metrics::DELAY_CONFIDENT_CORR
    {
        flags.push('L');
    }
    if record.delay_mismatch {
        flags.push('M');
    }
    if flags.is_empty() {
        flags.push('-');
    }
    flags
}

/// Renders the whole human-readable report: header, one block per scenario,
/// flag lists, and the AECMOS note. The same text is printed and preserved.
#[allow(clippy::too_many_arguments)]
fn render_report(
    clips: &[ClipRecord],
    flags: &Flags,
    set_name: &str,
    root: &Path,
    ctx: &RunContext,
    source_combined: &str,
    created: &str,
    aecmos_present: bool,
) -> String {
    let mut s = String::new();
    let cfg = &ctx.config;
    s.push_str("decibri-aec benchmark (internal metrics)\n");
    s.push_str(&format!(
        "  set:     {set_name} ({}), {} clips\n",
        root.display(),
        clips.len()
    ));
    s.push_str(&format!(
        "  engine:  {} at {} Hz, tail {} ms, delay window {} ms, suppression {}, \
         transition {}, estimator path (no delay hint)\n",
        cfg.model.as_str(),
        cfg.sample_rate,
        cfg.tail_ms,
        cfg.max_echo_delay_ms,
        suppression_name(cfg.suppression),
        transition_name(cfg.output_transition),
    ));
    s.push_str(&format!(
        "  source:  {} (sha-256 over src/)\n  created: {created}   crate {}\n",
        &source_combined[..16.min(source_combined.len())],
        env!("CARGO_PKG_VERSION"),
    ));
    s.push_str("\n\n");

    for scenario in Scenario::ALL {
        let rows: Vec<&ClipRecord> = clips.iter().filter(|c| c.scenario == scenario).collect();
        if rows.is_empty() {
            continue;
        }
        s.push_str(&format!(
            "{} ({} clips): {}\n",
            scenario.dir_name(),
            rows.len(),
            scenario.tagline()
        ));
        match scenario {
            Scenario::FarendSingletalk => {
                render_farend_block(&mut s, &rows, cfg.max_search_delay_ms)
            }
            Scenario::Doubletalk => render_doubletalk_block(&mut s, &rows, cfg.max_search_delay_ms),
            Scenario::NearendSingletalk => {
                render_nearend_block(&mut s, &rows, cfg.max_search_delay_ms)
            }
        }
        s.push('\n');
    }

    render_flag_lists(&mut s, clips, flags, ctx);

    if aecmos_present {
        s.push_str(&format!(
            "AECMOS: model present at {AECMOS_MODEL_REL}\n  score this run with: python \
             benchmarks{}run_aecmos.py\n",
            std::path::MAIN_SEPARATOR
        ));
    } else {
        s.push_str(&format!(
            "AECMOS: skipped (model not found at {AECMOS_MODEL_REL}); all internal \
             metrics above are complete without it\n"
        ));
    }
    s
}

fn suppression_name(suppression: Suppression) -> &'static str {
    match suppression {
        Suppression::Off => "off",
        Suppression::Conservative => "conservative",
        _ => "other",
    }
}

/// The recorded name of a policy. A graded policy carries its fade pair
/// so the recorded value names the exact configuration that ran.
fn transition_name(policy: OutputTransitionPolicy) -> String {
    match policy {
        OutputTransitionPolicy::PreserveCorrection => "preserve".to_string(),
        OutputTransitionPolicy::GradedReacquisition {
            fade_out_ms,
            fade_in_ms,
        } => format!("graded/{fade_out_ms}/{fade_in_ms}"),
        _ => "other".to_string(),
    }
}

fn render_farend_block(s: &mut String, rows: &[&ClipRecord], max_search_ms: u16) {
    s.push_str(&format!(
        "  {:<28} {:>8} {:>8} {:>7} {:>7} {:>7} {:>5} {:>6}  {}\n",
        "clip", "ERLEc dB", "redF dB", "frz%FA", "dly ms", "est ms", "corr", "xRT", "flags"
    ));
    for r in rows {
        let erle = if r.measured {
            fmt_opt_db(r.echo_reduction_conv_db, 8)
        } else {
            format!("{:>8}", "n/m")
        };
        s.push_str(&format!(
            "  {:<28} {} {:>8.1} {:>7} {:>7} {:>7} {:>5} {:>6.0}  {}\n",
            short_id(r),
            erle,
            r.reduction_full_db,
            r.freeze_far_active_pct
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".to_string()),
            r.engine_delay_ms()
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "none".to_string()),
            r.est
                .as_ref()
                .map(|e| format!("{}", e.lag_ms))
                .unwrap_or_else(|| "-".to_string()),
            r.est
                .as_ref()
                .map(|e| format!("{:.2}", e.corr))
                .unwrap_or_else(|| "-".to_string()),
            r.x_realtime,
            flags_column(r, max_search_ms),
        ));
    }
    let measured: Vec<&&ClipRecord> = rows.iter().filter(|r| r.measured).collect();
    let erles: Vec<f64> = measured
        .iter()
        .filter_map(|r| r.echo_reduction_conv_db)
        .filter(|v| v.is_finite())
        .collect();
    let worst = measured
        .iter()
        .filter(|r| r.echo_reduction_conv_db.is_some())
        .min_by(|a, b| {
            a.echo_reduction_conv_db
                .partial_cmp(&b.echo_reduction_conv_db)
                .expect("finite")
        });
    let frz: Vec<f64> = measured
        .iter()
        .filter_map(|r| r.freeze_far_active_pct)
        .collect();
    let locked = rows.iter().filter(|r| r.locked).count();
    let min_erle = erles.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_erle_text = if min_erle.is_finite() {
        format!("{min_erle:.1}")
    } else {
        "-".to_string()
    };
    s.push_str(&format!(
        "  summary: ERLE(conv) median {} dB, min {min_erle_text} dB{}; freeze%FA median {}; \
         locked {}/{}\n",
        median(&erles)
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "-".into()),
        worst
            .map(|r| format!(" ({})", short_id(r)))
            .unwrap_or_default(),
        median(&frz)
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "-".into()),
        locked,
        rows.len()
    ));
}

fn render_doubletalk_block(s: &mut String, rows: &[&ClipRecord], max_search_ms: u16) {
    s.push_str(&format!(
        "  {:<28} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>7} {:>7} {:>6}  {}\n",
        "clip",
        "ERc dB",
        "prjMed",
        "prjP5",
        "prjMin",
        "min@s",
        "<-3d%",
        "<-6d%",
        "frz%FA",
        "dly ms",
        "xRT",
        "flags"
    ));
    for r in rows {
        let (er, med, p5, min, at, b3, b6) = quality_cells(r);
        s.push_str(&format!(
            "  {:<28} {er} {med} {p5} {min} {at} {b3} {b6} {:>7} {:>7} {:>6.0}  {}\n",
            short_id(r),
            r.freeze_far_active_pct
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".to_string()),
            r.engine_delay_ms()
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "none".to_string()),
            r.x_realtime,
            flags_column(r, max_search_ms),
        ));
    }
    render_projection_summary(s, rows, "near-end projection");
}

fn render_nearend_block(s: &mut String, rows: &[&ClipRecord], max_search_ms: u16) {
    s.push_str(&format!(
        "  {:<28} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>7} {:>6}  {}\n",
        "clip", "prsMed", "prsP5", "prsMin", "min@s", "<-3d%", "<-6d%", "dt%", "xRT", "flags"
    ));
    for r in rows {
        let (_, med, p5, min, at, b3, b6) = quality_cells(r);
        s.push_str(&format!(
            "  {:<28} {med} {p5} {min} {at} {b3} {b6} {:>7.1} {:>6.0}  {}\n",
            short_id(r),
            r.double_talk_rate_pct,
            r.x_realtime,
            flags_column(r, max_search_ms),
        ));
    }
    render_projection_summary(s, rows, "preservation");
}

/// The quality cells shared by the doubletalk and nearend tables, rendered
/// as `n/m` for a NOT-MEASURED clip.
fn quality_cells(r: &ClipRecord) -> (String, String, String, String, String, String, String) {
    if !r.measured {
        let nm = |w: usize| format!("{:>w$}", "n/m");
        return (nm(7), nm(7), nm(7), nm(7), nm(8), nm(6), nm(6));
    }
    let er = fmt_opt_db(r.echo_reduction_conv_db, 7);
    match &r.near_end {
        Some(p) => (
            er,
            format!("{:>7.1}", p.median_db),
            format!("{:>7.1}", p.p5_db),
            format!("{:>7.1}", p.min_db),
            format!("{:>8.1}", p.min_at_s),
            format!("{:>6.1}", p.below_3db_pct()),
            format!("{:>6.1}", p.below_6db_pct()),
        ),
        None => (
            er,
            format!("{:>7}", "-"),
            format!("{:>7}", "-"),
            format!("{:>7}", "-"),
            format!("{:>8}", "-"),
            format!("{:>6}", "-"),
            format!("{:>6}", "-"),
        ),
    }
}

fn render_projection_summary(s: &mut String, rows: &[&ClipRecord], label: &str) {
    let measured: Vec<&&ClipRecord> = rows.iter().filter(|r| r.measured).collect();
    let medians: Vec<f64> = measured
        .iter()
        .filter_map(|r| r.near_end.as_ref().map(|p| p.median_db))
        .collect();
    let ers: Vec<f64> = measured
        .iter()
        .filter_map(|r| r.echo_reduction_conv_db)
        .filter(|v| v.is_finite())
        .collect();
    let mut worst: Option<(&&ClipRecord, f64, f64)> = None;
    let mut total_windows = 0usize;
    let mut below3 = 0usize;
    let mut below6 = 0usize;
    for r in &measured {
        if let Some(p) = &r.near_end {
            total_windows += p.n;
            below3 += p.below_3db;
            below6 += p.below_6db;
            if worst.map(|(_, db, _)| p.min_db < db).unwrap_or(true) {
                worst = Some((r, p.min_db, p.min_at_s));
            }
        }
    }
    if !ers.is_empty() {
        let med = median(&ers)
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "-".into());
        let min = format!("{:.1}", ers.iter().cloned().fold(f64::INFINITY, f64::min));
        s.push_str(&format!(
            "  summary: echo reduction (conv) median {med} dB, min {min} dB\n"
        ));
    }
    s.push_str(&format!(
        "  summary: {label} median-of-medians {} dB; pooled windows {} \
         (<-3 dB {:.1}%, <-6 dB {:.1}%){}\n",
        median(&medians)
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "-".into()),
        total_windows,
        below3 as f64 * 100.0 / total_windows.max(1) as f64,
        below6 as f64 * 100.0 / total_windows.max(1) as f64,
        worst
            .map(|(r, db, at)| format!("; worst {db:.1} dB at {at:.1} s ({})", short_id(r)))
            .unwrap_or_default(),
    ));
}

fn render_flag_lists(s: &mut String, clips: &[ClipRecord], flags: &Flags, ctx: &RunContext) {
    let max_echo_ms = ctx.config.max_echo_delay_ms;
    let max_search_ms = ctx.config.max_search_delay_ms;
    s.push_str("flags:\n");
    s.push_str(&format!(
        "  legend: S not-measured, R estimate past the {max_echo_ms} ms local fine window then \
         relocated and locked (normal), C credible estimate past the {max_search_ms} ms global \
         search ceiling (coverage failure), L no lock despite reference energy, M lock disagrees \
         with the harness estimate\n"
    ));
    let listed = |s: &mut String, title: &str, items: Vec<String>| {
        if items.is_empty() {
            s.push_str(&format!("  {title}: none\n"));
        } else {
            s.push_str(&format!("  {title}: {}\n", items.join(", ")));
        }
    };
    listed(
        s,
        "NOT-MEASURED",
        clips
            .iter()
            .filter(|c| !c.measured)
            .map(|c| c.id.clone())
            .collect(),
    );
    listed(
        s,
        &format!(
            "estimate beyond the {max_echo_ms} ms local fine window, reached by relocating the \
             fine search (normal, not a failure)"
        ),
        clips
            .iter()
            .filter(|c| {
                c.within_window == Some(false)
                    && !beyond_search_ceiling(c, max_search_ms)
                    && c.locked
            })
            .map(|c| {
                format!(
                    "{} (estimate {} ms, locked at {} ms)",
                    c.id,
                    c.est.as_ref().map(|e| e.lag_ms).unwrap_or(0),
                    c.engine_delay_ms()
                        .map(|v| format!("{v:.0}"))
                        .unwrap_or_else(|| "none".to_string()),
                )
            })
            .collect(),
    );
    listed(
        s,
        &format!(
            "credible estimate beyond the {max_search_ms} ms global search ceiling \
             (coverage failure)"
        ),
        clips
            .iter()
            .filter(|c| beyond_search_ceiling(c, max_search_ms))
            .map(|c| {
                format!(
                    "{} (estimate {} ms)",
                    c.id,
                    c.est.as_ref().map(|e| e.lag_ms).unwrap_or(0)
                )
            })
            .collect(),
    );
    listed(
        s,
        "estimator never locked despite reference energy",
        clips
            .iter()
            .filter(|c| {
                !c.locked
                    && c.est.as_ref().map(|e| e.corr).unwrap_or(0.0)
                        >= metrics::DELAY_CONFIDENT_CORR
            })
            .map(|c| c.id.clone())
            .collect(),
    );
    listed(
        s,
        &format!("lock disagrees with harness estimate by > {DELAY_MISMATCH_MS:.0} ms"),
        clips
            .iter()
            .filter(|c| c.delay_mismatch)
            .map(|c| {
                format!(
                    "{} (lock {:.0} ms, estimate {} ms)",
                    c.id,
                    c.engine_delay_ms().unwrap_or(0.0),
                    c.est.as_ref().map(|e| e.lag_ms).unwrap_or(0)
                )
            })
            .collect(),
    );
    listed(s, "unpaired files", flags.unpaired.clone());
    listed(
        s,
        "clips that failed to process",
        flags
            .read_errors
            .iter()
            .map(|(id, err)| format!("{id} ({err})"))
            .collect(),
    );
    s.push('\n');
}

// ---------------------------------------------------------------------------
// JSON result: hand-rendered; the schema is the field set emitted below.
// ---------------------------------------------------------------------------

fn jstr(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Numbers in the result are finite; an infinite energy ratio (an exactly
/// silent output span, unseen on real data) is encoded as the documented
/// sentinel 999.99.
fn jnum(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        "999.99".to_string()
    }
}

fn jopt(value: Option<f64>) -> String {
    value.map(jnum).unwrap_or_else(|| "null".to_string())
}

fn jopt_int(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn jbool(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

#[allow(clippy::too_many_arguments)]
fn render_json(
    clips: &[ClipRecord],
    flags: &Flags,
    set_name: &str,
    root: &Path,
    ctx: &RunContext,
    source_files: &[(String, String)],
    source_combined: &str,
    created: &str,
    aecmos_present: bool,
) -> String {
    let cfg = &ctx.config;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"schema\": {},\n", jstr(SCHEMA)));
    s.push_str(&format!("  \"protocol\": {PROTOCOL},\n"));
    s.push_str(&format!("  \"created_utc\": {},\n", jstr(created)));
    s.push_str(&format!(
        "  \"kit\": {{\"crate\": \"decibri-aec\", \"crate_version\": {}, \"example\": \
         \"benchmark\"}},\n",
        jstr(env!("CARGO_PKG_VERSION"))
    ));

    s.push_str("  \"source\": {\n");
    s.push_str(&format!(
        "    \"combined_sha256\": {},\n    \"files\": {{\n",
        jstr(source_combined)
    ));
    for (i, (path, hash)) in source_files.iter().enumerate() {
        let comma = if i + 1 < source_files.len() { "," } else { "" };
        s.push_str(&format!("      {}: {}{comma}\n", jstr(path), jstr(hash)));
    }
    s.push_str("    }\n  },\n");

    let mic_rates: Vec<u32> = clips.iter().map(|c| c.mic_rate).collect();
    let set_rate = mic_rates
        .first()
        .filter(|&&r| mic_rates.iter().all(|&x| x == r))
        .map(|r| r.to_string())
        .unwrap_or_else(|| "0".to_string());
    s.push_str(&format!(
        "  \"input\": {{\"set_name\": {}, \"root\": {}, \"pairing\": \"<stem>_mic.wav with \
         <stem>_lpb.wav\", \"clips\": {}, \"input_sample_rate_hz\": {set_rate}}},\n",
        jstr(set_name),
        jstr(&rel_display(root, &ctx.crate_root)),
        clips.len()
    ));

    s.push_str(&format!(
        "  \"engine\": {{\"model\": {}, \"sample_rate_hz\": {}, \"tail_ms\": {}, \
         \"max_echo_delay_ms\": {}, \"max_search_delay_ms\": {}, \"delay_hint_ms\": null, \
         \"suppression\": {}, \"output_transition\": {}, \"turn_samples\": {TURN}}},\n",
        jstr(cfg.model.as_str()),
        cfg.sample_rate,
        cfg.tail_ms,
        cfg.max_echo_delay_ms,
        cfg.max_search_delay_ms,
        jstr(suppression_name(cfg.suppression)),
        jstr(&transition_name(cfg.output_transition)),
    ));

    s.push_str(&format!(
        "  \"scoring\": {{\"window_ms\": {}, \"hop_ms\": {}, \"active_rel_db\": {}, \
         \"active_floor_dbfs\": {}, \"near_active_margin_db\": {}, \
         \"echo_dominant_margin_db\": {}, \"echo_coupling_q\": {}, \"converged_start_s\": {}, \
         \"starved_not_measured_pct\": {}, \"delay_search_max_ms\": {}, \
         \"delay_confident_corr\": {}, \"measured_rule\": \"\", \
         \"scored_portion\": {{\"echo_reduction\": \"\", \"near_end_projection\": \"\"}}}},\n",
        metrics::WINDOW_MS,
        metrics::HOP_MS,
        jnum(metrics::ACTIVE_REL_DB),
        jnum(metrics::ACTIVE_FLOOR_DBFS),
        jnum(metrics::NEAR_ACTIVE_MARGIN_DB),
        jnum(metrics::ECHO_DOMINANT_MARGIN_DB),
        jnum(metrics::ECHO_COUPLING_Q),
        jnum(metrics::CONVERGED_START_S),
        jnum(STARVED_MEASURED_LIMIT_PCT),
        metrics::DELAY_SEARCH_MAX_MS,
        jnum(metrics::DELAY_CONFIDENT_CORR),
    ));

    s.push_str("  \"clips\": [\n");
    for (i, c) in clips.iter().enumerate() {
        let comma = if i + 1 < clips.len() { "," } else { "" };
        s.push_str(&render_clip_json(c));
        s.push_str(&format!("{comma}\n"));
    }
    s.push_str("  ],\n");

    s.push_str("  \"flags\": {\n");
    let id_list = |ids: Vec<String>| -> String {
        let items: Vec<String> = ids.iter().map(|i| jstr(i)).collect();
        format!("[{}]", items.join(", "))
    };
    s.push_str(&format!(
        "    \"not_measured\": {},\n",
        id_list(
            clips
                .iter()
                .filter(|c| !c.measured)
                .map(|c| c.id.clone())
                .collect()
        )
    ));
    s.push_str(&format!(
        "    \"delay_window_exceeded\": {},\n",
        id_list(
            clips
                .iter()
                .filter(|c| c.within_window == Some(false))
                .map(|c| c.id.clone())
                .collect()
        )
    ));
    s.push_str(&format!(
        "    \"no_lock_with_reference\": {},\n",
        id_list(
            clips
                .iter()
                .filter(|c| !c.locked
                    && c.est.as_ref().map(|e| e.corr).unwrap_or(0.0)
                        >= metrics::DELAY_CONFIDENT_CORR)
                .map(|c| c.id.clone())
                .collect()
        )
    ));
    s.push_str(&format!(
        "    \"delay_mismatch\": {},\n",
        id_list(
            clips
                .iter()
                .filter(|c| c.delay_mismatch)
                .map(|c| c.id.clone())
                .collect()
        )
    ));
    s.push_str(&format!(
        "    \"unpaired\": {},\n",
        id_list(flags.unpaired.clone())
    ));
    s.push_str(&format!(
        "    \"read_errors\": {}\n",
        id_list(
            flags
                .read_errors
                .iter()
                .map(|(id, err)| format!("{id}: {err}"))
                .collect()
        )
    ));
    s.push_str("  },\n");

    s.push_str(&format!(
        "  \"aecmos\": {{\"model_present\": {}, \"model_path\": {}}}\n",
        jbool(aecmos_present),
        jstr(AECMOS_MODEL_REL)
    ));
    s.push_str("}\n");
    s
}

fn render_clip_json(c: &ClipRecord) -> String {
    let near_end = match &c.near_end {
        Some(p) => format!(
            "{{\"windows\": {}, \"median_db\": {}, \"p5_db\": {}, \"min_db\": {}, \
             \"min_at_s\": {}, \"below_3db\": {}, \"below_6db\": {}, \"below_3db_pct\": {}, \
             \"below_6db_pct\": {}}}",
            p.n,
            jnum(p.median_db),
            jnum(p.p5_db),
            jnum(p.min_db),
            jnum(p.min_at_s),
            p.below_3db,
            p.below_6db,
            jnum(p.below_3db_pct()),
            jnum(p.below_6db_pct()),
        ),
        None => "null".to_string(),
    };
    format!(
        "    {{\"id\": {}, \"scenario\": {}, \"talk_type\": {}, \"mic\": {}, \"lpb\": {}, \
         \"enhanced\": {}, \"aligned\": {{\"mic\": {}, \"lpb\": {}, \"enhanced\": {}, \
         \"sample_rate_hz\": {ENGINE_RATE}, \"samples\": {}}}, \
         \"mic_rate_hz\": {}, \"lpb_rate_hz\": {}, \"duration_s\": {}, \
         \"engine_samples\": {}, \"turns\": {}, \"measured\": {}, \"starved_samples\": {}, \
         \"starved_pct\": {}, \"parked_samples\": {}, \"parked_pct\": {}, \
         \"dropped_samples\": {}, \"divergence_resets\": {}, \
         \"internal_erle_db\": {}, \"delay\": {{\"locked\": {}, \"engine_ms\": {}, \
         \"engine_samples\": {}, \"lock_turn\": {}, \"harness_estimate_ms\": {}, \
         \"harness_corr\": {}, \"within_window\": {}, \"mismatch\": {}}}, \
         \"tracking\": {{\"moves\": {}, \"reacquisitions\": {}, \"coarse_rearms\": {}, \
         \"regions_rejected\": {}, \"last_trigger\": {}}}, \
         \"protection\": {{\"double_talk_rate_pct\": {}, \"freeze_far_active_pct\": {}}}, \
         \"echo_coupling_db\": {}, \"reduction_full_db\": {}, \
         \"echo_reduction_converged_db\": {}, \"echo_windows\": {}, \"near_end\": {near_end}, \
         \"throughput\": {{\"audio_s\": {}, \"engine_wall_s\": {}, \"x_realtime\": {}}}}}",
        jstr(&c.id),
        jstr(c.scenario.dir_name()),
        jstr(c.scenario.talk_type()),
        jstr(&c.mic_rel),
        jstr(&c.lpb_rel),
        jstr(&c.enh_rel),
        jstr(&c.aligned_mic_rel),
        jstr(&c.aligned_lpb_rel),
        jstr(&c.aligned_enh_rel),
        c.engine_samples,
        c.mic_rate,
        c.lpb_rate,
        jnum(c.duration_s),
        c.engine_samples,
        c.turns,
        jbool(c.measured),
        c.starved,
        jnum(c.starved_pct),
        c.parked,
        jnum(c.parked_pct),
        c.dropped,
        c.divergence_resets,
        jnum(c.internal_erle_db),
        jbool(c.locked),
        jopt(c.engine_delay_ms()),
        jopt_int(c.engine_delay_samples.map(|v| v as u64)),
        jopt_int(c.lock_turn),
        jopt_int(c.est.as_ref().map(|e| e.lag_ms as u64)),
        jopt(c.est.as_ref().map(|e| e.corr)),
        c.within_window
            .map(|b| jbool(b).to_string())
            .unwrap_or_else(|| "null".to_string()),
        jbool(c.delay_mismatch),
        c.tracking_moves,
        c.reacquisitions,
        c.coarse_rearms,
        c.regions_rejected,
        c.reacquire_trigger
            .as_deref()
            .map(jstr)
            .unwrap_or_else(|| "null".to_string()),
        jnum(c.double_talk_rate_pct),
        jopt(c.freeze_far_active_pct),
        jopt(c.echo_coupling_db),
        jnum(c.reduction_full_db),
        jopt(c.echo_reduction_conv_db),
        c.echo_windows,
        jnum(c.audio_s),
        jnum(c.engine_wall_s),
        jnum(c.x_realtime),
    )
}

// ---------------------------------------------------------------------------
// Path and name helpers.
// ---------------------------------------------------------------------------

/// A path rendered relative to the crate root when it sits inside it
/// (case-insensitive, for Windows drive-letter variance), with forward
/// slashes, so result files read the same on every machine.
fn rel_display(path: &Path, crate_root: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let full = absolute.to_string_lossy().replace('\\', "/");
    let root = crate_root.to_string_lossy().replace('\\', "/");
    let stripped = full
        .strip_prefix(&root)
        .or_else(|| {
            if full.to_lowercase().starts_with(&root.to_lowercase()) {
                full.get(root.len()..)
            } else {
                None
            }
        })
        .map(|rest| rest.trim_start_matches('/').to_string());
    stripped.unwrap_or(full)
}

/// The default set name: the clip root relative to the crate's `data/`
/// folder when it sits inside it, components joined with dashes; otherwise
/// the folder's own name.
fn derive_set_name(root: &Path, crate_root: &Path) -> String {
    let rel = rel_display(root, crate_root);
    let trimmed = rel.strip_prefix("data/").unwrap_or(&rel);
    let joined = trimmed.trim_matches('/').replace('/', "-");
    let name = if joined.is_empty() {
        root.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "set".into())
    } else {
        joined
    };
    sanitize_set_name(&name)
}

/// Set names appear in file names, so they are restricted to a safe set.
fn sanitize_set_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "set".to_string()
    } else {
        cleaned
    }
}

// ---------------------------------------------------------------------------
// Regression tests for the whole-pipeline triplet contract, run by `cargo test`
// through this example's test harness. The resampling contract itself is tested
// once in the shared `resample` module every consumer calls.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::resample::testkit::chirp;
    use super::*;

    #[test]
    fn to_engine_rate_uses_the_shared_contract() {
        // The wrapper adds only the duration bookkeeping: its samples are what
        // the shared helper returns for the same input.
        let samples = chirp(48_000, 1.0, 100.0, 3_000.0);
        let direct =
            resample::resample_aligned(&samples, 48_000, ENGINE_RATE).expect("shared helper");
        let clip = wav::MonoClip {
            samples: samples.clone(),
            sample_rate: 48_000,
        };
        let wrapped = to_engine_rate(clip).expect("wrapper");
        assert_eq!(wrapped.samples.len(), direct.len());
        for (i, (a, b)) in direct.iter().zip(wrapped.samples.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "sample {i} differs");
        }
        assert!((wrapped.input_seconds - 1.0).abs() < 1e-9);
    }

    #[test]
    fn passthrough_triplet_is_bit_exact() {
        // A near-end clip with a silent loopback: the engine passes the
        // capture through untouched, so the canonical mic and enhanced
        // signals must match bit for bit.
        let mic = wav::MonoClip {
            samples: chirp(48_000, 3.0, 100.0, 3_000.0),
            sample_rate: 48_000,
        };
        let lpb = wav::MonoClip {
            samples: vec![0.0; 48_000 * 3],
            sample_rate: 48_000,
        };
        let near = to_engine_rate(mic).expect("mic resample");
        let far = fit_len(
            to_engine_rate(lpb).expect("lpb resample").samples,
            near.samples.len(),
        );
        let mut config = AecConfig::default();
        config.sample_rate = ENGINE_RATE;
        let mut aec = Aec::new(config).expect("engine");
        let out = run_canceller(&mut aec, &near.samples, &far, |_, _| {}).expect("run");
        assert_eq!(far.len(), near.samples.len());
        assert_eq!(out.len(), near.samples.len());
        for (i, (a, b)) in near.samples.iter().zip(out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "sample {i} differs: {a} versus {b}"
            );
        }
    }
}
