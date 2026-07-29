//! Dataset split tool for the decibri-aec benchmark kit.
//!
//! ```text
//! cargo run --release --example make-split -- --size 50 [--pool DIR] [--seed U64]
//!     [--name LABEL] [--out PATH] [--test-frac FRACTION]
//! ```
//!
//! Draws a reproducible, scenario-stratified train/test split off a pool of
//! AEC-Challenge-style clip pairs and writes it as one JSON manifest per dev
//! cycle. The pool holds up to three scenario folders (`doubletalk/`,
//! `farend-singletalk/`, `nearend-singletalk/`), each with `<stem>_mic.wav` /
//! `<stem>_lpb.wav` pairs. The split is a REFERENCE only: the pool stays
//! intact and no audio is copied or moved. The manifest names the stems of
//! each set and starts with an empty `results` array that the benchmark later
//! appends to (see `examples/benchmark/`).
//!
//! Determinism: the split is drawn with a seeded generator, so the same
//! `--seed` and `--size` over the same pool produce byte-identical split fields
//! (`seed`, `n`, `split`, `train`, `test`). When `--seed` is omitted one is
//! generated from the system clock and RECORDED in the manifest, so the draw
//! is reproducible from that recorded seed alone.
//!
//! Stratification: `--size` is a total drawn from all three scenarios in
//! proportion to how many pairs each scenario offers (largest-remainder
//! rounding), then each scenario's draw is split into train and test within
//! itself by `--test-frac` (default 0.2), so both sets carry every scenario.
//!
//! Nothing here trains, tunes, or touches the engine: it selects and records.
//! The manifest lists licensed-data filenames, so its default location under
//! `data/splits/` is gitignored and it must never be committed.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The three AEC-Challenge scenarios, in the order the manifest lists them.
const SCENARIOS: [&str; 3] = ["doubletalk", "farend-singletalk", "nearend-singletalk"];

/// Default pool folder, relative to the crate root.
const DEFAULT_POOL: &str = "data/test_set_icassp2022";

/// Default fraction of each scenario's draw held back as the test set.
const DEFAULT_TEST_FRAC: f64 = 0.2;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Cli::parse(&args).and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: cargo run --release --example make-split -- --size <total> \
                 [--pool DIR] [--seed U64]\n\
                 \x20      [--name LABEL] [--out PATH] [--test-frac FRACTION]"
            );
            ExitCode::FAILURE
        }
    }
}

/// One parsed invocation.
struct Cli {
    n: usize,
    pool: PathBuf,
    seed: Option<u64>,
    name: Option<String>,
    out: Option<PathBuf>,
    test_frac: f64,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut n = None;
        let mut pool = None;
        let mut seed = None;
        let mut name = None;
        let mut out = None;
        let mut test_frac = None;

        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            let mut value = |flag: &str| -> Result<String, String> {
                iter.next()
                    .cloned()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match arg.as_str() {
                "--size" => {
                    let raw = value("--size")?;
                    n = Some(
                        raw.parse::<usize>()
                            .map_err(|_| format!("--size value '{raw}' is not a count"))?,
                    );
                }
                "--pool" => pool = Some(PathBuf::from(value("--pool")?)),
                "--seed" => {
                    let raw = value("--seed")?;
                    seed = Some(
                        raw.parse::<u64>()
                            .map_err(|_| format!("--seed value '{raw}' is not a u64"))?,
                    );
                }
                "--name" => name = Some(value("--name")?),
                "--out" => out = Some(PathBuf::from(value("--out")?)),
                "--test-frac" => {
                    let raw = value("--test-frac")?;
                    let f = raw
                        .parse::<f64>()
                        .map_err(|_| format!("--test-frac value '{raw}' is not a number"))?;
                    if !(0.0..=1.0).contains(&f) {
                        return Err(format!("--test-frac {f} must be between 0 and 1"));
                    }
                    test_frac = Some(f);
                }
                other => return Err(format!("unknown flag '{other}'")),
            }
        }

        Ok(Cli {
            n: n.ok_or("--size <total> is required")?,
            pool: pool.unwrap_or_else(|| PathBuf::from(DEFAULT_POOL)),
            seed,
            name,
            out,
            test_frac: test_frac.unwrap_or(DEFAULT_TEST_FRAC),
        })
    }
}

/// The per-scenario draw: which stems fell to train and which to test.
struct ScenarioDraw {
    scenario: &'static str,
    available: usize,
    train: Vec<String>,
    test: Vec<String>,
}

fn run(cli: Cli) -> Result<(), String> {
    if cli.n == 0 {
        return Err("--size must be at least 1".to_string());
    }
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // The pool path recorded in the manifest is kept relative to the crate
    // root so the manifest reads the same on every machine.
    let pool_abs = if cli.pool.is_absolute() {
        cli.pool.clone()
    } else {
        crate_root.join(&cli.pool)
    };
    if !pool_abs.is_dir() {
        return Err(format!("pool {} is not a folder", pool_abs.display()));
    }

    // Discover the available stems per scenario, sorted, so the draw depends
    // only on the seed and the pool contents.
    let mut available: Vec<(usize, Vec<String>)> = Vec::new();
    for (idx, scenario) in SCENARIOS.iter().enumerate() {
        let dir = pool_abs.join(scenario);
        let stems = discover_stems(&dir)?;
        available.push((idx, stems));
    }
    let total_available: usize = available.iter().map(|(_, s)| s.len()).sum();
    if total_available == 0 {
        return Err(format!(
            "no <stem>_mic.wav / <stem>_lpb.wav pairs under {}; expected doubletalk/, \
             farend-singletalk/, or nearend-singletalk/ subfolders",
            pool_abs.display()
        ));
    }

    // A pool smaller than the request draws everything it has; the recorded
    // n stays consistent with the split counts either way.
    let requested = cli.n;
    let target = requested.min(total_available);
    if target < requested {
        eprintln!(
            "warning: pool holds only {total_available} pairs; drawing all {target} \
             instead of the requested {requested}"
        );
    }

    // Allocate the total across scenarios in proportion to availability.
    let counts = allocate(&available, target);

    // Resolve or generate the seed. A generated seed is recorded so the draw
    // reproduces from the manifest alone.
    let seed = cli.seed.unwrap_or_else(generate_seed);
    if cli.seed.is_none() {
        eprintln!("seed: generated {seed} (recorded in the manifest)");
    }

    let test_frac = cli.test_frac;
    let mut draws: Vec<ScenarioDraw> = Vec::new();
    for ((idx, stems), &draw_count) in available.iter().zip(counts.iter()) {
        let scenario = SCENARIOS[*idx];
        let (train, test) = draw_scenario(seed, *idx, stems, draw_count, test_frac);
        draws.push(ScenarioDraw {
            scenario,
            available: stems.len(),
            train,
            test,
        });
    }

    let total_drawn: usize = draws.iter().map(|d| d.train.len() + d.test.len()).sum();
    let name = cli.name.unwrap_or_else(default_name);
    let pool_rel = rel_to_root(&pool_abs, &crate_root);

    let out_path = cli.out.clone().unwrap_or_else(|| {
        crate_root
            .join("data")
            .join("splits")
            .join(format!("{name}.json"))
    });

    // make-split never overwrites: a new cycle is a new manifest.
    if out_path.exists() {
        return Err(format!(
            "manifest already exists at {}; a new cycle needs a new --name or --out \
             (make-split never overwrites an existing split)",
            out_path.display()
        ));
    }

    let json = render_manifest(&name, seed, &pool_rel, total_drawn, &draws);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(&out_path, &json)
        .map_err(|e| format!("cannot write {}: {e}", out_path.display()))?;

    // A short human summary; the visible per-scenario counts are the
    // stratification proof.
    eprintln!("wrote {}", rel_to_root(&out_path, &crate_root));
    eprintln!("  name {name}, seed {seed}, total {total_drawn}, test-frac {test_frac}");
    for d in &draws {
        eprintln!(
            "  {:<20} train {:>3}  test {:>3}   (of {} available)",
            d.scenario,
            d.train.len(),
            d.test.len(),
            d.available
        );
    }
    Ok(())
}

/// Lists the `<stem>_mic.wav` / `<stem>_lpb.wav` pair identifiers in one
/// scenario folder, sorted, keeping only stems that carry both sides. A
/// missing folder is treated as an empty scenario, not an error, so a pool
/// that omits a scenario still produces a manifest.
fn discover_stems(dir: &Path) -> Result<Vec<String>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
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
    let paired: Vec<String> = mics
        .into_iter()
        .filter(|stem| lpbs.binary_search(stem).is_ok())
        .collect();
    Ok(paired)
}

/// Splits `target` across the scenarios in proportion to how many pairs each
/// offers, using largest-remainder rounding so the parts sum to `target` and
/// no scenario is allocated more than it has.
fn allocate(available: &[(usize, Vec<String>)], target: usize) -> Vec<usize> {
    let sizes: Vec<usize> = available.iter().map(|(_, s)| s.len()).collect();
    let total: usize = sizes.iter().sum();
    if total == 0 || target == 0 {
        return vec![0; sizes.len()];
    }
    // Exact quota per scenario, then floor, tracking the fractional remainder.
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
    // Distribute the leftover to the largest remainders first, skipping any
    // scenario already at its available ceiling. Ties break by scenario order.
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

/// Draws `count` stems from `stems` for scenario `idx` and splits them into
/// (train, test) by `test_frac`. The draw and the partition are both fixed by
/// the seed; the returned lists are sorted for a stable, readable manifest.
fn draw_scenario(
    seed: u64,
    idx: usize,
    stems: &[String],
    count: usize,
    test_frac: f64,
) -> (Vec<String>, Vec<String>) {
    let count = count.min(stems.len());
    if count == 0 {
        return (Vec::new(), Vec::new());
    }

    // A per-scenario substream so the scenarios draw independently: adding or
    // removing a scenario never reshuffles the others.
    let mut rng = Rng::new(seed ^ SCENARIO_SALT[idx]);
    let mut order: Vec<usize> = (0..stems.len()).collect();
    fisher_yates(&mut order, &mut rng);
    let drawn: Vec<String> = order[..count].iter().map(|&i| stems[i].clone()).collect();

    // Test size: the rounded fraction, forced to leave both sets non-empty
    // whenever the scenario drew at least two clips.
    let mut test_k = (count as f64 * test_frac).round() as usize;
    if count >= 2 {
        test_k = test_k.clamp(1, count - 1);
    } else {
        test_k = 0;
    }

    let mut test: Vec<String> = drawn[..test_k].to_vec();
    let mut train: Vec<String> = drawn[test_k..].to_vec();
    test.sort();
    train.sort();
    (train, test)
}

/// Distinct per-scenario salts so each scenario's substream is independent.
const SCENARIO_SALT: [u64; 3] = [
    0x1234_5678_9abc_def0,
    0x0fed_cba9_8765_4321,
    0xa5a5_5a5a_c3c3_3c3c,
];

/// SplitMix64, a small deterministic generator. Not cryptographic; used only
/// to make the dataset draw reproducible from a recorded seed.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `0..bound` (bound > 0) by rejection sampling, so the
    /// draw carries no modulo bias.
    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % bound;
            }
        }
    }
}

/// In-place Fisher-Yates shuffle driven by `rng`.
fn fisher_yates(items: &mut [usize], rng: &mut Rng) {
    let len = items.len();
    for i in (1..len).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        items.swap(i, j);
    }
}

/// A seed derived from the system clock, for when the caller supplies none.
/// Only the recorded value matters for reproducibility, so the source of
/// entropy is unimportant beyond being different between cycles.
fn generate_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // One SplitMix64 step spreads the low-entropy clock value across the word.
    let mut rng = Rng::new(nanos ^ 0xD1B5_4A32_D192_ED03);
    rng.next_u64()
}

// ---------------------------------------------------------------------------
// Manifest rendering (hand-rolled JSON, schema version 1).
// ---------------------------------------------------------------------------

fn render_manifest(
    name: &str,
    seed: u64,
    pool_rel: &str,
    n: usize,
    draws: &[ScenarioDraw],
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"version\": 1,\n");
    s.push_str(&format!("  \"created\": {},\n", jstr(&utc_now_iso())));
    s.push_str(&format!("  \"name\": {},\n", jstr(name)));
    s.push_str(&format!("  \"seed\": {seed},\n"));
    s.push_str(&format!("  \"pool\": {},\n", jstr(pool_rel)));
    s.push_str(&format!("  \"n\": {n},\n"));

    s.push_str("  \"split\": {\n");
    for (i, d) in draws.iter().enumerate() {
        let comma = if i + 1 < draws.len() { "," } else { "" };
        s.push_str(&format!(
            "    {}: {{ \"train\": {}, \"test\": {} }}{comma}\n",
            jstr(d.scenario),
            d.train.len(),
            d.test.len()
        ));
    }
    s.push_str("  },\n");

    s.push_str("  \"train\": {\n");
    render_set(&mut s, draws, |d| &d.train);
    s.push_str("  },\n");

    s.push_str("  \"test\": {\n");
    render_set(&mut s, draws, |d| &d.test);
    s.push_str("  },\n");

    s.push_str("  \"results\": []\n");
    s.push_str("}\n");
    s
}

/// Renders one of the `train` / `test` objects: a stem-list per scenario.
fn render_set<F: Fn(&ScenarioDraw) -> &Vec<String>>(
    s: &mut String,
    draws: &[ScenarioDraw],
    pick: F,
) {
    for (i, d) in draws.iter().enumerate() {
        let comma = if i + 1 < draws.len() { "," } else { "" };
        let stems: Vec<String> = pick(d).iter().map(|stem| jstr(stem)).collect();
        s.push_str(&format!(
            "    {}: [{}]{comma}\n",
            jstr(d.scenario),
            stems.join(", ")
        ));
    }
}

/// A JSON string literal with the escapes the manifest can encounter.
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

// ---------------------------------------------------------------------------
// Naming and paths.
// ---------------------------------------------------------------------------

/// The default cycle label, a dated `cycle-YYYY-MM-DD`.
fn default_name() -> String {
    let (year, month, day, _, _, _) = utc_now_parts();
    format!("cycle-{year:04}-{month:02}-{day:02}")
}

/// A path rendered relative to the crate root when it sits inside it
/// (case-insensitive for Windows drive-letter variance), with forward
/// slashes, matching the benchmark harness's own `rel_display`.
fn rel_to_root(path: &Path, crate_root: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let full = absolute.to_string_lossy().replace('\\', "/");
    let root = crate_root.to_string_lossy().replace('\\', "/");
    full.strip_prefix(&root)
        .or_else(|| {
            if full.to_lowercase().starts_with(&root.to_lowercase()) {
                full.get(root.len()..)
            } else {
                None
            }
        })
        .map(|rest| rest.trim_start_matches('/').to_string())
        .unwrap_or(full)
}

// ---------------------------------------------------------------------------
// UTC time, self-contained (mirrors the benchmark's provenance::utc_now so no
// date dependency enters the kit).
// ---------------------------------------------------------------------------

fn utc_now_iso() -> String {
    let (year, month, day, hour, minute, second) = utc_now_parts();
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// The current UTC time as (year, month, day, hour, minute, second) via the
/// standard days-from-civil inversion.
fn utc_now_parts() -> (i64, i64, i64, i64, i64, i64) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) = (
        (rem / 3600) as i64,
        ((rem % 3600) / 60) as i64,
        (rem % 60) as i64,
    );

    // Howard Hinnant's civil-from-days algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year_base = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year_base + 1 } else { year_base };
    (year, month, day, hour, minute, second)
}
