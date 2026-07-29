//! Delay-acquisition coverage probe: does the engine find the right delay
//! across the whole configured ceiling, on synthetic pairs whose true delay is
//! known exactly?
//!
//! ```text
//! cargo run --release --example delay-probe
//! cargo run --release --example delay-probe -- --ceiling-ms 250
//! cargo run --release --example delay-probe -- --track
//! ```
//!
//! `--ceiling-ms` sets [`AecConfig::max_search_delay_ms`].
//!
//! `--track` runs the tracking and reacquisition suite instead.

use std::process::ExitCode;
use std::time::Instant;

use decibri_aec::{Aec, AecConfig, DelayStatus};

/// One near-end block, the cadence the benchmark and the other examples use.
const TURN: usize = 256;
const RATE: u32 = 16000;
/// Clip length.
const CLIP_SAMPLES: usize = 240_000; // 15 s at 16 kHz.

/// The delays swept, in milliseconds.
const SWEEP_MS: [usize; 10] = [50, 150, 250, 325, 450, 650, 750, 900, 1100, 0];

fn main() -> ExitCode {
    let mut ceiling_ms: u16 = 1000;
    let mut track = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ceiling-ms" => match args.next().and_then(|v| v.parse::<u16>().ok()) {
                Some(value) => ceiling_ms = value,
                None => {
                    eprintln!("--ceiling-ms needs a number");
                    return ExitCode::FAILURE;
                }
            },
            "--track" => track = true,
            other => {
                eprintln!("unknown argument '{other}'");
                eprintln!("usage: delay-probe [--ceiling-ms N] [--track]");
                return ExitCode::FAILURE;
            }
        }
    }
    if track {
        return track_suite();
    }

    let tail_ms = AecConfig::default().tail_ms as usize;
    let fine_ms = AecConfig::default().max_echo_delay_ms as usize;
    println!("decibri-aec delay acquisition coverage probe");
    println!(
        "  engine: tau at {RATE} Hz, tail {tail_ms} ms, fine window {fine_ms} ms, \
         coarse ceiling {ceiling_ms} ms"
    );
    println!("  pairs:  synthetic far-end single talk, {CLIP_SAMPLES} samples");
    println!("  tail: {tail_ms} ms");
    println!();
    println!(
        "  {:>8}  {:>8}  {:>8}  {:>7}  {:>7}  {:>6}  {:>6}  {:>7}  {:>6}  verdict",
        "true ms", "lock ms", "onset ms", "err ms", "acq s", "reloc", "coarse", "src", "xRT",
    );

    let mut failures = 0usize;
    let mut rows = 0usize;
    for &delay_ms in SWEEP_MS.iter() {
        let bulk = (delay_ms * RATE as usize) / 1000;
        let (far, near) = delayed_pair(CLIP_SAMPLES, bulk);
        let outcome = run(ceiling_ms, &far, &near);

        // Synthesized delay plus the one-block anchor lead.
        let onset = bulk + TURN;
        let onset_ms = onset as f64 * 1000.0 / RATE as f64;
        let beyond_ceiling = delay_ms > ceiling_ms as usize;

        let (lock_ms, err_ms, verdict) = match outcome.delay {
            None => (
                "none".to_string(),
                "-".to_string(),
                if beyond_ceiling {
                    "NO-LOCK (correct: past ceiling)"
                } else {
                    "NO-LOCK"
                },
            ),
            Some(delay) => {
                let err = onset as i64 - delay as i64;
                let err_ms = err as f64 * 1000.0 / RATE as f64;
                let tail = (tail_ms * RATE as usize) / 1000;
                let verdict = if err < 0 {
                    "LATE"
                } else if err as usize > tail {
                    "EARLY"
                } else if beyond_ceiling {
                    "LOCKED PAST CEILING"
                } else {
                    "CORRECT"
                };
                (
                    format!("{:.0}", delay as f64 * 1000.0 / RATE as f64),
                    format!("{err_ms:+.1}"),
                    verdict,
                )
            }
        };
        if !verdict.starts_with("CORRECT") && !verdict.starts_with("NO-LOCK (correct") {
            failures += 1;
        }
        rows += 1;

        println!(
            "  {:>8}  {:>8}  {:>8.0}  {:>7}  {:>7}  {:>6}  {:>6}  {:>7}  {:>6.0}  {}",
            delay_ms,
            lock_ms,
            onset_ms,
            err_ms,
            outcome
                .acquired_s
                .map(|s| format!("{s:.2}"))
                .unwrap_or_else(|| "-".to_string()),
            if outcome.relocated { "yes" } else { "no" },
            outcome
                .coarse_region_ms
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".to_string()),
            outcome.source,
            outcome.realtime,
            verdict,
        );
    }

    println!();
    if failures == 0 {
        println!("delay-probe PASS: {rows}/{rows} delays resolved correctly");
        ExitCode::SUCCESS
    } else {
        println!("delay-probe FAIL: {failures} of {rows} delays did not resolve correctly");
        ExitCode::FAILURE
    }
}

/// What one probe run observed.
struct Outcome {
    delay: Option<usize>,
    acquired_s: Option<f64>,
    relocated: bool,
    coarse_region_ms: Option<f64>,
    source: &'static str,
    realtime: f64,
}

/// Drives the engine over a pair at the standard cadence.
fn run(ceiling_ms: u16, far: &[f32], near: &[f32]) -> Outcome {
    // `AecConfig` is non-exhaustive, so it is built by assignment from the
    // defaults rather than by struct literal, the way every other example does.
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.max_search_delay_ms = ceiling_ms;
    let mut aec = Aec::new(config).expect("probe configuration is valid");
    let mut out = Vec::with_capacity(near.len() + TURN);
    let mut far_chunks = far.chunks(TURN);
    let mut acquired_at = None;
    let mut processed = 0usize;

    let started = Instant::now();
    for near_chunk in near.chunks(TURN) {
        if let Some(far_chunk) = far_chunks.next() {
            aec.feed_reference(far_chunk);
        }
        aec.process(near_chunk, &mut out).expect("process succeeds");
        processed += near_chunk.len();
        if acquired_at.is_none() && aec.metrics().delay_samples.is_some() {
            acquired_at = Some(processed);
        }
    }
    aec.flush(&mut out).expect("flush succeeds");
    let wall = started.elapsed().as_secs_f64();

    let metrics = aec.metrics();
    Outcome {
        delay: metrics.delay_samples,
        acquired_s: acquired_at.map(|n| n as f64 / RATE as f64),
        relocated: metrics.delay.relocated,
        coarse_region_ms: metrics
            .delay
            .coarse_region_samples
            .map(|s| s as f64 * 1000.0 / RATE as f64),
        source: match metrics.delay.status {
            DelayStatus::Locked(source) => match source {
                decibri_aec::DelayLockSource::Hint => "hint",
                decibri_aec::DelayLockSource::GlobalAgreement => "global",
                decibri_aec::DelayLockSource::LocalEvidence => "local",
                decibri_aec::DelayLockSource::CoarseRegion => "coarse",
                _ => "other",
            },
            DelayStatus::Relocated => "-",
            _ => "-",
        },
        realtime: (near.len() as f64 / RATE as f64) / wall.max(f64::MIN_POSITIVE),
    }
}

/// A deterministic linear congruential generator, integer state mapped to
/// `f32`: the same generator the crate's other fixtures use.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }

    fn next_unit(&mut self) -> f32 {
        (self.next_f32() + 1.0) * 0.5
    }
}

/// The known sparse echo path, taps in samples relative to the path onset.
const ECHO_IR: [(usize, f32); 4] = [(0, 0.5), (80, -0.25), (240, 0.12), (410, -0.06)];
const ECHO_GAIN: f32 = 0.5;
const NOISE_FLOOR: f32 = 0.001;

/// A far-end single-talk pair whose echo path begins exactly `bulk` samples
/// after the far-end sample that caused it.
///
/// The far end is broadband noise under a syllabic amplitude envelope, with
/// segment lengths and levels both drawn from the generator.
fn delayed_pair(len: usize, bulk: usize) -> (Vec<f32>, Vec<f32>) {
    let mut carrier = Lcg(0x1234_5678);
    let mut shape = Lcg(0x00C0_FFEE);
    let mut floor = Lcg(0x0F10_0F10);

    let mut far = Vec::with_capacity(len);
    let mut level = 0.0_f32;
    let mut remaining = 0_usize;
    while far.len() < len {
        if remaining == 0 {
            remaining = 400 + (shape.next_unit() * 2400.0) as usize;
            level = if shape.next_unit() < 0.28 {
                0.0
            } else {
                0.15 + 0.85 * shape.next_unit()
            };
        }
        far.push(level * carrier.next_f32());
        remaining -= 1;
    }

    let mic: Vec<f32> = (0..len)
        .map(|i| {
            let mut echo = 0.0_f32;
            for &(tap, coeff) in &ECHO_IR {
                let lag = bulk + tap;
                if i >= lag {
                    echo += coeff * far[i - lag];
                }
            }
            ECHO_GAIN * echo + NOISE_FLOOR * floor.next_f32()
        })
        .collect();
    (far, mic)
}

// ---------------------------------------------------------------------------
// The tracking and reacquisition suite (`--track`).
// ---------------------------------------------------------------------------

/// A speech-shaped signal: broadband noise under a syllabic envelope, the same
/// construction `delayed_pair` uses for its far end.
fn speech_like(len: usize, carrier_seed: u64, shape_seed: u64) -> Vec<f32> {
    let mut carrier = Lcg(carrier_seed);
    let mut shape = Lcg(shape_seed);
    let mut signal = Vec::with_capacity(len);
    let mut level = 0.0_f32;
    let mut remaining = 0_usize;
    while signal.len() < len {
        if remaining == 0 {
            remaining = 400 + (shape.next_unit() * 2400.0) as usize;
            level = if shape.next_unit() < 0.28 {
                0.0
            } else {
                0.15 + 0.85 * shape.next_unit()
            };
        }
        signal.push(level * carrier.next_f32());
        remaining -= 1;
    }
    signal
}

/// How a case bounds its late excursion: which quantity the safety gate
/// measures.
enum LateGate {
    /// Cap on the worst lateness from the exact onset, ms.
    Absolute(f64),
    /// The post-lock excursion must stay inside the filter tail, so the echo
    /// onset stays within the filter's coverage. The bound is
    /// [`AecConfig::tail_ms`].
    ExcursionWithinTail,
}

/// One tracking-suite case: a synthesized scenario with an exact bulk-delay
/// trajectory and per-case expectations.
struct TrackCase {
    name: &'static str,
    /// Clip length in samples.
    len: usize,
    /// The bulk delay at mic sample `i`, in samples: the exact trajectory.
    bulk_at: fn(usize) -> usize,
    /// Far-end silence span (a render gap), if any.
    far_gap: Option<(usize, usize)>,
    /// Near-end talker: level, over a span (the whole clip when the span
    /// covers it).
    talker: Option<((usize, usize), f32)>,
    /// Echo present only from this sample on (0 for the whole clip).
    echo_from: usize,
    /// Capture stall: near samples in `[start, start + length)` are never
    /// processed while the far side keeps being fed, as a stalled consumer.
    stall: Option<(usize, usize)>,
    /// The sample at which the trajectory's disruptive event happens, for the
    /// recovery measurement; 0 when the case has none.
    event_at: usize,
    /// Expectations.
    expect_moves_min: u32,
    expect_reacq_min: u32,
    expect_reacq_max: u32,
    expect_rearms_min: u32,
    /// How this case bounds its late excursion (absolute worst, or post-lock).
    late: LateGate,
    /// The recovery budget after `event_at`, in seconds (ignored when 0).
    recovery_budget_s: f64,
}

/// What one tracked run measured.
struct TrackOutcome {
    locked_at_s: Option<f64>,
    final_delay: Option<usize>,
    final_status: &'static str,
    moves: u32,
    reacquisitions: u32,
    trigger: Option<decibri_aec::ReacquireTrigger>,
    rearms: u32,
    /// The worst lateness from the exact onset, ms (0.0 if never late).
    max_late_ms: f64,
    /// The signed lateness at the first lock, ms: negative when the lock lands
    /// early, positive if it locks late.
    initial_late_ms: f64,
    /// The post-lock late excursion, ms: the worst signed lateness minus
    /// [`initial_late_ms`](TrackOutcome::initial_late_ms).
    late_excursion_ms: f64,
    mean_abs_err_ms: f64,
    safe_pct: f64,
    recovery_s: Option<f64>,
    realtime: f64,
}

/// Drives one case at the standard cadence, scoring the engine's alignment
/// against the exact synthesized onset after every turn.
fn run_tracked(case: &TrackCase) -> TrackOutcome {
    let far_raw = speech_like(case.len, 0x1234_5678, 0x00C0_FFEE);
    let mut far = far_raw.clone();
    if let Some((from, to)) = case.far_gap {
        for slot in far[from..to].iter_mut() {
            *slot = 0.0;
        }
    }
    let talker = speech_like(case.len, 0x7E57_7A1E, 0x0B0B_0B0B);
    let mut floor = Lcg(0x0F10_0F10);
    let near: Vec<f32> = (0..case.len)
        .map(|i| {
            let mut echo = 0.0_f32;
            if i >= case.echo_from {
                let bulk = (case.bulk_at)(i);
                for &(tap, coeff) in &ECHO_IR {
                    let lag = bulk + tap;
                    if i >= lag {
                        echo += coeff * far[i - lag];
                    }
                }
            }
            let voice = match case.talker {
                Some(((from, to), level)) if (from..to).contains(&i) => level * talker[i],
                _ => 0.0,
            };
            ECHO_GAIN * echo + voice + NOISE_FLOOR * floor.next_f32()
        })
        .collect();

    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    let mut aec = Aec::new(config).expect("track configuration is valid");
    let mut out = Vec::with_capacity(near.len() + TURN);
    let tail = (AecConfig::default().tail_ms as usize * RATE as usize) / 1000;

    let mut locked_at: Option<usize> = None;
    let mut max_late = 0.0_f64;
    // The signed lateness at the first scored turn after lock, and the worst
    // signed lateness over the clip (positive = late).
    let mut initial_late: Option<f64> = None;
    let mut worst_signed_late = f64::NEG_INFINITY;
    let mut abs_err_sum = 0.0_f64;
    let mut scored_turns = 0_u64;
    let mut safe_turns = 0_u64;
    let mut recovery: Option<usize> = None;

    let started = Instant::now();
    let mut cursor = 0usize;
    let mut fed = 0usize;
    let feed_to = |aec: &mut Aec, upto: usize, fed: &mut usize| {
        let upto = upto.min(far.len());
        if upto > *fed {
            aec.feed_reference(&far[*fed..upto]);
            *fed = upto;
        }
    };
    while cursor < near.len() {
        if let Some((stall_at, stall_len)) = case.stall {
            if cursor == stall_at {
                // The consumer stalls: render continues, capture is lost.
                feed_to(&mut aec, cursor + stall_len + TURN, &mut fed);
                cursor += stall_len;
                continue;
            }
        }
        let take = TURN.min(near.len() - cursor);
        feed_to(&mut aec, cursor + TURN, &mut fed);
        aec.process(&near[cursor..cursor + take], &mut out)
            .expect("process succeeds");
        cursor += take;

        let metrics = aec.metrics();
        if locked_at.is_none() && metrics.delay_samples.is_some() {
            locked_at = Some(cursor);
        }
        // Score the alignment against the exact onset for THIS point of the
        // trajectory. Only meaningful once a first lock exists and while the
        // echo actually exists.
        if locked_at.is_some() && cursor > case.echo_from {
            if let Some(delay) = metrics.delay_samples {
                let bulk = (case.bulk_at)(cursor.saturating_sub(1));
                let onset = bulk + TURN + ECHO_IR[0].0;
                let err = onset as f64 - delay as f64;
                let err_ms = err * 1000.0 / RATE as f64;
                scored_turns += 1;
                abs_err_sum += err_ms.abs();
                // Signed lateness, positive on the fatal side. `initial_late`
                // is captured on the first scored turn (the acquisition offset);
                // `worst_signed_late` tracks the deepest lateness reached, which
                // stays negative on a case that is early throughout.
                let signed_late = -err_ms;
                if initial_late.is_none() {
                    initial_late = Some(signed_late);
                }
                worst_signed_late = worst_signed_late.max(signed_late);
                if err < 0.0 {
                    max_late = max_late.max(-err_ms);
                } else if (err as usize) < tail {
                    safe_turns += 1;
                }
                if recovery.is_none()
                    && case.event_at > 0
                    && cursor > case.event_at
                    && err >= 0.0
                    && (err as usize) < tail
                {
                    recovery = Some(cursor - case.event_at);
                }
            }
        }
    }
    aec.flush(&mut out).expect("flush succeeds");
    let wall = started.elapsed().as_secs_f64();

    let metrics = aec.metrics();
    TrackOutcome {
        locked_at_s: locked_at.map(|s| s as f64 / RATE as f64),
        final_delay: metrics.delay_samples,
        final_status: match metrics.delay.status {
            DelayStatus::Locked(_) => "locked",
            DelayStatus::Reacquiring => "reacq",
            DelayStatus::Relocated => "reloc",
            _ => "search",
        },
        moves: metrics.delay.tracking_moves,
        reacquisitions: metrics.delay.reacquisitions,
        trigger: metrics.delay.last_reacquire_trigger,
        rearms: metrics.delay.coarse_rearms,
        max_late_ms: max_late,
        initial_late_ms: initial_late.unwrap_or(0.0),
        // Undefined (0.0) when no turn was ever scored.
        late_excursion_ms: match initial_late {
            Some(init) => worst_signed_late - init,
            None => 0.0,
        },
        mean_abs_err_ms: if scored_turns > 0 {
            abs_err_sum / scored_turns as f64
        } else {
            0.0
        },
        safe_pct: if scored_turns > 0 {
            safe_turns as f64 * 100.0 / scored_turns as f64
        } else {
            0.0
        },
        recovery_s: recovery.map(|s| s as f64 / RATE as f64),
        realtime: (case.len as f64 / RATE as f64) / wall.max(f64::MIN_POSITIVE),
    }
}

/// Milliseconds to samples at the probe rate.
const fn ms(v: usize) -> usize {
    v * RATE as usize / 1000
}

/// The trajectory functions. Plain `fn` items so the case table stays a const
/// construction with no boxing.
fn bulk_static_100(_: usize) -> usize {
    ms(100)
}
fn bulk_static_150(_: usize) -> usize {
    ms(150)
}
fn bulk_static_300(_: usize) -> usize {
    ms(300)
}
fn bulk_static_400(_: usize) -> usize {
    ms(400)
}
/// 150 to 190 ms over 8 s to 28 s: 2 ms/s deeper.
fn bulk_drift_slow(i: usize) -> usize {
    ramp(i, ms(150), ms(190), 8 * RATE as usize, 28 * RATE as usize)
}
/// 200 to 320 ms over 8 s to 20 s: 10 ms/s deeper.
fn bulk_drift_fast(i: usize) -> usize {
    ramp(i, ms(200), ms(320), 8 * RATE as usize, 20 * RATE as usize)
}
/// 150 down to 118 ms over 8 s to 24 s: 2 ms/s toward the fatal side.
fn bulk_drift_down_slow(i: usize) -> usize {
    ramp(i, ms(150), ms(118), 8 * RATE as usize, 24 * RATE as usize)
}
/// 250 down to 150 ms over 8 s to 24 s: 6.25 ms/s toward the fatal side.
fn bulk_drift_down_fast(i: usize) -> usize {
    ramp(i, ms(250), ms(150), 8 * RATE as usize, 24 * RATE as usize)
}
/// 100 to 180 ms step at 12 s.
fn bulk_jump_in(i: usize) -> usize {
    if i < 12 * RATE as usize {
        ms(100)
    } else {
        ms(180)
    }
}
/// 100 to 600 ms step at 12 s.
fn bulk_jump_deep(i: usize) -> usize {
    if i < 12 * RATE as usize {
        ms(100)
    } else {
        ms(600)
    }
}
/// 600 to 100 ms step at 12 s.
fn bulk_jump_shallow(i: usize) -> usize {
    if i < 12 * RATE as usize {
        ms(600)
    } else {
        ms(100)
    }
}
/// 300 to 500 ms step at 12 s, across a capture stall at the same instant.
fn bulk_disc_jump(i: usize) -> usize {
    if i < 12 * RATE as usize {
        ms(300)
    } else {
        ms(500)
    }
}

/// Linear interpolation of a bulk delay between two times.
fn ramp(i: usize, from: usize, to: usize, start: usize, end: usize) -> usize {
    if i <= start {
        return from;
    }
    if i >= end {
        return to;
    }
    let span = end - start;
    let at = i - start;
    if to >= from {
        from + (to - from) * at / span
    } else {
        from - (from - to) * at / span
    }
}

fn track_suite() -> ExitCode {
    let s = RATE as usize;
    let cases: [TrackCase; 13] = [
        TrackCase {
            name: "control-static",
            len: 30 * s,
            bulk_at: bulk_static_100,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(0.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "drift-slow +2ms/s",
            len: 30 * s,
            bulk_at: bulk_drift_slow,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(4.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "drift-fast +10ms/s",
            len: 30 * s,
            bulk_at: bulk_drift_fast,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 3,
            expect_rearms_min: 0,
            late: LateGate::Absolute(4.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "drift-down -2ms/s",
            len: 30 * s,
            bulk_at: bulk_drift_down_slow,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 2,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(12.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "drift-down -6ms/s",
            len: 30 * s,
            bulk_at: bulk_drift_down_fast,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 3,
            expect_rearms_min: 0,
            late: LateGate::ExcursionWithinTail,
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "jump-in-window 100>180",
            len: 30 * s,
            bulk_at: bulk_jump_in,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 12 * s,
            expect_moves_min: 1,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(85.0),
            recovery_budget_s: 6.0,
        },
        TrackCase {
            name: "jump-deep 100>600",
            len: 40 * s,
            bulk_at: bulk_jump_deep,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 12 * s,
            expect_moves_min: 0,
            expect_reacq_min: 1,
            expect_reacq_max: 2,
            expect_rearms_min: 0,
            late: LateGate::Absolute(505.0),
            recovery_budget_s: 15.0,
        },
        TrackCase {
            name: "jump-shallow 600>100",
            len: 40 * s,
            bulk_at: bulk_jump_shallow,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: None,
            event_at: 12 * s,
            expect_moves_min: 0,
            expect_reacq_min: 1,
            expect_reacq_max: 2,
            expect_rearms_min: 0,
            late: LateGate::Absolute(505.0),
            recovery_budget_s: 15.0,
        },
        TrackCase {
            name: "near-only-gap 10-20s",
            len: 30 * s,
            bulk_at: bulk_static_150,
            far_gap: Some((10 * s, 20 * s)),
            talker: Some(((10 * s, 20 * s), 0.4)),
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(0.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "doubletalk-acquire 400",
            len: 30 * s,
            bulk_at: bulk_static_400,
            far_gap: None,
            talker: Some(((0, 30 * s), 0.35)),
            echo_from: 0,
            stall: None,
            event_at: 0,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 0,
            late: LateGate::Absolute(4.0),
            recovery_budget_s: 0.0,
        },
        TrackCase {
            name: "late-echo at 24s",
            len: 40 * s,
            bulk_at: bulk_static_300,
            far_gap: None,
            talker: None,
            echo_from: 24 * s,
            stall: None,
            event_at: 24 * s,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 0,
            expect_rearms_min: 1,
            late: LateGate::Absolute(4.0),
            recovery_budget_s: 10.0,
        },
        TrackCase {
            name: "stall-hold 300",
            len: 30 * s,
            bulk_at: bulk_static_300,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: Some((12 * s, 16384)),
            event_at: 12 * s,
            expect_moves_min: 0,
            expect_reacq_min: 0,
            expect_reacq_max: 1,
            expect_rearms_min: 0,
            late: LateGate::Absolute(4.0),
            recovery_budget_s: 8.0,
        },
        TrackCase {
            name: "stall-jump 300>500",
            len: 40 * s,
            bulk_at: bulk_disc_jump,
            far_gap: None,
            talker: None,
            echo_from: 0,
            stall: Some((12 * s, 16384)),
            event_at: 12 * s,
            expect_moves_min: 0,
            expect_reacq_min: 1,
            expect_reacq_max: 2,
            expect_rearms_min: 0,
            late: LateGate::Absolute(505.0),
            recovery_budget_s: 15.0,
        },
    ];

    let tail_ms = AecConfig::default().tail_ms as usize;
    println!("decibri-aec delay tracking and reacquisition suite");
    println!(
        "  engine: tau at {RATE} Hz, tail {tail_ms} ms, fine window {} ms, ceiling {} ms",
        AecConfig::default().max_echo_delay_ms,
        AecConfig::default().max_search_delay_ms
    );
    println!();
    println!(
        "  {:<24} {:>6} {:>7} {:>5} {:>5} {:>6} {:>8} {:>8} {:>6} {:>7} {:>6} {:>5}  verdict",
        "case",
        "lock s",
        "final",
        "moves",
        "reacq",
        "rearm",
        "maxlate",
        "mean|e|",
        "safe%",
        "recov s",
        "state",
        "xRT",
    );

    let mut failures = 0usize;
    for case in &cases {
        let outcome = run_tracked(case);
        let mut problems: Vec<String> = Vec::new();

        if outcome.final_delay.is_none() {
            problems.push("no final lock".to_string());
        }
        if outcome.moves < case.expect_moves_min {
            problems.push(format!(
                "moves {} < {}",
                outcome.moves, case.expect_moves_min
            ));
        }
        if outcome.reacquisitions < case.expect_reacq_min {
            problems.push(format!(
                "reacq {} < {}",
                outcome.reacquisitions, case.expect_reacq_min
            ));
        }
        if outcome.reacquisitions > case.expect_reacq_max {
            problems.push(format!(
                "reacq {} > {} (false reacquire)",
                outcome.reacquisitions, case.expect_reacq_max
            ));
        }
        if outcome.rearms < case.expect_rearms_min {
            problems.push(format!(
                "rearms {} < {}",
                outcome.rearms, case.expect_rearms_min
            ));
        }
        // The late-safety gate, in whichever quantity this case bounds.
        match case.late {
            LateGate::Absolute(bound) => {
                if outcome.max_late_ms > bound {
                    problems.push(format!("late {:.1} ms > {bound:.1}", outcome.max_late_ms));
                }
            }
            LateGate::ExcursionWithinTail => {
                let bound = tail_ms as f64;
                if outcome.late_excursion_ms > bound {
                    problems.push(format!(
                        "post-lock excursion {:.1} ms > tail {bound:.1}",
                        outcome.late_excursion_ms
                    ));
                }
            }
        }
        if case.recovery_budget_s > 0.0 {
            match outcome.recovery_s {
                Some(r) if r <= case.recovery_budget_s => {}
                Some(r) => problems.push(format!(
                    "recovery {r:.1} s > {:.1} s",
                    case.recovery_budget_s
                )),
                None => problems.push("never recovered".to_string()),
            }
        }
        // The final alignment must be causally safe for the end of the
        // trajectory, on every case, up to the tracker's late dead band.
        if let Some(delay) = outcome.final_delay {
            let bulk = (case.bulk_at)(case.len - 1);
            let onset = bulk + TURN;
            let dead_band = (4 * RATE as usize) / 1000;
            if delay > onset + dead_band {
                problems.push(format!("final {delay} late of onset {onset}"));
            } else if onset.saturating_sub(delay) >= (tail_ms * RATE as usize) / 1000 {
                problems.push(format!("final {delay} beyond a tail early of {onset}"));
            }
        }

        let verdict = if problems.is_empty() {
            "PASS".to_string()
        } else {
            failures += 1;
            format!("FAIL: {}", problems.join("; "))
        };
        println!(
            "  {:<24} {:>6} {:>7} {:>5} {:>5} {:>6} {:>8.1} {:>8.2} {:>6.1} {:>7} {:>6} {:>5.0}  {}",
            case.name,
            outcome
                .locked_at_s
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "-".to_string()),
            outcome
                .final_delay
                .map(|d| format!("{:.0}", d as f64 * 1000.0 / RATE as f64))
                .unwrap_or_else(|| "none".to_string()),
            outcome.moves,
            outcome.reacquisitions,
            outcome.rearms,
            outcome.max_late_ms,
            outcome.mean_abs_err_ms,
            outcome.safe_pct,
            outcome
                .recovery_s
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".to_string()),
            outcome.final_status,
            outcome.realtime,
            verdict,
        );
        if let Some(trigger) = outcome.trigger {
            println!("  {:<24} trigger: {trigger:?}", "");
        }
        // The lateness decomposition.
        println!(
            "  {:<24} init {:+.1} ms  worst-late {:.1} ms  post-lock excursion {:.1} ms",
            "", outcome.initial_late_ms, outcome.max_late_ms, outcome.late_excursion_ms
        );
    }

    println!();
    if failures == 0 {
        println!(
            "delay-probe --track PASS: {}/{} cases",
            cases.len(),
            cases.len()
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "delay-probe --track FAIL: {failures} of {} cases did not meet expectations",
            cases.len()
        );
        ExitCode::FAILURE
    }
}
