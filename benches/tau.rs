//! Criterion benchmark for the shipped Tau canceller, driven through the
//! public [`Aec`] engine the way an integrator drives it.
//!
//! Everything here runs on deterministic synthetic input minted in-code from
//! the same integer-only LCG and sparse echo path the crate's quality suite
//! (tests/quality.rs) pins its floors on. No fixture files, no licensed
//! recordings, no network: the numbers are reproducible on any machine and
//! nothing needs to be committed beside the code. Benchmarking against real
//! clips is a manual run of the `cancel` example over the gitignored `data/`
//! folder, never a committed input.
//!
//! The tracked numbers:
//!
//! - `tau/turn_256`: the steady-state cost of one 256-sample turn (feed the
//!   reference, process the capture), the per-block cost the decibri capture
//!   chain pays. Throughput is reported in elements (samples) per second;
//!   divide by 16000 for the realtime factor.
//! - `tau/stream_4s`: a fresh engine driven over the full four-second
//!   scenario including construction, convergence, and flush, the
//!   whole-stream figure.
//! - `tau/construction`: `Aec::new` alone (ring and transform setup), so a
//!   setup-cost regression is visible separately from the streaming cost.
//!
//! Before the timing runs, the harness prints a deterministic convergence
//! report: the number of 256-sample turns until the trailing-window ERLE
//! first clears each mark. Processing is bit-exact, so these counts are
//! stable across runs and platforms; a change is an algorithm change, not
//! noise.
//!
//! Run with `cargo bench`. Under `cargo test --all-targets` criterion runs
//! each benchmark once as a smoke test and the timing loops are skipped.

#![forbid(unsafe_code)]

use std::hint::black_box;

use criterion::{criterion_group, Criterion, Throughput};
use decibri_aec::{Aec, AecConfig};

/// Sample rate for every scenario, matching the configuration default.
const RATE: u32 = 16_000;

/// The per-turn chunk size: 256 samples is 16 ms at 16 kHz, one Tau block.
const TURN: usize = 256;

/// Scenario length: four seconds at 16 kHz, the quality suite's length, long
/// enough that the stream benchmark spans convergence and steady state.
const SCENARIO_LEN: usize = 64_000;

/// The delay hint that makes the one-turn reference lead read exactly on
/// time at 16 kHz, so the benches measure the canceller, not the estimator.
const ALIGNED_HINT_MS: u16 = 16;

/// Trailing-window length for the convergence report's ERLE, in samples.
const CONVERGENCE_WINDOW: usize = 4_096;

/// The two ERLE marks the convergence report counts turns to.
const CONVERGENCE_MARKS_DB: [f64; 2] = [20.0, 45.0];

/// The default configuration with the scenario alignment supplied as a hint.
fn aligned_config() -> AecConfig {
    let mut config = AecConfig::default();
    config.delay_hint_ms = Some(ALIGNED_HINT_MS);
    config
}

/// One 256-sample turn: feed the reference chunk, process the capture chunk.
fn drive_turn(aec: &mut Aec, far_chunk: &[f32], mic_chunk: &[f32], out: &mut Vec<f32>) {
    aec.feed_reference(far_chunk);
    aec.process(mic_chunk, out)
        .expect("the Tau canceller never fails after construction");
}

/// Steady-state per-turn cost. The engine is converged before timing starts,
/// and each iteration cycles through the scenario's turns so the input is
/// never a degenerate repeated block.
fn bench_turn(c: &mut Criterion) {
    let (far, mic) = synth::echo_pair(SCENARIO_LEN);
    let mut aec = Aec::new(aligned_config()).expect("the default configuration is valid");
    let mut out = Vec::with_capacity(SCENARIO_LEN + TURN);

    // Converge once so the timed turns measure steady state.
    for (far_chunk, mic_chunk) in far.chunks(TURN).zip(mic.chunks(TURN)) {
        drive_turn(&mut aec, far_chunk, mic_chunk, &mut out);
    }
    out.clear();

    let far_chunks: Vec<&[f32]> = far.chunks(TURN).collect();
    let mic_chunks: Vec<&[f32]> = mic.chunks(TURN).collect();
    let mut turn = 0_usize;

    let mut group = c.benchmark_group("tau");
    group.throughput(Throughput::Elements(TURN as u64));
    group.bench_function("turn_256", |b| {
        b.iter(|| {
            drive_turn(&mut aec, far_chunks[turn], mic_chunks[turn], &mut out);
            turn = (turn + 1) % far_chunks.len();
            out.clear();
        });
    });
    group.finish();
}

/// Whole-stream cost: a fresh engine over the full four-second scenario,
/// construction and flush included. Elements per second over 64000 samples;
/// divide by 16000 for the realtime factor.
fn bench_stream(c: &mut Criterion) {
    let (far, mic) = synth::echo_pair(SCENARIO_LEN);

    let mut group = c.benchmark_group("tau");
    group.throughput(Throughput::Elements(SCENARIO_LEN as u64));
    group.bench_function("stream_4s", |b| {
        b.iter(|| {
            let mut aec = Aec::new(aligned_config()).expect("the default configuration is valid");
            let mut out = Vec::with_capacity(SCENARIO_LEN + TURN);
            for (far_chunk, mic_chunk) in far.chunks(TURN).zip(mic.chunks(TURN)) {
                drive_turn(&mut aec, far_chunk, mic_chunk, &mut out);
            }
            aec.flush(&mut out)
                .expect("the Tau canceller never fails after construction");
            black_box(out)
        });
    });
    group.finish();
}

/// Construction alone: ring sizing and transform setup.
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tau");
    group.bench_function("construction", |b| {
        b.iter(|| {
            Aec::new(black_box(aligned_config())).expect("the default configuration is valid")
        });
    });
    group.finish();
}

/// Prints the deterministic convergence report: turns until the ERLE over
/// the trailing [`CONVERGENCE_WINDOW`] samples first clears each mark.
fn print_convergence_report() {
    let (far, mic) = synth::echo_pair(SCENARIO_LEN);
    let mut aec = Aec::new(aligned_config()).expect("the default configuration is valid");
    let mut out = Vec::with_capacity(SCENARIO_LEN + TURN);
    let mut reached: [Option<usize>; CONVERGENCE_MARKS_DB.len()] = [None; 2];

    for (turn, (far_chunk, mic_chunk)) in far.chunks(TURN).zip(mic.chunks(TURN)).enumerate() {
        drive_turn(&mut aec, far_chunk, mic_chunk, &mut out);
        if out.len() < CONVERGENCE_WINDOW {
            continue;
        }
        let start = out.len() - CONVERGENCE_WINDOW;
        let erle = erle_db(&mic[start..out.len()], &out[start..]);
        for (mark, slot) in CONVERGENCE_MARKS_DB.iter().zip(reached.iter_mut()) {
            if slot.is_none() && erle >= *mark {
                *slot = Some(turn + 1);
            }
        }
        if reached.iter().all(Option::is_some) {
            break;
        }
    }

    println!("tau convergence on the deterministic echo-only scenario (bit-exact, not timing):");
    for (mark, slot) in CONVERGENCE_MARKS_DB.iter().zip(reached.iter()) {
        match slot {
            Some(turns) => println!(
                "  {mark:.0} dB trailing-window ERLE after {turns} turns \
                 ({:.0} ms of audio)",
                *turns as f64 * TURN as f64 * 1000.0 / RATE as f64
            ),
            None => println!("  {mark:.0} dB not reached within {SCENARIO_LEN} samples"),
        }
    }
}

/// Echo-return-loss enhancement in decibels, accumulated in `f64` the way
/// the quality suite measures it.
fn erle_db(mic: &[f32], residual: &[f32]) -> f64 {
    let mic_energy: f64 = mic.iter().map(|&s| s as f64 * s as f64).sum();
    let residual_energy: f64 = residual.iter().map(|&s| s as f64 * s as f64).sum();
    if residual_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (mic_energy / residual_energy).log10()
}

/// The deterministic synthesis: the same integer-only LCG and sparse echo
/// path as tests/quality.rs.
mod synth {
    /// A deterministic linear congruential generator, integer state mapped
    /// to `f32`, no platform-dependent transcendentals.
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
    }

    /// The known sparse echo impulse response, taps in samples at 16 kHz.
    const ECHO_IR: [(usize, f32); 4] = [(40, 0.5), (120, -0.25), (280, 0.12), (450, -0.06)];

    /// Echo path gain.
    const ECHO_GAIN: f32 = 0.5;

    /// Amplitude of the deterministic noise floor on the microphone signal.
    const NOISE_FLOOR: f32 = 0.001;

    /// Far-end single-talk pair of `len` samples: a broadband far end, and
    /// a microphone signal holding its echo through the known path plus a
    /// tiny noise floor.
    pub fn echo_pair(len: usize) -> (Vec<f32>, Vec<f32>) {
        let mut far_lcg = Lcg(0x1234_5678);
        let mut floor_lcg = Lcg(0x0F10_0F10);
        let far: Vec<f32> = (0..len).map(|_| far_lcg.next_f32()).collect();
        let mic: Vec<f32> = (0..len)
            .map(|i| {
                let mut echo = 0.0_f32;
                for &(delay, coeff) in &ECHO_IR {
                    if i >= delay {
                        echo += coeff * far[i - delay];
                    }
                }
                ECHO_GAIN * echo + NOISE_FLOOR * floor_lcg.next_f32()
            })
            .collect();
        (far, mic)
    }
}

criterion_group!(benches, bench_turn, bench_stream, bench_construction);

/// Custom entry point instead of `criterion_main!`: the convergence report
/// prints once, then the criterion groups run (or smoke-run in test mode).
fn main() {
    print_convergence_report();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
