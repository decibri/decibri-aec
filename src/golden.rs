//! Golden-pair validation of the internal Rho reference canceller and, through
//! it, of the whole engine pipeline.
//!
//! The fixtures are minted in-repo from a deterministic linear congruential
//! generator and a known sparse echo impulse response, so they are reproducible
//! on every platform with no external tooling: a far-end reference, the echo it
//! produces through the known path, and (for the double-talk pair) a near-end
//! talker. The echo delay is known by construction and supplied to the engine
//! as a delay hint on the pairs that measure cancellation, so those measure the
//! canceller rather than the alignment; the estimator has its own fixture, which
//! withholds the hint and places a known bulk delay for it to find.
//!
//! The suite mirrors the public quality harness in `tests/quality.rs`, and adds
//! the two things that harness cannot reach from outside the crate: the shipped
//! Tau canceller measured against this reference on the same pairs, and the
//! delay estimator driven on a pair with a genuine bulk delay and no hint. The
//! bit-exact golden vector at the bottom is the artifact that makes Rho a
//! reference: it pins Rho's numerical behavior so an accidental change to the
//! algorithm is caught as a bit mismatch, the same way the decibri-resampler
//! pins its golden vectors. Regenerate it via
//! `DECIBRI_REGEN_AEC_GOLDEN=1 cargo test rho_matches_the_bit_exact_golden -- --nocapture`,
//! paste the printed const here, then rerun without the variable to confirm.

use crate::canceller::EchoCanceller;
use crate::config::{AecConfig, Suppression};
use crate::delay::LOCK_MARGIN_SAMPLES;
use crate::engine::Aec;
use crate::rho::RhoCanceller;
use crate::tau::TauCanceller;

/// Sample rate for every golden pair.
const RATE: u32 = 16_000;

/// Filter tail for the scenario pairs: 32 ms is 512 taps at 16 kHz, covering
/// the longest echo-path reflection below with margin while keeping the
/// per-sample reference filter cheap enough for debug-build test runs.
const TAIL_MS: u16 = 32;

/// Scenario length: one second at 16 kHz. NLMS with a 512-tap filter reaches
/// steady state well inside the first half, so the last quarter measures
/// converged behavior.
const SCENARIO_LEN: usize = 16_000;

/// The known within-tail echo impulse response: sparse early reflections, all
/// inside the 512-tap filter span.
const ECHO_IR: [(usize, f32); 4] = [(40, 0.5), (120, -0.25), (280, 0.12), (450, -0.06)];

/// Echo path gain. 0.5 keeps the echo-return loss in the several-dB regime the
/// Geigel detector's 0.5 threshold assumes, so far-end single-talk does not
/// read as double-talk and adaptation actually runs.
const ECHO_GAIN: f32 = 0.5;

/// Amplitude of the deterministic noise floor added to the microphone signal.
const NOISE_FLOOR: f32 = 0.001;

/// Amplitude of the near-end talker burst in the double-talk pair.
const NEAR_AMPLITUDE: f32 = 0.3;

// ---- Deterministic synthesis ------------------------------------------------

/// A deterministic linear congruential generator, identical to the public
/// harness's: integer-only state mapped to `f32`, no platform-dependent
/// transcendentals, so every fixture is bit-identical everywhere.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    /// The next pseudo-random sample in `[-1.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32; // top 24 bits
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
}

/// Convolves the far-end signal with a sparse impulse response at a gain:
/// the known echo path behind every golden pair.
fn convolve_echo(far: &[f32], ir: &[(usize, f32)], gain: f32) -> Vec<f32> {
    (0..far.len())
        .map(|i| {
            let mut echo = 0.0_f32;
            for &(delay, coeff) in ir {
                if i >= delay {
                    echo += coeff * far[i - delay];
                }
            }
            gain * echo
        })
        .collect()
}

/// A golden pair: the far-end reference, the microphone signal containing the
/// known echo, and the clean near-end component (zero outside the burst).
struct GoldenPair {
    far: Vec<f32>,
    mic: Vec<f32>,
    near: Vec<f32>,
}

/// Far-end single-talk: broadband far end, echo through the known path, a tiny
/// noise floor, and no near-end talker. The convergence and ERLE fixture.
fn echo_only_pair() -> GoldenPair {
    let mut far_lcg = Lcg::new(0x5EED_FA4E);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    let far: Vec<f32> = (0..SCENARIO_LEN).map(|_| far_lcg.next_f32()).collect();
    let echo = convolve_echo(&far, &ECHO_IR, ECHO_GAIN);
    let mic: Vec<f32> = echo
        .iter()
        .map(|&e| e + NOISE_FLOOR * floor_lcg.next_f32())
        .collect();
    let near = vec![0.0; SCENARIO_LEN];
    GoldenPair { far, mic, near }
}

/// Double-talk: the same echo path, with a deterministic near-end talker
/// active through the middle third of the stream.
fn double_talk_pair() -> GoldenPair {
    let base = echo_only_pair();
    let burst = SCENARIO_LEN / 3..2 * SCENARIO_LEN / 3;
    let near: Vec<f32> = (0..SCENARIO_LEN)
        .map(|i| {
            if burst.contains(&i) {
                ((i % 41) as f32 / 20.0 - 1.0) * NEAR_AMPLITUDE
            } else {
                0.0
            }
        })
        .collect();
    let mic: Vec<f32> = base.mic.iter().zip(&near).map(|(&m, &n)| m + n).collect();
    GoldenPair {
        far: base.far,
        mic,
        near,
    }
}

// ---- Measurement helpers (mirroring the public harness) ---------------------

/// The total energy of a block of samples, accumulated in `f64`.
fn energy(samples: &[f32]) -> f64 {
    samples.iter().map(|&s| s as f64 * s as f64).sum()
}

/// Echo-return-loss enhancement in decibels of `residual` against `mic`.
fn erle_db(mic: &[f32], residual: &[f32]) -> f64 {
    let residual_energy = energy(residual);
    if residual_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (energy(mic) / residual_energy).log10()
}

/// Scale-invariant signal-to-distortion ratio in decibels of `estimate`
/// against the clean `reference`.
fn si_sdr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    let reference_energy = energy(reference);
    if reference_energy <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let dot: f64 = reference
        .iter()
        .zip(estimate)
        .map(|(&r, &e)| r as f64 * e as f64)
        .sum();
    let scale = dot / reference_energy;

    let mut target_energy = 0.0_f64;
    let mut noise_energy = 0.0_f64;
    for (&r, &e) in reference.iter().zip(estimate) {
        let target = scale * r as f64;
        let noise = e as f64 - target;
        target_energy += target * target;
        noise_energy += noise * noise;
    }
    if noise_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (target_energy / noise_energy).log10()
}

/// Asserts two sample vectors are bit-identical, reporting the first
/// mismatching index. A bit mismatch in a golden run is a determinism leak or
/// an unacknowledged algorithm change; investigate before regenerating.
fn assert_bit_exact(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "golden length changed for {context}: regenerate (see DECIBRI_REGEN_AEC_GOLDEN)"
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "golden mismatch at {i} for {context}: got {g:?}, expected {e:?}. \
             A bit-exact mismatch is a determinism leak (FMA, denormals, reorder) \
             or an unacknowledged algorithm change; investigate as a bug before \
             regenerating."
        );
    }
}

/// Prints a sample vector as a pasteable `const`, the regeneration path for
/// the golden vector below.
fn print_golden(name: &str, data: &[f32]) {
    let mut body = String::new();
    for (i, p) in data.iter().enumerate() {
        if i % 14 == 0 {
            body.push_str("\n    ");
        }
        body.push_str(&format!("{p:?}, "));
    }
    println!("const {name}: &[f32] = &[{body}\n];");
}

/// Drives a whole aligned pair through a fresh Rho instance.
fn run_rho(near: &[f32], far: &[f32], tail_ms: u16) -> (Vec<f32>, RhoCanceller) {
    let mut rho = RhoCanceller::new(RATE, tail_ms);
    let mut out = Vec::new();
    rho.process(near, far, &mut out)
        .expect("rho never fails after construction");
    rho.flush(&mut out).expect("rho flush never fails");
    (out, rho)
}

// ---- Live algorithm tests against the internal reference --------------------

/// Peer of the ignored `tau_reaches_target_erle`: steady-state ERLE on the
/// far-end single-talk golden pair, measured over the converged last quarter.
#[test]
fn rho_reaches_steady_state_erle_on_the_golden_pair() {
    let pair = echo_only_pair();
    let (out, rho) = run_rho(&pair.mic, &pair.far, TAIL_MS);
    let tail_start = SCENARIO_LEN - SCENARIO_LEN / 4;
    let erle = erle_db(&pair.mic[tail_start..], &out[tail_start..]);
    let metrics = rho.metrics();
    println!(
        "rho steady-state ERLE: {erle:.2} dB (metric estimate {:.2} dB)",
        metrics.erle_db
    );
    assert!(
        erle >= 40.0,
        "steady-state ERLE {erle:.2} dB must clear the measured floor of 40 dB"
    );
    assert!(
        metrics.erle_db > 10.0,
        "the smoothed ERLE metric must reflect the convergence, got {:.2} dB",
        metrics.erle_db
    );
    assert_eq!(metrics.divergence_resets, 0);
}

/// Peer of the ignored `tau_preserves_near_end_during_double_talk`: with the
/// filter converged, the near-end burst must survive the canceller, and the
/// Geigel freeze must be visible through the metrics while it is active.
#[test]
fn rho_preserves_near_end_during_double_talk() {
    let pair = double_talk_pair();
    let mut rho = RhoCanceller::new(RATE, TAIL_MS);
    let mut out = Vec::new();
    let mut double_talk_seen = false;
    for (mic_chunk, far_chunk) in pair.mic.chunks(256).zip(pair.far.chunks(256)) {
        rho.process(mic_chunk, far_chunk, &mut out)
            .expect("rho never fails after construction");
        if rho.metrics().double_talk {
            double_talk_seen = true;
        }
    }
    let burst = SCENARIO_LEN / 3..2 * SCENARIO_LEN / 3;
    let sdr = si_sdr_db(&pair.near[burst.clone()], &out[burst]);
    println!("rho near-end SI-SDR through double-talk: {sdr:.2} dB");
    assert!(
        sdr >= 35.0,
        "near-end SI-SDR {sdr:.2} dB must clear the measured floor of 35 dB"
    );
    assert!(
        double_talk_seen,
        "the Geigel freeze must surface through metrics() during the burst"
    );
    assert_eq!(rho.metrics().divergence_resets, 0);
}

/// Peer of the ignored `tau_recovers_erle_after_double_talk`: the frozen
/// filter holds through the burst instead of diverging, so cancellation is
/// immediately back once the near end goes quiet.
#[test]
fn rho_recovers_erle_after_double_talk() {
    let pair = double_talk_pair();
    let (out, rho) = run_rho(&pair.mic, &pair.far, TAIL_MS);
    let tail_start = SCENARIO_LEN - SCENARIO_LEN / 4;
    let erle = erle_db(&pair.mic[tail_start..], &out[tail_start..]);
    println!("rho post-double-talk ERLE: {erle:.2} dB");
    assert!(
        erle >= 40.0,
        "post-double-talk ERLE {erle:.2} dB must clear the measured floor of 40 dB"
    );
    assert_eq!(rho.metrics().divergence_resets, 0);
}

/// Peer of the ignored `tau_output_is_chunk_seam_invariant`: Rho is
/// per-sample, so chunking must not change one bit of output.
#[test]
fn rho_output_is_chunk_seam_invariant() {
    let pair = echo_only_pair();
    let (whole, _) = run_rho(&pair.mic, &pair.far, TAIL_MS);

    let mut rho = RhoCanceller::new(RATE, TAIL_MS);
    let mut chunked = Vec::new();
    for (mic_chunk, far_chunk) in pair.mic.chunks(160).zip(pair.far.chunks(160)) {
        rho.process(mic_chunk, far_chunk, &mut chunked)
            .expect("rho never fails after construction");
    }
    rho.flush(&mut chunked).expect("rho flush never fails");

    assert_bit_exact(&chunked, &whole, "chunk-seam invariance");
}

/// Peer of the ignored `tau_bounds_non_finite_damage`, made exact: Rho's
/// defensive sanitize maps a non-finite input sample to `0.0`, so the damaged
/// run must be bit-identical to a run with zeros at those positions, and every
/// output must be finite.
#[test]
fn rho_bounds_non_finite_damage() {
    let pair = echo_only_pair();

    let mut damaged_mic = pair.mic.clone();
    damaged_mic[100] = f32::NAN;
    damaged_mic[101] = f32::INFINITY;
    damaged_mic[102] = f32::NEG_INFINITY;
    let mut damaged_far = pair.far.clone();
    damaged_far[200] = f32::NAN;

    let mut clean_mic = pair.mic.clone();
    clean_mic[100] = 0.0;
    clean_mic[101] = 0.0;
    clean_mic[102] = 0.0;
    let mut clean_far = pair.far.clone();
    clean_far[200] = 0.0;

    let (damaged_out, rho) = run_rho(&damaged_mic, &damaged_far, TAIL_MS);
    let (clean_out, _) = run_rho(&clean_mic, &clean_far, TAIL_MS);

    assert!(damaged_out.iter().all(|s| s.is_finite()));
    assert_bit_exact(&damaged_out, &clean_out, "non-finite damage bounding");
    assert_eq!(rho.metrics().divergence_resets, 0);
}

/// Two fresh instances over the same pair must produce bit-identical output:
/// the run-to-run half of the determinism guarantee (the golden vector below
/// pins the cross-platform half).
#[test]
fn rho_is_deterministic_across_runs() {
    let pair = echo_only_pair();
    let (first, _) = run_rho(&pair.mic, &pair.far, TAIL_MS);
    let (second, _) = run_rho(&pair.mic, &pair.far, TAIL_MS);
    assert_bit_exact(&second, &first, "run-to-run determinism");
}

/// The engine pipeline (sanitize, ring, absolute-index alignment with a
/// supplied delay hint) must hand Rho exactly the aligned pair the direct
/// drive uses: the outputs are bit-identical and nothing starves. This is Rho
/// acting as the pipeline validator.
#[test]
fn engine_pipeline_matches_the_direct_reference() {
    let pair = echo_only_pair();
    let (direct, _) = run_rho(&pair.mic, &pair.far, TAIL_MS);

    // 256-sample turns with a 16 ms (256-sample) hint: each near-end read
    // lands exactly one turn behind the reference frontier, so engine
    // alignment reproduces the direct pairing sample for sample.
    let config = AecConfig {
        tail_ms: TAIL_MS,
        delay_hint_ms: Some(16),
        ..AecConfig::default()
    };
    let mut aec = Aec::with_internal_reference(config).expect("config is valid");
    let mut out = Vec::new();
    for (far_chunk, mic_chunk) in pair.far.chunks(256).zip(pair.mic.chunks(256)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("the internal reference processes every block");
    }
    aec.flush(&mut out).expect("flush never fails");

    let metrics = aec.metrics();
    assert_eq!(metrics.reference_starved, 0, "no aligned read may starve");
    assert_eq!(metrics.reference_dropped, 0);
    assert_eq!(aec.latency_samples(), 0);
    assert_bit_exact(&out, &direct, "engine pipeline against direct reference");
}

// ---- Tau against the reference on the same golden pairs ---------------------

/// Drives a whole aligned pair through a fresh Tau instance, including the
/// end-of-stream flush, and returns the output alongside the canceller.
fn run_tau(
    near: &[f32],
    far: &[f32],
    tail_ms: u16,
    suppression: Suppression,
) -> (Vec<f32>, TauCanceller) {
    let mut tau = TauCanceller::new(RATE, tail_ms, suppression);
    let mut out = Vec::new();
    tau.process(near, far, &mut out)
        .expect("tau never fails after construction");
    tau.flush(&mut out).expect("tau flush never fails");
    (out, tau)
}

/// Preservation floor for a stream whose microphone holds no echo at all
/// while the reference is genuinely active: the ideal output is the input
/// exactly, and every deviation is invented.
const NO_ECHO_PRESERVATION_FLOOR_DB: f64 = 60.0;

/// A microphone with no echo in it must come through untouched even while
/// the reference is playing.
#[test]
fn tau_leaves_a_no_echo_stream_untouched() {
    // Four seconds: establishment completes within the first half second and
    // the wandering shadow then gets several seconds to try to reach the
    // output.
    let len = 250 * crate::tau::BLOCK;
    let mut far_lcg = Lcg::new(0x00EC_0A11);
    let mut near_lcg = Lcg::new(0x7A1C_E55A);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    // The reference plays throughout; none of it reaches the microphone. The
    // talker is independent noise under a syllabic envelope, not the suite's
    // periodic ramp.
    let far: Vec<f32> = (0..len).map(|_| far_lcg.next_f32()).collect();
    let near: Vec<f32> = (0..len)
        .map(|i| {
            let syllable = ((i / 2048) % 3) as f32 / 2.0 + 0.5;
            NEAR_AMPLITUDE * syllable * near_lcg.next_f32() + NOISE_FLOOR * floor_lcg.next_f32()
        })
        .collect();

    let (out, tau) = run_tau(&near, &far, TAIL_MS, Suppression::Conservative);

    assert_eq!(
        tau.transfers(),
        0,
        "the shadow earned {} transfer blocks on a stream with no echo",
        tau.transfers()
    );
    let mut near_energy = 0.0_f64;
    let mut artifact_energy = 0.0_f64;
    for (&n, &o) in near.iter().zip(&out) {
        near_energy += n as f64 * n as f64;
        let d = o as f64 - n as f64;
        artifact_energy += d * d;
    }
    let preservation = if artifact_energy > 0.0 {
        10.0 * (near_energy / artifact_energy).log10()
    } else {
        f64::INFINITY
    };
    println!(
        "no-echo preservation: {preservation:.2} dB, transfers {}",
        tau.transfers()
    );
    assert!(
        preservation >= NO_ECHO_PRESERVATION_FLOOR_DB,
        "no-echo preservation is {preservation:.2} dB, under the floor of \
         {NO_ECHO_PRESERVATION_FLOOR_DB} dB: the canceller altered a stream that held no echo"
    );
}

/// The partition count must follow the configured tail, not the default: a
/// 32 ms tail at 16 kHz is 512 taps, which is two 256-sample partitions, and the
/// default 200 ms tail is 3200 taps, which is thirteen.
#[test]
fn tau_partitions_follow_the_configured_tail() {
    assert_eq!(
        TauCanceller::new(RATE, 32, Suppression::Off).partitions(),
        2
    );
    assert_eq!(
        TauCanceller::new(RATE, 200, Suppression::Off).partitions(),
        13
    );
    assert_eq!(
        TauCanceller::new(RATE, 16, Suppression::Off).partitions(),
        1
    );
}

/// The linear filter on its own, with suppression bypassed.
#[test]
fn tau_linear_filter_reaches_steady_state_erle() {
    let pair = echo_only_pair();
    let (out, tau) = run_tau(&pair.mic, &pair.far, TAIL_MS, Suppression::Off);
    let tail_start = SCENARIO_LEN - SCENARIO_LEN / 4;
    let erle = erle_db(&pair.mic[tail_start..], &out[tail_start..]);
    let metrics = tau.metrics();
    println!(
        "tau linear-only steady-state ERLE: {erle:.2} dB (metric estimate {:.2} dB)",
        metrics.erle_db
    );
    assert!(
        erle >= 40.0,
        "linear-only steady-state ERLE {erle:.2} dB must clear the measured floor of 40 dB"
    );
    assert!(
        (metrics.erle_db as f64 - erle).abs() < 5.0,
        "the smoothed ERLE metric ({:.2} dB) must track the measured ERLE ({erle:.2} dB)",
        metrics.erle_db
    );
    assert_eq!(metrics.divergence_resets, 0);
}

/// Tau against the time-domain reference on the same pair.
///
/// They are different algorithms, one per-sample in the time domain and one
/// block-partitioned in the frequency domain, so this is a tolerance agreement
/// and never a bit comparison.
#[test]
fn tau_agrees_with_the_reference_on_the_golden_pair() {
    let pair = echo_only_pair();
    let tail_start = SCENARIO_LEN - SCENARIO_LEN / 4;
    let (tau_out, _) = run_tau(&pair.mic, &pair.far, TAIL_MS, Suppression::Off);
    let (rho_out, rho) = run_rho(&pair.mic, &pair.far, TAIL_MS);
    assert_eq!(tau_out.len(), rho_out.len());

    let difference: Vec<f32> = tau_out[tail_start..]
        .iter()
        .zip(&rho_out[tail_start..])
        .map(|(a, b)| a - b)
        .collect();
    let worst = difference.iter().map(|d| d.abs()).fold(0.0_f32, f32::max);
    let below_input = erle_db(&pair.mic[tail_start..], &difference);
    let rho_erle = erle_db(&pair.mic[tail_start..], &rho_out[tail_start..]);
    let tau_erle = erle_db(&pair.mic[tail_start..], &tau_out[tail_start..]);
    println!(
        "tau vs reference: worst |diff| {worst:e}, difference sits {below_input:.2} dB \
         below the microphone (reference ERLE {rho_erle:.2} dB, tau ERLE {tau_erle:.2} dB, \
         reference metric {:.2} dB)",
        rho.metrics().erle_db
    );

    assert!(
        below_input >= 45.0,
        "tau and the reference disagree only {below_input:.2} dB below the microphone, \
         under the measured floor of 45 dB"
    );
    // Neither may be quietly worse than the other.
    assert!(
        (tau_erle - rho_erle).abs() < 6.0,
        "tau ERLE {tau_erle:.2} dB and reference ERLE {rho_erle:.2} dB must agree within 6 dB"
    );
}

/// The suppressed path, and the near-end preservation that gates it.
///
/// The suppressor is only allowed to be the default if it buys real echo
/// rejection without eating the near-end talker, so both halves of that bargain
/// are measured here against the linear-only path and floored. This is also
/// where Tau clears the reference: the frequency-domain filter plus its
/// residual stage reaches an ERLE the per-sample time-domain reference does not.
#[test]
fn tau_suppressor_buys_erle_without_eating_the_near_end() {
    let echo = echo_only_pair();
    let tail_start = SCENARIO_LEN - SCENARIO_LEN / 4;

    let (linear, _) = run_tau(&echo.mic, &echo.far, TAIL_MS, Suppression::Off);
    let (suppressed, _) = run_tau(&echo.mic, &echo.far, TAIL_MS, Suppression::Conservative);
    let (reference, _) = run_rho(&echo.mic, &echo.far, TAIL_MS);
    let linear_erle = erle_db(&echo.mic[tail_start..], &linear[tail_start..]);
    let suppressed_erle = erle_db(&echo.mic[tail_start..], &suppressed[tail_start..]);
    let reference_erle = erle_db(&echo.mic[tail_start..], &reference[tail_start..]);
    println!(
        "tau ERLE: linear-only {linear_erle:.2} dB, conservative {suppressed_erle:.2} dB, \
         reference {reference_erle:.2} dB"
    );

    let talk = double_talk_pair();
    let burst = SCENARIO_LEN / 3..2 * SCENARIO_LEN / 3;
    let (linear_talk, _) = run_tau(&talk.mic, &talk.far, TAIL_MS, Suppression::Off);
    let (suppressed_talk, _) = run_tau(&talk.mic, &talk.far, TAIL_MS, Suppression::Conservative);
    let linear_sdr = si_sdr_db(&talk.near[burst.clone()], &linear_talk[burst.clone()]);
    let suppressed_sdr = si_sdr_db(&talk.near[burst.clone()], &suppressed_talk[burst]);
    let cost = linear_sdr - suppressed_sdr;
    println!(
        "tau near-end SI-SDR through double-talk: linear-only {linear_sdr:.2} dB, \
         conservative {suppressed_sdr:.2} dB (cost {cost:.2} dB)"
    );

    assert!(
        suppressed_erle - linear_erle >= 3.0,
        "the suppressor must buy real echo rejection: {linear_erle:.2} dB to {suppressed_erle:.2} dB"
    );
    // The differential preservation gate.
    assert!(
        cost <= 2.0,
        "the suppressor costs {cost:.2} dB of near-end preservation, over the 2 dB bound"
    );
    // And the absolute floor, not just the differential.
    assert!(
        suppressed_sdr >= 30.0,
        "near-end SI-SDR under the suppressor is {suppressed_sdr:.2} dB, \
         under the measured floor of 30 dB"
    );
    // Tau's own higher bar: the frequency-domain filter with its residual stage
    // clears the time-domain reference, which is the payoff for the partitioned
    // design.
    assert!(
        suppressed_erle >= reference_erle + 5.0,
        "tau at {suppressed_erle:.2} dB must clear the reference at {reference_erle:.2} dB"
    );
}

/// The most the talker's level may move, in either direction, while the
/// talker sits inside the establishment window.
const WARMUP_GAIN_BOUND_DB: f64 = 1.0;

/// Floor on the talker-to-introduced-residue ratio inside the establishment
/// window: what the delivered signal holds beyond the talker at its measured
/// gain and some share of the true echo is content the canceller invented.
const WARMUP_INTRODUCED_FLOOR_DB: f64 = 20.0;

/// The warm-up gate.
#[test]
fn tau_preserves_a_talker_inside_its_own_warmup_window() {
    let len = crate::tau::DTD_ESTABLISH_BLOCKS as usize * crate::tau::BLOCK;
    let onset = len / 3;

    let mut far_lcg = Lcg::new(0x5EED_FA4E);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    let far: Vec<f32> = (0..len).map(|_| far_lcg.next_f32()).collect();
    let echo = convolve_echo(&far, &ECHO_IR, ECHO_GAIN);
    let near: Vec<f32> = (0..len)
        .map(|i| {
            if i >= onset {
                ((i % 41) as f32 / 20.0 - 1.0) * NEAR_AMPLITUDE
            } else {
                0.0
            }
        })
        .collect();
    let mic: Vec<f32> = echo
        .iter()
        .zip(&near)
        .map(|(&e, &n)| e + n + NOISE_FLOOR * floor_lcg.next_f32())
        .collect();

    let (out, tau) = run_tau(&mic, &far, TAIL_MS, Suppression::Conservative);

    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>()
    };
    let out_span = &out[onset..];
    let near_span = &near[onset..];
    let echo_span = &echo[onset..];

    // The talker's delivered level: the output projected onto the true near
    // end. A canceller that captures or shapes the talker moves this away
    // from unity; deliberate passthrough and clean cancellation both hold it.
    let near_energy = dot(near_span, near_span);
    let gain = dot(out_span, near_span) / near_energy;
    let gain_db = 20.0 * gain.abs().log10();
    assert!(
        gain_db.abs() <= WARMUP_GAIN_BOUND_DB,
        "talker gain inside the establishment window is {gain_db:.2} dB, \
         outside the +-{WARMUP_GAIN_BOUND_DB} dB bound"
    );

    // Whatever the output holds beyond the talker at that gain and the
    // best-fit share of the true echo is content the canceller introduced.
    let after_talker: Vec<f64> = out_span
        .iter()
        .zip(near_span)
        .map(|(&o, &n)| o as f64 - gain * n as f64)
        .collect();
    let echo_energy = dot(echo_span, echo_span);
    let echo_share = after_talker
        .iter()
        .zip(echo_span)
        .map(|(&r, &e)| r * e as f64)
        .sum::<f64>()
        / echo_energy;
    let introduced: f64 = after_talker
        .iter()
        .zip(echo_span)
        .map(|(&r, &e)| {
            let residue = r - echo_share * e as f64;
            residue * residue
        })
        .sum();
    let introduced_db = 10.0 * (near_energy / introduced.max(1e-30)).log10();
    println!(
        "warm-up window ({len} samples, onset {onset}) talker gain {gain_db:.2} dB, \
         echo share {echo_share:.2}, talker-to-introduced {introduced_db:.2} dB"
    );
    assert!(
        introduced_db >= WARMUP_INTRODUCED_FLOOR_DB,
        "talker-to-introduced ratio inside the establishment window is {introduced_db:.2} dB, \
         under the measured floor of {WARMUP_INTRODUCED_FLOOR_DB} dB"
    );
    assert_eq!(tau.metrics().divergence_resets, 0);
}

// ---- The automatic delay estimator ------------------------------------------

/// The bulk delay the estimator fixture places between the far-end reference and
/// its echo: 100 ms at 16 kHz, well inside the default 250 ms search window and
/// far outside the 32 ms filter tail, so only a correct estimate can align it.
const BULK_DELAY: usize = 1_600;

/// Length of the estimator fixture: three seconds, long enough that the
/// estimator locks and the filter then converges against the corrected
/// alignment inside the same stream.
const DELAYED_LEN: usize = 48_000;

/// A far-end reference whose echo arrives a known bulk delay later, on top of
/// the same within-tail impulse response the other fixtures use.
fn delayed_echo_pair() -> GoldenPair {
    let mut far_lcg = Lcg::new(0x0DE1_A400);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    let far: Vec<f32> = (0..DELAYED_LEN).map(|_| far_lcg.next_f32()).collect();
    let echo = convolve_echo(&far, &ECHO_IR, ECHO_GAIN);
    let mic: Vec<f32> = (0..DELAYED_LEN)
        .map(|i| {
            let delayed = if i >= BULK_DELAY {
                echo[i - BULK_DELAY]
            } else {
                0.0
            };
            delayed + NOISE_FLOOR * floor_lcg.next_f32()
        })
        .collect();
    GoldenPair {
        far,
        mic,
        near: vec![0.0; DELAYED_LEN],
    }
}

/// The estimator must find the known delay with no hint supplied, and the
/// canceller must then converge against the alignment it produced.
#[test]
fn the_delay_estimator_locks_onto_the_known_delay() {
    let pair = delayed_echo_pair();
    let dominant_tap = ECHO_IR[0].0;
    let peak = BULK_DELAY + dominant_tap + 256;
    let expected = peak - LOCK_MARGIN_SAMPLES;

    let config = AecConfig {
        tail_ms: TAIL_MS,
        // No hint: this is the path a caller who cannot measure their platform
        // latency takes, and the whole point of the estimator.
        delay_hint_ms: None,
        ..AecConfig::default()
    };
    let mut aec = Aec::new(config).expect("config is valid");
    let mut out = Vec::new();
    let mut locked_at = None;
    let mut processed = 0_usize;
    for (far_chunk, mic_chunk) in pair.far.chunks(256).zip(pair.mic.chunks(256)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("the public model processes every block");
        processed += mic_chunk.len();
        if locked_at.is_none() && aec.metrics().delay_samples.is_some() {
            locked_at = Some(processed);
        }
    }
    aec.flush(&mut out).expect("flush never fails");

    let estimate = aec.metrics().delay_samples;
    let locked_at = locked_at.expect("the estimator must lock on a broadband delayed pair");
    println!(
        "delay estimator: locked at {locked_at} samples ({:.0} ms), estimate {estimate:?}, expected {expected}",
        locked_at as f64 * 1000.0 / RATE as f64
    );

    assert_eq!(
        estimate,
        Some(expected),
        "the estimator must find the known delay exactly, less its safety margin"
    );
    assert!(
        estimate.expect("locked") <= peak,
        "the adopted offset must never sit later than the correlation peak"
    );

    assert!(
        locked_at <= RATE as usize,
        "the estimator must lock within a second of audio, took {locked_at} samples"
    );

    let tail_start = DELAYED_LEN - DELAYED_LEN / 4;
    let erle = erle_db(&pair.mic[tail_start..], &out[tail_start..]);
    println!("erle after the estimator locked: {erle:.2} dB");
    assert!(
        erle >= 45.0,
        "ERLE after the lock is {erle:.2} dB, below the measured floor of 45 dB"
    );
}

/// With a hint supplied the estimator does not run at all: the caller's measured
/// value is adopted as-is and reported straight back through the metrics.
#[test]
fn a_supplied_hint_is_adopted_without_estimation() {
    let config = AecConfig {
        tail_ms: TAIL_MS,
        delay_hint_ms: Some(16),
        ..AecConfig::default()
    };
    let aec = Aec::new(config).expect("config is valid");
    assert_eq!(aec.metrics().delay_samples, Some(256));
}

/// With no hint and nothing correlated to find, the estimator must decline to
/// lock rather than commit to a spurious peak: an unlocked estimate leaves the
/// caller where they already were, a wrong one breaks a working canceller.
#[test]
fn the_delay_estimator_declines_an_uncorrelated_pair() {
    let mut far_lcg = Lcg::new(0xAAAA_0001);
    let mut mic_lcg = Lcg::new(0x5555_0002);
    let far: Vec<f32> = (0..DELAYED_LEN).map(|_| far_lcg.next_f32()).collect();
    // An independent near-end signal: no echo of the reference anywhere in it.
    let mic: Vec<f32> = (0..DELAYED_LEN).map(|_| 0.3 * mic_lcg.next_f32()).collect();

    let config = AecConfig {
        tail_ms: TAIL_MS,
        delay_hint_ms: None,
        ..AecConfig::default()
    };
    let mut aec = Aec::new(config).expect("config is valid");
    let mut out = Vec::new();
    for (far_chunk, mic_chunk) in far.chunks(256).zip(mic.chunks(256)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("processing succeeds");
    }
    println!(
        "uncorrelated pair estimate: {:?}",
        aec.metrics().delay_samples
    );
    assert_eq!(
        aec.metrics().delay_samples,
        None,
        "the estimator must decline an uncorrelated pair rather than commit"
    );
}

// ---- The bit-exact golden vector --------------------------------------------

/// Tail for the golden vector: 16 ms is 256 taps at 16 kHz, enough to cover
/// the golden echo path below while keeping the pinned const compact.
const GOLDEN_TAIL_MS: u16 = 16;

/// The golden echo path, inside the 256-tap golden tail.
const GOLDEN_IR: [(usize, f32); 3] = [(10, 0.5), (50, -0.3), (120, 0.1)];

/// The golden pair: broadband far end, echo through [`GOLDEN_IR`] at 0.8 gain,
/// and a continuous near-end talker, so the pinned samples exercise
/// adaptation, the Geigel freeze, and their transitions.
fn golden_vector_pair() -> GoldenPair {
    let mut far_lcg = Lcg::new(0xC0FF_EE00);
    let mut near_lcg = Lcg::new(0xFACE_FEED);
    let far: Vec<f32> = (0..512).map(|_| far_lcg.next_f32()).collect();
    let near: Vec<f32> = (0..512).map(|_| 0.3 * near_lcg.next_f32()).collect();
    let echo = convolve_echo(&far, &GOLDEN_IR, 0.8);
    let mic: Vec<f32> = echo.iter().zip(&near).map(|(&e, &n)| e + n).collect();
    GoldenPair { far, mic, near }
}

/// Peer of the ignored `tau_matches_the_bit_exact_golden`, and the artifact
/// that makes Rho a reference: the committed const pins Rho's output on the
/// golden pair with zero tolerance, so any numerical change to the reference
/// is caught. Regenerate deliberately via `DECIBRI_REGEN_AEC_GOLDEN=1`.
#[test]
fn rho_matches_the_bit_exact_golden() {
    let pair = golden_vector_pair();
    let (out, _) = run_rho(&pair.mic, &pair.far, GOLDEN_TAIL_MS);

    if std::env::var("DECIBRI_REGEN_AEC_GOLDEN").is_ok() {
        print_golden("EXPECTED_RHO_GOLDEN", &out);
        panic!(
            "DECIBRI_REGEN_AEC_GOLDEN is set: copy the printed const into \
             src/golden.rs and rerun without the variable"
        );
    }

    assert_bit_exact(&out, EXPECTED_RHO_GOLDEN, "rho golden pair");
}

/// Rho's pinned output for [`golden_vector_pair`]. Regenerate via
/// `DECIBRI_REGEN_AEC_GOLDEN=1 cargo test rho_matches_the_bit_exact_golden -- --nocapture`,
/// paste the printed const here, then rerun without the variable to confirm.
const EXPECTED_RHO_GOLDEN: &[f32] = &[
    0.22828765,
    0.06530954,
    -0.17608099,
    -0.11770649,
    -0.18323475,
    -0.13725683,
    0.19401309,
    0.19334182,
    -0.019414831,
    -0.12060832,
    -0.077098764,
    0.31137106,
    0.24130507,
    -0.15018176,
    0.0067413747,
    0.08098669,
    -0.28201813,
    -0.28408504,
    0.11480364,
    -0.37741578,
    0.38918793,
    -0.039120585,
    0.25480115,
    0.031081658,
    0.1999982,
    0.45615342,
    0.27669024,
    0.5266491,
    0.32042643,
    -0.0034170747,
    -0.579636,
    -0.5255182,
    -0.18367673,
    -0.17532839,
    0.03732173,
    -0.1520684,
    0.041738927,
    0.05292444,
    0.26206142,
    0.21755622,
    -0.4993707,
    -0.32229888,
    0.12666872,
    0.22882885,
    0.22305566,
    0.3919682,
    0.3691523,
    0.4521572,
    -0.37453336,
    0.12398842,
    0.33268484,
    0.09964232,
    -0.70840573,
    0.073963344,
    0.27447373,
    0.40415052,
    0.50427353,
    0.61709183,
    -0.39963922,
    -0.1789375,
    -0.48803174,
    -0.6009495,
    0.13660774,
    0.2549617,
    -0.01148966,
    -0.6685681,
    0.3397557,
    -0.47833508,
    0.23765886,
    -0.2638618,
    0.08079375,
    0.3448012,
    0.36499086,
    -0.040269684,
    0.07891707,
    0.34853697,
    0.11495787,
    0.1402524,
    -0.36772934,
    0.20304283,
    0.14469416,
    0.19704156,
    -0.44900402,
    -0.34508264,
    0.13525544,
    0.028007165,
    -0.25396413,
    -0.20511419,
    0.3668392,
    -0.51217705,
    0.13411346,
    0.32736322,
    0.61179763,
    0.02978827,
    -0.60224855,
    0.22560148,
    -0.5267095,
    0.322133,
    0.57506526,
    0.22250171,
    -0.51990247,
    0.2922151,
    -0.13688438,
    -0.20550825,
    -0.8292963,
    0.56353474,
    -0.01573928,
    -0.29820895,
    0.35411167,
    -0.2628238,
    -0.08701134,
    -0.3735898,
    -0.019414425,
    0.15426232,
    -0.021316886,
    -0.18521613,
    -0.071505375,
    0.13990214,
    0.15029709,
    -0.6803494,
    -0.488851,
    0.0359535,
    0.45367718,
    -0.25411245,
    0.124745965,
    -0.14485914,
    -0.46972772,
    0.0542223,
    0.1801314,
    0.34213614,
    0.07743001,
    -0.59774673,
    -0.56740355,
    -0.021450121,
    0.022588804,
    -0.084515035,
    0.5003331,
    -0.29813385,
    -0.16500352,
    -0.24100316,
    0.06422792,
    0.43295172,
    0.20017031,
    0.18609184,
    -0.07890161,
    -0.17492485,
    0.1016539,
    0.59702396,
    0.27739587,
    -0.33250463,
    0.04565096,
    -0.032947086,
    0.18151045,
    0.09472013,
    0.14893505,
    0.21531105,
    -0.012622356,
    -0.028344145,
    0.42478907,
    0.6313474,
    0.030566856,
    -0.072638646,
    -0.7671651,
    0.15350954,
    -0.1471438,
    0.14223824,
    0.3370439,
    0.1169847,
    -0.25887057,
    -0.13524742,
    0.10770531,
    0.023042247,
    -0.08285475,
    0.31073734,
    -0.28373912,
    -0.3386232,
    -0.15346013,
    -0.036875874,
    0.086958036,
    -0.38508552,
    -0.36527646,
    -0.36433068,
    -0.5421856,
    0.117941946,
    0.32154965,
    0.09068868,
    0.5427742,
    0.002490662,
    0.009283438,
    -0.10316968,
    -0.40202105,
    -0.09957598,
    -0.61853373,
    0.362472,
    -0.45432675,
    0.053849835,
    0.47988075,
    -0.46782714,
    -0.14874521,
    0.34496516,
    0.60795796,
    -0.29430905,
    0.089597836,
    0.22890423,
    -0.3958976,
    0.59100133,
    -0.489316,
    -0.17499974,
    0.064399496,
    -0.098535,
    -0.16323715,
    -0.14893547,
    0.13122153,
    -0.2271341,
    0.46738786,
    -0.06709147,
    0.579809,
    0.2065188,
    0.049949944,
    0.31047362,
    -0.29471886,
    -0.0793107,
    -0.040700734,
    -0.15184271,
    -0.06630734,
    0.41443193,
    -0.44296566,
    0.10001625,
    -0.00012668967,
    -0.5006292,
    -0.16152588,
    0.34413564,
    0.68284106,
    0.20032105,
    -0.22567186,
    0.17330983,
    -0.3848587,
    0.5086057,
    0.5214288,
    -0.36178535,
    -0.25405234,
    0.3956294,
    -0.11063644,
    -0.12801997,
    -0.36518216,
    -0.094800085,
    0.50889045,
    -0.22461878,
    -0.5047648,
    -0.047589287,
    -0.21150905,
    -0.45966566,
    0.35740852,
    0.35491127,
    -0.4830294,
    -0.12409095,
    -0.020512156,
    -0.0660446,
    -0.31610626,
    0.104853675,
    0.30965877,
    0.052743487,
    0.4278958,
    0.40691414,
    0.46391267,
    -0.24915834,
    0.30176142,
    0.16957131,
    -0.17139459,
    -0.20174204,
    -0.23380834,
    -0.16776341,
    -0.34722728,
    -0.3443746,
    -0.22411802,
    0.35918373,
    -0.31582707,
    -0.34603852,
    -0.30447274,
    0.058741137,
    0.47368318,
    -0.068500504,
    0.7347133,
    0.5862215,
    0.48722467,
    0.09770313,
    -0.45474553,
    0.06838125,
    -0.13069752,
    0.24727395,
    0.053753793,
    -0.29591733,
    0.54374486,
    -0.07393353,
    0.21451318,
    0.08444017,
    -0.23766467,
    -0.3773973,
    0.22461125,
    0.26901534,
    0.0050897896,
    0.022925526,
    0.14793712,
    -0.26407695,
    -0.41997027,
    0.007731244,
    0.24495687,
    -0.2194951,
    0.11218029,
    0.60162103,
    -0.31095165,
    -0.19981164,
    0.60817486,
    0.3618686,
    0.0028746873,
    -0.038457714,
    0.53719866,
    0.17649437,
    0.22453393,
    -0.17311524,
    -0.118498914,
    0.44261748,
    -0.3714317,
    0.10667838,
    0.14212152,
    -0.06256025,
    -0.15109593,
    0.5115217,
    0.21234329,
    0.12499665,
    -0.49574414,
    0.14042865,
    -0.33575776,
    -0.27898228,
    0.018034697,
    -0.27402493,
    0.10224457,
    0.083775654,
    -0.053317726,
    -0.23277877,
    -0.23830295,
    0.11034894,
    -0.33097148,
    0.1602071,
    -0.4441334,
    0.27244633,
    0.1967408,
    -0.10875544,
    -0.14473364,
    -0.13227975,
    0.038029775,
    0.22849654,
    -0.08930158,
    -0.28270447,
    -0.2142395,
    -0.489515,
    0.06552107,
    -0.50790226,
    -0.39509353,
    0.2252044,
    -0.10984249,
    -0.5954031,
    -0.33122975,
    -0.27324384,
    0.31597567,
    -0.21677269,
    -0.23124677,
    0.20368215,
    -0.35540488,
    0.3113496,
    -0.29125625,
    0.23483132,
    -0.02791468,
    0.26240948,
    -0.2231426,
    0.31190884,
    -0.16098964,
    -0.5548317,
    0.09061332,
    0.03745514,
    0.08034399,
    0.40995878,
    0.14363399,
    -0.61841905,
    0.433566,
    -0.33986613,
    -0.44682723,
    0.30074584,
    -0.38267165,
    -0.0898229,
    0.03654656,
    0.34958157,
    0.04935293,
    0.06110803,
    -0.046016157,
    0.8180887,
    0.27886155,
    0.5944164,
    -0.2609567,
    0.27890274,
    0.16317776,
    0.33476356,
    0.24422365,
    0.2487921,
    -0.07489465,
    0.25002924,
    -0.0393762,
    -0.21503806,
    0.1265331,
    -0.7740984,
    0.3626418,
    -0.20695606,
    0.4253009,
    -0.20132512,
    0.18898922,
    -0.38766187,
    -0.15096498,
    -0.25211257,
    0.15880571,
    -0.43971372,
    -0.16121317,
    -0.4932125,
    -0.4560203,
    0.34739235,
    -0.2929073,
    0.1943098,
    0.43114847,
    -0.6452151,
    0.5746881,
    0.45542246,
    -0.5553136,
    0.5536539,
    0.0133535275,
    -0.17207101,
    -0.28792754,
    -0.06488961,
    -0.056334987,
    0.0155834565,
    0.57298124,
    0.20424342,
    -0.12280667,
    0.23174913,
    -0.23648567,
    -0.20001823,
    -0.21239722,
    -0.36217842,
    0.48259252,
    0.17621467,
    0.20290834,
    0.31311834,
    -0.22477126,
    0.086139545,
    -0.17338943,
    -0.34752405,
    -0.10839681,
    0.12188561,
    -0.10251622,
    0.38279492,
    -0.05742778,
    0.6443758,
    -0.0006274432,
    -0.20584619,
    0.42603865,
    0.1443646,
    0.11585261,
    -0.31328762,
    -0.032235697,
    0.7292306,
    -0.3631106,
    -0.48881388,
    0.37497538,
    0.13378216,
    -0.1353902,
    0.22246474,
    0.6788965,
    0.20667017,
    -0.4734509,
    0.08553499,
    -0.16664967,
    0.2221978,
    -0.06502204,
    0.11489524,
    -0.32743028,
    0.18846183,
    0.37957028,
    -0.16208535,
    -0.22010168,
    0.1703568,
    -0.15486014,
    -0.3170847,
    -0.0006484762,
    -0.059295468,
    0.11663507,
    -0.31516552,
    0.32374364,
    0.0035253763,
    0.57590425,
    -0.15032367,
    -0.04681982,
    -0.17478272,
    0.05993472,
    0.25001055,
    0.054717537,
    -0.4980104,
    -0.24120682,
    -0.06116886,
    0.3742594,
    -0.22927846,
    -0.27026206,
    -0.11184198,
    -0.074340746,
    0.26718953,
];
