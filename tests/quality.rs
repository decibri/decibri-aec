//! Quality-bar suite for the echo canceller's public API.
//!
//! Everything here drives the crate the way an integrator does: through the
//! `Aec` engine, with the public model selected by `AecConfig::model`, over
//! fixtures minted in-repo from a deterministic linear congruential generator
//! and a known echo path. No external fixture files, no network, and no
//! platform-dependent transcendentals in the synthesis, so every measurement
//! below is reproducible on any machine.

use std::str::FromStr;

use decibri_aec::{
    Aec, AecConfig, AecError, AecMetrics, AecModel, CancellerMetrics, DelayStatus, EchoCanceller,
    Suppression,
};

// ---- Shared helpers --------------------------------------------------------

/// A deterministic linear congruential generator: integer-only state mapped to
/// `f32`, with no platform-dependent transcendental, so every fixture and every
/// bit-exact golden below is reproducible across platforms.
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

/// Sample rate for every scenario, matching the configuration default.
const RATE: u32 = 16_000;

/// Scenario length: four seconds at 16 kHz.
const SCENARIO_LEN: usize = 64_000;

/// The known echo impulse response: sparse early reflections, all well inside
/// the modelled tail.
const ECHO_IR: [(usize, f32); 4] = [(40, 0.5), (120, -0.25), (280, 0.12), (450, -0.06)];

/// Echo path gain.
const ECHO_GAIN: f32 = 0.5;

/// Amplitude of the deterministic noise floor added to the microphone signal.
const NOISE_FLOOR: f32 = 0.001;

/// Amplitude of the near-end talker burst in the double-talk scenario.
const NEAR_AMPLITUDE: f32 = 0.3;

/// A synthesized echo scenario: the far-end reference, the microphone signal
/// carrying the echo, and the clean near-end component the microphone also
/// carries (zero where no near-end talker is active).
struct Scenario {
    far: Vec<f32>,
    mic: Vec<f32>,
    near: Vec<f32>,
}

/// Convolves the far-end signal with the sparse impulse response at a gain: the
/// known echo path behind every scenario.
fn convolve_echo(far: &[f32]) -> Vec<f32> {
    (0..far.len())
        .map(|i| {
            let mut echo = 0.0_f32;
            for &(delay, coeff) in &ECHO_IR {
                if i >= delay {
                    echo += coeff * far[i - delay];
                }
            }
            ECHO_GAIN * echo
        })
        .collect()
}

/// Far-end single-talk: broadband far end, echo through the known path, a tiny
/// noise floor, and no near-end talker. The convergence and ERLE fixture.
fn echo_only_scenario() -> Scenario {
    let mut far_lcg = Lcg::new(0x1234_5678);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    let far: Vec<f32> = (0..SCENARIO_LEN).map(|_| far_lcg.next_f32()).collect();
    let echo = convolve_echo(&far);
    let mic: Vec<f32> = echo
        .iter()
        .map(|&e| e + NOISE_FLOOR * floor_lcg.next_f32())
        .collect();
    Scenario {
        far,
        mic,
        near: vec![0.0; SCENARIO_LEN],
    }
}

/// Double-talk: the same echo path, with a deterministic near-end talker active
/// through the middle third of the stream, so the stream contains a converged
/// span before the burst, the burst itself, and a recovery span after it.
fn double_talk_scenario() -> Scenario {
    let base = echo_only_scenario();
    let near: Vec<f32> = (0..SCENARIO_LEN)
        .map(|i| {
            if near_burst().contains(&i) {
                ((i % 41) as f32 / 20.0 - 1.0) * NEAR_AMPLITUDE
            } else {
                0.0
            }
        })
        .collect();
    let mic: Vec<f32> = base.mic.iter().zip(&near).map(|(&m, &n)| m + n).collect();
    Scenario {
        far: base.far,
        mic,
        near,
    }
}

/// The span of the double-talk scenario the near-end talker occupies.
fn near_burst() -> std::ops::Range<usize> {
    SCENARIO_LEN / 3..2 * SCENARIO_LEN / 3
}

/// The converged span every steady-state measurement is taken over: the last
/// quarter of the scenario, which is also past the double-talk burst.
fn converged() -> std::ops::Range<usize> {
    SCENARIO_LEN - SCENARIO_LEN / 4..SCENARIO_LEN
}

/// The total energy of a block of samples, accumulated in `f64`.
fn energy(samples: &[f32]) -> f64 {
    samples.iter().map(|&s| s as f64 * s as f64).sum()
}

/// Echo-return-loss enhancement in decibels: how much the residual reduces the
/// microphone energy. `f64::INFINITY` when the residual is exactly silent.
fn erle_db(mic: &[f32], residual: &[f32]) -> f64 {
    let mic_energy = energy(mic);
    let residual_energy = energy(residual);
    if residual_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (mic_energy / residual_energy).log10()
}

/// Scale-invariant signal-to-distortion ratio in decibels of `estimate` against
/// the clean `reference`. Scale-invariant, so a pure gain change is a perfect
/// reconstruction.
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

/// Whether every sample is finite (no `NaN`, no infinities).
fn contains_only_finite(samples: &[f32]) -> bool {
    samples.iter().all(|s| s.is_finite())
}

/// Whether every sample is at or below `eps` in magnitude.
fn is_silent(samples: &[f32], eps: f32) -> bool {
    samples.iter().all(|&s| s.abs() <= eps)
}

/// Asserts two sample vectors are bit-identical, reporting the first mismatching
/// index.
fn assert_bit_exact(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "length changed for {context}: got {}, expected {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "mismatch at {i} for {context}: got {g:?}, expected {e:?}"
        );
    }
}

/// The turn size for [`drive_aligned`]: 256 samples is 16 ms at 16 kHz, so a
/// one-turn reference lead is exactly the [`ALIGNED_HINT_MS`] delay hint.
const CHUNK: usize = 256;

/// The delay hint that makes [`drive_aligned`]'s one-turn reference lead read
/// exactly on time at 16 kHz.
const ALIGNED_HINT_MS: u16 = 16;

/// The default configuration with the known scenario delay supplied as a hint.
fn aligned_config() -> AecConfig {
    let mut config = AecConfig::default();
    config.delay_hint_ms = Some(ALIGNED_HINT_MS);
    config
}

/// Drives the engine over an aligned `(far, mic)` pair the way an integrator
/// would: interleaved [`CHUNK`]-sample turns (feed the reference, then process
/// the capture), so with [`aligned_config`] every far-end read lands on the
/// sample that produced the echo and nothing starves.
fn drive_aligned(aec: &mut Aec, far: &[f32], mic: &[f32]) -> Result<Vec<f32>, AecError> {
    let mut out = Vec::new();
    for (far_chunk, mic_chunk) in far.chunks(CHUNK).zip(mic.chunks(CHUNK)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)?;
    }
    aec.flush(&mut out)?;
    Ok(out)
}

// ---- Engine-level tests ----------------------------------------------------

#[test]
fn construction_accepts_the_default_config() {
    let aec = Aec::new(AecConfig::default()).expect("default config is valid");
    // The default model reports one block of framing latency.
    assert_eq!(aec.latency_samples(), 256);
    assert_eq!(aec.metrics().canceller, CancellerMetrics::default());
}

#[test]
fn construction_rejects_an_out_of_range_sample_rate() {
    let mut config = AecConfig::default();
    config.sample_rate = 96_000;
    assert!(matches!(
        Aec::new(config),
        Err(AecError::SampleRateOutOfRange { requested: 96_000 })
    ));
}

#[test]
fn a_custom_configuration_is_accepted() {
    let mut config = AecConfig::default();
    config.model = AecModel::Tau;
    config.suppression = Suppression::Off;
    config.tail_ms = 120;
    config.max_echo_delay_ms = 400;
    config.delay_hint_ms = Some(80);
    assert!(Aec::new(config).is_ok());
}

/// Selecting the public model yields a working canceller.
#[test]
fn the_public_model_resolves_to_a_working_canceller() {
    let mut aec = Aec::new(aligned_config()).unwrap();
    aec.feed_reference(&[0.1; CHUNK]);
    let mut out = Vec::new();
    aec.process(&[0.2; CHUNK], &mut out)
        .expect("selecting the public model produces a working canceller");
    assert_eq!(out.len(), CHUNK);
    assert!(contains_only_finite(&out));
}

/// The same, reached through the string boundary a binding would use.
#[test]
fn the_model_string_resolves_to_a_working_canceller() {
    let mut config = aligned_config();
    config.model = "tau".parse::<AecModel>().expect("'tau' parses");
    let mut aec = Aec::new(config).unwrap();
    aec.feed_reference(&[0.1; CHUNK]);
    let mut out = Vec::new();
    aec.process(&[0.2; CHUNK], &mut out)
        .expect("the parsed model produces a working canceller");
    assert_eq!(out.len(), CHUNK);
}

#[test]
fn metrics_track_the_reference_transport() {
    let mut aec = Aec::new(AecConfig::default()).unwrap();
    aec.feed_reference(&[0.5; 128]);
    let mut out = Vec::new();
    let _ = aec.process(&[0.5; 128], &mut out);
    let metrics: AecMetrics = aec.metrics();
    // With no delay hint there is no alignment yet.
    assert_eq!(metrics.acquisition_parked, 128);
    assert_eq!(metrics.reference_starved, 0);
    assert_eq!(metrics.reference_dropped, 0);
    // Too little audio for the estimator to have locked, and no hint to adopt.
    assert_eq!(metrics.delay_samples, None);
}

#[test]
fn a_delay_hint_aligns_the_reads_onto_the_fed_reference() {
    let mut config = AecConfig::default();
    config.delay_hint_ms = Some(16); // 256 samples at 16 kHz
    let mut aec = Aec::new(config).unwrap();
    aec.feed_reference(&[0.5; 256]);
    let mut out = Vec::new();
    let _ = aec.process(&[0.5; 256], &mut out);
    assert_eq!(aec.metrics().reference_starved, 0);
    // A supplied hint is the active alignment and is reported straight back.
    assert_eq!(aec.metrics().delay_samples, Some(256));
}

#[test]
fn flush_and_reset_are_safe_with_nothing_processed() {
    let mut aec = Aec::new(AecConfig::default()).unwrap();
    aec.feed_reference(&[0.5; 256]);
    let mut out = vec![7.0];
    aec.flush(&mut out).expect("flush succeeds");
    assert_eq!(out, vec![7.0]);
    aec.reset();
    assert_eq!(aec.metrics().reference_dropped, 0);
    assert_eq!(aec.metrics().reference_starved, 0);
}

#[test]
fn the_engine_and_the_trait_object_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Aec>();
    assert_send::<Box<dyn EchoCanceller>>();
}

// ---- Model-string parse ----------------------------------------------------

#[test]
fn the_public_model_name_parses() {
    assert_eq!(AecModel::from_str("tau").unwrap(), AecModel::Tau);
    assert_eq!(AecModel::Tau.as_str(), "tau");
}

#[test]
fn an_unknown_model_string_reports_the_received_name_and_the_available_set() {
    let err = AecModel::from_str("tao").unwrap_err();
    assert!(matches!(
        &err,
        AecError::UnknownModel { requested } if requested == "tao"
    ));
    assert_eq!(err.to_string(), "model must be one of: 'tau'; got 'tao'");
}

/// The available-model list is the public set only.
#[test]
fn the_available_model_list_is_public_models_only() {
    assert_eq!(AecModel::PUBLIC_MODEL_NAMES, &["tau"]);
    assert!(AecModel::from_str("nope").is_err());
}

// ---- Measurement and guard machinery ---------------------------------------

#[test]
fn erle_db_measures_a_known_reduction() {
    let mic: Vec<f32> = (0..512).map(|i| ((i % 7) as f32) - 3.0).collect();
    let residual: Vec<f32> = mic.iter().map(|&s| s * 0.1).collect();
    let erle = erle_db(&mic, &residual);
    println!("known-reduction ERLE: {erle:.4} dB");
    // A tenfold amplitude reduction is a hundredfold energy reduction: 20 dB.
    assert!((erle - 20.0).abs() < 1e-6, "expected ~20 dB, got {erle:.4}");
}

#[test]
fn si_sdr_db_is_scale_invariant_and_finite_under_noise() {
    let reference: Vec<f32> = (0..512).map(|i| ((i % 5) as f32) - 2.0).collect();
    let scaled: Vec<f32> = reference.iter().map(|&s| s * 2.5).collect();
    assert!(
        si_sdr_db(&reference, &scaled) > 100.0,
        "a pure scaling must read as a near-perfect reconstruction"
    );
    let mut noisy = reference.clone();
    noisy[0] += 0.01;
    let sdr = si_sdr_db(&reference, &noisy);
    println!("noisy SI-SDR: {sdr:.2} dB");
    assert!(sdr.is_finite() && sdr > 30.0, "got {sdr:.2}");
}

#[test]
fn guard_predicates_flag_non_finite_and_silence() {
    assert!(contains_only_finite(&[0.0, 1.0, -0.5]));
    assert!(!contains_only_finite(&[0.0, f32::NAN]));
    assert!(!contains_only_finite(&[f32::INFINITY]));
    assert!(is_silent(&[0.0, 0.0, 0.0], 1e-6));
    assert!(!is_silent(&[0.0, 0.2, 0.0], 1e-6));
}

// ---- Algorithm tests -------------------------------------------------------

/// Steady-state echo rejection floor.
const ERLE_FLOOR_DB: f64 = 45.0;

/// Near-end preservation floor through double-talk.
const NEAR_END_FLOOR_DB: f64 = 20.0;

/// Echo-rejection floor after the double-talk burst ends.
const RECOVERY_FLOOR_DB: f64 = 25.0;

/// Echo rejection the filter must already have reached one second in.
const EARLY_ERLE_FLOOR_DB: f64 = 12.0;

/// Steady-state echo rejection on far-end single-talk, measured over the
/// converged last quarter.
#[test]
fn tau_reaches_target_erle() {
    let scenario = echo_only_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &scenario.far, &scenario.mic)
        .expect("the Tau canceller cancels the echo");
    assert_eq!(out.len(), scenario.mic.len());
    let span = converged();
    let erle = erle_db(&scenario.mic[span.clone()], &out[span]);
    let metrics = aec.metrics();
    println!(
        "steady-state ERLE: {erle:.2} dB (metric estimate {:.2} dB)",
        metrics.canceller.erle_db
    );
    assert!(
        erle >= ERLE_FLOOR_DB,
        "steady-state ERLE {erle:.2} dB must clear the measured floor of {ERLE_FLOOR_DB} dB"
    );
    assert_eq!(metrics.canceller.divergence_resets, 0);
    assert_eq!(metrics.reference_starved, 0);
}

/// Echo rejection one second in, before the filter has finished converging.
#[test]
fn tau_converges_usefully_within_one_second() {
    let scenario = echo_only_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &scenario.far, &scenario.mic)
        .expect("the Tau canceller cancels the echo");
    // The last quarter of the first second: past the initial transient, well
    // short of convergence.
    let span = 3 * RATE as usize / 4..RATE as usize;
    let erle = erle_db(&scenario.mic[span.clone()], &out[span]);
    println!("ERLE one second in: {erle:.2} dB");
    assert!(
        erle >= EARLY_ERLE_FLOOR_DB,
        "ERLE one second in is {erle:.2} dB, under the measured floor of          {EARLY_ERLE_FLOOR_DB} dB"
    );
}

/// The near-end talker must survive the canceller and its residual suppressor,
/// and the double-talk freeze must be visible through the metrics while the
/// talker is active.
#[test]
fn tau_preserves_near_end_during_double_talk() {
    let scenario = double_talk_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let mut out = Vec::new();
    let mut double_talk_seen = false;
    for (far_chunk, mic_chunk) in scenario.far.chunks(CHUNK).zip(scenario.mic.chunks(CHUNK)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("the Tau canceller cancels the echo");
        if aec.metrics().canceller.double_talk {
            double_talk_seen = true;
        }
    }
    aec.flush(&mut out).expect("flush never fails");

    let burst = near_burst();
    let sdr = si_sdr_db(&scenario.near[burst.clone()], &out[burst]);
    println!("near-end SI-SDR through double-talk: {sdr:.2} dB");
    assert!(
        sdr >= NEAR_END_FLOOR_DB,
        "near-end SI-SDR {sdr:.2} dB must clear the measured floor of {NEAR_END_FLOOR_DB} dB"
    );
    assert!(
        double_talk_seen,
        "the double-talk freeze must surface through metrics() during the burst"
    );
    assert_eq!(aec.metrics().canceller.divergence_resets, 0);
}

/// Cancellation returns after the double-talk burst ends.
#[test]
fn tau_recovers_erle_after_double_talk() {
    let scenario = double_talk_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &scenario.far, &scenario.mic)
        .expect("the Tau canceller cancels the echo");
    let span = converged();
    let erle = erle_db(&scenario.mic[span.clone()], &out[span]);
    println!("post-double-talk ERLE: {erle:.2} dB");
    assert!(
        erle >= RECOVERY_FLOOR_DB,
        "post-double-talk ERLE {erle:.2} dB must recover past the measured floor \
         of {RECOVERY_FLOOR_DB} dB"
    );
    assert_eq!(aec.metrics().canceller.divergence_resets, 0);
}

// ---- The realistic scenario ------------------------------------------------

/// The realistic scenario's length: eight seconds at 16 kHz.
const REAL_LEN: usize = 8 * RATE as usize;

/// The near-end talker's span: the middle of the scenario, so there is a clean
/// span before it to converge on and a clean span after it to score echo removal
/// over.
const REAL_TALK: std::ops::Range<usize> = 3 * RATE as usize..6 * RATE as usize;

/// The span the realistic scenario's echo removal is scored over: after the
/// talker stops, so the microphone holds echo alone and the number is a true
/// ERLE.
const REAL_ECHO_ONLY: std::ops::Range<usize> = 6 * RATE as usize..REAL_LEN;

/// How far the microphone sits ABOVE the loopback, in decibels. Positive is the
/// point: this is echo-return gain.
const REAL_ECHO_RETURN_GAIN_DB: f64 = 12.0;

/// The loopback's level, well below full scale as a real one is.
const REAL_FAR_RMS: f64 = 0.010;

/// The near-end talker's level.
const REAL_NEAR_RMS: f64 = 0.05;

/// The microphone's noise floor.
const REAL_NOISE_RMS: f64 = 0.0003;

/// The diffuse echo path's length in taps, 50 ms at 16 kHz.
const REAL_PATH_TAPS: usize = 800;

/// Where the diffuse path begins, before which it is silent.
const REAL_PATH_ONSET: usize = 64;

/// Samples the path takes to rise from its onset to its peak.
const REAL_PATH_RISE: usize = 12;

/// Per-tap decay of the diffuse tail after the direct arrival, applied by
/// repeated multiplication so no `exp` is involved and the taps are reproducible
/// to the bit.
const REAL_PATH_DECAY: f64 = 0.9975;

/// The reverberant tail's level against the direct arrival.
const REAL_PATH_TAIL_GAIN: f64 = 0.45;

/// Two-pole all-pole colouring, so the reference is not white.
fn colour(x: &[f32], a1: f64, a2: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let mut p1 = 0.0_f64;
    let mut p2 = 0.0_f64;
    for &sample in x {
        let value = sample as f64 + a1 * p1 + a2 * p2;
        p2 = p1;
        p1 = value;
        out.push(value);
    }
    out
}

/// A piecewise-linear syllabic envelope: `on` samples of each `period` are
/// voiced, with linear `ramp` edges. Straight lines only, so there is no
/// transcendental and nothing to drift across platforms.
fn syllables(len: usize, period: usize, on: usize, ramp: usize, offset: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let phase = (i + offset) % period;
            if phase < ramp {
                phase as f64 / ramp as f64
            } else if phase < on - ramp {
                1.0
            } else if phase < on {
                (on - phase) as f64 / ramp as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// The root mean square of a signal, in `f64`.
fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&s| s * s).sum::<f64>() / x.len() as f64).sqrt()
}

/// Scales a signal to a target root mean square.
fn scale_to_rms(x: &[f64], target: f64) -> Vec<f32> {
    let current = rms(x);
    if current <= 0.0 {
        return x.iter().map(|&s| s as f32).collect();
    }
    let gain = target / current;
    x.iter().map(|&s| (s * gain) as f32).collect()
}

/// A diffuse echo path whose onset precedes its peak: a short linear rise into
/// the direct arrival, then a decaying reverberant tail of pseudo-random taps.
fn diffuse_path(seed: u64) -> Vec<f64> {
    let mut lcg = Lcg::new(seed);
    let mut path = vec![0.0_f64; REAL_PATH_TAPS];
    for j in 0..=REAL_PATH_RISE {
        path[REAL_PATH_ONSET + j] = (j + 1) as f64 / (REAL_PATH_RISE + 1) as f64;
    }
    let mut gain = 1.0_f64;
    for tap in path
        .iter_mut()
        .take(REAL_PATH_TAPS)
        .skip(REAL_PATH_ONSET + REAL_PATH_RISE + 1)
    {
        *tap = REAL_PATH_TAIL_GAIN * gain * lcg.next_f32() as f64;
        gain *= REAL_PATH_DECAY;
    }
    path
}

/// The realistic scenario: echo-return gain, a diffuse path, a quiet coloured
/// and gated reference, and a near-end talker over the middle span. `near` holds
/// the talker alone.
fn realistic_scenario() -> Scenario {
    let mut far_lcg = Lcg::new(0x5EED_FA12);
    let mut near_lcg = Lcg::new(0x5EED_4EA5);
    let mut noise_lcg = Lcg::new(0x5EED_0F10);

    let far_source: Vec<f32> = (0..REAL_LEN).map(|_| far_lcg.next_f32()).collect();
    let far_envelope = syllables(REAL_LEN, 4800, 2880, 320, 0);
    let far_shaped: Vec<f64> = colour(&far_source, 1.6, -0.72)
        .iter()
        .zip(&far_envelope)
        .map(|(&s, &e)| s * e)
        .collect();
    let far = scale_to_rms(&far_shaped, REAL_FAR_RMS);

    let path = diffuse_path(0x5EED_9A78);
    let echo_raw: Vec<f64> = (0..REAL_LEN)
        .map(|i| {
            let mut sum = 0.0_f64;
            let first = i.saturating_sub(REAL_PATH_TAPS - 1);
            for (tap, &coeff) in path.iter().enumerate() {
                if i >= tap && i - tap >= first {
                    sum += coeff * far[i - tap] as f64;
                }
            }
            sum
        })
        .collect();
    let echo = scale_to_rms(
        &echo_raw,
        REAL_FAR_RMS * 10.0_f64.powf(REAL_ECHO_RETURN_GAIN_DB / 20.0),
    );

    let near_source: Vec<f32> = (0..REAL_LEN).map(|_| near_lcg.next_f32()).collect();
    let near_envelope = syllables(REAL_LEN, 5600, 3600, 280, 1700);
    let near_shaped: Vec<f64> = colour(&near_source, 1.4, -0.6)
        .iter()
        .zip(&near_envelope)
        .map(|(&s, &e)| s * e)
        .collect();
    let near_full = scale_to_rms(&near_shaped, REAL_NEAR_RMS);
    let near: Vec<f32> = (0..REAL_LEN)
        .map(|i| {
            if REAL_TALK.contains(&i) {
                near_full[i]
            } else {
                0.0
            }
        })
        .collect();

    let mic: Vec<f32> = (0..REAL_LEN)
        .map(|i| echo[i] + near[i] + (REAL_NOISE_RMS as f32) * noise_lcg.next_f32())
        .collect();

    Scenario { far, mic, near }
}

/// Drives the engine over the realistic scenario the way the bench harness and
/// an integrator do: no delay hint, so the estimator supplies the alignment and
/// the alignment margin is exercised, in interleaved reference-then-capture
/// turns.
fn drive_estimated(aec: &mut Aec, far: &[f32], mic: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for (far_chunk, mic_chunk) in far.chunks(CHUNK).zip(mic.chunks(CHUNK)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("processing succeeds");
    }
    aec.flush(&mut out).expect("flush never fails");
    out
}

/// Echo rejection floor on the realistic scenario, over the span after the
/// talker stops.
const REAL_ERLE_FLOOR_DB: f64 = 15.0;

/// Near-end preservation floor on the realistic scenario, through the talker's
/// span.
const REAL_NEAR_FLOOR_DB: f64 = 15.0;

/// The realistic scenario's echo must be cancelled, under echo-return gain, a
/// quiet coloured reference and a diffuse path.
#[test]
fn tau_cancels_under_echo_return_gain() {
    let scenario = realistic_scenario();
    let mut aec = Aec::new(AecConfig::default()).unwrap();
    let out = drive_estimated(&mut aec, &scenario.far, &scenario.mic);

    let span = REAL_ECHO_ONLY;
    let compared = span.end.min(out.len());
    let erle = erle_db(
        &scenario.mic[span.start..compared],
        &out[span.start..compared],
    );
    let metrics = aec.metrics();
    println!(
        "realistic scenario ERLE: {erle:.2} dB (delay {:?}, divergence resets {})",
        metrics.delay_samples, metrics.canceller.divergence_resets
    );
    assert!(
        erle >= REAL_ERLE_FLOOR_DB,
        "ERLE under echo-return gain is {erle:.2} dB, under the measured floor \
         of {REAL_ERLE_FLOOR_DB} dB"
    );
}

/// And the talker must survive it.
#[test]
fn tau_preserves_the_talker_under_echo_return_gain() {
    let scenario = realistic_scenario();
    let mut aec = Aec::new(AecConfig::default()).unwrap();
    let out = drive_estimated(&mut aec, &scenario.far, &scenario.mic);

    let span = REAL_TALK;
    let compared = span.end.min(out.len());
    let sdr = si_sdr_db(
        &scenario.near[span.start..compared],
        &out[span.start..compared],
    );
    println!("realistic scenario near-end SI-SDR through double-talk: {sdr:.2} dB");
    assert!(
        sdr >= REAL_NEAR_FLOOR_DB,
        "near-end SI-SDR under echo-return gain is {sdr:.2} dB, under the \
         measured floor of {REAL_NEAR_FLOOR_DB} dB"
    );
    assert_eq!(aec.metrics().canceller.divergence_resets, 0);
}

/// Non-finite input must not poison the filter or the output.
#[test]
fn tau_bounds_non_finite_damage() {
    let scenario = echo_only_scenario();

    let mut damaged_mic = scenario.mic.clone();
    damaged_mic[100] = f32::NAN;
    damaged_mic[101] = f32::INFINITY;
    damaged_mic[102] = f32::NEG_INFINITY;
    let mut damaged_far = scenario.far.clone();
    damaged_far[200] = f32::NAN;

    let mut clean_mic = scenario.mic.clone();
    clean_mic[100] = 0.0;
    clean_mic[101] = 0.0;
    clean_mic[102] = 0.0;
    let mut clean_far = scenario.far.clone();
    clean_far[200] = 0.0;

    let mut damaged_aec = Aec::new(aligned_config()).unwrap();
    let damaged = drive_aligned(&mut damaged_aec, &damaged_far, &damaged_mic)
        .expect("the Tau canceller cancels the echo");
    let mut clean_aec = Aec::new(aligned_config()).unwrap();
    let clean = drive_aligned(&mut clean_aec, &clean_far, &clean_mic)
        .expect("the Tau canceller cancels the echo");

    assert!(
        contains_only_finite(&damaged),
        "a non-finite input must not produce a non-finite output"
    );
    assert_bit_exact(&damaged, &clean, "non-finite damage bounding");
    assert_eq!(damaged_aec.metrics().canceller.divergence_resets, 0);
}

/// Silence in, exactly silence out.
#[test]
fn tau_silence_in_yields_silence_out() {
    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &[0.0; 2048], &[0.0; 2048])
        .expect("the Tau canceller cancels the echo");
    assert_eq!(out.len(), 2048);
    assert!(
        is_silent(&out, 0.0),
        "silence in must yield exact silence out"
    );
}

/// A never-active far end is a bit-exact passthrough.
#[test]
fn tau_is_exact_passthrough_when_far_is_never_active() {
    let scenario = double_talk_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let silent_far = vec![0.0; scenario.near.len()];
    let out = drive_aligned(&mut aec, &silent_far, &scenario.near)
        .expect("the Tau canceller passes silence-referenced audio through");
    assert_bit_exact(&out, &scenario.near, "silent-far-end passthrough");
    let sdr = si_sdr_db(&scenario.near, &out);
    println!("passthrough SI-SDR: {sdr} dB (bit-exact)");
}

/// The canceller's internal block framing must be invisible to the caller's
/// chunking.
#[test]
fn tau_output_is_chunk_seam_invariant() {
    let scenario = echo_only_scenario();

    let mut whole = Aec::new(aligned_config()).unwrap();
    let whole_out = drive_aligned(&mut whole, &scenario.far, &scenario.mic)
        .expect("the Tau canceller cancels the echo");

    let mut chunked = Aec::new(aligned_config()).unwrap();
    let mut chunked_out = Vec::new();
    for (far_chunk, mic_chunk) in scenario.far.chunks(CHUNK).zip(scenario.mic.chunks(CHUNK)) {
        chunked.feed_reference(far_chunk);
        // The same turn's capture, handed in 64 samples at a time.
        for piece in mic_chunk.chunks(64) {
            chunked
                .process(piece, &mut chunked_out)
                .expect("the Tau canceller cancels the echo");
        }
    }
    chunked
        .flush(&mut chunked_out)
        .expect("the Tau canceller drains its carry");

    assert_bit_exact(&chunked_out, &whole_out, "chunk-seam invariance");
}

/// Two fresh engines over the same scenario must produce bit-identical output:
/// the run-to-run half of the determinism guarantee. The committed golden vector
/// below pins the cross-platform, cross-toolchain half.
#[test]
fn tau_is_deterministic_across_runs() {
    let scenario = echo_only_scenario();
    let mut first_aec = Aec::new(aligned_config()).unwrap();
    let first = drive_aligned(&mut first_aec, &scenario.far, &scenario.mic).unwrap();
    let mut second_aec = Aec::new(aligned_config()).unwrap();
    let second = drive_aligned(&mut second_aec, &scenario.far, &scenario.mic).unwrap();
    assert_bit_exact(&second, &first, "run-to-run determinism");
}

// ---- The measurement-rig gates ---------------------------------------------

/// Length of the warm-up gate's fixture: no longer than the detector's
/// establishment window.
const WARMUP_LEN: usize = 32 * CHUNK;

/// Where the warm-up fixture's talker starts: inside the establishment window.
const WARMUP_ONSET: usize = 10 * CHUNK;

/// The echo path and levels are the standard fixture's; only the geometry
/// differs, and the geometry is the test.
fn warmup_scenario() -> Scenario {
    let mut far_lcg = Lcg::new(0x1234_5678);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    let far: Vec<f32> = (0..WARMUP_LEN).map(|_| far_lcg.next_f32()).collect();
    let echo = convolve_echo(&far);
    let near: Vec<f32> = (0..WARMUP_LEN)
        .map(|i| {
            if i >= WARMUP_ONSET {
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
    Scenario { far, mic, near }
}

/// The most the talker's level may move, in either direction, while the
/// talker starts inside the warm-up window.
const WARMUP_GAIN_BOUND_DB: f64 = 1.0;

/// Floor on the talker-to-introduced-residue ratio inside the warm-up
/// window.
const WARMUP_INTRODUCED_FLOOR_DB: f64 = 20.0;

/// Preservation floor for the no-echo gate below.
const NO_ECHO_PRESERVATION_FLOOR_DB: f64 = 60.0;

/// A microphone that holds no echo must come through untouched even while
/// the reference is genuinely playing: the ideal output is the input
/// exactly, and every deviation is invented.
#[test]
fn tau_leaves_a_no_echo_stream_untouched() {
    let len = 250 * CHUNK;
    let mut far_lcg = Lcg::new(0x00EC_0A11);
    let mut near_lcg = Lcg::new(0x7A1C_E55A);
    let mut floor_lcg = Lcg::new(0x0F10_0F10);
    // The talker is independent noise under a syllabic envelope rather than
    // the suite's periodic ramp.
    let far: Vec<f32> = (0..len).map(|_| far_lcg.next_f32()).collect();
    let mic: Vec<f32> = (0..len)
        .map(|i| {
            let syllable = ((i / 2048) % 3) as f32 / 2.0 + 0.5;
            NEAR_AMPLITUDE * syllable * near_lcg.next_f32() + NOISE_FLOOR * floor_lcg.next_f32()
        })
        .collect();

    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &far, &mic).expect("the engine processes the pair");

    let mut mic_energy = 0.0_f64;
    let mut artifact_energy = 0.0_f64;
    for (&m, &o) in mic.iter().zip(&out) {
        mic_energy += m as f64 * m as f64;
        let d = o as f64 - m as f64;
        artifact_energy += d * d;
    }
    let preservation = if artifact_energy > 0.0 {
        10.0 * (mic_energy / artifact_energy).log10()
    } else {
        f64::INFINITY
    };
    println!("no-echo preservation through the engine: {preservation:.2} dB");
    assert!(
        preservation >= NO_ECHO_PRESERVATION_FLOOR_DB,
        "no-echo preservation is {preservation:.2} dB, under the floor of \
         {NO_ECHO_PRESERVATION_FLOOR_DB} dB: the canceller altered a stream that held no echo"
    );
    assert_eq!(aec.metrics().canceller.divergence_resets, 0);
}

/// A talker who starts speaking before the detector's baseline is established
/// must come through undamaged.
#[test]
fn tau_preserves_a_talker_inside_the_warmup_window() {
    let scenario = warmup_scenario();
    let echo = convolve_echo(&scenario.far);
    let mut aec = Aec::new(aligned_config()).unwrap();
    let out = drive_aligned(&mut aec, &scenario.far, &scenario.mic)
        .expect("the Tau canceller cancels the echo");

    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>()
    };
    let out_span = &out[WARMUP_ONSET..];
    let near_span = &scenario.near[WARMUP_ONSET..WARMUP_ONSET + out_span.len()];
    let echo_span = &echo[WARMUP_ONSET..WARMUP_ONSET + out_span.len()];

    let near_energy = dot(near_span, near_span);
    let gain = dot(out_span, near_span) / near_energy;
    let gain_db = 20.0 * gain.abs().log10();
    assert!(
        gain_db.abs() <= WARMUP_GAIN_BOUND_DB,
        "talker gain inside the warm-up window is {gain_db:.2} dB, \
         outside the +-{WARMUP_GAIN_BOUND_DB} dB bound"
    );

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
        "warm-up window talker gain {gain_db:.2} dB, echo share {echo_share:.2}, \
         talker-to-introduced {introduced_db:.2} dB"
    );
    assert!(
        introduced_db >= WARMUP_INTRODUCED_FLOOR_DB,
        "talker-to-introduced ratio inside the warm-up window is {introduced_db:.2} dB, \
         under the measured floor of {WARMUP_INTRODUCED_FLOOR_DB} dB"
    );
    assert_eq!(aec.metrics().canceller.divergence_resets, 0);
}

/// The nonstationary fixture's length: eight seconds at 16 kHz, room for the
/// filter to converge as far as this path lets it and a long scored span after.
const NONSTATIONARY_LEN: usize = 8 * RATE as usize;

/// Strength of the deterministic cubic distortion in the echo path, applied to
/// a unit-RMS signal as `x * (1 - a * x^2)`.
const NONSTATIONARY_DISTORTION: f64 = 0.08;

/// Period, in samples, of the triangular gain wobble on the echo path: about
/// 1.3 s, deliberately coprime with the syllable period so the two modulations
/// never settle into a repeating pattern.
const NONSTATIONARY_WOBBLE_PERIOD: usize = 21_000;

/// Depth of that wobble: the path gain traverses `1.0 +- this` and back each
/// period, a slow physical drift like a talker shifting in a chair.
const NONSTATIONARY_WOBBLE_DEPTH: f64 = 0.08;

/// Far-end-only audio through a nonstationary, mildly nonlinear echo path: a
/// coloured, syllabically gated reference, a diffuse room response, cubic
/// loudspeaker distortion, and a slow gain drift. No near-end talker exists
/// anywhere in this scenario, so every double-talk verdict on it is false by
/// construction.
fn nonstationary_far_only_scenario() -> Scenario {
    let mut far_lcg = Lcg::new(0x0DD5_EED5);
    let mut noise_lcg = Lcg::new(0x0DD5_0F10);

    let far_source: Vec<f32> = (0..NONSTATIONARY_LEN).map(|_| far_lcg.next_f32()).collect();
    let far_envelope = syllables(NONSTATIONARY_LEN, 4800, 2880, 320, 0);
    let far_shaped: Vec<f64> = colour(&far_source, 1.6, -0.72)
        .iter()
        .zip(&far_envelope)
        .map(|(&s, &e)| s * e)
        .collect();
    let far = scale_to_rms(&far_shaped, REAL_FAR_RMS);

    // The loudspeaker's contribution: unit-RMS drive through a cubic
    // softening.
    let drive = scale_to_rms(&far_shaped, 1.0);
    let distorted: Vec<f64> = drive
        .iter()
        .map(|&s| {
            let x = s as f64;
            x * (1.0 - NONSTATIONARY_DISTORTION * x * x)
        })
        .collect();

    let path = diffuse_path(0x0DD5_9A78);
    let echo_raw: Vec<f64> = (0..NONSTATIONARY_LEN)
        .map(|i| {
            let mut sum = 0.0_f64;
            for (tap, &coeff) in path.iter().enumerate() {
                if i >= tap {
                    sum += coeff * distorted[i - tap];
                }
            }
            sum
        })
        .collect();

    // The slow path drift: a piecewise-linear triangular gain, no
    // transcendentals, applied before the level is set so the drift never
    // changes the scenario's overall loudness.
    let wobbled: Vec<f64> = echo_raw
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let phase = i % NONSTATIONARY_WOBBLE_PERIOD;
            let half = NONSTATIONARY_WOBBLE_PERIOD / 2;
            let ramp = if phase < half {
                phase as f64 / half as f64
            } else {
                (NONSTATIONARY_WOBBLE_PERIOD - phase) as f64 / half as f64
            };
            s * (1.0 - NONSTATIONARY_WOBBLE_DEPTH + 2.0 * NONSTATIONARY_WOBBLE_DEPTH * ramp)
        })
        .collect();
    let echo = scale_to_rms(
        &wobbled,
        REAL_FAR_RMS * 10.0_f64.powf(REAL_ECHO_RETURN_GAIN_DB / 20.0),
    );

    let mic: Vec<f32> = (0..NONSTATIONARY_LEN)
        .map(|i| echo[i] + (REAL_NOISE_RMS as f32) * noise_lcg.next_f32())
        .collect();

    Scenario {
        far,
        mic,
        near: vec![0.0; NONSTATIONARY_LEN],
    }
}

/// Turns at the head of the nonstationary run left out of the freeze count:
/// one second.
const FREEZE_GRACE_TURNS: usize = RATE as usize / CHUNK;

/// The double-talk flag rate, in percent, over far-active turns of an aligned
/// run, after the grace span.
fn far_active_freeze_rate_pct(aec: &mut Aec, far: &[f32], mic: &[f32]) -> f64 {
    let mut out = Vec::new();
    let mut frozen = Vec::new();
    for (far_chunk, mic_chunk) in far.chunks(CHUNK).zip(mic.chunks(CHUNK)) {
        aec.feed_reference(far_chunk);
        aec.process(mic_chunk, &mut out)
            .expect("processing succeeds");
        frozen.push(aec.metrics().canceller.double_talk);
    }
    let rms: Vec<f64> = far
        .chunks(CHUNK)
        .map(|chunk| {
            let energy: f64 = chunk.iter().map(|&s| s as f64 * s as f64).sum();
            (energy / chunk.len() as f64).sqrt()
        })
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let loud = sorted[(sorted.len() - 1).min(sorted.len() * 95 / 100)];
    let floor = loud * 0.1;
    let mut active = 0_u64;
    let mut active_frozen = 0_u64;
    for turn in FREEZE_GRACE_TURNS..frozen.len().min(rms.len()) {
        if rms[turn] > floor {
            active += 1;
            if frozen[turn] {
                active_frozen += 1;
            }
        }
    }
    assert!(active > 0, "the fixture must actually drive the far end");
    active_frozen as f64 * 100.0 / active as f64
}

/// The false-freeze ceiling on echo-only nonstationary audio. There is no
/// talker anywhere in the fixture, so every flagged turn is a false verdict.
const FALSE_FREEZE_CEILING_PCT: f64 = 15.0;

/// Echo-only nonstationary audio (path drift, loudspeaker distortion, syllabic
/// gating) contains no near-end talker, so the double-talk detector must not
/// flag it.
#[test]
fn tau_does_not_freeze_on_echo_only_nonstationarity() {
    let scenario = nonstationary_far_only_scenario();
    let mut aec = Aec::new(aligned_config()).unwrap();
    let rate = far_active_freeze_rate_pct(&mut aec, &scenario.far, &scenario.mic);
    println!("nonstationary echo-only false-freeze rate: {rate:.1}% of far-active turns");
    assert!(
        rate <= FALSE_FREEZE_CEILING_PCT,
        "the detector froze {rate:.1}% of far-active turns on audio with no \
         near-end talker, over the {FALSE_FREEZE_CEILING_PCT}% ceiling"
    );
}

// ---- Reference transport: chunk-size independence --------------------------

/// Clip length for the transport suite: ten seconds at 16 kHz.
const TRANSPORT_LEN: usize = 160_000;

/// Bulk delay for the transport suite: 100 ms.
const TRANSPORT_DELAY: usize = 1_600;

/// The near-end block the transport suite processes, matching the cadence the
/// examples and the benchmark use.
const TRANSPORT_TURN: usize = 256;

/// Reference chunk sizes a caller might plausibly feed against a 256-sample
/// capture block.
const TRANSPORT_CHUNKS: [usize; 5] = [128, 160, 256, 512, 1024];

/// A far-end single-talk pair whose echo path begins exactly `bulk` samples
/// after the far-end sample that caused it, under a syllabic envelope.
fn transport_pair(len: usize, bulk: usize) -> (Vec<f32>, Vec<f32>) {
    let mut carrier = Lcg::new(0x7A11_0001);
    let mut shape = Lcg::new(0x7A11_0002);
    let mut noise = Lcg::new(0x7A11_0003);

    let mut far = Vec::with_capacity(len);
    let mut level = 0.0_f32;
    let mut remaining = 0_usize;
    while far.len() < len {
        if remaining == 0 {
            remaining = 400 + ((shape.next_f32() + 1.0) * 0.5 * 2400.0) as usize;
            level = if (shape.next_f32() + 1.0) * 0.5 < 0.28 {
                0.0
            } else {
                0.15 + 0.85 * ((shape.next_f32() + 1.0) * 0.5)
            };
        }
        far.push(level * carrier.next_f32());
        remaining -= 1;
    }

    let mic = (0..len)
        .map(|i| {
            let mut echo = 0.0_f32;
            for &(tap, coeff) in &ECHO_IR {
                let lag = bulk + tap;
                if i >= lag {
                    echo += coeff * far[i - lag];
                }
            }
            ECHO_GAIN * echo + NOISE_FLOOR * noise.next_f32()
        })
        .collect();
    (far, mic)
}

/// What one transport run observed about the acquisition's evidence.
struct TransportRun {
    metrics: AecMetrics,
    /// `fine_scans` at the moment the ring first overwrote a sample, and at the
    /// end of the clip.
    scans_at_overflow: u64,
    scans_at_end: u64,
}

/// Drives a whole clip with the reference fed in `far_chunk`-sample pieces and
/// the capture processed in [`TRANSPORT_TURN`]-sample blocks, keeping the
/// reference frontier just ahead of the block about to be processed. This is
/// the render/capture callback pattern: a device whose render buffer size is
/// not its capture buffer size, feeding and reading at the same rate.
fn drive_with_reference_chunk(far: &[f32], mic: &[f32], far_chunk: usize) -> TransportRun {
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    let mut aec = Aec::new(config).expect("default config is valid");

    let mut out = Vec::new();
    let mut fed = 0usize;
    let mut near = 0usize;
    let mut scans_at_overflow = None;
    while near + TRANSPORT_TURN <= mic.len() {
        while fed < near + TRANSPORT_TURN && fed < far.len() {
            let end = (fed + far_chunk).min(far.len());
            aec.feed_reference(&far[fed..end]);
            fed = end;
        }
        aec.process(&mic[near..near + TRANSPORT_TURN], &mut out)
            .expect("processing succeeds");
        near += TRANSPORT_TURN;
        if scans_at_overflow.is_none() && aec.metrics().reference_dropped > 0 {
            scans_at_overflow = Some(aec.metrics().delay.fine_scans);
        }
    }
    aec.flush(&mut out).expect("flush never fails");

    let metrics = aec.metrics();
    TransportRun {
        scans_at_overflow: scans_at_overflow.expect("the clip must outrun the ring"),
        scans_at_end: metrics.delay.fine_scans,
        metrics,
    }
}

/// The size of the caller's reference chunk must not change what the delay
/// acquisition sees.
#[test]
fn the_reference_chunk_size_does_not_change_the_acquisition() {
    let (far, mic) = transport_pair(TRANSPORT_LEN, TRANSPORT_DELAY);

    let runs: Vec<(usize, TransportRun)> = TRANSPORT_CHUNKS
        .iter()
        .map(|&chunk| (chunk, drive_with_reference_chunk(&far, &mic, chunk)))
        .collect();

    for (chunk, run) in &runs {
        assert!(
            run.metrics.reference_dropped > 0,
            "chunk {chunk}: the clip must outrun the ring for this to test anything"
        );
        // Feeding and reading at the same rate is not falling behind, whatever
        // chunk sizes the two sides use.
        assert_eq!(
            run.metrics.reference_reanchors, 0,
            "chunk {chunk}: a caller that keeps pace must never be re-anchored"
        );
        assert!(
            matches!(run.metrics.delay.status, DelayStatus::Locked(_)),
            "chunk {chunk}: the acquisition must lock, got {:?}",
            run.metrics.delay.status
        );
        // Evidence must keep accumulating past the point the ring first overflows.
        assert!(
            run.scans_at_end > run.scans_at_overflow,
            "chunk {chunk}: the fine estimator ran {} scans before the ring \
             first overflowed and {} by the end of the clip; it must keep \
             scanning after the ring fills",
            run.scans_at_overflow,
            run.scans_at_end,
        );
    }

    // The invariant: same clip, same evidence, whatever the caller's chunk size.
    let (first_chunk, first) = &runs[0];
    for (chunk, run) in &runs[1..] {
        assert_eq!(
            run.scans_at_end, first.scans_at_end,
            "chunk {chunk} ran {} fine scans where chunk {first_chunk} ran {}",
            run.scans_at_end, first.scans_at_end,
        );
        assert_eq!(
            run.metrics.delay.tracking_moves, first.metrics.delay.tracking_moves,
            "chunk {chunk} disagreed with chunk {first_chunk} on tracking moves"
        );
        assert_eq!(
            run.metrics.delay.reacquisitions, first.metrics.delay.reacquisitions,
            "chunk {chunk} disagreed with chunk {first_chunk} on reacquisitions"
        );
        assert_eq!(
            run.metrics.delay.last_reacquire_trigger, first.metrics.delay.last_reacquire_trigger,
            "chunk {chunk} disagreed with chunk {first_chunk} on the reacquisition trigger"
        );
    }
}

/// A consumer that genuinely stalls is still re-anchored, and told so.
#[test]
fn a_stalled_consumer_is_re_anchored_and_reported() {
    let (far, mic) = transport_pair(TRANSPORT_LEN, TRANSPORT_DELAY);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.delay_hint_ms = Some(100);
    let mut aec = Aec::new(config).expect("default config is valid");

    let mut out = Vec::new();
    // Establish the alignment and let the baseline settle.
    const SETTLE_BLOCKS: usize = 64;
    let mut cursor = 0usize;
    for _ in 0..SETTLE_BLOCKS {
        aec.feed_reference(&far[cursor..cursor + TRANSPORT_TURN]);
        aec.process(&mic[cursor..cursor + TRANSPORT_TURN], &mut out)
            .expect("processing succeeds");
        cursor += TRANSPORT_TURN;
    }
    assert_eq!(aec.metrics().reference_reanchors, 0);

    // The renderer runs a whole second ahead while the capture side loses
    // everything in between.
    let stall_end = cursor + RATE as usize;
    aec.feed_reference(&far[cursor..stall_end]);
    cursor = stall_end;

    // The stall becomes visible over the capture blocks that follow it.
    const INFER_BLOCKS: usize = 8;
    for _ in 0..INFER_BLOCKS {
        aec.feed_reference(&far[cursor..cursor + TRANSPORT_TURN]);
        aec.process(&mic[cursor..cursor + TRANSPORT_TURN], &mut out)
            .expect("processing succeeds after a stall");
        cursor += TRANSPORT_TURN;
    }

    assert_eq!(
        aec.metrics().reference_reanchors,
        1,
        "a capture stall must be inferred and reported"
    );

    // The re-anchor is not a failure to process: the stream continues from the
    // reference frontier, which is what the counter is telling the caller.
    assert!(out.iter().all(|s| s.is_finite()));

    // Back in step, the rebuilt alignment holds: one stall, one re-anchor.
    for _ in 0..32 {
        aec.feed_reference(&far[cursor..cursor + TRANSPORT_TURN]);
        aec.process(&mic[cursor..cursor + TRANSPORT_TURN], &mut out)
            .expect("processing succeeds");
        cursor += TRANSPORT_TURN;
    }
    assert_eq!(aec.metrics().reference_reanchors, 1);
}

// ---- The bit-exact golden vector -------------------------------------------

/// Tail for the golden vector: 32 ms is 512 taps at 16 kHz, two partitions,
/// enough to cover the golden echo path while keeping the pinned const compact.
const GOLDEN_TAIL_MS: u16 = 32;

/// The golden pair: a broadband far end, its echo through the known path, and a
/// continuous near-end talker, so the pinned samples exercise adaptation, the
/// double-talk freeze, the residual suppressor, and their transitions.
fn golden_pair() -> (Vec<f32>, Vec<f32>) {
    let mut far_lcg = Lcg::new(0xC0FF_EE00);
    let mut near_lcg = Lcg::new(0xFACE_FEED);
    let far: Vec<f32> = (0..1024).map(|_| far_lcg.next_f32()).collect();
    let near: Vec<f32> = (0..1024).map(|_| 0.3 * near_lcg.next_f32()).collect();
    let echo = convolve_echo(&far);
    let mic: Vec<f32> = echo.iter().zip(&near).map(|(&e, &n)| e + n).collect();
    (far, mic)
}

/// Prints a sample vector as a pasteable `const`, the regeneration path for the
/// golden vector below.
fn print_golden(name: &str, data: &[f32]) {
    let mut body = String::new();
    for (i, sample) in data.iter().enumerate() {
        if i % 8 == 0 {
            body.push_str("\n    ");
        }
        body.push_str(&format!("{sample:?}, "));
    }
    println!("const {name}: &[f32] = &[{body}\n];");
}

/// The committed bit-exact golden: the public engine's output on a fixed
/// deterministic pair, pinned to the bit. This is the artifact that catches an
/// accidental numerical change (a reordered accumulation, a fused multiply-add,
/// a tuning constant nudged without acknowledgement) as a bit mismatch rather
/// than a silent drift in a decibel figure. Regenerate deliberately via
/// `DECIBRI_REGEN_AEC_TAU_GOLDEN=1 cargo test tau_matches_the_bit_exact_golden -- --nocapture`,
/// paste the printed const here, then rerun without the variable to confirm.
#[test]
fn tau_matches_the_bit_exact_golden() {
    let (far, mic) = golden_pair();
    let mut config = aligned_config();
    config.tail_ms = GOLDEN_TAIL_MS;
    let mut aec = Aec::new(config).unwrap();
    let out = drive_aligned(&mut aec, &far, &mic).expect("the Tau canceller cancels the echo");

    if std::env::var("DECIBRI_REGEN_AEC_TAU_GOLDEN").is_ok() {
        print_golden("EXPECTED_TAU_GOLDEN", &out);
        panic!(
            "DECIBRI_REGEN_AEC_TAU_GOLDEN is set: copy the printed const into \
             tests/quality.rs and rerun without the variable"
        );
    }

    assert_bit_exact(&out, EXPECTED_TAU_GOLDEN, "tau golden pair");
}

/// The public engine's pinned output for [`golden_pair`]. Regenerate via
/// `DECIBRI_REGEN_AEC_TAU_GOLDEN=1 cargo test tau_matches_the_bit_exact_golden -- --nocapture`,
/// paste the printed const here, then rerun without the variable to confirm.
const EXPECTED_TAU_GOLDEN: &[f32] = &[
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
    -0.050817028,
    0.2973713,
    -0.13401376,
    -0.1813659,
    0.15558863,
    0.17320962,
    -0.13976204,
    0.08095461,
    -0.10457225,
    -0.0023405314,
    0.15431318,
    -0.28130013,
    -0.034372225,
    0.0635969,
    0.054479316,
    0.18446639,
    0.1080072,
    0.14447409,
    0.2916075,
    -0.27042204,
    -0.22773083,
    -0.2682312,
    -0.16170245,
    0.1256115,
    0.17837203,
    -0.21444549,
    0.17397112,
    -0.12965345,
    0.08653275,
    0.21315588,
    -0.19238701,
    -0.24965039,
    0.025297612,
    0.06486009,
    -0.16943872,
    0.06609051,
    0.13113227,
    -0.059173673,
    0.030833766,
    -0.5011085,
    0.13831913,
    0.24117194,
    -0.09633666,
    -0.28708446,
    -0.043000087,
    0.16128334,
    0.30494583,
    0.40991104,
    0.023231417,
    0.14521562,
    -0.44967842,
    -0.23563981,
    0.06831646,
    0.07666114,
    -0.36492816,
    -0.13073115,
    0.019981958,
    0.116493024,
    0.40171972,
    0.096491784,
    0.06314796,
    0.12530667,
    0.17469177,
    0.10775438,
    0.058104128,
    0.29384542,
    0.3230951,
    0.15659842,
    -0.33960938,
    0.21813825,
    0.3181358,
    -0.08136453,
    -0.16983011,
    0.17139287,
    0.19825491,
    0.46242,
    -0.07561527,
    0.27697146,
    -0.058229275,
    -0.36395,
    0.21235886,
    -0.26194197,
    0.24238306,
    -0.15170428,
    0.20710573,
    -0.16738924,
    0.1993281,
    0.027110562,
    0.2693123,
    -0.28463614,
    -0.49468094,
    0.20714808,
    -0.037544966,
    -0.3045867,
    -0.2220513,
    0.19644536,
    -0.05963635,
    -0.07530087,
    0.24264064,
    -0.12949565,
    -0.04527754,
    0.05655759,
    0.13455644,
    0.06056772,
    0.45316365,
    0.0410614,
    -0.111837804,
    0.13577364,
    0.11363201,
    -0.31110293,
    -0.25373644,
    0.19142362,
    -0.0072586983,
    0.2141864,
    -0.027488485,
    0.30491367,
    -0.43288592,
    0.0750985,
    -0.031897742,
    0.3913876,
    -0.57178676,
    -0.43859673,
    0.07644219,
    -0.09387119,
    -0.4445333,
    0.1465056,
    0.15041634,
    -0.065464556,
    0.10018152,
    -0.4087875,
    -0.091298446,
    -0.020452507,
    -0.006620329,
    -0.06396927,
    -0.27246103,
    -0.008644819,
    0.29728195,
    0.043465763,
    -0.077873915,
    -0.35979575,
    -0.26111066,
    0.27034768,
    0.0732552,
    -0.1015559,
    -0.101928964,
    -0.2626914,
    0.12734784,
    0.018731192,
    0.47183067,
    0.06122291,
    0.29191643,
    -0.35130933,
    -0.34312963,
    0.19942957,
    0.10553667,
    0.16292048,
    0.20291288,
    -0.40466312,
    -0.004700033,
    0.123333305,
    0.06256232,
    0.1472028,
    -0.070049435,
    0.27753687,
    -0.23748326,
    -0.0434442,
    -0.258642,
    0.3196811,
    0.0015005618,
    -0.30702877,
    -0.025323227,
    -0.31528172,
    -0.04978472,
    -0.11795737,
    0.19330065,
    0.21702954,
    0.12578729,
    -0.00012496859,
    0.11872857,
    -0.15318225,
    -0.4459945,
    -0.038048252,
    -0.41388732,
    0.17834702,
    -0.466529,
    -0.20638041,
    0.0008697044,
    0.05985725,
    -0.24693581,
    0.31562123,
    0.4199006,
    -0.25748003,
    -0.33542693,
    0.300861,
    -0.27808863,
    -0.21831778,
    0.0528318,
    -0.31315494,
    -0.16352336,
    -0.22921032,
    -0.15622261,
    -0.2493467,
    -0.26482934,
    0.0059125572,
    0.2289308,
    -0.37341595,
    0.35418516,
    0.025961086,
    0.038615838,
    0.12022096,
    0.0468968,
    -0.29506052,
    -0.32564658,
    0.22106172,
    -0.2541277,
    0.20990506,
    -0.13520291,
    0.10481578,
    0.047088616,
    0.16807652,
    0.18580629,
    0.027539097,
    0.020678684,
    0.20249636,
    -0.53276694,
    0.5609093,
    -0.27216196,
    -0.014521323,
    0.3349021,
    -0.08647223,
    0.058464363,
    0.3013692,
    0.32941076,
    -0.1548687,
    -0.098493144,
    0.23985606,
    0.30522233,
    0.4206869,
    -0.0034195036,
    0.2952728,
    -0.09477513,
    -0.20611419,
    0.1393911,
    0.069775015,
    0.18239896,
    0.16379851,
    -0.056261525,
    -0.4100337,
    -0.24821714,
    0.14181355,
    0.12158547,
    0.18104422,
    0.3954634,
    0.3935154,
    0.14927422,
    0.14853536,
    0.0432515,
    -0.024093702,
    0.29748064,
    -0.25209183,
    -0.32738662,
    0.11856358,
    -0.15746021,
    0.15662585,
    -0.12432388,
    0.0367788,
    0.018543378,
    -0.34495398,
    -0.24244054,
    -0.34253323,
    0.10898885,
    0.064602986,
    0.24855897,
    0.28834838,
    -0.104270816,
    -0.28379104,
    -0.08208233,
    -0.075057566,
    0.13460955,
    -0.09639993,
    -0.2970478,
    0.022105634,
    0.46896654,
    -0.3227142,
    0.39747447,
    0.13476445,
    0.39419246,
    0.11711534,
    -0.15501244,
    -0.14589538,
    0.110160336,
    0.073061116,
    0.25311008,
    -0.5103938,
    -0.16402988,
    -0.061070457,
    -0.2535534,
    -0.016817022,
    -0.23873591,
    0.06535625,
    -0.31404498,
    -0.2573835,
    0.5343537,
    0.13185039,
    0.117843136,
    -0.035116084,
    0.19746655,
    0.27479362,
    -0.4005918,
    0.14620532,
    0.056302182,
    0.018024296,
    0.40130627,
    -0.022816435,
    -0.11636919,
    0.07871368,
    -0.27367476,
    0.12311125,
    -0.26257232,
    0.09070284,
    -0.08281997,
    0.07484782,
    0.18499377,
    -0.004438106,
    -0.23110023,
    -0.23112321,
    0.07306431,
    -0.19392036,
    -0.16872056,
    0.19844691,
    -0.28992382,
    0.00523825,
    0.029999182,
    0.32464087,
    -0.045847535,
    0.3406252,
    0.42020485,
    -0.25290707,
    -0.19289792,
    -0.23863646,
    0.3260848,
    0.20611444,
    -0.012764536,
    0.25432158,
    0.26239145,
    0.042543214,
    -0.055066332,
    0.030533835,
    0.15247296,
    0.34294462,
    -0.070341095,
    -0.12754002,
    -0.28654265,
    -0.27807468,
    0.26266587,
    -0.021202028,
    -0.21860678,
    0.25341696,
    0.003421232,
    0.02793312,
    -0.19206692,
    0.10592854,
    -0.15481085,
    -0.09193091,
    -0.48992473,
    -0.010020256,
    -0.16811238,
    -0.3366385,
    0.38399866,
    0.05332634,
    -0.5072185,
    0.16738994,
    -0.11622296,
    -0.30034217,
    0.09410947,
    -0.38377398,
    -0.02165687,
    0.09809983,
    -0.053986996,
    0.11777805,
    -0.34045258,
    0.1087392,
    0.025569364,
    0.08669142,
    -0.12870634,
    -0.016244292,
    -0.053397745,
    0.35935766,
    -0.1609534,
    0.3567858,
    -0.4780805,
    0.17697468,
    -0.35360944,
    0.36377853,
    0.049157076,
    0.52122504,
    -0.08983792,
    -0.40085065,
    -0.2009462,
    -0.14255409,
    0.22904587,
    0.0042575896,
    0.37334615,
    -0.2521446,
    0.2811427,
    -0.5031328,
    -0.17684038,
    0.0029394254,
    -0.14158472,
    -0.2580443,
    -0.18066135,
    -0.18672791,
    -0.3496793,
    -0.02626811,
    0.18077974,
    -0.12112008,
    0.29004037,
    -0.051492184,
    -0.19595863,
    0.40392995,
    0.028760359,
    0.24118757,
    0.10590445,
    -0.20647082,
    -0.028449668,
    0.17880931,
    -0.004712209,
    0.09572095,
    0.3413346,
    0.023832783,
    0.23407231,
    0.16031162,
    0.09474471,
    0.080503404,
    -0.11860763,
    0.15948693,
    -0.040164977,
    -0.22983126,
    0.062543035,
    -0.33341384,
    -0.060549974,
    -0.3289677,
    -0.45945656,
    -0.03391514,
    0.096719414,
    0.24685188,
    0.105307445,
    -0.3822216,
    0.06719752,
    0.30870396,
    0.046558514,
    -0.02392003,
    -0.28017032,
    -0.3203047,
    -0.3341822,
    -0.07769182,
    0.295384,
    0.24636102,
    -0.027418181,
    -0.11138491,
    0.3657241,
    0.47421777,
    -0.17234051,
    0.090347305,
    -0.037642203,
    -0.12685725,
    0.0076149404,
    0.0810366,
    0.020393506,
    0.02140563,
    0.06857382,
    0.16020755,
    -0.13570483,
    -0.3557843,
    0.15880191,
    -0.38935846,
    -0.19261068,
    0.41791302,
    0.27580768,
    0.015740518,
    -0.06534086,
    -0.1078497,
    -0.0064964145,
    0.0817233,
    0.04471076,
    -0.16515556,
    -0.05296111,
    0.30711883,
    0.37383547,
    0.033325583,
    -0.25280148,
    0.06521352,
    0.20410867,
    -0.15487368,
    0.033744,
    -0.04868696,
    -0.035232544,
    -0.19725452,
    0.21161573,
    0.05920423,
    0.31403613,
    0.35376543,
    -0.35144252,
    0.23504484,
    0.045223333,
    0.028205559,
    -0.06892912,
    -0.037543938,
    0.038403347,
    -0.26004148,
    -0.21550253,
    -0.11475621,
    0.3345465,
    0.13033806,
    -0.1144564,
    -0.098570436,
    0.484603,
    -0.09829936,
    0.13393666,
    -0.36897454,
    0.0121098785,
    0.10639157,
    0.49585903,
    -0.06458846,
    -0.18789522,
    -0.30919728,
    0.07733684,
    0.45947766,
    -0.022689566,
    -0.0029543936,
    -0.29970056,
    -0.026091367,
    0.098752744,
    -0.05639237,
    0.204457,
    -0.29723155,
    0.17096005,
    -0.22447751,
    -0.12396854,
    0.22570014,
    0.26174414,
    0.20553334,
    0.32035753,
    0.27770644,
    0.3192807,
    0.19149855,
    0.20341045,
    0.015975997,
    -0.10108429,
    0.07188188,
    -0.50765944,
    -0.16035463,
    0.19983599,
    0.36729607,
    0.35505003,
    -0.08968092,
    0.52616733,
    -0.10535307,
    -0.15185158,
    0.3252439,
    -0.26272404,
    -0.2449174,
    -0.043446466,
    -0.44126272,
    0.033593297,
    0.052183226,
    -0.3517002,
    -0.2527191,
    -0.010672718,
    -0.1307874,
    -0.27201676,
    0.050790016,
    -0.19745633,
    0.07262272,
    -0.1257823,
    -0.16816384,
    0.13026491,
    0.041079223,
    0.029911548,
    0.06589639,
    -0.38955748,
    -0.020089636,
    0.269401,
    0.06773522,
    -0.19121836,
    0.18161511,
    -0.33943045,
    0.024612606,
    0.12662198,
    0.10525586,
    -0.06529279,
    -0.54541034,
    -0.26657167,
    0.41105843,
    -0.038133606,
    -0.31179836,
    0.29666427,
    0.24332704,
    -0.25406823,
    -0.08269121,
    -0.067843825,
    0.011690155,
    -0.031586826,
    0.105194956,
    -0.12097056,
    0.18790668,
    0.35910606,
    -0.06761988,
    -0.0229925,
    0.08700225,
    0.06296536,
    -0.40021807,
    0.06286265,
    -0.028358668,
    -0.14226617,
    -0.05863978,
    -0.10003039,
    -0.15735523,
    -0.08650157,
    -0.02130881,
    -0.15166813,
    -0.35115308,
    0.2706523,
    -0.06857175,
    -0.11730187,
    0.127831,
    -0.48172566,
    0.16378249,
    0.41509593,
    0.18310353,
    0.14901909,
    0.18133065,
    -0.533811,
    -0.06992557,
    0.2702997,
    0.14308712,
    -0.029816031,
    -0.3644849,
    0.09936442,
    -0.29972023,
    -0.038111478,
    0.13511503,
    -0.10077059,
    0.20903619,
    0.20269479,
    -0.44322982,
    0.06649687,
    -0.4155044,
    -0.1593338,
    0.08341874,
    0.0684129,
    0.17502113,
    -0.08307102,
    -0.30814236,
    0.07962914,
    0.02730804,
    -0.11313658,
    0.1846244,
    0.23789135,
    0.2226251,
    -0.20169698,
    -0.41068125,
    0.04917285,
    0.08291912,
    0.3181569,
    -0.44678143,
    0.26081344,
    -0.028190233,
    0.03016705,
    -0.46561882,
    -0.49253213,
    -0.09884636,
    -0.067585684,
    -0.35215944,
    0.19606078,
    0.10867969,
    0.1532177,
    -0.20746237,
    0.009903189,
    -0.2542492,
    -0.12676598,
    0.16947907,
    -0.005485043,
    -0.46144372,
    -0.13317873,
    0.11540687,
    -0.23452678,
    -0.10967161,
    -0.19466078,
    -0.05952534,
    -0.23800473,
    -0.17476556,
    0.06718506,
    0.22519961,
    -0.28083083,
    0.12948649,
    -0.34423694,
    0.3176502,
    0.021542013,
    0.4216491,
    0.23381329,
    0.1066563,
    0.30141947,
    -0.08137062,
    0.387289,
    -0.07675469,
    -0.038548924,
    0.43199742,
    0.15907192,
    0.072303034,
    0.07105138,
    -0.11480619,
    0.2232571,
    -0.15858959,
    -0.14930166,
    -0.5009968,
    0.11432876,
    0.33704558,
    0.055178076,
    0.064125896,
    -0.043449506,
    -0.20648554,
    0.010422125,
    -0.06391122,
    0.34498742,
    -0.277532,
    -0.03951273,
    0.15108478,
    -0.12640053,
    -0.079585224,
    0.35258436,
    -0.057350762,
    0.0865316,
    -0.21223637,
    0.05327774,
    -0.32260162,
    -0.24913579,
    0.29470065,
    0.35723478,
    0.40376005,
    -0.0047913454,
    -0.09203944,
    -0.39992577,
    0.05428753,
    0.16352001,
    0.19129682,
    -0.32034078,
    -0.050670043,
    0.30018604,
    -0.15228687,
    0.08670345,
    0.11550844,
    0.52047265,
    0.09353356,
    -0.33545202,
    -0.25259158,
    -0.1076287,
    -0.04128109,
    0.114385575,
    0.048324574,
    0.29406673,
    0.26633546,
    0.04782568,
    -0.20178299,
    -0.1453025,
    0.025534116,
    -0.03918928,
    -0.18203782,
    -0.037490644,
    -0.3669767,
    0.30027092,
    0.09244824,
    -0.19744667,
    -0.08077549,
    -0.16951796,
    -0.1462581,
    0.053339045,
    -0.30884445,
    0.054002922,
    0.14029883,
    0.060290977,
    0.38305572,
    -0.118008405,
    -0.02421002,
    0.018779784,
    -0.158664,
    -0.31657952,
    -0.06861522,
    -0.32018337,
    0.022433698,
    -0.024515744,
    -0.080362104,
    0.4881634,
    -0.084528044,
    0.53342223,
    0.21001779,
    -0.030858383,
    -0.1285167,
    0.13945383,
    -0.039106525,
    0.09317875,
    -0.28656092,
    -0.060544163,
    -0.14365064,
    -0.30479845,
    -0.1543869,
    0.24619816,
    0.07461685,
    -0.19058324,
    -0.35595065,
    0.25888997,
    -0.2750655,
    -0.18803325,
    -0.11174507,
    0.3776188,
    0.54610217,
    0.3266489,
    -0.11675075,
    -0.4073722,
    0.019992884,
    0.39480796,
    0.25614312,
    0.23231968,
    -0.03197667,
    -0.40044388,
    -0.026413865,
    0.36998993,
    -0.25782946,
    0.43460244,
    -0.09834822,
    0.16068688,
    0.24303162,
    0.14093304,
    -0.13107711,
    0.013326421,
    0.21159083,
    0.010167778,
    0.32851326,
    -0.41584098,
    0.057303652,
    0.014807615,
    0.06997428,
    -0.16889378,
    -0.19028114,
    -0.424858,
    0.05425404,
    0.01132004,
    -0.17906487,
    0.3116254,
    -0.107275575,
    0.08927489,
    -0.035920393,
    0.08932924,
    0.33417827,
    -0.25141418,
    0.07607308,
    0.34763557,
    0.36418957,
    -0.35879004,
    -0.06499459,
    -0.24797587,
    0.017549261,
    0.13562912,
    0.46138227,
    -0.009593889,
    0.019427866,
    0.113668755,
    0.4521178,
    0.14065668,
    0.041010603,
    -0.046527684,
    0.121045746,
    -0.44324413,
    0.5287685,
    -0.27646637,
    -0.16906416,
    -0.04625575,
    -0.21924108,
    -0.39781404,
    0.14688385,
    0.030063067,
    -0.3166999,
    -0.52603555,
    0.2847375,
    -0.004978545,
    -0.103872396,
    0.07177416,
    0.15549447,
    -0.08869029,
    0.30527344,
    -0.34247008,
    0.26965344,
    0.078345485,
    0.4196058,
    -0.1909121,
    0.14259219,
    0.108189546,
    -0.16677576,
    0.34495115,
    0.472315,
    0.14752935,
    0.11222298,
    0.28200158,
    0.123448476,
    0.04208448,
    0.10118098,
    -0.18315962,
    -0.20503059,
    -0.13776034,
    0.38763228,
    0.24564123,
    -0.044949874,
    -0.10949126,
    -0.16171291,
    0.41196054,
    -0.14995697,
    -0.20107023,
    -0.13945484,
    0.44267288,
    0.36349934,
    0.11507931,
    0.28709003,
    0.11335548,
    0.39449388,
    0.38993704,
    0.086149305,
    0.21948186,
    0.016717121,
    -0.39315915,
    0.062028497,
    -0.21244538,
    0.16241708,
    -0.30891263,
    0.3306774,
    -0.0151328,
    0.45496467,
    0.200484,
    0.07688928,
    -0.023199692,
    -0.2295102,
    0.25564057,
    0.50128925,
    0.3513738,
    -0.07422376,
    0.15600929,
    -0.21763489,
    -0.029259257,
    -0.4309446,
    0.22209692,
    -0.19436744,
    -0.14031635,
    -0.12665161,
    0.09248672,
    0.3093769,
    -0.090404265,
    0.14298986,
    0.3311106,
    0.4702943,
    0.35399425,
    -0.22639664,
    0.014701918,
    -0.12418794,
    -0.115320966,
    0.25504506,
    -0.06350073,
    0.04782331,
    -0.18327202,
    0.107629225,
    0.15147136,
    -0.2053522,
    0.28599414,
    -0.031657197,
    -0.041558564,
    -0.25355738,
    0.16319594,
    -0.006639406,
    -0.11632357,
    0.018866256,
    0.30992356,
    -0.06745736,
    -0.45366743,
    -0.26545364,
    0.056162,
    -0.26000124,
    -0.2806766,
    0.04718879,
    -0.11935869,
    0.3497,
    0.108135596,
    -0.33475855,
    -0.084001,
    -0.46144974,
    0.32694966,
    -0.18386255,
    0.39845535,
    -0.091850996,
    -0.12885602,
    0.058078766,
    0.1708295,
    0.0603541,
    -0.24004239,
    0.25489882,
    0.06273575,
    0.1615884,
    -0.49116752,
    -0.2339638,
    0.014377318,
    0.09612881,
    0.31995165,
    0.21385407,
    0.44208568,
    -0.32378337,
    -0.05868312,
    -0.16734236,
];
