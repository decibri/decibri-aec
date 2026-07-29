//! WAV-in, WAV-out bench harness for the [`Aec`] engine.
//!
//! Runs the shipped canceller (Tau, the default configuration) over a real
//! recording pair and reports what it did: the measured energy reduction, the
//! double-talk flag rate, the delay the estimator locked (or the hint used),
//! the reference-transport counters, and the processing throughput.
//!
//! ```text
//! cargo run --release --example cancel -- <far.wav> <near.wav> <out.wav> [--delay-ms N]
//! cargo run --release --example cancel -- --selftest
//! ```
//!
//! `<far.wav>` is the far-end loopback (what the loudspeaker played) and
//! `<near.wav>` is the near-end microphone capture (speech plus the playback
//! echo).
//!
//! The engine is a 16 kHz canceller, so the harness converts on the way in:
//! an input WAV at any other rate is resampled to 16 kHz through the shared
//! `resample_aligned` contract, which uses the decibri-resampler crate (a
//! dev-dependency of this example, never of the shipped library) and returns
//! the signal on the input's own timeline, at the theoretical output length
//! and with no leading delay. The far end and the near end are converted
//! independently to the same 16 kHz target, so their relative timing is
//! preserved. The cleaned near-end signal is written to `<out.wav>` as 32-bit
//! float mono at 16 kHz, the rate the canceller operated at.
//!
//! With no `--delay-ms`, the engine's automatic delay estimator supplies the
//! alignment. `--delay-ms N` supplies a caller-measured hint instead, and the
//! summary states which path was used (and whether the engine clamped an
//! out-of-window hint).
//!
//! `--selftest` proves the pipeline end to end without any external data: it
//! synthesizes a deterministic far/near pair with a known echo path, writes
//! the pair under the crate's `data/selftest/`, then runs the normal WAV path
//! on those files and checks the measured ERLE against a floor. `--delay-ms`
//! is rejected alongside `--selftest`.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use decibri_aec::{Aec, AecConfig};

#[path = "shared/resample.rs"]
mod resample;

/// The rate the engine runs at; the harness brings every input to this rate
/// before the engine sees it.
const ENGINE_RATE: u32 = 16_000;

/// The per-turn chunk size the harness feeds and processes: 256 samples is
/// 16 ms at 16 kHz, one Tau block.
const TURN: usize = 256;

/// The ERLE floor, in dB, the self-test must clear over the converged last
/// quarter of its synthetic echo-only pair.
const SELFTEST_ERLE_FLOOR_DB: f64 = 40.0;

/// Self-test scenario length: eight seconds at 16 kHz.
const SELFTEST_LEN: usize = 128_000;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Args::parse(&args) {
        Ok(Args::Run(run)) => match cancel(&run) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        },
        Ok(Args::Selftest) => match selftest() {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("selftest FAILED: {message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!(
                "usage: cargo run --release --example cancel -- \
                 <far.wav> <near.wav> <out.wav> [--delay-ms N]\n\
                 \x20      cargo run --release --example cancel -- --selftest"
            );
            ExitCode::FAILURE
        }
    }
}

/// One parsed invocation: a real WAV run, or the synthetic self-test.
enum Args {
    Run(RunArgs),
    Selftest,
}

/// The three WAV paths and the optional delay hint of a real run.
struct RunArgs {
    far: PathBuf,
    near: PathBuf,
    out: PathBuf,
    delay_ms: Option<u16>,
}

impl Args {
    fn parse(args: &[String]) -> Result<Args, String> {
        let mut positional: Vec<&str> = Vec::new();
        let mut delay_ms: Option<u16> = None;
        let mut selftest = false;

        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--selftest" => selftest = true,
                "--delay-ms" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--delay-ms needs a value in milliseconds".to_string())?;
                    let parsed: u16 = value
                        .parse()
                        .map_err(|_| format!("--delay-ms value '{value}' is not a valid u16"))?;
                    delay_ms = Some(parsed);
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag '{other}'"));
                }
                other => positional.push(other),
            }
        }

        if selftest {
            if !positional.is_empty() {
                return Err("--selftest takes no positional arguments".to_string());
            }
            if delay_ms.is_some() {
                return Err("--selftest does not take --delay-ms".to_string());
            }
            return Ok(Args::Selftest);
        }
        match positional.as_slice() {
            [far, near, out] => Ok(Args::Run(RunArgs {
                far: PathBuf::from(far),
                near: PathBuf::from(near),
                out: PathBuf::from(out),
                delay_ms,
            })),
            other => Err(format!(
                "expected exactly three paths (<far.wav> <near.wav> <out.wav>), got {}",
                other.len()
            )),
        }
    }
}

/// What one full run measured, returned so the self-test can check it.
struct Report {
    /// The rate the engine ran at and the metrics were computed at.
    sample_rate: u32,
    /// Energy reduction from near-end input to output over the whole clip, dB.
    full_clip_reduction_db: f64,
    /// The same reduction over the last quarter of the clip, dB.
    converged_reduction_db: f64,
}

/// A clip made ready for the engine: mono `f32` at [`ENGINE_RATE`], plus what
/// it was on disk, so the summary can report what the harness did to it.
struct EngineClip {
    /// The samples the engine will see, at [`ENGINE_RATE`].
    samples: Vec<f32>,
    /// The WAV header rate the clip had on disk.
    input_rate: u32,
    /// The clip duration at its on-disk rate, for the throughput figure.
    input_seconds: f64,
    /// The decoder's note (channel count, sample format).
    source_description: String,
}

/// Brings a decoded clip to [`ENGINE_RATE`] through the shared
/// [`resample::resample_aligned`] contract, keeping the clip's on-disk rate,
/// duration and decoder note for the summary.
fn to_engine_rate(clip: wav::MonoClip) -> Result<EngineClip, String> {
    let input_seconds = clip.samples.len() as f64 / clip.sample_rate as f64;
    let samples = resample::resample_aligned(&clip.samples, clip.sample_rate, ENGINE_RATE)?;
    Ok(EngineClip {
        samples,
        input_rate: clip.sample_rate,
        input_seconds,
        source_description: clip.source_description,
    })
}

/// One summary line for an input clip: duration and rate on disk, the
/// decoder's note, and whether the harness resampled it for the engine.
fn print_input(label: &str, clip: &EngineClip) {
    if clip.input_rate == ENGINE_RATE {
        println!(
            "  {label}: {:.2} s at {} Hz ({})",
            clip.input_seconds, clip.input_rate, clip.source_description
        );
    } else {
        println!(
            "  {label}: {:.2} s at {} Hz ({}); resampled to {ENGINE_RATE} Hz",
            clip.input_seconds, clip.input_rate, clip.source_description
        );
    }
}

/// Runs the canceller over the WAV pair and prints the summary.
fn cancel(run: &RunArgs) -> Result<(), String> {
    cancel_measured(run).map(|_| ())
}

/// The full pipeline: read both WAVs, bring each to [`ENGINE_RATE`], drive
/// the engine in [`TURN`]-sample turns, write the cleaned output, print the
/// summary, return the numbers.
fn cancel_measured(run: &RunArgs) -> Result<Report, String> {
    let far = to_engine_rate(wav::read_mono(&run.far)?)?;
    let near = to_engine_rate(wav::read_mono(&run.near)?)?;

    let mut config = AecConfig::default();
    // The engine always runs at ENGINE_RATE: any other input rate was
    // resampled above, so the canceller only ever sees that rate.
    config.sample_rate = ENGINE_RATE;
    config.delay_hint_ms = run.delay_ms;
    let mut aec =
        Aec::new(config).map_err(|e| format!("engine rejected the configuration: {e}"))?;

    let mut out: Vec<f32> = Vec::with_capacity(near.samples.len());
    let mut turns = 0_u64;
    let mut double_talk_turns = 0_u64;

    // One turn: feed the reference chunk, then process the capture chunk, the
    // interleaved cadence an integrator uses. The metrics snapshot after each
    // turn is what the double-talk rate is counted over.
    let started = Instant::now();
    let mut far_chunks = far.samples.chunks(TURN);
    for near_chunk in near.samples.chunks(TURN) {
        if let Some(far_chunk) = far_chunks.next() {
            aec.feed_reference(far_chunk);
        }
        aec.process(near_chunk, &mut out)
            .map_err(|e| format!("processing failed: {e}"))?;
        turns += 1;
        if aec.metrics().canceller.double_talk {
            double_talk_turns += 1;
        }
    }
    aec.flush(&mut out)
        .map_err(|e| format!("flush failed: {e}"))?;
    let wall = started.elapsed();

    wav::write_mono(&run.out, &out, ENGINE_RATE)?;

    let metrics = aec.metrics();
    let compared = near.samples.len().min(out.len());
    let full_clip_reduction_db = reduction_db(&near.samples[..compared], &out[..compared]);
    let converged_start = compared - compared / 4;
    let converged_reduction_db = reduction_db(
        &near.samples[converged_start..compared],
        &out[converged_start..compared],
    );

    // Throughput is judged against the real recorded duration, which the
    // resampling preserves: the near end's length at its on-disk rate.
    let audio_seconds = near.input_seconds;
    let wall_seconds = wall.as_secs_f64();

    println!(
        "decibri-aec cancel: {} + {} -> {}",
        run.far.display(),
        run.near.display(),
        run.out.display()
    );
    print_input("near", &near);
    print_input("far", &far);
    println!("  engine: {ENGINE_RATE} Hz; output written at {ENGINE_RATE} Hz");
    if far.samples.len() != near.samples.len() {
        println!(
            "  note: far and near lengths differ at the engine rate; the near end was \
             processed in full and the reference was fed while it lasted"
        );
    }
    if out.len() != near.samples.len() {
        println!(
            "  note: output length {} differs from near-end input length {}; \
             measurements compare the overlapping {compared} samples",
            out.len(),
            near.samples.len(),
        );
    }
    match (run.delay_ms, metrics.delay_samples) {
        (Some(hint), Some(samples)) => {
            // The engine clamps a hint beyond its search window to the window
            // bound, so the active offset it reports back, not the raw hint, is
            // what this harness makes visible here.
            let requested = hint as u64 * ENGINE_RATE as u64 / 1000;
            if samples as u64 == requested {
                println!(
                    "  delay: caller hint {hint} ms ({samples} samples); the estimator did not run"
                );
            } else {
                println!(
                    "  delay: caller hint {hint} ms exceeds the engine's search window and \
                     was clamped; active alignment {samples} samples ({:.1} ms); the \
                     estimator did not run",
                    samples as f64 * 1000.0 / ENGINE_RATE as f64
                );
            }
        }
        (None, Some(samples)) => println!(
            "  delay: estimator locked {samples} samples ({:.1} ms)",
            samples as f64 * 1000.0 / ENGINE_RATE as f64
        ),
        (None, None) => println!(
            "  delay: estimator never locked; alignment stayed at the reference frontier, \
             so expect little or no cancellation above"
        ),
        (Some(hint), None) => {
            println!("  delay: caller hint {hint} ms supplied but not reported back; unexpected")
        }
    }
    println!(
        "  double-talk: flagged on {:.1}% of {turns} {TURN}-sample turns",
        percent(double_talk_turns, turns),
    );
    println!(
        "  reference: starved {} samples, dropped {} samples",
        metrics.reference_starved, metrics.reference_dropped
    );
    println!(
        "  canceller: internal smoothed ERLE estimate {:.1} dB; divergence resets {}",
        metrics.canceller.erle_db, metrics.canceller.divergence_resets
    );
    println!("  energy reduction, near-end in -> cancelled out (10*log10(E_in/E_out)):");
    println!("    full clip:    {full_clip_reduction_db:.2} dB");
    println!("    last quarter: {converged_reduction_db:.2} dB");
    println!(
        "  throughput: {audio_seconds:.2} s of audio in {wall_seconds:.3} s ({:.0}x realtime)",
        audio_seconds / wall_seconds.max(1e-9)
    );

    Ok(Report {
        sample_rate: ENGINE_RATE,
        full_clip_reduction_db,
        converged_reduction_db,
    })
}

/// Synthesizes the deterministic echo pair, writes it under the crate's own
/// `data/selftest/` (anchored to the manifest directory, not the invoker's
/// working directory, so the WAVs always land inside the gitignored folder),
/// runs the normal WAV path on those files, and checks the measured ERLE.
fn selftest() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("selftest");
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let (far, mic) = synth::echo_pair(SELFTEST_LEN);
    let far_path = dir.join("far.wav");
    let near_path = dir.join("near.wav");
    let out_path = dir.join("out.wav");
    wav::write_mono(&far_path, &far, 16_000)?;
    wav::write_mono(&near_path, &mic, 16_000)?;

    println!(
        "selftest: synthetic far-end single-talk pair, {:.0} s at 16 kHz;\n\
         \x20 written to {} for listening",
        SELFTEST_LEN as f64 / 16_000.0,
        dir.display()
    );
    let report = cancel_measured(&RunArgs {
        far: far_path,
        near: near_path,
        out: out_path,
        delay_ms: None,
    })?;

    debug_assert_eq!(report.sample_rate, 16_000);
    if report.converged_reduction_db < SELFTEST_ERLE_FLOOR_DB {
        return Err(format!(
            "converged ERLE {:.2} dB is under the {SELFTEST_ERLE_FLOOR_DB} dB floor \
             (full clip {:.2} dB); the pipeline is misrouting or misaligning",
            report.converged_reduction_db, report.full_clip_reduction_db
        ));
    }
    println!(
        "selftest PASS: converged ERLE {:.2} dB clears the {SELFTEST_ERLE_FLOOR_DB} dB floor",
        report.converged_reduction_db
    );
    Ok(())
}

/// Energy reduction from `input` to `output` in decibels, accumulated in
/// `f64`. Infinite when the output is exactly silent; zero when the input is.
fn reduction_db(input: &[f32], output: &[f32]) -> f64 {
    let input_energy: f64 = input.iter().map(|&s| s as f64 * s as f64).sum();
    let output_energy: f64 = output.iter().map(|&s| s as f64 * s as f64).sum();
    if input_energy <= 0.0 {
        return 0.0;
    }
    if output_energy <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (input_energy / output_energy).log10()
}

/// `part` of `total` as a percentage, zero when there is no total.
fn percent(part: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    part as f64 * 100.0 / total as f64
}

/// Minimal WAV read/write on the `hound` dev-dependency: mono or multichannel
/// in (downmixed by averaging), 32-bit float mono out.
mod wav {
    use std::path::Path;

    /// A decoded mono clip and a human-readable note of what decoding did.
    pub struct MonoClip {
        pub samples: Vec<f32>,
        pub sample_rate: u32,
        pub source_description: String,
    }

    /// Reads a WAV as mono `f32` in `[-1.0, 1.0]`. Integer PCM of any depth
    /// is scaled by its full-scale value; multichannel audio is downmixed by
    /// averaging the channels of each frame.
    pub fn read_mono(path: &Path) -> Result<MonoClip, String> {
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let spec = reader.spec();

        // Validate the declared bit depth before it feeds a shift: hound's
        // header parsing admits depths its decoder will reject, and a depth
        // of 65 or more would otherwise overflow the shift below in a debug
        // build instead of producing this clean error.
        let supported = match spec.sample_format {
            hound::SampleFormat::Float => spec.bits_per_sample == 32,
            hound::SampleFormat::Int => (1..=32).contains(&spec.bits_per_sample),
        };
        if !supported {
            return Err(format!(
                "unsupported bit depth in {}: {} bits {:?}; expected 32-bit float or \
                 integer PCM of at most 32 bits",
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

        let format = match spec.sample_format {
            hound::SampleFormat::Float => format!("{}-bit float", spec.bits_per_sample),
            hound::SampleFormat::Int => format!("{}-bit int", spec.bits_per_sample),
        };
        let source_description = if channels == 1 {
            format!("mono, {format}")
        } else {
            format!("{channels} channels averaged to mono, {format}")
        };

        Ok(MonoClip {
            samples,
            sample_rate: spec.sample_rate,
            source_description,
        })
    }

    /// Writes mono `f32` samples as a 32-bit float WAV at `sample_rate`.
    pub fn write_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        for &sample in samples {
            writer
                .write_sample(sample)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("cannot finalize {}: {e}", path.display()))
    }
}

/// The deterministic synthesis behind `--selftest`: an integer-only LCG and a
/// sparse echo path.
mod synth {
    /// A deterministic linear congruential generator, integer state mapped to
    /// `f32`, no platform-dependent transcendentals.
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

    /// Far-end single-talk pair of `len` samples: a broadband far end, and a
    /// microphone signal holding its echo through the known path plus a tiny
    /// noise floor.
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
