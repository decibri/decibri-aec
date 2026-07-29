//! Sample-rate conversion shared by the harness examples.
//!
//! This file is not an example target of its own: `examples/shared/` holds no
//! `main.rs`, so cargo does not build it as one. Each example that needs the
//! conversion declares this file as a module with an explicit `#[path]`
//! relative to that example's own source, so all of them run the same code:
//!
//! ```ignore
//! #[path = "shared/resample.rs"]      // examples/<name>.rs
//! #[path = "../shared/resample.rs"]   // examples/<name>/main.rs
//! mod resample;
//! ```
//!
//! Nothing here is shipped: the library does not depend on this file, and
//! decibri-resampler is a dev-dependency of the examples only.

use decibri_resampler::{PolyphaseResampler, Resampler};

/// Converts `samples` from `input_rate` to `output_rate` on the input's own
/// timeline.
///
/// A signal already at `output_rate` passes through untouched and carries no
/// latency. Any other rate goes through one streaming [`PolyphaseResampler`]:
/// the whole signal in one `process` call, then one `flush` to drain the
/// filter tail and partial-frame carry, the resampler's reported latency
/// removed from the front, and the result truncated to the theoretical output
/// length `ceil(len * output_rate / input_rate)`. The returned signal is
/// therefore lag-zero against the input and exactly that many samples long; a
/// shorter result is an error rather than a silently short return.
pub fn resample_aligned(
    samples: &[f32],
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f32>, String> {
    if input_rate == output_rate {
        return Ok(samples.to_vec());
    }
    let mut resampler = PolyphaseResampler::new(input_rate, output_rate)
        .map_err(|e| format!("cannot resample {input_rate} Hz to {output_rate} Hz: {e}"))?;
    let expected =
        (samples.len() as u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate)) as usize;
    let latency = resampler.latency_samples();
    let mut out = Vec::with_capacity(expected + 2 * latency);
    resampler
        .process(samples, &mut out)
        .map_err(|e| format!("cannot resample {input_rate} Hz to {output_rate} Hz: {e}"))?;
    resampler.flush(&mut out);
    if out.len() < latency + expected {
        return Err(format!(
            "resampler produced {} samples, fewer than the {latency} latency \
             plus {expected} expected",
            out.len()
        ));
    }
    out.drain(..latency);
    out.truncate(expected);
    if out.len() != expected {
        return Err(format!(
            "resampled length {} does not match the expected {expected}",
            out.len()
        ));
    }
    Ok(out)
}

#[cfg(test)]
pub mod testkit {
    /// A linear chirp rendered analytically at `rate`, so the same waveform
    /// exists at two rates with zero relative delay by construction.
    pub fn chirp(rate: u32, seconds: f64, f0: f64, f1: f64) -> Vec<f32> {
        let n = (seconds * f64::from(rate)).round() as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let phase =
                    2.0 * std::f64::consts::PI * (f0 * t + 0.5 * (f1 - f0) / seconds * t * t);
                (phase.sin() * 0.5) as f32
            })
            .collect()
    }

    /// A fixed-frequency tone of `n` samples at `rate`, active to the very
    /// last sample.
    pub fn tone(rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                ((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5) as f32
            })
            .collect()
    }

    /// The lag of `b` relative to `a` (positive when `b` is late) with the
    /// largest cross-correlation over `[-max_lag, max_lag]`.
    pub fn best_lag(a: &[f32], b: &[f32], max_lag: usize) -> isize {
        let n = a.len().min(b.len());
        let dot = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(&p, &q)| f64::from(p) * f64::from(q))
                .sum()
        };
        let mut best = 0isize;
        let mut best_v = f64::NEG_INFINITY;
        for lag in -(max_lag as isize)..=(max_lag as isize) {
            let l = lag.unsigned_abs();
            let v = if lag >= 0 {
                dot(&a[..n - l], &b[l..n])
            } else {
                dot(&a[l..n], &b[..n - l])
            };
            if v > best_v {
                best_v = v;
                best = lag;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::resample_aligned;
    use super::testkit::{best_lag, chirp, tone};

    const OUT_RATE: u32 = 16_000;

    /// The theoretical output length the contract pins the result to.
    fn theoretical_len(len: usize, input_rate: u32, output_rate: u32) -> usize {
        (len as u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate)) as usize
    }

    #[test]
    fn resample_aligned_lands_at_lag_zero() {
        let input = chirp(48_000, 2.0, 100.0, 3_000.0);
        let reference = chirp(OUT_RATE, 2.0, 100.0, 3_000.0);
        let out = resample_aligned(&input, 48_000, OUT_RATE).expect("resample");
        let lag = best_lag(&reference, &out, 300);
        assert_eq!(lag, 0, "converted output must align at lag zero, got {lag}");
    }

    #[test]
    fn resample_aligned_length_is_exact() {
        for &in_rate in &[48_000u32, 44_100] {
            for &len in &[
                7usize, 1_000, 12_345, 16_001, 44_101, 47_999, 48_000, 48_001,
            ] {
                let out = resample_aligned(&vec![0.25; len], in_rate, OUT_RATE).expect("resample");
                assert_eq!(
                    out.len(),
                    theoretical_len(len, in_rate, OUT_RATE),
                    "{in_rate} Hz input of {len} samples"
                );
            }
        }
    }

    #[test]
    fn resample_aligned_keeps_the_signal_tail() {
        let len = 48_001usize;
        let expected = theoretical_len(len, 48_000, OUT_RATE);
        let out = resample_aligned(&tone(48_000, len), 48_000, OUT_RATE).expect("resample");
        assert_eq!(out.len(), expected);
        let reference = tone(OUT_RATE, expected);
        let tail = 160usize;
        let tail_energy = |s: &[f32]| -> f64 {
            s.iter()
                .rev()
                .take(tail)
                .map(|&x| f64::from(x) * f64::from(x))
                .sum()
        };
        let ratio = tail_energy(&out) / tail_energy(&reference);
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "tail energy ratio {ratio} outside [0.5, 2.0]"
        );
    }

    #[test]
    fn matching_rates_pass_through_untouched() {
        let input = tone(OUT_RATE, 4_096);
        let out = resample_aligned(&input, OUT_RATE, OUT_RATE).expect("passthrough");
        assert_eq!(out.len(), input.len());
        for (i, (a, b)) in input.iter().zip(out.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "sample {i} differs");
        }
    }
}
