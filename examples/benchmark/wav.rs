//! WAV decode and encode for the benchmark harness: the same implementation
//! the `cancel` example carries.

use std::path::Path;

/// A decoded mono clip.
pub struct MonoClip {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Reads a WAV as mono `f32` in `[-1.0, 1.0]`. Integer PCM of any depth is
/// scaled by its full-scale value; multichannel audio is downmixed by
/// averaging the channels of each frame.
pub fn read_mono(path: &Path) -> Result<MonoClip, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let spec = reader.spec();

    // Validate the declared bit depth before it feeds a shift: hound's header
    // parsing admits depths its decoder will reject, and a depth of 65 or
    // more would otherwise overflow the shift below in a debug build instead
    // of producing this clean error.
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

    Ok(MonoClip {
        samples,
        sample_rate: spec.sample_rate,
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
