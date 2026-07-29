//! The owned, deterministic radix-2 fast Fourier transform.
//!
//! This is the crate's frequency-domain primitive: an owned Cooley-Tukey radix-2
//! transform written from first principles, with no third-party FFT dependency.
//! It is checked against an independent naive discrete Fourier transform (DFT)
//! and pinned to a committed golden vector.
//!
//! # What it provides
//!
//! - [`ComplexFft`]: the general complex-to-complex radix-2 transform, forward
//!   and inverse, in place. It is the engine the real path is built on and the
//!   thing the naive DFT is checked against directly.
//! - [`RealFft`]: the real-input path the filter actually uses. A real block of
//!   `n` samples transforms to the `n/2 + 1` non-redundant complex bins (the
//!   rest are the conjugate mirror), via a half-length complex transform and an
//!   `O(n)` recombination, and back. This is the standard efficient real FFT,
//!   not a full complex transform on zero-padded imaginary parts.
//! - [`Complex`]: the `f64` complex number the transforms carry.
//!
//! # Size and the block model
//!
//! Every transform size is a power of two, checked at construction. The
//! partitioned-block frequency-domain filter uses a length-512 real transform
//! over a length-256 complex one. The transforms here are generic over any
//! power-of-two size so the filter can pick its partition, and the committed
//! golden pins the block-model size of 512 concretely. The power-of-two
//! constraint is what keeps the decimation a clean radix-2 recursion with a
//! fixed, auditable butterfly order.
//!
//! # Normalization
//!
//! The forward transforms are unscaled, and the entire `1/n` normalization lives
//! in the inverse, so `inverse(forward(x)) == x` up to floating-point roundoff.
//! A frequency-domain filter that multiplies spectra therefore applies no scale
//! of its own beyond the single inverse it already runs per block.
//!
//! # Determinism
//!
//! The same input produces bit-identical output on every platform, every run,
//! and both supported toolchains:
//!
//! - The butterflies evaluate in a fixed, explicit order. Nothing iterates an
//!   unordered container, and there is no parallelism, no threading, no time,
//!   and no randomness anywhere in the module.
//! - The transform path uses only IEEE-exact operations (`+`, `-`, `*`, and one
//!   `*` by a precomputed reciprocal for the inverse scale). There is no
//!   `mul_add`/FMA, so the result never depends on whether a target contracts a
//!   multiply-add, and rustc applies no fast-math reassociation at any
//!   optimization level. Complex multiplication is written as two independent
//!   products per component precisely so no fused form can arise.
//! - There are no transcendentals in the transform path. The twiddle factors are
//!   the only place `cos`/`sin` are needed, and they are computed once at
//!   construction by a deterministic power-series evaluation (see [`det_cos`] and
//!   [`det_sin`]), not the platform `f64::cos`/`f64::sin`, whose precision the
//!   standard library documents as platform-dependent. The resulting twiddle
//!   table is therefore bit-identical everywhere and is captured as part of the
//!   constructed transform.
//! - The committed golden vector in the tests pins a full length-512 real
//!   transform to the bit, so an accidental change to any of the above is caught
//!   as a bit mismatch rather than a silent low-bit drift.
//!
//! This module is an internal primitive: it is `pub(crate)`. Its items are
//! unused in a plain library build, which is what the module-level
//! `allow(dead_code)` acknowledges.

#![allow(dead_code)]

use std::f64::consts::PI;

/// `2 * pi`, the full-turn angle used to build the twiddle table.
const TWO_PI: f64 = 2.0 * PI;

/// `pi / 2`, the fold boundary in the deterministic trig range reduction.
const HALF_PI: f64 = 0.5 * PI;

/// A complex number carried by the transforms, in `f64`.
///
/// The transforms compute in `f64` throughout: the crate's audio is `f32`, so
/// the wider type is carried here and narrowed only at the `f32` audio boundary
/// of the real path. The arithmetic is written out as explicit method calls
/// rather than operator overloads so the evaluation order behind the determinism
/// claim is visible at every call site.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Complex {
    /// The real part.
    pub(crate) re: f64,
    /// The imaginary part.
    pub(crate) im: f64,
}

impl Complex {
    /// The additive identity, `0 + 0i`.
    const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    /// Constructs a complex number from its real and imaginary parts.
    pub(crate) fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// The complex conjugate: the imaginary part negated.
    fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// Componentwise sum.
    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Componentwise difference.
    fn sub(self, other: Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Complex product, written as two independent products per component so no
    /// fused multiply-add can arise and the operation is bit-identical on every
    /// target: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`.
    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Scales both parts by a real factor.
    fn scale(self, factor: f64) -> Complex {
        Complex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

/// Sine of `x` in radians, evaluated deterministically.
///
/// The platform `f64::sin` has platform-dependent precision, which would make
/// the twiddle table (and therefore every transform) differ across targets. This
/// evaluates instead in a fixed operation order with no fused multiply-add: the
/// argument is reduced to `[-pi, pi]` by subtracting the nearest multiple of
/// `2*pi`, folded into `[-pi/2, pi/2]` via `sin(pi - r) = sin(r)`, then a
/// fixed-degree odd Taylor polynomial (through the term in `r^23`) is evaluated
/// by Horner's method. Over the folded range the truncation error is far below
/// `f64` epsilon, and the twiddle angles used here already lie in `[-pi, 0]`, so
/// the reduction multiple is zero and contributes no error.
fn det_sin(x: f64) -> f64 {
    // Reduce to [-pi, pi].
    let k = (x / TWO_PI).round();
    let mut r = x - k * TWO_PI;

    // Fold into [-pi/2, pi/2]: sin(pi - r) = sin(r), sin(-pi - r) = sin(r).
    if r > HALF_PI {
        r = PI - r;
    } else if r < -HALF_PI {
        r = -PI - r;
    }

    // Odd Taylor coefficients: a_k = (-1)^k / (2k + 1)!.
    let a0 = 1.0;
    let a1 = -1.0 / 6.0;
    let a2 = 1.0 / 120.0;
    let a3 = -1.0 / 5_040.0;
    let a4 = 1.0 / 362_880.0;
    let a5 = -1.0 / 39_916_800.0;
    let a6 = 1.0 / 6_227_020_800.0;
    let a7 = -1.0 / 1_307_674_368_000.0;
    let a8 = 1.0 / 355_687_428_096_000.0;
    let a9 = -1.0 / 121_645_100_408_832_000.0;
    let a10 = 1.0 / 51_090_942_171_709_440_000.0;
    let a11 = -1.0 / 25_852_016_738_884_976_640_000.0;

    let r2 = r * r;
    // Horner in ascending order, fixed evaluation order, no fused multiply-add.
    let mut p = a11;
    p = p * r2 + a10;
    p = p * r2 + a9;
    p = p * r2 + a8;
    p = p * r2 + a7;
    p = p * r2 + a6;
    p = p * r2 + a5;
    p = p * r2 + a4;
    p = p * r2 + a3;
    p = p * r2 + a2;
    p = p * r2 + a1;
    p = p * r2 + a0;
    p * r
}

/// Cosine of `x` in radians, evaluated deterministically.
///
/// The companion to [`det_sin`], with the same rationale and the same fixed,
/// fused-free evaluation. The argument is reduced to `[-pi, pi]`, folded into
/// `[0, pi/2]` using that cosine is even and `cos(r) = -cos(pi - r)`, then a
/// fixed-degree even Taylor polynomial (through the term in `r^22`) is evaluated
/// by Horner's method. `det_cos(0)` returns exactly `1.0`, which makes the zeroth
/// twiddle exactly `1 + 0i` and keeps the trivial butterflies and the DC bin
/// bit-exact.
fn det_cos(x: f64) -> f64 {
    // Reduce to [-pi, pi].
    let k = (x / TWO_PI).round();
    let r = x - k * TWO_PI;

    // Fold into [0, pi/2]: cosine is even, and cos(r) = -cos(pi - r).
    let mut r = r.abs();
    let mut sign = 1.0;
    if r > HALF_PI {
        r = PI - r;
        sign = -1.0;
    }

    // Even Taylor coefficients: b_k = (-1)^k / (2k)!.
    let b0 = 1.0;
    let b1 = -1.0 / 2.0;
    let b2 = 1.0 / 24.0;
    let b3 = -1.0 / 720.0;
    let b4 = 1.0 / 40_320.0;
    let b5 = -1.0 / 3_628_800.0;
    let b6 = 1.0 / 479_001_600.0;
    let b7 = -1.0 / 87_178_291_200.0;
    let b8 = 1.0 / 20_922_789_888_000.0;
    let b9 = -1.0 / 6_402_373_705_728_000.0;
    let b10 = 1.0 / 2_432_902_008_176_640_000.0;
    let b11 = -1.0 / 1_124_000_727_777_607_680_000.0;

    let r2 = r * r;
    // Horner in ascending order, fixed evaluation order, no fused multiply-add.
    let mut p = b11;
    p = p * r2 + b10;
    p = p * r2 + b9;
    p = p * r2 + b8;
    p = p * r2 + b7;
    p = p * r2 + b6;
    p = p * r2 + b5;
    p = p * r2 + b4;
    p = p * r2 + b3;
    p = p * r2 + b2;
    p = p * r2 + b1;
    p = p * r2 + b0;
    sign * p
}

/// The direction of a complex transform.
#[derive(Clone, Copy)]
enum Direction {
    /// The unscaled forward transform, twiddle `exp(-2*pi*i*k/n)`.
    Forward,
    /// The inverse transform, conjugate twiddle and a final `1/n` scale.
    Inverse,
}

/// A complex-to-complex radix-2 fast Fourier transform for a fixed power-of-two
/// size.
///
/// Construction precomputes the bit-reversal permutation and the twiddle table
/// once; [`forward`](ComplexFft::forward) and [`inverse`](ComplexFft::inverse)
/// then run in place over a caller-owned buffer of exactly `n` elements with no
/// allocation and no transcendental. See the module documentation for the
/// determinism guarantees.
pub(crate) struct ComplexFft {
    /// The transform size, a power of two.
    n: usize,
    /// `twiddles[r] = exp(-2*pi*i*r/n)` for `r` in `0..n/2`, built once at
    /// construction with [`det_cos`]/[`det_sin`] so the table is bit-identical on
    /// every platform. The forward transform indexes it directly; the inverse
    /// conjugates on read.
    twiddles: Vec<Complex>,
    /// `bit_reversal[i]` is `i` with its `log2(n)` low bits reversed: the
    /// decimation-in-time input permutation, precomputed so the hot path only
    /// indexes it.
    bit_reversal: Vec<usize>,
    /// `1.0 / n`, the inverse normalization, computed once so the inverse applies
    /// it as a single deterministic multiply.
    inv_n: f64,
}

impl ComplexFft {
    /// Constructs the transform for a power-of-two size.
    ///
    /// Panics if `n` is not a power of two: the size is chosen by crate-internal
    /// code, so a non-power-of-two is a construction bug, not a runtime input.
    pub(crate) fn new(n: usize) -> ComplexFft {
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of two, got {n}"
        );
        ComplexFft {
            n,
            twiddles: build_twiddles(n),
            bit_reversal: build_bit_reversal(n),
            inv_n: 1.0 / n as f64,
        }
    }

    /// The transform size.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Runs the unscaled forward transform in place. `buf` must have length `n`.
    pub(crate) fn forward(&self, buf: &mut [Complex]) {
        self.run(buf, Direction::Forward);
    }

    /// Runs the inverse transform in place, including the `1/n` normalization, so
    /// it exactly inverts [`forward`](ComplexFft::forward) up to roundoff. `buf`
    /// must have length `n`.
    pub(crate) fn inverse(&self, buf: &mut [Complex]) {
        self.run(buf, Direction::Inverse);
    }

    /// The shared decimation-in-time engine.
    ///
    /// Applies the precomputed bit-reversal permutation, then the `log2(n)`
    /// butterfly stages in ascending order. The twiddle for stage span `m` and
    /// position `j` is `twiddles[j * (n / m)]`, which is `exp(-2*pi*i*j/m)`; the
    /// inverse conjugates it and scales the result by `1/n` at the end. Every
    /// arithmetic operation is a plain IEEE `+`, `-`, or `*`.
    fn run(&self, buf: &mut [Complex], direction: Direction) {
        assert_eq!(buf.len(), self.n, "transform buffer must have length n");

        // Decimation-in-time input permutation, each pair swapped exactly once.
        for (i, &j) in self.bit_reversal.iter().enumerate() {
            if i < j {
                buf.swap(i, j);
            }
        }

        // Butterfly stages, span doubling from 2 up to n.
        let mut span = 2;
        while span <= self.n {
            let half = span / 2;
            let stride = self.n / span;
            let mut base = 0;
            while base < self.n {
                for j in 0..half {
                    let twiddle = match direction {
                        Direction::Forward => self.twiddles[j * stride],
                        Direction::Inverse => self.twiddles[j * stride].conj(),
                    };
                    let upper = buf[base + j];
                    let lower = twiddle.mul(buf[base + j + half]);
                    buf[base + j] = upper.add(lower);
                    buf[base + j + half] = upper.sub(lower);
                }
                base += span;
            }
            span <<= 1;
        }

        // The inverse carries the whole 1/n normalization.
        if let Direction::Inverse = direction {
            for value in buf.iter_mut() {
                *value = value.scale(self.inv_n);
            }
        }
    }
}

/// A real-input fast Fourier transform for a fixed power-of-two size.
///
/// A real block of `n` samples has a conjugate-symmetric spectrum, so only the
/// `n/2 + 1` bins from DC through Nyquist are independent; the rest are their
/// mirror. This computes exactly those bins (and inverts from them) using a
/// single half-length complex transform plus an `O(n)` recombination, which is
/// the transform an overlap-save frequency-domain filter runs per block. The
/// forward path takes `f32` audio and yields `f64` bins; the inverse takes bins
/// and yields `f32` audio, converting only at that boundary.
pub(crate) struct RealFft {
    /// The real block length, an even power of two.
    n: usize,
    /// The half-length complex transform the real path is built on: length
    /// `n/2`. The real samples are packed two-per-element into it.
    half: ComplexFft,
    /// `recombination[k] = exp(-2*pi*i*k/n)` for `k` in `0..=n/2`, the full-length
    /// twiddles that split the half-length spectrum into the real spectrum. Built
    /// once with [`det_cos`]/[`det_sin`], so `recombination[0]` is exactly
    /// `1 + 0i` and `recombination[n/2]` is exactly `-1 + 0i`.
    recombination: Vec<Complex>,
}

impl RealFft {
    /// Constructs the real transform for an even power-of-two block length.
    ///
    /// Panics if `n` is not a power of two of at least 2. The size comes from
    /// crate-internal code, so this guards a construction bug rather than runtime
    /// input.
    pub(crate) fn new(n: usize) -> RealFft {
        assert!(
            n >= 2 && n.is_power_of_two(),
            "real FFT size must be a power of two of at least 2, got {n}"
        );
        let half = n / 2;
        let recombination = (0..=half)
            .map(|k| {
                let theta = -TWO_PI * (k as f64) / (n as f64);
                Complex::new(det_cos(theta), det_sin(theta))
            })
            .collect();
        RealFft {
            n,
            half: ComplexFft::new(half),
            recombination,
        }
    }

    /// The real block length.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// The number of independent complex bins, `n/2 + 1` (DC through Nyquist).
    pub(crate) fn spectrum_len(&self) -> usize {
        self.n / 2 + 1
    }

    /// Transforms `input` (exactly `n` real `f32` samples) into its `n/2 + 1`
    /// complex bins, overwriting `out`.
    ///
    /// Unlike the appending canceller convention, a transform yields a complete
    /// result, so `out` is cleared and refilled. The transform is unscaled.
    ///
    /// The method packs the even-indexed samples into the real parts and the
    /// odd-indexed samples into the imaginary parts of a half-length complex
    /// vector, runs the half-length forward transform, then recombines: with
    /// `Z` the half-length spectrum, the even/odd sub-spectra are recovered by
    /// conjugate symmetry as `E[k] = (Z[k] + conj(Z[m-k]))/2` and
    /// `O[k] = -i*(Z[k] - conj(Z[m-k]))/2`, and the real spectrum is
    /// `X[k] = E[k] + W_n^k * O[k]`.
    pub(crate) fn forward(&self, input: &[f32], out: &mut Vec<Complex>) {
        assert_eq!(input.len(), self.n, "real FFT input must have length n");
        let m = self.n / 2;

        // Pack consecutive sample pairs into the half-length complex vector.
        let mut packed: Vec<Complex> = input
            .chunks_exact(2)
            .map(|pair| Complex::new(pair[0] as f64, pair[1] as f64))
            .collect();
        self.half.forward(&mut packed);

        out.clear();
        out.reserve(m + 1);
        for k in 0..=m {
            let z_k = packed[k % m];
            let z_mirror = packed[(m - k) % m].conj();
            // E[k] = (Z[k] + conj(Z[m-k])) / 2.
            let even = z_k.add(z_mirror).scale(0.5);
            // O[k] = -i * (Z[k] - conj(Z[m-k])) / 2. Multiplying c by -i/2 maps
            // (c.re, c.im) to (c.im/2, -c.re/2).
            let diff = z_k.sub(z_mirror);
            let odd = Complex::new(0.5 * diff.im, -0.5 * diff.re);
            out.push(even.add(self.recombination[k].mul(odd)));
        }
    }

    /// Inverts a spectrum of `n/2 + 1` complex bins back to `n` real `f32`
    /// samples, overwriting `out`.
    ///
    /// This inverts [`forward`](RealFft::forward) up to roundoff. It reconstructs
    /// the even/odd sub-spectra from the half-spectrum and its conjugate mirror,
    /// forms the half-length complex spectrum, runs the half-length inverse
    /// (which carries its own `1/(n/2)` scale), and unpacks. The bins above
    /// Nyquist are supplied implicitly by conjugate symmetry. It expects a
    /// Hermitian spectrum, the kind [`forward`](RealFft::forward) produces, with
    /// real DC and Nyquist bins; it does not sanitize a malformed spectrum, so a
    /// caller that fabricates one gets a correspondingly malformed result rather
    /// than a silent correction.
    pub(crate) fn inverse(&self, spectrum: &[Complex], out: &mut Vec<f32>) {
        assert_eq!(
            spectrum.len(),
            self.n / 2 + 1,
            "real FFT spectrum must have length n/2 + 1"
        );
        let m = self.n / 2;

        let mut packed = vec![Complex::ZERO; m];
        for (k, slot) in packed.iter_mut().enumerate() {
            let x_k = spectrum[k];
            // The mirror bin, supplied by conjugate symmetry for k = 0.
            let x_mirror = spectrum[m - k].conj();
            // E[k] = (X[k] + conj(X[m-k])) / 2.
            let even = x_k.add(x_mirror).scale(0.5);
            // W_n^k * O[k] = (X[k] - conj(X[m-k])) / 2, so
            // O[k] = conj(W_n^k) * (X[k] - conj(X[m-k])) / 2.
            let diff = x_k.sub(x_mirror);
            let odd = self.recombination[k].conj().mul(diff).scale(0.5);
            // Z[k] = E[k] + i * O[k].
            *slot = Complex::new(even.re - odd.im, even.im + odd.re);
        }
        self.half.inverse(&mut packed);

        out.clear();
        out.reserve(self.n);
        for value in &packed {
            out.push(value.re as f32);
            out.push(value.im as f32);
        }
    }
}

/// Builds the twiddle table `exp(-2*pi*i*r/n)` for `r` in `0..n/2`, using the
/// deterministic trig so the table is bit-identical on every platform. For
/// `n <= 1` the table is empty (a size-one transform is the identity and runs no
/// butterflies).
fn build_twiddles(n: usize) -> Vec<Complex> {
    (0..n / 2)
        .map(|r| {
            let theta = -TWO_PI * (r as f64) / (n as f64);
            Complex::new(det_cos(theta), det_sin(theta))
        })
        .collect()
}

/// Builds the bit-reversal permutation for a power-of-two size: `table[i]` is `i`
/// with its `log2(n)` low bits reversed.
fn build_bit_reversal(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros();
    (0..n).map(|i| reverse_low_bits(i, bits)).collect()
}

/// Reverses the low `bits` bits of `value`.
fn reverse_low_bits(mut value: usize, bits: u32) -> usize {
    let mut reversed = 0;
    for _ in 0..bits {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic linear congruential generator: integer-only state mapped
    /// to a float, no platform-dependent transcendental, so every test input is
    /// bit-identical everywhere.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }

        /// The next pseudo-random value in `[-1.0, 1.0)`.
        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (self.0 >> 40) as u32; // top 24 bits
            (bits as f64 / (1u32 << 23) as f64) - 1.0
        }
    }

    /// Independent, obviously-correct O(n^2) complex DFT reference. It uses the
    /// platform `cos`/`sin` deliberately: this is a test reference, not the
    /// shipped path, so its platform-dependent transcendental precision is fine.
    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::ZERO;
                for (idx, &sample) in x.iter().enumerate() {
                    let angle = -2.0 * PI * (k as f64) * (idx as f64) / (n as f64);
                    let twiddle = Complex::new(angle.cos(), angle.sin());
                    acc = acc.add(twiddle.mul(sample));
                }
                acc
            })
            .collect()
    }

    /// Independent naive real DFT: the `n/2 + 1` non-redundant bins of a real
    /// input, by the same O(n^2) definition.
    fn naive_real_dft(x: &[f64]) -> Vec<Complex> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let mut acc = Complex::ZERO;
                for (idx, &sample) in x.iter().enumerate() {
                    let angle = -2.0 * PI * (k as f64) * (idx as f64) / (n as f64);
                    acc = acc.add(Complex::new(angle.cos() * sample, angle.sin() * sample));
                }
                acc
            })
            .collect()
    }

    /// The largest magnitude of the componentwise difference between two complex
    /// vectors.
    fn max_abs_diff(a: &[Complex], b: &[Complex]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(p, q)| {
                let dr = p.re - q.re;
                let di = p.im - q.im;
                (dr * dr + di * di).sqrt()
            })
            .fold(0.0_f64, f64::max)
    }

    /// The magnitude of a complex number.
    fn magnitude(c: Complex) -> f64 {
        (c.re * c.re + c.im * c.im).sqrt()
    }

    /// The forward transform sizes exercised against the naive DFT. Small sizes
    /// make the O(n^2) reference cheap; 512 is the block-model size.
    const SIZES: [usize; 9] = [2, 4, 8, 16, 32, 64, 128, 256, 512];

    /// A spread of complex inputs of length `n` for the agreement sweep:
    /// impulse, DC, a single complex exponential at each of a few bins, and
    /// deterministic random vectors.
    fn complex_cases(n: usize) -> Vec<Vec<Complex>> {
        let mut cases = Vec::new();

        let mut impulse = vec![Complex::ZERO; n];
        impulse[0] = Complex::new(1.0, 0.0);
        cases.push(impulse);

        cases.push(vec![Complex::new(1.0, 0.0); n]);

        for &bin in &[1usize, 2, n / 2, n / 2 + 1] {
            if bin < n {
                let wave: Vec<Complex> = (0..n)
                    .map(|idx| {
                        let angle = 2.0 * PI * (bin as f64) * (idx as f64) / (n as f64);
                        Complex::new(angle.cos(), angle.sin())
                    })
                    .collect();
                cases.push(wave);
            }
        }

        let mut lcg = Lcg::new(0x0FF1_CE00 ^ n as u64);
        for _ in 0..3 {
            let random: Vec<Complex> = (0..n)
                .map(|_| Complex::new(lcg.next_unit(), lcg.next_unit()))
                .collect();
            cases.push(random);
        }

        cases
    }

    /// The fast complex transform must match the independent naive DFT within a
    /// fixed tolerance.
    #[test]
    fn complex_fft_matches_naive_dft() {
        let mut worst = 0.0_f64;
        for &n in &SIZES {
            let fft = ComplexFft::new(n);
            for input in complex_cases(n) {
                let mut buf = input.clone();
                fft.forward(&mut buf);
                let reference = naive_dft(&input);
                worst = worst.max(max_abs_diff(&buf, &reference));
            }
        }
        println!("complex FFT vs naive DFT worst absolute error = {worst:e}");
        assert!(
            worst < 1e-9,
            "complex FFT worst error {worst:e} exceeds the 1e-9 bound"
        );
    }

    /// The real transform must match the independent naive real DFT within a
    /// fixed tolerance.
    #[test]
    fn real_fft_matches_naive_real_dft() {
        let mut worst = 0.0_f64;
        for &n in &SIZES {
            let fft = RealFft::new(n);
            let mut lcg = Lcg::new(0xD1CE_5EED ^ n as u64);

            let mut cases: Vec<Vec<f32>> = Vec::new();
            // Impulse.
            let mut impulse = vec![0.0_f32; n];
            impulse[0] = 1.0;
            cases.push(impulse);
            // DC.
            cases.push(vec![1.0_f32; n]);
            // Nyquist alternation.
            cases.push(
                (0..n)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
            );
            // Sinusoids at a bin and near an inter-bin frequency.
            for &bin in &[1usize, 3, n / 4, n / 2] {
                if bin <= n / 2 {
                    cases.push(
                        (0..n)
                            .map(|i| {
                                (2.0 * PI * (bin as f64) * (i as f64) / (n as f64)).cos() as f32
                            })
                            .collect(),
                    );
                }
            }
            // Deterministic random vectors.
            for _ in 0..3 {
                cases.push((0..n).map(|_| lcg.next_unit() as f32).collect());
            }

            let mut out = Vec::new();
            for input in cases {
                fft.forward(&input, &mut out);
                let reference_input: Vec<f64> = input.iter().map(|&s| s as f64).collect();
                let reference = naive_real_dft(&reference_input);
                worst = worst.max(max_abs_diff(&out, &reference));
            }
        }
        println!("real FFT vs naive real DFT worst absolute error = {worst:e}");
        assert!(
            worst < 1e-9,
            "real FFT worst error {worst:e} exceeds the 1e-9 bound"
        );
    }

    /// An impulse transforms to a perfectly flat spectrum, and this is bit-exact:
    /// the impulse propagates through the butterflies multiplying only zeros, so
    /// every bin is exactly `1 + 0i` regardless of twiddle precision. This is the
    /// indexing-and-scaling check a tolerance-based comparison can mask.
    #[test]
    fn impulse_transforms_to_a_flat_spectrum() {
        let n = 512;

        let fft = ComplexFft::new(n);
        let mut buf = vec![Complex::ZERO; n];
        buf[0] = Complex::new(1.0, 0.0);
        fft.forward(&mut buf);
        for (k, bin) in buf.iter().enumerate() {
            assert_eq!(bin.re.to_bits(), 1.0_f64.to_bits(), "complex bin {k} re");
            assert_eq!(bin.im.to_bits(), 0.0_f64.to_bits(), "complex bin {k} im");
        }

        let real_fft = RealFft::new(n);
        let mut input = vec![0.0_f32; n];
        input[0] = 1.0;
        let mut out = Vec::new();
        real_fft.forward(&input, &mut out);
        assert_eq!(out.len(), n / 2 + 1);
        for (k, bin) in out.iter().enumerate() {
            assert_eq!(bin.re.to_bits(), 1.0_f64.to_bits(), "real bin {k} re");
            assert_eq!(bin.im.to_bits(), 0.0_f64.to_bits(), "real bin {k} im");
        }
    }

    /// A DC input transforms to a single DC bin. The DC bin is the exact sum of
    /// the inputs, so it is asserted bit-exactly (`= n`), while the vanishing
    /// bins are held to a small tolerance.
    #[test]
    fn dc_transforms_to_a_single_bin() {
        let n = 256;

        let fft = ComplexFft::new(n);
        let mut buf = vec![Complex::new(1.0, 0.0); n];
        fft.forward(&mut buf);
        assert_eq!(buf[0].re.to_bits(), (n as f64).to_bits(), "complex DC re");
        assert_eq!(buf[0].im.to_bits(), 0.0_f64.to_bits(), "complex DC im");
        for (k, bin) in buf.iter().enumerate().skip(1) {
            assert!(magnitude(*bin) < 1e-11, "complex bin {k} should vanish");
        }

        let real_fft = RealFft::new(n);
        let mut out = Vec::new();
        real_fft.forward(&vec![1.0_f32; n], &mut out);
        assert_eq!(out[0].re.to_bits(), (n as f64).to_bits(), "real DC re");
        assert_eq!(out[0].im.to_bits(), 0.0_f64.to_bits(), "real DC im");
        for (k, bin) in out.iter().enumerate().skip(1) {
            assert!(magnitude(*bin) < 1e-11, "real bin {k} should vanish");
        }
    }

    /// A pure cosine at an exact bin lands in that bin (magnitude `n/2`, real and
    /// positive), and every other bin vanishes.
    #[test]
    fn a_sinusoid_lands_in_its_bin() {
        let n = 512;
        let bin = 37;
        let real_fft = RealFft::new(n);
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * (bin as f64) * (i as f64) / (n as f64)).cos() as f32)
            .collect();
        let mut out = Vec::new();
        real_fft.forward(&input, &mut out);

        let peak = out[bin];
        let peak_error = (peak.re - n as f64 / 2.0).abs().max(peak.im.abs());
        let leakage = out
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != bin)
            .map(|(_, value)| magnitude(*value))
            .fold(0.0_f64, f64::max);
        println!("sinusoid peak error = {peak_error:e}, worst off-bin leakage = {leakage:e}");
        assert!(
            peak_error < 3e-4,
            "bin {bin} should hold n/2, got {peak:?} (error {peak_error:e})"
        );
        assert!(leakage < 3e-5, "off-bin leakage {leakage:e} too large");
    }

    /// A Nyquist alternation `(-1)^n` lands in the Nyquist bin and nowhere else.
    #[test]
    fn a_nyquist_alternation_lands_in_the_nyquist_bin() {
        let n = 256;
        let real_fft = RealFft::new(n);
        let input: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut out = Vec::new();
        real_fft.forward(&input, &mut out);

        let nyquist = out[n / 2];
        let peak_error = (nyquist.re - n as f64).abs().max(nyquist.im.abs());
        let leakage = out
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != n / 2)
            .map(|(_, value)| magnitude(*value))
            .fold(0.0_f64, f64::max);
        println!("nyquist peak error = {peak_error:e}, worst off-bin leakage = {leakage:e}");
        assert!(
            peak_error < 1e-9,
            "Nyquist bin should hold n, got {nyquist:?} (error {peak_error:e})"
        );
        assert!(leakage < 1e-9, "off-bin leakage {leakage:e} too large");
    }

    /// The transform is linear: `F(a*x + b*y) = a*F(x) + b*F(y)` up to roundoff.
    #[test]
    fn the_transform_is_linear() {
        let n = 128;
        let fft = ComplexFft::new(n);
        let mut lcg = Lcg::new(0x11EA_0000);
        let x: Vec<Complex> = (0..n)
            .map(|_| Complex::new(lcg.next_unit(), lcg.next_unit()))
            .collect();
        let y: Vec<Complex> = (0..n)
            .map(|_| Complex::new(lcg.next_unit(), lcg.next_unit()))
            .collect();
        let a = Complex::new(0.7, -0.3);
        let b = Complex::new(-0.2, 1.1);

        let mut combined: Vec<Complex> = (0..n).map(|i| a.mul(x[i]).add(b.mul(y[i]))).collect();
        fft.forward(&mut combined);

        let mut fx = x.clone();
        fft.forward(&mut fx);
        let mut fy = y.clone();
        fft.forward(&mut fy);
        let expected: Vec<Complex> = (0..n).map(|i| a.mul(fx[i]).add(b.mul(fy[i]))).collect();

        let err = max_abs_diff(&combined, &expected);
        println!("linearity worst absolute error = {err:e}");
        assert!(err < 1e-11, "linearity error {err:e} exceeds 1e-11");
    }

    /// Forward then inverse recovers a complex input up to roundoff, confirming
    /// the `1/n` normalization lives entirely in the inverse.
    #[test]
    fn complex_roundtrip_recovers_the_input() {
        let mut worst = 0.0_f64;
        for &n in &SIZES {
            let fft = ComplexFft::new(n);
            let mut lcg = Lcg::new(0xC0DE_0000 ^ n as u64);
            let input: Vec<Complex> = (0..n)
                .map(|_| Complex::new(lcg.next_unit(), lcg.next_unit()))
                .collect();
            let mut buf = input.clone();
            fft.forward(&mut buf);
            fft.inverse(&mut buf);
            worst = worst.max(max_abs_diff(&buf, &input));
        }
        println!("complex roundtrip worst absolute error = {worst:e}");
        assert!(
            worst < 1e-12,
            "complex roundtrip error {worst:e} exceeds 1e-12"
        );
    }

    /// Forward then inverse recovers a real input up to the `f32` boundary
    /// quantization, which dominates the error.
    #[test]
    fn real_roundtrip_recovers_the_input() {
        let mut worst = 0.0_f32;
        for &n in &SIZES {
            let fft = RealFft::new(n);
            let mut lcg = Lcg::new(0xBEEF_0000 ^ n as u64);
            let input: Vec<f32> = (0..n).map(|_| lcg.next_unit() as f32).collect();
            let mut spectrum = Vec::new();
            fft.forward(&input, &mut spectrum);
            let mut recovered = Vec::new();
            fft.inverse(&spectrum, &mut recovered);
            assert_eq!(recovered.len(), n);
            for (a, b) in recovered.iter().zip(&input) {
                worst = worst.max((a - b).abs());
            }
        }
        println!("real roundtrip worst absolute error = {worst:e}");
        assert!(worst < 1e-6, "real roundtrip error {worst:e} exceeds 1e-6");
    }

    /// The zeroth twiddle is exactly `1 + 0i`, which is what keeps the trivial
    /// butterflies and the DC bin bit-exact.
    #[test]
    fn the_zeroth_twiddle_is_exactly_unity() {
        let twiddles = build_twiddles(512);
        assert_eq!(twiddles[0].re.to_bits(), 1.0_f64.to_bits());
        assert_eq!(twiddles[0].im.to_bits(), 0.0_f64.to_bits());
    }

    /// The deterministic trig agrees with the platform trig well within the
    /// precision the twiddle table needs, over the angle range the table spans.
    #[test]
    fn deterministic_trig_matches_the_platform_over_the_twiddle_range() {
        let mut worst = 0.0_f64;
        let steps = 100_000;
        for i in 0..=steps {
            let theta = -TWO_PI * (i as f64) / (steps as f64); // [-2*pi, 0]
            worst = worst.max((det_cos(theta) - theta.cos()).abs());
            worst = worst.max((det_sin(theta) - theta.sin()).abs());
        }
        println!("deterministic trig worst absolute error vs platform = {worst:e}");
        assert!(
            worst < 1e-13,
            "deterministic trig error {worst:e} too large"
        );
    }

    /// Pure functions: identical inputs yield identical bits, so the twiddle
    /// table is reproducible by construction.
    #[test]
    fn deterministic_trig_is_reproducible() {
        for &x in &[0.0, -0.25, -1.0, -PI, -TWO_PI, -2.5, -3.1] {
            assert_eq!(det_sin(x).to_bits(), det_sin(x).to_bits());
            assert_eq!(det_cos(x).to_bits(), det_cos(x).to_bits());
        }
    }

    /// Two fresh transforms over the same input produce bit-identical output:
    /// the run-to-run half of the determinism guarantee. The committed golden
    /// below pins the cross-platform, cross-toolchain half.
    #[test]
    fn the_transform_is_deterministic_across_runs() {
        let n = 512;
        let fft = RealFft::new(n);
        let mut lcg = Lcg::new(0x5EED_0FF7);
        let input: Vec<f32> = (0..n).map(|_| lcg.next_unit() as f32).collect();

        let mut first = Vec::new();
        fft.forward(&input, &mut first);
        let mut second = Vec::new();
        fft.forward(&input, &mut second);

        assert_eq!(first.len(), second.len());
        for (k, (a, b)) in first.iter().zip(&second).enumerate() {
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "bin {k} re");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "bin {k} im");
        }

        // The complex inverse is deterministic run to run as well: a full
        // round-trip run twice must produce bit-identical output both times.
        let cfft = ComplexFft::new(n);
        let mut first_trip: Vec<Complex> =
            input.iter().map(|&s| Complex::new(s as f64, 0.0)).collect();
        let mut second_trip = first_trip.clone();
        for trip in [&mut first_trip, &mut second_trip] {
            cfft.forward(trip);
            cfft.inverse(trip);
        }
        for (x, y) in first_trip.iter().zip(&second_trip) {
            assert_eq!(x.re.to_bits(), y.re.to_bits());
            assert_eq!(x.im.to_bits(), y.im.to_bits());
        }
    }

    /// The block-model input the golden pins: a deterministic length-512 real
    /// vector.
    fn golden_input() -> Vec<f32> {
        let mut lcg = Lcg::new(0x600D_F17E);
        (0..512).map(|_| lcg.next_unit() as f32).collect()
    }

    /// A deterministic, well-formed Hermitian spectrum of 257 bins (the `n/2 + 1`
    /// bins of a length-512 real transform) for the inverse golden: arbitrary
    /// bins from the LCG, with the DC and Nyquist bins made real as a valid real
    /// spectrum requires. This is not derived from [`golden_input`], so the
    /// inverse golden pins the inverse path on independent input rather than
    /// merely re-checking the roundtrip.
    fn golden_spectrum() -> Vec<Complex> {
        let m = 256;
        let mut lcg = Lcg::new(0xBEEF_CAFE);
        (0..=m)
            .map(|k| {
                let re = 4.0 * lcg.next_unit();
                let im = if k == 0 || k == m {
                    0.0
                } else {
                    4.0 * lcg.next_unit()
                };
                Complex::new(re, im)
            })
            .collect()
    }

    /// Prints the real spectrum as a pasteable `const`, the regeneration path for
    /// the golden vector below. Each bin contributes its real then imaginary
    /// part, formatted with the round-tripping `Debug` representation.
    fn print_golden(spectrum: &[Complex]) {
        let mut body = String::new();
        let mut count = 0;
        for bin in spectrum {
            for part in [bin.re, bin.im] {
                if count % 4 == 0 {
                    body.push_str("\n    ");
                }
                body.push_str(&format!("{part:?}, "));
                count += 1;
            }
        }
        println!("const EXPECTED_FFT_GOLDEN: &[f64] = &[{body}\n];");
    }

    /// The committed bit-exact golden: the length-512 real transform of a fixed
    /// deterministic input, pinned to the bit via `to_bits`. This is the artifact
    /// that catches an accidental numeric change (a twiddle slip, a butterfly
    /// reorder, a fused multiply-add) as a bit mismatch. Regenerate deliberately
    /// via
    /// `DECIBRI_REGEN_AEC_FFT_GOLDEN=1 cargo test fft_matches_the_bit_exact_golden -- --nocapture`,
    /// paste the printed const, then rerun without the variable to confirm.
    #[test]
    fn fft_matches_the_bit_exact_golden() {
        let fft = RealFft::new(512);
        let mut out = Vec::new();
        fft.forward(&golden_input(), &mut out);

        if std::env::var("DECIBRI_REGEN_AEC_FFT_GOLDEN").is_ok() {
            print_golden(&out);
            panic!(
                "DECIBRI_REGEN_AEC_FFT_GOLDEN is set: copy the printed const into \
                 src/fft.rs and rerun without the variable"
            );
        }

        assert_eq!(
            out.len() * 2,
            EXPECTED_FFT_GOLDEN.len(),
            "golden length changed: regenerate (see DECIBRI_REGEN_AEC_FFT_GOLDEN)"
        );
        for (k, bin) in out.iter().enumerate() {
            let expected_re = EXPECTED_FFT_GOLDEN[2 * k];
            let expected_im = EXPECTED_FFT_GOLDEN[2 * k + 1];
            assert_eq!(
                bin.re.to_bits(),
                expected_re.to_bits(),
                "golden re mismatch at bin {k}: got {}, expected {}. A bit mismatch \
                 is a determinism leak (FMA, reorder, twiddle drift) or an \
                 unacknowledged change; investigate before regenerating.",
                bin.re,
                expected_re
            );
            assert_eq!(
                bin.im.to_bits(),
                expected_im.to_bits(),
                "golden im mismatch at bin {k}: got {}, expected {}.",
                bin.im,
                expected_im
            );
        }
    }

    /// The real transform of [`golden_input`], pinned bit for bit. Interleaved
    /// real then imaginary part per bin, 257 bins. Regenerate via
    /// `DECIBRI_REGEN_AEC_FFT_GOLDEN=1`.
    const EXPECTED_FFT_GOLDEN: &[f64] = &[
        -10.425267696380615,
        0.0,
        -2.5037175545669266,
        -3.3684983997133697,
        -11.059800079555695,
        -5.4825821445596965,
        -14.672774767788665,
        4.76340291983234,
        -24.263671128777958,
        0.9349155016232966,
        -3.7136550981415644,
        -3.695446807048178,
        -12.307264649143313,
        3.3968423249424777,
        -16.631341322474324,
        -12.946303453799244,
        10.03940519616435,
        11.23325718946845,
        -17.952491755968587,
        3.632668118303812,
        5.003304116090249,
        6.157843689021387,
        7.436450951340566,
        1.2063702344736633,
        12.053560658706719,
        -2.2265764634793808,
        -13.779544380660939,
        0.7863646474894281,
        6.458662859217743,
        10.086031369191831,
        -1.1659555308678344,
        7.9254708128406985,
        -5.151359063663131,
        8.22027333101176,
        19.857505335728433,
        4.993072012401091,
        5.431082746312782,
        3.8116579978271625,
        1.5321427329957835,
        1.898780652159155,
        -15.33433343950332,
        -1.5512269372641523,
        2.4870088079985457,
        -10.313568153693154,
        10.860296970769992,
        2.8185803211770697,
        3.3283111183850647,
        8.51075465621852,
        -8.761411410011478,
        10.067154902207221,
        3.5409479150246277,
        -8.301082874324358,
        8.176172339294865,
        -3.3003606429869534,
        -5.771012651305721,
        -17.087241929962467,
        -2.1714334156112116,
        -8.619792088775895,
        -5.603023536781756,
        -11.916633571498,
        -8.420448630501985,
        -4.323609120804191,
        22.689448404998974,
        14.828909069934689,
        0.7217185783993498,
        20.944513662898125,
        1.7543952053937657,
        9.65793013336884,
        1.3630516831430413,
        13.350566690771052,
        2.6296932695750894,
        -15.511307696320511,
        -0.27996298864619784,
        7.8645332174368585,
        18.302495389669417,
        16.137836511190375,
        1.8585054151975369,
        -18.190331796677444,
        4.116821964309178,
        -13.565595743400898,
        -2.5034409206631114,
        -9.44168190679325,
        -6.4749243996100425,
        -4.718626115665389,
        -3.7077120768522875,
        -7.191453927092409,
        -7.617044104282105,
        0.6997235409704272,
        10.828569888990648,
        3.8499244444947176,
        -18.971646445242374,
        -5.153371529825306,
        8.232634595632387,
        9.041437857873326,
        14.465010576705009,
        12.206883119474824,
        7.323869507619502,
        17.45388029456781,
        15.757591338016734,
        -8.369362595274925,
        -13.306780747510405,
        4.736618378077534,
        10.918165999746872,
        -7.590358918950343,
        -3.001739043090499,
        2.7915845800049013,
        -11.412475805979003,
        6.423017470628148,
        5.66966914983071,
        -0.17110047324104016,
        -7.649733632626386,
        6.318752905436831,
        3.2024336906482276,
        2.426354325199159,
        -19.401511450791222,
        -2.6380106078565926,
        1.868717340319531,
        2.238438635059211,
        0.6512213851745687,
        -7.0002484241260206,
        -5.114316330127553,
        -1.0602407228200494,
        1.907699426485351,
        2.5047688313051877,
        -7.285030873838027,
        22.66220111651395,
        22.113802685079538,
        6.301580706393954,
        -6.214463536666237,
        0.9299343974996104,
        -11.583697354820977,
        -7.397806727349242,
        0.2893105229577553,
        -4.992555206267712,
        -8.473609289708751,
        8.251440002457311,
        13.580852735312689,
        -11.871496887856646,
        7.267049827615686,
        -5.745528691624361,
        -4.477257622235518,
        0.5805353711290442,
        15.882214286271667,
        -19.476595574212837,
        -9.996906957440146,
        6.171950128491212,
        15.915783614467196,
        -3.3122541687791553,
        -8.286081658468976,
        -3.0661159632970216,
        -0.7875449615307799,
        -7.286603786254728,
        -3.623363123700596,
        -11.64712817000269,
        9.624532956922774,
        20.75823379373834,
        3.0860589044192084,
        -10.554861071353141,
        5.268112137291704,
        7.228117370263268,
        -11.835295054492907,
        6.4873576608032515,
        -9.914191508681093,
        -0.4773514334652931,
        -11.92005654940025,
        3.96973821594482,
        -17.181428952952363,
        12.69463035895427,
        -3.807151106791534,
        -7.209699537450504,
        4.385432534871999,
        8.160890266220363,
        1.7829394919472774,
        3.224868112159012,
        3.3652381666807996,
        -7.419828880294725,
        4.201816579650011,
        -9.715886227443422,
        9.037530543474002,
        10.430874095894033,
        21.069481102297782,
        4.372929446612888,
        17.022464898686437,
        -4.672004095902994,
        11.292507415826364,
        -1.0922888052079256,
        0.22970133556171302,
        4.3574923251389075,
        12.127838526438264,
        -3.727166990195246,
        -2.3450101544625914,
        -6.441109485897158,
        12.537052015374417,
        7.021506313608919,
        -5.4342665689676615,
        19.667211517343148,
        -7.417765012708973,
        4.5823711292376785,
        -5.953995508203208,
        -8.816346423475203,
        9.56686756822472,
        -0.44341660799068094,
        -3.733558327416291,
        -5.000167096977737,
        10.645056069372782,
        3.674771583880404,
        10.108929414499716,
        1.427124242991669,
        -2.3445386488052478,
        -6.646798390564475,
        2.7902091076026716,
        -2.248265868425106,
        -6.876749694784156,
        -3.1578793243831216,
        10.161274009209547,
        -6.4683395756341895,
        4.015854867990522,
        -2.857705253774913,
        2.8094654450859338,
        12.664749906387886,
        -17.38252741220549,
        -4.1688007546126435,
        14.56373980567712,
        -14.20216083392043,
        -1.5367927313605838,
        -0.61187807170241,
        6.1056614831605325,
        0.2985156418144257,
        15.02349437938014,
        -9.933201805383103,
        -6.017949290344665,
        11.54100305550506,
        -8.312007941925668,
        2.985064631641065,
        -11.77904224317006,
        -7.114824182260179,
        3.5002550789684372,
        -7.866348145366567,
        7.851660416765556,
        7.798504461630456,
        -5.845878080200435,
        4.197546361999237,
        -11.69325107789458,
        -0.47123094480883587,
        1.1820917795892143,
        10.337519586828888,
        -13.567498270278202,
        1.5384288321072423,
        -10.477939701529722,
        1.5351870511525885,
        -4.113840518581819,
        -19.583123934050843,
        7.1592167407633625,
        -3.5741319876841176,
        4.834043578233806,
        -14.977548252100108,
        2.031334638595581,
        11.726141214370728,
        -4.050833952215384,
        12.963716315003841,
        4.656190187972404,
        8.420079606435236,
        7.245081878775508,
        11.655820008034391,
        -10.669666655512275,
        -11.974670442499177,
        -9.550713897540986,
        4.405517978555994,
        -4.150970119216206,
        -0.7900548761336967,
        -8.120578739882447,
        -16.69751126109889,
        -4.358928803186797,
        -25.62048576286295,
        -14.346777230878084,
        -7.793090998903903,
        12.029448387332081,
        -10.606394642472255,
        -17.642958338923737,
        -3.223515824744142,
        -10.161734544479376,
        -1.2385176835857774,
        -0.5652053753637314,
        5.675930294794897,
        -7.340337603733374,
        -8.474749172606188,
        -4.8421883233235326,
        -12.557167356120678,
        6.46821983689187,
        1.0657848407432997,
        0.6331317841418809,
        -12.143510968262266,
        -0.5521328062255666,
        -10.975732796067653,
        3.6678883763614314,
        0.21255824742050589,
        0.013605695108887872,
        -1.7865310999196233,
        -1.4558779447900783,
        9.478164263997527,
        5.74834567068954,
        12.858292168777457,
        -0.4841091980106651,
        -3.6711522660966125,
        -13.705240460978068,
        -4.844693299023469,
        -6.036122601199321,
        -2.69765405625019,
        -3.881038549477511,
        5.789352201443791,
        2.9809342283675795,
        -1.121633929207897,
        -2.198040867950549,
        -8.562234276875124,
        -6.1820722101871635,
        -9.98614444987414,
        0.9489702190441651,
        0.6664939504114429,
        4.443263638466959,
        1.657230369504413,
        1.3072797631393271,
        -4.064247292063863,
        -0.5278254068802324,
        -0.3064166833005548,
        4.732773302671949,
        -15.48647918878391,
        13.959908465939094,
        6.403604653465497,
        -5.642025248607724,
        0.4500872680396055,
        -11.074732417169969,
        -4.4767752240347205,
        1.5732142938744644,
        10.95670648345829,
        -5.818406617233528,
        12.06544470129293,
        -9.563850287298086,
        2.9388898989404617,
        -2.9808817023747105,
        -9.286054781962052,
        8.215673480848398,
        -0.1328572062541915,
        3.8421225285958194,
        10.295238422214819,
        -1.1196194563932023,
        -0.49360189423643286,
        -9.778432273438685,
        18.075649441937287,
        -11.204438256044405,
        -3.6185214120190636,
        -8.068583703365661,
        -3.441905263235361,
        0.9375155329993232,
        -3.4120722029461543,
        -7.82374198234726,
        8.73537916480702,
        -2.89859343055105,
        8.354176427452579,
        3.048613462547671,
        -1.2267281755338786,
        -10.38243208154702,
        -15.558506388311828,
        -5.927163200299472,
        5.708591136019397,
        -11.635588233319378,
        3.1179791847403964,
        -4.768990011350317,
        -8.041337066889959,
        -14.323939265073307,
        8.60952345044715,
        1.6223241624338005,
        -2.146516189982984,
        -5.08634422679397,
        -9.20332817721249,
        5.995558266341542,
        -6.773734825935542,
        -8.455540115573987,
        16.882887777720338,
        23.449650136952013,
        -5.353431593288276,
        1.027931197729271,
        -2.380499531217845,
        9.788903491785696,
        3.4126900175456205,
        -14.976620371414818,
        5.307894602577735,
        12.572386542244747,
        -7.726859446770408,
        0.862088587493326,
        7.929844445735206,
        -0.5415028606951033,
        5.332915499457909,
        -3.1614688365469945,
        -19.09801766989424,
        4.651971091572754,
        -5.347912984029875,
        1.3380803307991687,
        -10.987710861610179,
        -2.1353248641244456,
        8.298418647165935,
        -12.570210725265033,
        14.737849703951404,
        -2.508141757583678,
        9.459822175268275,
        -5.910882436311454,
        -15.200677808885708,
        6.211624806130854,
        -8.197972147567715,
        -14.066798151656368,
        2.144992465572649,
        -14.920322092818768,
        2.3944720215227058,
        -0.6376414757682243,
        -10.202790698224813,
        2.7997639398643788,
        -12.86728707557301,
        -6.867541061232462,
        -11.806388985158808,
        2.896721420201265,
        -6.350961922591971,
        12.104798059256336,
        -3.5867716184233327,
        -11.323621588973932,
        -6.74237519741341,
        -6.115557450971634,
        6.398177207611003,
        20.68968881307755,
        3.5729517902914347,
        3.318617356515309,
        -1.959966833333878,
        -8.19686645912181,
        -8.954531073975508,
        -10.32240463895309,
        6.102313554073094,
        1.230509031370953,
        0.574140462007847,
        -4.13943924109439,
        -6.611191800910071,
        8.87570533621038,
        -11.105298905360495,
        -1.5377529398760081,
        -6.901095294559261,
        -10.408430144648904,
        3.760937210691222,
        -5.524753743398703,
        -4.0517237181027586,
        7.848987373836004,
        -2.863389822515718,
        -12.032954520273691,
        20.72229546143066,
        22.74668959438952,
        -13.403149656036113,
        1.211776155502585,
        4.611589315792162,
        -18.983695812989748,
        -6.236552550554316,
        3.5856304855808463,
        0.07074489768531755,
        -16.34613177448213,
        -5.21440778362177,
        -10.62476423200606,
        5.081295713470183,
        -3.5258313950823688,
        5.873089287951476,
        4.507389257582293,
        -10.52286520855821,
        13.456306276249729,
        -9.277214990673503,
        -7.168128203416666,
        -2.485106425408386,
        -10.929279399218993,
        -5.127692351203023,
        -12.230017939499781,
        -25.79857226331073,
        22.611983964431356,
        -5.061751347526611,
        -4.518160829902692,
        5.987935326586193,
        5.98924482132892,
        -0.28618807646236455,
        8.143093467198351,
        -21.09460845663839,
        -13.892396920277026,
        -16.52431491740026,
        -4.858493021209339,
        20.811963745286846,
        -2.904443560543931,
        -12.555216850341292,
        -15.769597992634854,
        -6.079592789796338,
        -2.7102310128604574,
        -15.70305985392158,
        -18.326077818579126,
        -4.251755411882296,
        -1.4514543527250865,
        -3.8056670872340024,
        3.5711788777605853,
        4.256694415058009,
        16.939716627084735,
        -19.31029948854878,
        -8.16757337341555,
        3.679571998566415,
        4.938392388867122,
        -18.75498579708494,
        9.440139794831792,
        0.858745643207676,
        14.984170202220929,
        -7.254719124526908,
        4.615207209063095,
        -2.4920850119942535,
        -14.585803043584114,
        3.18057962573195,
        3.04634952545166,
        0.0,
    ];

    /// Prints the real signal as a pasteable `const`, the regeneration path for
    /// the inverse golden below.
    fn print_golden_f32(signal: &[f32]) {
        let mut body = String::new();
        for (i, sample) in signal.iter().enumerate() {
            if i % 6 == 0 {
                body.push_str("\n    ");
            }
            body.push_str(&format!("{sample:?}, "));
        }
        println!("const EXPECTED_IFFT_GOLDEN: &[f32] = &[{body}\n];");
    }

    /// The committed bit-exact inverse golden: the length-512 real inverse of a
    /// fixed deterministic Hermitian spectrum ([`golden_spectrum`]), pinned to the
    /// bit via `to_bits`. It closes the inverse path (the conjugated twiddles, the
    /// `1/n` scale, and the inverse recombination) under the same cross-platform
    /// guarantee the forward golden gives the forward path. Regenerate via
    /// `DECIBRI_REGEN_AEC_FFT_GOLDEN=1 cargo test inverse_matches_the_bit_exact_golden -- --nocapture`.
    #[test]
    fn inverse_matches_the_bit_exact_golden() {
        let fft = RealFft::new(512);
        let mut out = Vec::new();
        fft.inverse(&golden_spectrum(), &mut out);

        if std::env::var("DECIBRI_REGEN_AEC_FFT_GOLDEN").is_ok() {
            print_golden_f32(&out);
            panic!(
                "DECIBRI_REGEN_AEC_FFT_GOLDEN is set: copy the printed const into \
                 src/fft.rs and rerun without the variable"
            );
        }

        assert_eq!(
            out.len(),
            EXPECTED_IFFT_GOLDEN.len(),
            "inverse golden length changed: regenerate (see DECIBRI_REGEN_AEC_FFT_GOLDEN)"
        );
        for (i, (got, expected)) in out.iter().zip(EXPECTED_IFFT_GOLDEN).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "inverse golden mismatch at {i}: got {got}, expected {expected}. A bit \
                 mismatch is a determinism leak or an unacknowledged change; \
                 investigate before regenerating."
            );
        }
    }

    /// The length-512 real inverse of [`golden_spectrum`], pinned bit for bit.
    /// Regenerate via `DECIBRI_REGEN_AEC_FFT_GOLDEN=1`.
    const EXPECTED_IFFT_GOLDEN: &[f32] = &[
        0.06302998,
        0.28812072,
        -0.08174661,
        -0.12678152,
        0.008283026,
        -0.12255078,
        0.011952432,
        -0.02889483,
        0.41076913,
        0.023405919,
        0.20888734,
        -0.027558977,
        -0.27545974,
        -0.0032017638,
        0.19085501,
        -0.080802985,
        0.031563763,
        0.098846965,
        0.02135037,
        -0.06701913,
        -0.016346376,
        0.03915449,
        -0.123429604,
        -0.15482074,
        0.031738766,
        0.034895618,
        0.05444525,
        -0.14506051,
        0.21988067,
        0.3752621,
        -0.11865443,
        0.1718669,
        -0.03012534,
        0.119480595,
        -0.09844812,
        -0.04871595,
        0.21530238,
        0.15674153,
        -0.17307502,
        -0.044486247,
        0.26416022,
        -0.061382826,
        0.101783596,
        0.0028487449,
        -0.12666748,
        -0.036199454,
        -0.1270768,
        0.1653435,
        0.047658514,
        -0.089707494,
        -0.15346956,
        0.08895725,
        -0.19808978,
        0.095273204,
        -0.13964093,
        0.25627252,
        0.07091857,
        0.054137006,
        -0.06322579,
        -0.08793628,
        -0.14199711,
        -0.14581966,
        -0.019727496,
        0.2632647,
        0.30934766,
        -0.02646449,
        0.018417075,
        0.017334284,
        0.16503634,
        -0.08410108,
        0.14303684,
        0.015575036,
        0.026512358,
        -0.16606589,
        0.10412953,
        -0.0093009565,
        -0.1345609,
        0.03535923,
        -0.18730557,
        0.010563686,
        -0.043718357,
        0.15596132,
        -0.04425003,
        0.017131412,
        0.025909122,
        0.106244095,
        -0.079515465,
        0.07000335,
        -0.13604157,
        0.030143583,
        0.03286438,
        -0.21633711,
        0.14649147,
        0.12728305,
        -0.08532618,
        -0.0054530017,
        -0.18653214,
        0.24166892,
        0.06393635,
        -0.16626671,
        -0.12761164,
        -0.07785929,
        0.1860316,
        0.032817382,
        -0.08253856,
        0.12284774,
        -0.025635405,
        0.22658634,
        -0.0054343278,
        0.26166594,
        0.1264827,
        -0.08053098,
        -0.09309249,
        -0.19973479,
        0.1231683,
        0.26838586,
        0.13463072,
        0.033989612,
        0.13009483,
        -0.12973939,
        -0.019023117,
        -0.12842445,
        0.13149074,
        0.00045310188,
        -0.12026049,
        0.12153276,
        -0.26311582,
        0.18500696,
        -0.122058794,
        -0.053380966,
        -0.14633405,
        0.049809914,
        -0.19870923,
        -0.031149423,
        -0.2871646,
        0.15845479,
        0.06023968,
        0.017923977,
        -0.36468023,
        0.09755819,
        0.13070305,
        0.23949715,
        0.2008224,
        -0.2192139,
        -0.03247198,
        0.17869587,
        0.03711365,
        -0.10568945,
        -0.05119626,
        -0.0035702726,
        -0.08744795,
        0.042224973,
        -0.049913634,
        0.15483911,
        0.08029666,
        0.05928207,
        -0.17267345,
        -0.124972984,
        0.10515226,
        0.07885628,
        0.13576882,
        -0.14767642,
        0.10325609,
        -0.029838197,
        -0.22696811,
        -0.29835823,
        -0.19199501,
        -0.17058912,
        -0.18338269,
        0.13395175,
        0.07661596,
        -0.07443781,
        -0.012015797,
        0.11706473,
        0.2527737,
        0.010065079,
        -0.18110579,
        -0.07396812,
        -0.1339353,
        -0.048784688,
        0.09782229,
        -0.075311854,
        0.106546074,
        -0.25615665,
        0.082506076,
        -0.04745256,
        -0.021574073,
        -0.2246283,
        -0.1979268,
        -0.0726023,
        0.32488808,
        0.08563068,
        0.29732788,
        0.07600513,
        -0.1523171,
        0.1581961,
        0.15141404,
        0.11498424,
        0.039380386,
        0.0073903184,
        0.0879988,
        -0.15182035,
        -0.0740957,
        -0.093049034,
        0.009219403,
        0.29165927,
        -0.04401558,
        -0.06579706,
        -0.28930902,
        -0.21105827,
        -0.18892483,
        0.124010235,
        0.022520367,
        -0.15404502,
        -0.08827899,
        -0.08275043,
        0.0086016,
        0.053393498,
        0.14816388,
        0.06735632,
        -0.08570139,
        -0.09182744,
        -0.029441413,
        -0.23838913,
        -0.15675679,
        -0.12918366,
        0.23156841,
        -0.09197939,
        0.056168955,
        0.09330649,
        0.15974198,
        -0.24252942,
        0.10886095,
        0.236655,
        0.0672414,
        0.19270697,
        0.103776865,
        -0.11960768,
        0.11179299,
        0.031228624,
        0.24651277,
        -0.10790567,
        -0.0032621324,
        0.035310242,
        0.11643112,
        0.19377182,
        0.021227296,
        -0.04623683,
        -0.01301583,
        0.023138093,
        0.0007499156,
        0.21508984,
        -0.10562917,
        0.026313178,
        0.03150969,
        0.015863277,
        -0.124725096,
        -0.19038822,
        0.10149802,
        -0.17143986,
        -0.006970848,
        -0.11072391,
        -0.24779621,
        0.029524034,
        -0.27560565,
        0.17441444,
        -0.14213172,
        0.19400555,
        -0.056991495,
        0.10454619,
        0.2180568,
        -0.2960209,
        -0.21019517,
        -0.031631526,
        0.07307846,
        0.09593478,
        0.06483016,
        0.067742236,
        0.006161654,
        0.06312376,
        0.004825272,
        -0.13307661,
        0.09749614,
        0.13660534,
        -0.04773135,
        0.32305208,
        0.042795766,
        -0.11622949,
        -0.14590573,
        -0.08863321,
        -0.045945108,
        -0.02945453,
        -0.059908472,
        -0.052364595,
        0.02242849,
        -0.12887393,
        0.30974275,
        0.085546695,
        -0.115155004,
        0.12162058,
        -0.19300707,
        0.022068417,
        0.27438444,
        -0.26453465,
        -0.33038083,
        -0.29103372,
        0.109188564,
        -0.06258128,
        0.099031255,
        -0.25234216,
        0.16688538,
        -0.099951446,
        -0.4039805,
        -0.19164535,
        -0.068562746,
        -0.06440827,
        -0.0017451923,
        -0.024835054,
        0.034130856,
        -0.113814585,
        0.1269454,
        0.22418542,
        0.120283455,
        -0.053925473,
        0.16924313,
        -0.069839515,
        0.2760269,
        0.23023662,
        0.20600939,
        -0.081870906,
        -0.09436051,
        0.04776874,
        0.07668536,
        -0.19603519,
        -0.09916764,
        -0.047397375,
        -0.13768312,
        0.025087323,
        0.09191404,
        0.104929835,
        0.22234763,
        0.14875872,
        0.11324102,
        0.15341575,
        0.32417646,
        -0.08439456,
        0.24863549,
        0.22729042,
        0.13246588,
        0.002160263,
        0.20307529,
        -0.028480949,
        0.1679456,
        0.11744453,
        0.11304163,
        0.07495274,
        -0.062171254,
        0.19998299,
        0.07302651,
        -0.040018033,
        0.0799946,
        -0.33158705,
        -0.1443476,
        0.020299247,
        -0.27476287,
        -0.069917254,
        0.14657412,
        -0.08356746,
        0.27333304,
        0.15148507,
        -0.04208513,
        0.017223785,
        -0.0659876,
        -0.16780145,
        0.00040329492,
        0.020424658,
        0.08838225,
        -0.027995933,
        -0.15820973,
        0.04192641,
        0.034187388,
        -0.13016011,
        0.15541852,
        0.24725436,
        -0.050475594,
        -0.049611147,
        -0.038542446,
        -0.3303179,
        0.11785414,
        -0.007913675,
        -0.03897911,
        0.004900466,
        -0.111238405,
        0.17423156,
        0.14368176,
        -0.18765981,
        -0.05908763,
        -0.006123665,
        -0.10161777,
        0.08476822,
        -0.18296136,
        0.16038841,
        -0.02795675,
        0.1172937,
        -0.07307826,
        0.1058133,
        -0.031586904,
        -0.014693714,
        0.2718757,
        0.07717528,
        0.14502083,
        -0.012951131,
        0.2432431,
        0.022940565,
        -0.00842255,
        -0.012050823,
        -0.041451104,
        -0.050382026,
        0.10879702,
        -0.097439945,
        0.29300573,
        -0.08377587,
        0.056109186,
        -0.06250436,
        0.08341857,
        -0.07259322,
        -0.11416027,
        0.15832944,
        -0.044282034,
        -0.2341093,
        0.029575896,
        -0.10938703,
        -0.29932272,
        -0.11770287,
        0.15118104,
        -0.20086038,
        0.28363177,
        0.41942802,
        0.15250325,
        -0.06995579,
        0.07594987,
        0.06124238,
        -0.1319902,
        -0.06996808,
        -0.27702612,
        0.17880693,
        -0.07026561,
        0.04902329,
        0.005045132,
        -0.15587755,
        -0.018710377,
        0.15517613,
        -0.3928552,
        -0.010492233,
        0.207549,
        -0.022543438,
        -0.10349851,
        0.1476347,
        0.07263461,
        -0.12787828,
        0.12933762,
        0.1586354,
        -0.02021348,
        -0.061450716,
        -0.2519217,
        0.20490032,
        -0.08500736,
        0.30513832,
        -0.11347248,
        0.17540593,
        0.006166844,
        -0.18100356,
        0.03289541,
        0.17417568,
        0.001467443,
        0.06610873,
        -0.116976246,
        0.08520695,
        0.026306167,
        -0.24389423,
        -0.022958431,
        0.1624419,
        0.13562275,
        -0.063282594,
        0.23527652,
        0.080347255,
        -0.04275136,
        -0.011706013,
        0.006350756,
        -0.12657735,
        0.034005634,
        0.07363511,
        -0.25337377,
        -0.20239156,
        -0.085269526,
        -0.08745511,
        0.0365433,
        0.19535263,
        0.10304742,
        0.023385024,
        -0.024083063,
        0.12509824,
        0.17468189,
        0.0171946,
        -0.17246693,
        0.050104104,
        0.047678027,
        0.26170132,
        0.16858879,
        0.21090238,
        0.014294049,
        -0.058475353,
    ];
}
