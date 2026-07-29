//! Capture-continuity suite.
//!
//! Fixtures are minted here from the crate's deterministic generator: no
//! external files, no licensed data, no network.

use decibri_aec::{Aec, AecConfig, CaptureContinuity, DelayStatus};

/// One near-end block, the cadence every example and the benchmark use.
const TURN: usize = 256;
const RATE: u32 = 16_000;
/// The default tail. 200 ms is 3200 samples at 16 kHz.
const TAIL_MS: u16 = 200;
/// The bulk echo delay every case is synthesized around.
const BULK_MS: usize = 100;

/// Seconds of matched streaming before the loss.
const CONVERGE_SECONDS: usize = 8;
/// Seconds of matched streaming after the loss.
const AFTER_SECONDS: usize = 5;

/// A deterministic linear congruential generator.
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

/// Speech-shaped far end: broadband noise under a syllabic amplitude envelope,
/// with segment lengths and levels both drawn from the generator.
fn speech_like(len: usize) -> Vec<f32> {
    let mut carrier = Lcg(0x1234_5678);
    let mut shape = Lcg(0x00C0_FFEE);
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

/// A reverberant echo path spanning `span_ms`, exponentially decaying and
/// normalized to a fixed whole-path gain.
fn echo_path(span_ms: usize) -> Vec<f32> {
    let span = (span_ms * RATE as usize) / 1000;
    let mut rng = Lcg(0x0BAD_F00D);
    let mut taps = vec![0.0_f32; span];
    for (index, tap) in taps.iter_mut().enumerate() {
        let decay = (-6.0 * index as f32 / span as f32).exp();
        *tap = decay * rng.next_f32();
    }
    let norm: f32 = taps.iter().map(|t| t * t).sum::<f32>().sqrt();
    for tap in taps.iter_mut() {
        *tap /= norm;
    }
    taps
}

/// The near end: the far end through the echo path, delayed by `bulk`, under a
/// low noise floor.
fn near_end(far: &[f32], path: &[f32], bulk: usize) -> Vec<f32> {
    let mut floor = Lcg(0x0F10_0F10);
    (0..far.len())
        .map(|i| {
            let mut echo = 0.0_f32;
            for (tap, &coeff) in path.iter().enumerate() {
                let lag = bulk + tap;
                if i >= lag {
                    echo += coeff * far[i - lag];
                }
            }
            0.5 * echo + 0.001 * floor.next_f32()
        })
        .collect()
}

fn energy(samples: &[f32]) -> f64 {
    samples.iter().map(|&s| s as f64 * s as f64).sum()
}

/// Echo-return-loss enhancement of `residual` against `mic`, in decibels.
fn erle_db(mic: &[f32], residual: &[f32]) -> f64 {
    let residual_energy = energy(residual);
    if residual_energy <= 0.0 {
        return f64::INFINITY;
    }
    let mic_energy = energy(mic);
    if mic_energy <= 0.0 {
        return f64::NAN;
    }
    10.0 * (mic_energy / residual_energy).log10()
}

/// What the host does about the capture loss.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Host {
    /// Says nothing, leaving the engine to infer the seam. The automatic path.
    Silent,
    /// Declares the loss with the sample count it knows.
    Declares,
    /// Declares the loss without a count, which is all many platforms report.
    DeclaresUncounted,
}

/// What one case measured.
#[derive(Debug)]
struct Outcome {
    /// ERLE over the two seconds before the loss: the converged baseline.
    erle_before: f64,
    /// ERLE over the last two seconds of the run: where it ended up.
    erle_settled: f64,
    /// Automatic re-anchors.
    reanchors: u64,
    /// Applied host-declared discontinuities.
    declared: u64,
    /// Near samples the host reported lost.
    reported_lost: u64,
    /// The alignment before and after the loss.
    delay_before: Option<usize>,
    delay_after: Option<usize>,
}

/// The far-end and near-end pair one loss size needs, synthesized once.
struct Pair {
    far: Vec<f32>,
    near: Vec<f32>,
    lag: usize,
}

fn synthesize(loss_ms: usize) -> Pair {
    let lag = ms(loss_ms);
    let converge = RATE as usize * CONVERGE_SECONDS;
    let total = converge + lag + RATE as usize * AFTER_SECONDS + TURN * 4;
    let far = speech_like(total);
    let near = near_end(&far, &echo_path(60), (BULK_MS * RATE as usize) / 1000);
    Pair { far, near, lag }
}

/// Converges the engine, loses `lag` near samples the far stream did not lose,
/// then keeps streaming and measures where the cancellation ends up.
fn run_case(pair: &Pair, host: Host) -> Outcome {
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let lag = pair.lag;
    let converge = RATE as usize * CONVERGE_SECONDS;
    let total = pair.far.len();
    let far = &pair.far;
    let near = &pair.near;

    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::with_capacity(total);
    let mut mic_kept: Vec<f32> = Vec::new();

    // Feed then process, equal sizes.
    let mut cursor = 0usize;
    while cursor + TURN <= converge {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out)
            .expect("process succeeds");
        mic_kept.extend_from_slice(&near[cursor..cursor + TURN]);
        cursor += TURN;
    }
    let converged_mark = out.len();
    let delay_before = aec.metrics().delay_samples;
    assert!(
        matches!(aec.metrics().delay.status, DelayStatus::Locked(_)),
        "the case is only meaningful once a lock stands: {:?}",
        aec.metrics().delay.status
    );

    // The capture loss: the renderer keeps feeding, the capture side loses
    // `lag` samples outright. Those near samples are never processed.
    aec.feed_reference(&far[cursor..cursor + lag]);
    cursor += lag;

    // Where a real host learns of the hole: between the interrupted callback
    // and the next one.
    match host {
        Host::Silent => {}
        Host::Declares => aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(lag as u64),
        }),
        Host::DeclaresUncounted => {
            aec.declare_capture_continuity(CaptureContinuity::Discontinuity { lost_samples: None })
        }
    }

    while cursor + TURN <= total {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out)
            .expect("process succeeds");
        mic_kept.extend_from_slice(&near[cursor..cursor + TURN]);
        cursor += TURN;
    }
    aec.flush(&mut out).expect("flush succeeds");

    let metrics = aec.metrics();
    let window = RATE as usize * 2;
    let limit = out.len().min(mic_kept.len());
    let base_lo = converged_mark.saturating_sub(window);
    let tail_lo = limit.saturating_sub(window);

    Outcome {
        erle_before: erle_db(
            &mic_kept[base_lo..converged_mark.min(limit)],
            &out[base_lo..converged_mark.min(limit)],
        ),
        erle_settled: erle_db(&mic_kept[tail_lo..limit], &out[tail_lo..limit]),
        reanchors: metrics.reference_reanchors,
        declared: metrics.capture_discontinuities,
        reported_lost: metrics.capture_samples_lost,
        delay_before,
        delay_after: metrics.delay_samples,
    }
}

/// Samples for a duration in milliseconds at the fixture rate.
fn ms(value: usize) -> usize {
    (value * RATE as usize) / 1000
}

/// The floor a recovered case must clear.
const RECOVERED_FLOOR_DB: f64 = 40.0;

/// How far apart the declared and inferred recoveries may land, in decibels.
const RECOVERY_AGREEMENT_DB: f64 = 3.0;

/// A loss below the filter tail.
fn below_tail_case(loss_ms: usize) {
    let pair = synthesize(loss_ms);
    assert!(
        pair.lag < ms(TAIL_MS as usize),
        "this case must sit below the tail to be testing anything"
    );

    let silent = run_case(&pair, Host::Silent);
    let declared = run_case(&pair, Host::Declares);

    assert!(
        silent.erle_before > RECOVERED_FLOOR_DB,
        "{loss_ms} ms: the run must converge before the loss, got {:.2} dB",
        silent.erle_before
    );
    assert_eq!(
        silent.reanchors, 1,
        "{loss_ms} ms: the silent host's loss must be inferred exactly once"
    );
    assert_eq!(
        silent.declared, 0,
        "{loss_ms} ms: nothing was host-declared, so the host-declared counter \
         must stay at zero"
    );
    assert!(
        silent.erle_settled > RECOVERED_FLOOR_DB,
        "{loss_ms} ms: an inferred loss must re-anchor and recover, got {:.2} dB",
        silent.erle_settled
    );
    assert!(
        declared.erle_settled > RECOVERED_FLOOR_DB,
        "{loss_ms} ms: a declared loss must re-anchor and recover, got {:.2} dB \
         (silent {:.2} dB)",
        declared.erle_settled,
        silent.erle_settled
    );
    assert!(
        (declared.erle_settled - silent.erle_settled).abs() < RECOVERY_AGREEMENT_DB,
        "{loss_ms} ms: declared and inferred must reach the same place, got \
         {:.2} dB declared against {:.2} dB inferred",
        declared.erle_settled,
        silent.erle_settled
    );
    assert_eq!(
        declared.declared, 1,
        "{loss_ms} ms: one applied declaration"
    );
    assert_eq!(
        declared.reanchors, 0,
        "{loss_ms} ms: a declared loss must not also produce an inferred re-anchor"
    );
    assert_eq!(
        declared.reported_lost, pair.lag as u64,
        "{loss_ms} ms: the reported loss is carried through to the metrics"
    );
}

/// A quarter of the tail.
#[test]
fn an_undeclared_50_ms_capture_loss_is_inferred_and_recovers() {
    below_tail_case(50);
}

/// Three quarters of the tail.
#[test]
fn an_undeclared_150_ms_capture_loss_is_inferred_and_recovers() {
    below_tail_case(150);
}

/// A capture loss LONGER than the tail.
fn above_tail_case(loss_ms: usize) {
    let pair = synthesize(loss_ms);
    assert!(
        pair.lag > ms(TAIL_MS as usize),
        "this case must sit above the tail to be testing anything"
    );

    let silent = run_case(&pair, Host::Silent);
    let declared = run_case(&pair, Host::Declares);

    assert_eq!(
        silent.reanchors, 1,
        "{loss_ms} ms: the inference re-anchors an above-tail loss exactly once"
    );
    assert!(
        silent.erle_settled > RECOVERED_FLOOR_DB,
        "{loss_ms} ms: the silent path must recover, got {:.2} dB",
        silent.erle_settled
    );
    assert!(
        declared.erle_settled > RECOVERED_FLOOR_DB,
        "{loss_ms} ms: the declared path must recover too, got {:.2} dB",
        declared.erle_settled
    );
    assert_eq!(
        declared.reanchors, 0,
        "{loss_ms} ms: a declared loss must not also produce an inferred re-anchor"
    );
    assert_eq!(
        declared.declared, 1,
        "{loss_ms} ms: one applied declaration"
    );
    assert!(
        (declared.erle_settled - silent.erle_settled).abs() < RECOVERY_AGREEMENT_DB,
        "{loss_ms} ms: a declared loss and a silent one must reach the same \
         place: {:.2} dB declared against {:.2} dB inferred",
        declared.erle_settled,
        silent.erle_settled
    );
}

/// Half a tail past the boundary.
#[test]
fn a_declared_300_ms_capture_loss_matches_the_automatic_re_anchor() {
    above_tail_case(300);
}

/// Four tails past the boundary.
#[test]
fn a_declared_800_ms_capture_loss_matches_the_automatic_re_anchor() {
    above_tail_case(800);
}

/// A host that knows a hole exists but not how big it is is served identically.
/// Only the reported-loss metric differs.
#[test]
fn a_declared_loss_with_no_sample_count_recovers_identically() {
    let pair = synthesize(150);
    let sized = run_case(&pair, Host::Declares);
    let uncounted = run_case(&pair, Host::DeclaresUncounted);

    assert!(
        uncounted.erle_settled > RECOVERED_FLOOR_DB,
        "an uncounted declaration must recover, got {:.2} dB",
        uncounted.erle_settled
    );
    assert_eq!(
        format!("{:.2}", uncounted.erle_settled),
        format!("{:.2}", sized.erle_settled),
        "the count is informational; it must not change the alignment"
    );
    assert_eq!(uncounted.declared, 1);
    assert_eq!(
        uncounted.reported_lost, 0,
        "a host that supplied no count contributes nothing to the loss total"
    );
    assert_eq!(sized.reported_lost, pair.lag as u64);

    assert!(
        sized.delay_before.is_some(),
        "the case converges to a lock before the loss"
    );
    assert_eq!(
        sized.delay_after, sized.delay_before,
        "the standing delay is unchanged across the loss"
    );
}

// ---- Ageing -----------------------------------------------------------------

/// A caller driving one continuous stream, so a case can compose segments of
/// different shapes (matched turns, one oversized block, an undeclared loss)
/// and measure the cancellation between them.
struct Stream<'a> {
    aec: Aec,
    far: &'a [f32],
    near: &'a [f32],
    out: Vec<f32>,
    /// The near samples actually handed over, which is what the ERLE is
    /// measured against.
    mic: Vec<f32>,
    cursor: usize,
}

impl<'a> Stream<'a> {
    fn new(far: &'a [f32], near: &'a [f32]) -> Stream<'a> {
        let mut config = AecConfig::default();
        config.sample_rate = RATE;
        config.tail_ms = TAIL_MS;
        Stream {
            aec: Aec::new(config).expect("configuration is valid"),
            far,
            near,
            out: Vec::new(),
            mic: Vec::new(),
            cursor: 0,
        }
    }

    /// One turn at whatever granularity the caller is using: `len` samples of
    /// reference in, the matching `len` samples of capture processed.
    fn turn(&mut self, len: usize) {
        let end = self.cursor + len;
        self.aec.feed_reference(&self.far[self.cursor..end]);
        self.aec
            .process(&self.near[self.cursor..end], &mut self.out)
            .expect("process succeeds");
        self.mic.extend_from_slice(&self.near[self.cursor..end]);
        self.cursor = end;
    }

    /// Matched turns at the ordinary cadence for `samples` of stream.
    fn matched(&mut self, samples: usize) {
        let end = self.cursor + samples;
        while self.cursor + TURN <= end {
            self.turn(TURN);
        }
    }

    /// A capture loss the host never declares: the renderer keeps feeding, the
    /// capture side loses the span outright, and those near samples are never
    /// handed over.
    fn undeclared_loss(&mut self, samples: usize) {
        let end = self.cursor + samples;
        self.aec.feed_reference(&self.far[self.cursor..end]);
        self.cursor = end;
    }

    /// ERLE over the last `window` samples processed.
    fn erle_over_last(&self, window: usize) -> f64 {
        let limit = self.out.len().min(self.mic.len());
        let lo = limit.saturating_sub(window);
        erle_db(&self.mic[lo..limit], &self.out[lo..limit])
    }

    fn inferred(&self) -> u64 {
        self.aec.metrics().reference_reanchors
    }
}

/// This case hands over the oversized block, streams normally, and only then
/// loses capture: twice, at two sizes, with no declaration.
#[test]
fn oversized_block_does_not_suppress_later_inference() {
    /// Half a second of capture in one block.
    const HITCH: usize = 8192;
    let converge = RATE as usize * CONVERGE_SECONDS;
    let age = RATE as usize * 6;
    let settle = RATE as usize * 4;
    let window = RATE as usize * 2;
    let first = ms(50);
    let second = ms(150);
    let total = converge + HITCH + age + first + settle + second + settle + TURN * 4;

    let far = speech_like(total);
    let near = near_end(&far, &echo_path(60), (BULK_MS * RATE as usize) / 1000);
    let mut stream = Stream::new(&far, &near);

    stream.matched(converge);
    let converged = stream.erle_over_last(window);
    assert!(
        converged > RECOVERED_FLOOR_DB,
        "the run must converge before anything is asked of it, got {converged:.2} dB"
    );

    // The hitch, and then a long stretch of ordinary streaming after it.
    stream.turn(HITCH);
    stream.matched(age);
    assert_eq!(
        stream.inferred(),
        0,
        "the oversized block lost nothing and must not itself be inferred as a loss"
    );

    // A quarter of the tail, undeclared.
    stream.undeclared_loss(first);
    stream.matched(settle);
    let after_first = stream.erle_over_last(window);
    assert_eq!(
        stream.inferred(),
        1,
        "a 50 ms capture loss must still be inferred {} s after the oversized \
         block",
        age / RATE as usize
    );
    assert!(
        after_first > RECOVERED_FLOOR_DB,
        "and it must still recover, got {after_first:.2} dB"
    );

    // And again at three quarters of the tail.
    stream.undeclared_loss(second);
    stream.matched(settle);
    let after_second = stream.erle_over_last(window);
    assert_eq!(
        stream.inferred(),
        2,
        "a second undeclared loss must be inferred too"
    );
    assert!(
        after_second > RECOVERED_FLOOR_DB,
        "and recover, got {after_second:.2} dB"
    );

    assert_eq!(
        stream.aec.metrics().capture_discontinuities,
        0,
        "nothing was host-declared in this case"
    );
    println!(
        "oversized-block stream: converged {converged:.2} dB, after a 50 ms loss \
         {after_first:.2} dB, after a 150 ms loss {after_second:.2} dB"
    );
}

// ---- The caller-shape matrix -----------------------------------------------
//
// Every shape below is a legitimate transport: it feeds and reads at the same
// rate, and none of them has lost a near-end sample. None of them may produce
// an inferred seam.

/// How a caller hands reference to the engine relative to the blocks it
/// processes.
#[derive(Clone, Copy, Debug)]
enum Caller {
    /// One block in, one block out.
    Matched,
    /// A fixed reference chunk that does not divide the capture block.
    FixedChunk(usize),
    /// Reference chunks that alternate size from block to block.
    Alternating(usize, usize),
    /// Several reference chunks handed over before each process call.
    SeveralPerBlock(usize),
    /// A caller holding `ahead` samples of reference in front of the engine
    /// from the first block, and keeping that distance for the whole stream.
    BufferedAhead(usize),
    /// A caller whose reference feed creeps `per_block` samples ahead of its
    /// capture every block until it is `ahead` in front: a render clock running
    /// slightly fast, or a buffer filling gradually. Nothing is ever lost.
    RampingAhead { per_block: usize, ahead: usize },
}

impl Caller {
    fn name(self) -> String {
        let in_ms = |samples: usize| samples * 1000 / RATE as usize;
        match self {
            Caller::Matched => "matched 256/256".to_string(),
            Caller::FixedChunk(chunk) => format!("fixed {chunk}/256 chunking"),
            Caller::Alternating(a, b) => format!("alternating {a}/{b} chunks"),
            Caller::SeveralPerBlock(n) => format!("{n} reference feeds per block"),
            Caller::BufferedAhead(ahead) => {
                format!("contiguous buffered caller {} ms ahead", in_ms(ahead))
            }
            Caller::RampingAhead { per_block, ahead } => format!(
                "caller creeping {per_block}/block to {} ms ahead",
                in_ms(ahead)
            ),
        }
    }
}

/// What one transport run observed.
#[derive(Debug)]
struct Transport {
    /// Inferred capture discontinuities.
    inferred: u64,
    /// Host-declared ones.
    declared: u64,
    /// Process calls the run made.
    blocks: u64,
    /// Whether the acquisition held a lock at the end.
    locked: bool,
}

/// A far end and a near end carrying its echo through a short path.
fn transport_pair(len: usize) -> (Vec<f32>, Vec<f32>) {
    let far = speech_like(len);
    let near = near_end(&far, &echo_path(4), (BULK_MS * RATE as usize) / 1000);
    (far, near)
}

/// Drives one caller shape over a whole clip, feeding and reading at the same
/// rate throughout, and losing nothing.
fn drive(caller: Caller, far: &[f32], near: &[f32]) -> Transport {
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let len = far.len().min(near.len());
    let mut fed = 0usize;
    let mut done = 0usize;
    let mut blocks = 0u64;

    // A caller that starts out in front puts its buffer in before the first
    // block, which is exactly what a host with a pre-rolled render queue does.
    if let Caller::BufferedAhead(ahead) = caller {
        let end = ahead.min(len);
        aec.feed_reference(&far[..end]);
        fed = end;
    }

    while done + TURN <= len {
        let (target, chunk) = match caller {
            Caller::Matched => (done + TURN, TURN),
            Caller::FixedChunk(chunk) => (done + TURN, chunk),
            Caller::Alternating(a, b) => {
                (done + TURN, if blocks.is_multiple_of(2) { a } else { b })
            }
            Caller::SeveralPerBlock(n) => (done + TURN, TURN / n),
            Caller::BufferedAhead(ahead) => (done + TURN + ahead, TURN),
            Caller::RampingAhead { per_block, ahead } => (
                (done + TURN + (blocks as usize * per_block).min(ahead)),
                TURN,
            ),
        };
        while fed < target.min(len) {
            let end = (fed + chunk).min(len);
            aec.feed_reference(&far[fed..end]);
            fed = end;
        }
        aec.process(&near[done..done + TURN], &mut out)
            .expect("process succeeds");
        done += TURN;
        blocks += 1;
    }
    aec.flush(&mut out).expect("flush succeeds");

    let metrics = aec.metrics();
    Transport {
        inferred: metrics.reference_reanchors,
        declared: metrics.capture_discontinuities,
        blocks,
        locked: matches!(metrics.delay.status, DelayStatus::Locked(_)),
    }
}

/// Seconds of continuous streaming each caller shape is watched for.
const MATRIX_SECONDS: usize = 60;

/// Every legitimate caller shape, watched for a minute each, must produce zero
/// inferred seams.
#[test]
fn no_legitimate_caller_shape_is_ever_inferred_to_have_lost_capture() {
    let (far, near) = transport_pair(RATE as usize * MATRIX_SECONDS);
    let shapes = [
        Caller::Matched,
        Caller::FixedChunk(160),
        Caller::FixedChunk(1024),
        Caller::Alternating(128, 512),
        Caller::SeveralPerBlock(4),
        // A fifth of a second.
        Caller::BufferedAhead(ms(200)),
        Caller::BufferedAhead(ms(400)),
        Caller::RampingAhead {
            per_block: 8,
            ahead: ms(300),
        },
    ];

    let mut total_blocks = 0u64;
    let mut total_inferred = 0u64;
    for shape in shapes {
        let run = drive(shape, &far, &near);
        assert_eq!(
            run.inferred,
            0,
            "{}: inferred {} capture discontinuities from a caller that lost \
             nothing, over {} blocks",
            shape.name(),
            run.inferred,
            run.blocks
        );
        assert_eq!(run.declared, 0, "{}: nothing was declared", shape.name());
        total_blocks += run.blocks;
        total_inferred += run.inferred;
    }

    let seconds = total_blocks as f64 * TURN as f64 / RATE as f64;
    println!(
        "no false inference observed: {total_inferred} inferred seams across \
         {seconds:.1} s and {total_blocks} process calls covering {} continuous \
         caller patterns",
        shapes.len(),
    );
    assert_eq!(total_inferred, 0);
}

/// The caller's chunk partition must not change what the engine decides.
#[test]
fn the_inference_is_invariant_to_the_callers_chunk_partition() {
    let (far, near) = transport_pair(RATE as usize * 20);
    let shapes = [
        Caller::Matched,
        Caller::FixedChunk(160),
        Caller::FixedChunk(1024),
        Caller::Alternating(128, 512),
        Caller::SeveralPerBlock(4),
    ];
    let runs: Vec<(Caller, Transport)> =
        shapes.iter().map(|&s| (s, drive(s, &far, &near))).collect();

    let (first_shape, first) = &runs[0];
    assert!(first.locked, "the reference run must lock");
    for (shape, run) in &runs[1..] {
        assert_eq!(
            run.inferred,
            first.inferred,
            "{} inferred {} seams where {} inferred {}",
            shape.name(),
            run.inferred,
            first_shape.name(),
            first.inferred
        );
        assert_eq!(run.declared, 0);
        assert!(
            run.locked,
            "{}: the inference must not cost this caller its lock",
            shape.name()
        );
    }
}

/// A reference stall is the far feed pausing while capture keeps going. Nothing
/// was lost, no timeline skipped, and no seam may be forged, however long the
/// pause or however abruptly the feed catches back up afterwards.
#[test]
fn a_reference_stall_with_no_timeline_skip_forges_no_seam() {
    let (far, near) = transport_pair(RATE as usize * 20);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let mut cursor = 0usize;
    let settle = RATE as usize * 4;
    while cursor + TURN <= settle {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }
    assert_eq!(aec.metrics().reference_reanchors, 0);

    // The render side stops feeding for a second while capture continues. The
    // reads starve, which is the honest report; nothing was lost from the near
    // stream, so nothing may be inferred.
    let stall = RATE as usize;
    let resume = cursor + stall;
    while cursor + TURN <= resume {
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }
    // It resumes and catches up in one burst, then runs matched again.
    aec.feed_reference(&far[resume - stall..resume]);
    let len = far.len().min(near.len());
    while cursor + TURN <= len {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }

    let metrics = aec.metrics();
    assert_eq!(
        metrics.reference_reanchors, 0,
        "a reference stall skipped no near-end sample and must forge no seam"
    );
    assert!(
        metrics.reference_starved > 0,
        "the stall must have starved reads for this to be testing anything"
    );
}

/// A delay jump with no capture loss: the echo path moved but the transport did
/// not, so no capture discontinuity may be inferred.
#[test]
fn a_delay_jump_without_capture_loss_is_not_inferred_as_a_loss() {
    let far = speech_like(RATE as usize * 24);
    let path = echo_path(4);
    let first = near_end(&far, &path, ms(100));
    let second = near_end(&far, &path, ms(180));
    // One near stream whose echo path relocates 80 ms mid-clip, with the
    // transport perfectly matched throughout: not one near sample is lost.
    let switch = RATE as usize * 12;
    let near: Vec<f32> = first[..switch]
        .iter()
        .chain(second[switch..].iter())
        .copied()
        .collect();

    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor + TURN <= near.len() {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }

    let metrics = aec.metrics();
    assert_eq!(
        metrics.reference_reanchors, 0,
        "a delay jump is not a capture loss and must not be inferred as one"
    );
    assert_eq!(metrics.capture_discontinuities, 0);
    assert!(
        metrics.delay.tracking_moves > 0 || metrics.delay.reacquisitions > 0,
        "the jump must register as delay movement: moves {} reacquisitions {}",
        metrics.delay.tracking_moves,
        metrics.delay.reacquisitions
    );
}

/// A host that declares a loss and also presents the large lead that loss
/// produces gets exactly one seam, the host-declared one.
#[test]
fn an_explicit_declaration_with_a_large_lead_produces_no_inferred_duplicate() {
    let (far, near) = transport_pair(RATE as usize * 20);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let mut cursor = 0usize;
    let settle = RATE as usize * 6;
    while cursor + TURN <= settle {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }

    // Half a second lost, and the host says so.
    let lost = ms(500);
    aec.feed_reference(&far[cursor..cursor + lost]);
    cursor += lost;
    aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
        lost_samples: Some(lost as u64),
    });

    let len = far.len().min(near.len());
    while cursor + TURN <= len {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }

    let metrics = aec.metrics();
    assert_eq!(metrics.capture_discontinuities, 1, "one host-declared seam");
    assert_eq!(
        metrics.reference_reanchors, 0,
        "and no inferred duplicate for the same event"
    );
    assert_eq!(metrics.capture_samples_lost, lost as u64);
}

/// A declaration on every block is counted in the metrics.
#[test]
fn declaring_on_every_block_is_visible_in_the_metrics() {
    let (far, near) = transport_pair(RATE as usize * 8);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let mut cursor = 0usize;
    let len = far.len().min(near.len());
    while cursor + TURN <= len {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        aec.declare_capture_continuity(CaptureContinuity::Discontinuity { lost_samples: None });
        assert!(
            aec.metrics().capture_declaration_pending,
            "a declaration is latched until a process consumes it"
        );
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        assert!(!aec.metrics().capture_declaration_pending);
        cursor += TURN;
    }

    let metrics = aec.metrics();
    assert!(metrics.capture_discontinuities > 100);
    assert_eq!(
        metrics.capture_declarations_without_decision, metrics.capture_discontinuities,
        "no declaration was followed by an acquisition decision, so the \
         without-decision counter equals the declaration count"
    );
    assert_eq!(
        metrics.capture_declarations_without_decision_max, metrics.capture_discontinuities,
        "the high-water mark survives the run that produced it"
    );
    assert_eq!(metrics.delay_samples, None, "the alignment never converges");
}

/// A host that declares only on real events, with the acquisition reaching
/// decisions between them, does not accumulate the without-decision counter.
#[test]
fn a_declaration_the_acquisition_recovers_from_does_not_accumulate() {
    let (far, near) = transport_pair(RATE as usize * 24);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let len = far.len().min(near.len());
    let mut cursor = 0usize;
    let mut blocks = 0usize;
    let every = RATE as usize * 4 / TURN;
    while cursor + TURN <= len {
        aec.feed_reference(&far[cursor..cursor + TURN]);
        // One declaration every four seconds.
        if blocks > 0 && blocks.is_multiple_of(every) {
            aec.declare_capture_continuity(CaptureContinuity::Discontinuity { lost_samples: None });
        }
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
        blocks += 1;
    }

    let metrics = aec.metrics();
    assert!(metrics.capture_discontinuities >= 4);
    assert!(
        metrics.capture_declarations_without_decision <= 1,
        "the without-decision counter must not accumulate when decisions land \
         between declarations: got {}",
        metrics.capture_declarations_without_decision
    );
    assert!(
        metrics.delay_samples.is_some(),
        "and the alignment still converges"
    );
}

/// A caller that is matched for a while and then STEPS permanently further
/// ahead (a host that switches to a deeper render buffer mid-stream, say).
#[test]
fn permanent_lead_step_produces_one_reanchor() {
    let (far, near) = transport_pair(RATE as usize * 30);
    let mut config = AecConfig::default();
    config.sample_rate = RATE;
    config.tail_ms = TAIL_MS;
    let mut aec = Aec::new(config).expect("configuration is valid");
    let mut out = Vec::new();

    let len = far.len().min(near.len());
    let step_at = RATE as usize * 8;
    let step = ms(200);
    let mut fed = 0usize;
    let mut cursor = 0usize;
    while cursor + TURN <= len {
        // Matched, except for the one block where the caller hands over an
        // extra 200 ms and stays that far in front for the rest of the run.
        let target = if cursor >= step_at {
            cursor + TURN + step
        } else {
            cursor + TURN
        };
        while fed < target.min(len) {
            let end = (fed + TURN).min(len);
            aec.feed_reference(&far[fed..end]);
            fed = end;
        }
        aec.process(&near[cursor..cursor + TURN], &mut out).unwrap();
        cursor += TURN;
    }

    assert_eq!(
        aec.metrics().reference_reanchors,
        1,
        "a permanent lead step must cost exactly one re-anchor, not one per block"
    );
}
