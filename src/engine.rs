//! The public [`Aec`] engine, its [`AecMetrics`] snapshot, and the
//! [`CaptureContinuity`] declaration the host makes about its capture stream.
//!
//! The engine is the batteries-included entry point. It owns the messy
//! real-world parts once: the deep far-end reference ring, the absolute
//! sample-count alignment between the near and far streams, input sanitization
//! for non-finite samples, and the counters behind [`Aec::metrics`]. It selects
//! a canceller by [`AecConfig::model`] and drives it through the
//! [`EchoCanceller`] seam.

// Diagnostics shim: with the `tracing` feature enabled the emit sites below
// forward to `tracing`; without it they expand to nothing and the crate has no
// tracing dependency. The macro definitions keep the call sites identical in
// both configurations.
#[cfg(feature = "tracing")]
use tracing::{debug, warn};

#[cfg(not(feature = "tracing"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}
#[cfg(not(feature = "tracing"))]
macro_rules! warn {
    ($($arg:tt)*) => {};
}

use crate::acquire::{AcquireAction, DelayAcquirer};
use crate::canceller::{CancellerMetrics, EchoCanceller};
use crate::config::{AecConfig, AecModel, OutputTransitionPolicy};
use crate::delay::{DelayEstimate, DelayLockSource, DelayStatus, WindowSupport};
use crate::error::AecError;
use crate::ring::ReferenceRing;
use crate::tau::TauCanceller;

/// Reference ring slack beyond the modelled delay-plus-tail span, in seconds.
const RING_SLACK_SECONDS: u64 = 4;

/// How close a re-promoted lock must sit to the standing offset for the engine
/// to keep the offset and the canceller's learned state instead of adopting
/// and resetting, in milliseconds.
const RELOCK_KEEP_MS: usize = 8;

/// Process calls the lead model folds into its envelope before it will infer
/// anything at all.
const LEAD_BASELINE_BLOCKS: u32 = 32;

/// Process calls one envelope bucket spans.
const LEAD_WINDOW_BLOCKS: u32 = 128;

/// The fewest consecutive process calls a step must survive before it is
/// believed, whatever the caller.
const LEAD_STEP_BLOCKS: u32 = 4;

/// The caller's learned frontier-lead behaviour, and the inference drawn from a
/// departure from it.
///
/// The lead is `reference_frontier - expected_reference_frontier`: how far the
/// far-end feed has run ahead of where the standing alignment says the near
/// stream's next sample sits.
#[derive(Debug, Default)]
struct LeadModel {
    /// Leads folded since construction or the last reseed.
    blocks: u32,
    /// The retired bucket's `(min, max)`, once one has retired.
    previous: Option<(u64, u64)>,
    /// The filling bucket's `(min, max)`.
    current: Option<(u64, u64)>,
    /// Leads folded into the filling bucket.
    current_blocks: u32,
    /// The retired bucket's largest near block, once one has retired.
    previous_block: Option<u64>,
    /// The filling bucket's largest near block.
    current_block: Option<u64>,
    /// Consecutive observations sitting outside the bound and not draining.
    run: u32,
    /// The highest lead the standing run has seen, which a drain falls away
    /// from and a plateau does not.
    run_max: u64,
}

impl LeadModel {
    /// Returns the model to its just-constructed state, so the baseline is
    /// learned again from scratch.
    ///
    /// Called at every seam, declared or inferred, and at
    /// [`Aec::reset`]: past the seam the anchor is rebuilt onto the reference
    /// frontier, so the lead restarts at zero and every envelope learned
    /// against the old anchor describes a stream that no longer exists.
    fn reseed(&mut self) {
        *self = LeadModel::default();
    }

    /// The learned envelope over the remembered window.
    fn envelope(&self) -> Option<(u64, u64)> {
        match (self.previous, self.current) {
            (Some((a_lo, a_hi)), Some((b_lo, b_hi))) => Some((a_lo.min(b_lo), a_hi.max(b_hi))),
            (Some(one), None) | (None, Some(one)) => Some(one),
            (None, None) => None,
        }
    }

    /// How far a lead may sit above the envelope's top, or fall back from a
    /// run's high-water mark, and still be this caller's own behaviour: the
    /// span its ordinary oscillation covers, floored at the block size it
    /// delivers audio in, because the model cannot resolve a step finer than
    /// that whatever the envelope says.
    fn margin(&self) -> u64 {
        let span = self.envelope().map_or(0, |(min, max)| max - min);
        span.max(self.block())
    }

    /// The largest near block the caller has delivered over the remembered
    /// window, never zero.
    fn block(&self) -> u64 {
        let recent = match (self.previous_block, self.current_block) {
            (Some(a), Some(b)) => a.max(b),
            (Some(one), None) | (None, Some(one)) => one,
            (None, None) => 0,
        };
        recent.max(1)
    }

    /// The lead above which an observation is no longer explicable as this
    /// caller's ordinary behaviour. [`None`] until something has been learned.
    fn bound(&self) -> Option<u64> {
        let (_, max) = self.envelope()?;
        Some(max + self.margin())
    }

    /// How many consecutive outside-the-bound observations this caller must
    /// produce before a plateau is believed.
    fn required_run(&self) -> u32 {
        let blocks_to_outgrow_jitter = (self.margin() / self.block()) as u32;
        LEAD_STEP_BLOCKS.max(blocks_to_outgrow_jitter.saturating_add(3))
    }

    /// Folds one process call's lead in, and reports whether a capture
    /// discontinuity is now inferred.
    fn observe(&mut self, lead: u64, block_len: u64) -> bool {
        self.current_block = Some(match self.current_block {
            Some(max) => max.max(block_len),
            None => block_len,
        });
        let outside = self.bound().is_some_and(|bound| lead > bound);
        if !outside {
            self.run = 0;
            self.run_max = 0;
            self.fold(lead);
            return false;
        }

        self.run = self.run.saturating_add(1);
        self.run_max = self.run_max.max(lead);

        if self.run_max - lead > self.margin() {
            self.run = 1;
            self.run_max = lead;
            return false;
        }

        self.blocks >= LEAD_BASELINE_BLOCKS && self.run >= self.required_run()
    }

    /// Folds a lead the model accepts as ordinary into the sliding envelope.
    fn fold(&mut self, lead: u64) {
        self.blocks = self.blocks.saturating_add(1);
        self.current = Some(match self.current {
            Some((min, max)) => (min.min(lead), max.max(lead)),
            None => (lead, lead),
        });
        self.current_blocks += 1;
        if self.current_blocks >= LEAD_WINDOW_BLOCKS {
            self.previous = self.current;
            self.current = None;
            // The block size retires on the same rotation. It is refilled by
            // the next observation, which happens before anything reads it,
            // so the floor is never momentarily absent while a bucket is empty.
            self.previous_block = self.current_block;
            self.current_block = None;
            self.current_blocks = 0;
        }
    }
}

/// What happened to the capture (near-end) stream between two
/// [`Aec::process`] calls, as the HOST observed it.
///
/// Declared through [`Aec::declare_capture_continuity`]. The host is the honest
/// source of this fact: its audio callback is told directly that capture was
/// interrupted (a WASAPI data-discontinuity flag, a CoreAudio timestamp gap, an
/// ALSA overrun, a device change), while the engine downstream sees only a near
/// stream that arrives with a hole already in it, indistinguishable from a
/// caller whose blocks are merely late until the hole grows large enough to be
/// unmistakable.
///
/// A capture loss shifts the echo to a lag no causal filter can reach at any
/// tail length, so the fix belongs to the alignment, and a declared loss
/// re-anchors regardless of tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CaptureContinuity {
    /// The near stream continues the one the previous block ended: nothing was
    /// lost. The resting state, and declaring it is a no-op, so a host holding
    /// a per-callback platform flag can declare unconditionally rather than
    /// branching.
    Continuous,
    /// The near stream lost samples, or was restarted, since the previous
    /// block. The next [`Aec::process`] re-anchors.
    Discontinuity {
        /// How many near-end samples were lost, when the host knows.
        ///
        /// INFORMATIONAL. Reported through
        /// [`AecMetrics::capture_samples_lost`] and used for nothing else: the
        /// re-anchor rebuilds the alignment from the reference frontier, which
        /// needs no count, so `Some(n)` and `None` produce the same re-anchor,
        /// the same seam and the same recovery for every `n`. A host that knows
        /// only THAT a hole exists passes `None` and is served identically.
        lost_samples: Option<u64>,
    },
}

/// The per-sample output-transition gain ramp behind
/// [`OutputTransitionPolicy::GradedReacquisition`].
///
/// It turns the engine's own published delay status into a per-sample blend
/// gain in `[0.0, 1.0]`: `1.0` emits the canceller correction, `0.0` emits the
/// untouched near-end capture, and the blend between them is linear. The ramp
/// advances by a fixed per-sample increment precomputed once from the fade
/// lengths, so driving it costs one add and one clamp per sample and never
/// allocates. Because it always moves from the CURRENT gain, a status change
/// mid-ramp reverses smoothly rather than stepping the signal (the anti-flap
/// property).
///
/// A ramp increment of `1.0 / fade_samples` reaches the target after
/// `fade_samples` samples from a full 1.0 or 0.0 start, and the `min`/`max`
/// clamp pins the endpoints to exactly `1.0` and `0.0`. Pinning `1.0` exactly
/// is load-bearing: a sample emitted at gain `1.0` is left bit-for-bit as the
/// canceller wrote it (see [`Aec::process`]), so a stream that never reacquires
/// is byte-identical to [`OutputTransitionPolicy::PreserveCorrection`].
struct GradedGate {
    /// Per-sample decrement toward the untouched capture while reacquiring,
    /// `1.0 / fade_out_samples`.
    down_step: f32,
    /// Per-sample increment back toward full correction otherwise,
    /// `1.0 / fade_in_samples`.
    up_step: f32,
    /// The current blend gain, in `[0.0, 1.0]`. Starts at full correction and
    /// is restored to it by [`Aec::reset`].
    gain: f32,
    /// Near-end samples handed to the canceller but not yet emitted, oldest at
    /// the front: the engine's mirror of the canceller's internal block carry.
    /// Blending happens at emit time because a framed canceller delivers a
    /// sample one block after it is fed, and the blend must pair each emitted
    /// correction with the SAME near sample (lag 0). Reused across calls: it
    /// grows to hold the largest block-and-carry the stream has encountered, and
    /// once its capacity covers that high-water pending length the per-call
    /// `reserve` is a no-op and no steady-state allocation occurs. A caller that
    /// later hands over a larger block raises the high-water length and
    /// reallocates the mirror once, via `Vec::reserve`, before it settles again.
    pending_near: Vec<f32>,
    /// The blend gain computed for each pending near sample, in the same order
    /// and of the same length as [`GradedGate::pending_near`]. Computed when the
    /// sample is fed (in near order, which is emit order) so the gain a sample
    /// blends at is the gain in force when it was captured, not when its block
    /// happens to complete.
    pending_gain: Vec<f32>,
}

impl GradedGate {
    /// Builds the gate for two fade lengths in samples. A zero-length fade
    /// (a fade duration below one sample at the rate) collapses to a one-sample
    /// step, which is the hardest cut the linear ramp can express.
    fn new(fade_out_samples: u64, fade_in_samples: u64) -> GradedGate {
        GradedGate {
            down_step: 1.0 / fade_out_samples.max(1) as f32,
            up_step: 1.0 / fade_in_samples.max(1) as f32,
            gain: 1.0,
            pending_near: Vec::new(),
            pending_gain: Vec::new(),
        }
    }

    /// Advances the ramp one sample toward the target the status implies and
    /// returns the new gain. A reacquisition targets the untouched capture
    /// (`0.0`); every other status, initial acquisition included, targets full
    /// correction (`1.0`). The result is always in `[0.0, 1.0]`, and the
    /// endpoints are pinned exactly by the clamp.
    fn step(&mut self, reacquiring: bool) -> f32 {
        if reacquiring {
            self.gain = (self.gain - self.down_step).max(0.0);
        } else {
            self.gain = (self.gain + self.up_step).min(1.0);
        }
        self.gain
    }

    /// Restores the gate to its just-constructed state: full correction, no
    /// pending samples. Called from [`Aec::reset`], the whole-stream restart,
    /// so the next stream begins at full correction exactly as construction
    /// does.
    fn reset(&mut self) {
        self.gain = 1.0;
        self.pending_near.clear();
        self.pending_gain.clear();
    }
}

/// The resolved per-engine output-transition behavior, built once from
/// [`AecConfig::output_transition`] and fixed for the engine's life.
enum OutputBlend {
    /// [`OutputTransitionPolicy::PreserveCorrection`]: the canceller output is
    /// delivered unchanged, on the exact code path the engine used before the
    /// policy existed.
    Preserve,
    /// [`OutputTransitionPolicy::GradedReacquisition`]: the correction is faded
    /// toward the untouched capture while a trusted alignment is being
    /// reacquired.
    Graded(GradedGate),
}

/// A streaming acoustic echo canceller: the batteries-included engine.
///
/// Construct it from an [`AecConfig`] with [`Aec::new`], feed the far-end
/// reference with [`Aec::feed_reference`], and cancel near-end capture blocks
/// with [`Aec::process`]. The engine sanitizes both inputs, aligns them by
/// absolute sample count off the reference ring, and drives the canceller
/// selected by the configuration.
///
/// # Alignment
///
/// A caller who supplies [`AecConfig::delay_hint_ms`] is taken at their word:
/// the offset is seeded from the hint and kept. A caller who supplies no hint
/// gets the automatic estimator, which correlates the two streams and locks the
/// offset once it is confident. Until it locks the offset is zero, so the
/// near-end reads sit at the reference frontier and the canceller has little to
/// cancel; when it locks, the engine adopts the offset and resets the
/// canceller, whose coefficients were learned against the alignment that just
/// changed. The active offset, from either source, is reported through
/// [`AecMetrics::delay_samples`].
///
/// # Capture continuity
///
/// The alignment is a statement about two streams advancing together. A host
/// whose capture stream loses samples should say so with
/// [`Aec::declare_capture_continuity`], which re-anchors on the next
/// [`Aec::process`]. See [`CaptureContinuity`].
///
/// A host that CANNOT say so is not left without a repair. The engine learns
/// the caller's own frontier-lead behaviour, and infers a capture discontinuity
/// from a sudden step outside that learned envelope which persists across more
/// than one `process` boundary, re-anchoring exactly as a declaration does and
/// counting it under [`AecMetrics::reference_reanchors`]. The inference is the
/// fallback and the declaration is the ground truth: a declaration always takes
/// precedence and reseeds the learned model, so a host that declares is never
/// second-guessed and never gets two seams for one event.
///
/// The fallback has a warm-up, and a host should size it in its own terms: it
/// infers nothing until it has watched the caller for 32 [`Aec::process`]
/// calls, and it scales with the BLOCK SIZE rather than the rate. The window
/// restarts past every seam. A loss inside it is not inferred, which is one
/// more reason a host that CAN declare should: a declaration is applied on the
/// next `process` with no warm-up at all.
pub struct Aec {
    /// The configuration the engine was constructed from. Read by the
    /// diagnostic emit sites only, so a build without the `tracing` feature
    /// never reads it.
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    config: AecConfig,
    ring: ReferenceRing,
    /// The canceller [`AecConfig::model`] selected, constructed with the
    /// engine. Every publicly selectable model is compiled in and always
    /// available, so this is never absent. The crate-internal test constructor
    /// replaces it with the internal reference canceller, which is how the
    /// pipeline is validated end to end.
    canceller: Box<dyn EchoCanceller>,
    /// The alignment offset in samples, seeded from the delay hint. The near-end
    /// sample at absolute index `n` reads the far-end sample at absolute index
    /// `anchor + n - delay_offset`.
    delay_offset: u64,
    /// Far absolute index that near-end index 0 anchors to, captured on the first
    /// `process` call after each (re-)anchor. `None` until the next `process`
    /// re-establishes it.
    anchor: Option<u64>,
    /// Count of near-end samples processed since the current anchor.
    near_processed: u64,
    /// Total near-end samples rendered silent because the aligned far-end sample
    /// was dropped (too old) or not yet fed (in the future), counted only while
    /// an alignment (hint or lock) was active.
    reference_starved: u64,
    /// Total near-end samples processed while no alignment was active: the
    /// engine parks the far-end read during acquisition and renders silence by
    /// design, which is a search state, not a transport failure.
    acquisition_parked: u64,
    /// [`RELOCK_KEEP_MS`] in samples at the configured rate.
    relock_keep: u64,
    /// Times the alignment was re-anchored for a capture discontinuity the
    /// engine INFERRED from the caller's frontier lead, the host having
    /// declared nothing.
    reference_reanchors: u64,
    /// The learned frontier-lead behaviour of this caller, and the inference
    /// drawn from a departure from it. See [`LeadModel`].
    lead: LeadModel,
    /// A host-declared capture discontinuity awaiting the next
    /// [`Aec::process`], which consumes it. [`CaptureContinuity::Continuous`]
    /// when nothing is pending, which is the state every caller that never
    /// declares anything stays in for the life of the stream.
    declared_continuity: CaptureContinuity,
    /// Times a pending declaration was consumed and the alignment re-anchored
    /// for it.
    capture_discontinuities: u64,
    /// Near-end samples the host reported lost across those discontinuities.
    /// Saturating: telemetry must not panic or wrap on a stream that runs for
    /// weeks.
    capture_samples_lost: u64,
    /// Applied declarations since the delay acquisition last reached a decision
    /// of its own. See
    /// [`AecMetrics::capture_declarations_without_decision`].
    declarations_without_decision: u64,
    /// The high-water mark of that run.
    declarations_without_decision_max: u64,
    /// The automatic coarse-to-fine delay acquisition, present only when the
    /// caller supplied no hint. A supplied hint is a measurement the caller
    /// already made, so the engine does not second-guess it and neither search
    /// stage is constructed at all.
    acquirer: Option<DelayAcquirer>,
    /// Whether the active offset is a known alignment (a supplied hint, or a
    /// promoted lock) rather than the unlocked default.
    delay_known: bool,
    /// Scratch for the far-end window one FINE frame consumes. The coarse scan
    /// keeps its own decimated far history and needs no window from the engine.
    estimator_scratch: Vec<f32>,
    near_scratch: Vec<f32>,
    far_scratch: Vec<f32>,
    reference_scratch: Vec<f32>,
    /// The resolved output-transition behavior. Fades the emitted signal toward
    /// the untouched capture while the alignment is being reacquired, or leaves
    /// the correction untouched, per [`AecConfig::output_transition`].
    output_blend: OutputBlend,
}

impl Aec {
    /// Constructs the engine for a validated configuration.
    ///
    /// Validates the configuration, sizes the reference ring, and seeds the
    /// alignment offset from [`AecConfig::delay_hint_ms`]. This is the only
    /// fallible operation: it returns [`AecError`] when a field is out of range.
    /// A delay hint outside the search window is clamped rather than rejected.
    ///
    /// A hint is measured from the reference frontier the caller's own feeding
    /// establishes, and a hint longer than that offset cancels nothing without
    /// reporting an error. See [`AecConfig::delay_hint_ms`] before supplying
    /// one.
    pub fn new(config: AecConfig) -> Result<Aec, AecError> {
        if !(8000..=48000).contains(&config.sample_rate) {
            warn!(
                sample_rate = config.sample_rate,
                error = "sample_rate_out_of_range",
                "rejected AEC configuration"
            );
            return Err(AecError::SampleRateOutOfRange {
                requested: config.sample_rate,
            });
        }
        if !(16..=500).contains(&config.tail_ms) {
            warn!(
                tail_ms = config.tail_ms,
                error = "tail_out_of_range",
                "rejected AEC configuration"
            );
            return Err(AecError::TailOutOfRange {
                requested_ms: config.tail_ms,
            });
        }
        if !(10..=1000).contains(&config.max_echo_delay_ms) {
            warn!(
                max_echo_delay_ms = config.max_echo_delay_ms,
                error = "echo_delay_out_of_range",
                "rejected AEC configuration"
            );
            return Err(AecError::EchoDelayOutOfRange {
                requested_ms: config.max_echo_delay_ms,
            });
        }

        if config.max_search_delay_ms < config.max_echo_delay_ms
            || config.max_search_delay_ms > 2000
        {
            warn!(
                max_search_delay_ms = config.max_search_delay_ms,
                max_echo_delay_ms = config.max_echo_delay_ms,
                error = "search_delay_out_of_range",
                "rejected AEC configuration"
            );
            return Err(AecError::SearchDelayOutOfRange {
                requested_ms: config.max_search_delay_ms,
                fine_window_ms: config.max_echo_delay_ms,
            });
        }

        let delay_known = config.delay_hint_ms.is_some();
        let acquirer = match config.delay_hint_ms {
            Some(_) => None,
            None => Some(DelayAcquirer::new(
                config.sample_rate,
                config.max_echo_delay_ms,
                config.max_search_delay_ms,
            )),
        };

        let delay_offset = match config.delay_hint_ms {
            Some(hint_ms) => {
                let clamped_ms = hint_ms.min(config.max_echo_delay_ms);
                if clamped_ms != hint_ms {
                    warn!(
                        requested_ms = hint_ms,
                        clamped_ms = clamped_ms,
                        "delay hint outside the search window; clamped to the window bound"
                    );
                }
                ms_to_samples(clamped_ms, config.sample_rate)
            }
            None => 0,
        };

        let capacity = derive_ring_capacity(&config);
        let ring = ReferenceRing::new(capacity);

        // Every publicly selectable model is compiled into the library and
        // constructed here, so selecting one can never fail at this point.
        let canceller: Box<dyn EchoCanceller> = match config.model {
            AecModel::Tau => Box::new(TauCanceller::new(
                config.sample_rate,
                config.tail_ms,
                config.suppression,
            )),
        };

        debug!(
            model = ?config.model,
            sample_rate = config.sample_rate,
            tail_ms = config.tail_ms,
            max_echo_delay_ms = config.max_echo_delay_ms,
            suppression = ?config.suppression,
            ring_capacity = capacity,
            delay_offset_samples = delay_offset,
            "constructed AEC engine"
        );

        // Resolve the output-transition policy once. The graded gate's ramp
        // increments are fixed for the engine's life, so they are precomputed
        // here and the audio path never derives them again. This is the only
        // place the policy is read: `process` and `flush` act on the resolved
        // `OutputBlend`, never on the configuration.
        let output_blend = match config.output_transition {
            OutputTransitionPolicy::PreserveCorrection => OutputBlend::Preserve,
            OutputTransitionPolicy::GradedReacquisition {
                fade_out_ms,
                fade_in_ms,
            } => OutputBlend::Graded(GradedGate::new(
                fade_ms_to_samples(fade_out_ms, config.sample_rate),
                fade_ms_to_samples(fade_in_ms, config.sample_rate),
            )),
        };

        let relock_keep = ms_to_samples(RELOCK_KEEP_MS as u16, config.sample_rate);
        Ok(Aec {
            config,
            ring,
            canceller,
            delay_offset,
            anchor: None,
            near_processed: 0,
            reference_starved: 0,
            acquisition_parked: 0,
            relock_keep,
            reference_reanchors: 0,
            lead: LeadModel::default(),
            declared_continuity: CaptureContinuity::Continuous,
            capture_discontinuities: 0,
            capture_samples_lost: 0,
            declarations_without_decision: 0,
            declarations_without_decision_max: 0,
            delay_known,
            acquirer,
            estimator_scratch: Vec::new(),
            near_scratch: Vec::new(),
            far_scratch: Vec::new(),
            reference_scratch: Vec::new(),
            output_blend,
        })
    }

    /// Crate-internal, test-only: constructs the engine with the internal
    /// reference canceller (Rho) populated behind the seam, so the pipeline
    /// (ring, alignment, sanitization, metrics, and the canceller drive) can
    /// be validated end to end.
    ///
    /// This is deliberately the only way a Rho instance ever reaches the
    /// engine: it is `pub(crate)` and compiled only into test builds, so no
    /// public string, selector, or constructor can produce it.
    #[cfg(all(test, feature = "internal-tests"))]
    pub(crate) fn with_internal_reference(config: AecConfig) -> Result<Aec, AecError> {
        let mut aec = Aec::new(config)?;
        aec.canceller = Box::new(crate::rho::RhoCanceller::new(
            aec.config.sample_rate,
            aec.config.tail_ms,
        ));
        Ok(aec)
    }

    /// The running delay acquisition, when one was constructed.
    ///
    /// Test-only. Reaching the acquisition directly is how a test forces a
    /// state the audio alone would reach only incidentally, such as a spurious
    /// reacquisition on an otherwise static delay.
    #[cfg(test)]
    pub(crate) fn acquirer_mut(&mut self) -> Option<&mut DelayAcquirer> {
        self.acquirer.as_mut()
    }

    /// Appends far-end reference samples: mono, at the configured rate, in played
    /// order, including any renderer-inserted silence.
    ///
    /// Never blocks and never fails. Non-finite samples are sanitized to `0.0`
    /// before entering the ring, so the ring holds only finite values. When the
    /// ring is full the oldest samples are overwritten and counted (see
    /// [`AecMetrics::reference_dropped`]); that is what a sliding window of
    /// finite depth does on every push, not a failure, and it does not disturb
    /// the alignment. Whether the alignment still describes the stream is
    /// decided in [`Aec::process`], against the near stream rather than the
    /// ring (see [`AecMetrics::reference_reanchors`]).
    ///
    /// # A reference at the wrong rate is accepted silently
    ///
    /// Both streams are plain `&[f32]` with no rate attached, so a reference at
    /// a rate other than [`AecConfig::sample_rate`] cannot be detected: it is
    /// accepted with no error and no warning, and nothing is cancelled. This is
    /// the likeliest integration mistake, because a host's playback and capture
    /// devices commonly run at different rates.
    ///
    /// The signature is [`AecMetrics::acquisition_parked`] climbing while
    /// [`AecMetrics::delay_samples`] stays `None`, on a stream where audio is
    /// definitely playing. That combination means the reference is not at the
    /// configured rate, or is not the signal that produced the echo. Resample
    /// the reference to the configured rate before feeding it.
    ///
    /// # Automatic acquisition needs broadband far-end material
    ///
    /// The delay search is correlation-based, so it needs a far end with an
    /// unambiguous correlation peak. Sustained periodic material (a held tone, a
    /// steady harmonic complex, and some music) has no such peak, because every
    /// period is an equally good match. On material like that the acquisition
    /// can stay parked for the whole stream: the reference flows, nothing is
    /// cancelled, [`AecMetrics::delay_samples`] stays `None`, and no error is
    /// returned. Speech and most broadband program material lock normally.
    ///
    /// A caller that has to cancel against periodic far-end material can supply
    /// [`AecConfig::delay_hint_ms`], which skips the search entirely and locks
    /// on the supplied offset. Read that field's documentation before doing so:
    /// the hint is measured from the reference frontier as the caller's own
    /// feeding establishes it, not from an absolute platform latency, and a hint
    /// longer than that offset cancels nothing.
    pub fn feed_reference(&mut self, reference: &[f32]) {
        sanitize_into(reference, &mut self.reference_scratch);
        self.ring.push(&self.reference_scratch);
        // The coarse scan keeps its own decimated far history, addressed by the
        // same absolute index the ring uses. Pushed here, immediately after the
        // ring, so the two counters cannot drift apart. A no-op once the
        // acquisition has promoted, so steady-state cost is unchanged.
        if let Some(acquirer) = &mut self.acquirer {
            acquirer.push_far(&self.reference_scratch);
        }

        // Overwriting the oldest retained sample is what a fixed-capacity ring
        // does on every push once it is full, so ANY stream longer than the
        // ring's depth does it forever. It says nothing about the consumer, and
        // the alignment is decided against the consumer: see the lag check at
        // the head of [`Aec::process`].
    }

    /// Declares what happened to the capture stream since the previous
    /// [`Aec::process`] call.
    ///
    /// [`CaptureContinuity::Continuous`] is a no-op, so a host holding a
    /// per-callback platform flag can declare unconditionally.
    ///
    /// [`CaptureContinuity::Discontinuity`] takes effect on the NEXT `process`,
    /// which re-anchors the near stream onto the reference frontier exactly as
    /// a stream start does, discards the acquisition evidence that straddles
    /// the seam, and keeps the standing delay only as a PRIOR: the alignment
    /// the tracker re-confirms against fresh evidence, on a tightened leash,
    /// rather than one taken as still true. The canceller's learned state is
    /// NOT discarded, because the echo path did not change; only the timeline
    /// it is read against did.
    ///
    /// A declared discontinuity is LATCHED until a `process` consumes it, and a
    /// later `Continuous` declaration in the same gap does not cancel it: a
    /// discontinuity that goes unheard costs the cancellation outright, while a
    /// spurious one costs a re-anchor the engine performs at every stream start
    /// anyway. Two declarations before one `process` describe one seam, so they
    /// latch into one re-anchor and their reported losses add.
    ///
    /// Declare on the EVENT, not on every block. Each applied declaration
    /// discards the partial evidence both delay searches were accumulating, so
    /// a host that declares a discontinuity continuously never banks enough of
    /// it to lock.
    ///
    /// Not to be confused with [`Aec::reset`], which is the whole-stream form:
    /// it clears the reference ring, the delay, and the converged filter. This
    /// is the mid-stream form, which keeps all three.
    pub fn declare_capture_continuity(&mut self, continuity: CaptureContinuity) {
        let CaptureContinuity::Discontinuity { lost_samples } = continuity else {
            return;
        };
        // Counts add when both are known: two holes in one gap lost the sum of
        // their samples. A hole the host could not size contributes nothing
        // rather than erasing a count that was supplied.
        let latched = match self.declared_continuity {
            CaptureContinuity::Discontinuity { lost_samples: held } => match (held, lost_samples) {
                (Some(held), Some(fresh)) => Some(held.saturating_add(fresh)),
                (held, fresh) => held.or(fresh),
            },
            CaptureContinuity::Continuous => lost_samples,
        };
        self.declared_continuity = CaptureContinuity::Discontinuity {
            lost_samples: latched,
        };
    }

    /// Cancels one near-end capture block, appending the echo-reduced samples to
    /// `out`.
    ///
    /// Sanitizes the near-end block, pulls the aligned far-end block from the
    /// reference ring (rendering unavailable spans as silence and counting them
    /// as starved), and hands the equal-length, time-aligned pair to the
    /// selected canceller. `out` is appended to, never cleared.
    ///
    /// A framed canceller re-blocks internally, so a call may append fewer or
    /// more samples than it consumed; the totals balance after [`Aec::flush`].
    pub fn process(&mut self, near: &[f32], out: &mut Vec<f32>) -> Result<(), AecError> {
        // A capture discontinuity the HOST declared, consumed here and applied
        // before anything else. It supersedes the frontier-lag inference below
        // and does not consult it: the host observed the hole directly, so
        // there is nothing left to infer. See [`CaptureContinuity`].
        let declared = self.declared_continuity;
        self.declared_continuity = CaptureContinuity::Continuous;
        if let CaptureContinuity::Discontinuity { lost_samples } = declared {
            self.capture_discontinuities = self.capture_discontinuities.saturating_add(1);
            // Saturating: a count the host supplies is unvalidated telemetry,
            // and a stream that runs for weeks must not be able to wrap or
            // panic it. Saturation is visible in the metric itself and costs
            // nothing that matters, because the value is informational.
            self.capture_samples_lost = self
                .capture_samples_lost
                .saturating_add(lost_samples.unwrap_or(0));
            // A standing alignment IS the decision this run counts the absence
            // of, so a host declaring real seams against a healthy lock never
            // accumulates one: a settled tracker can idle for seconds at a time
            // without acting, and counting that as starvation would make the
            // metric climb on exactly the hosts doing it right. The run
            // accumulates only while declarations keep arriving and no
            // alignment has ever been reached, which is the failure this exists
            // to make visible.
            self.declarations_without_decision = if self.delay_known {
                0
            } else {
                self.declarations_without_decision.saturating_add(1)
            };
            self.declarations_without_decision_max = self
                .declarations_without_decision_max
                .max(self.declarations_without_decision);
            // Rebuild the alignment from the reference frontier below, exactly
            // as a stream start does. The standing offset survives as a prior:
            // the timeline moved, the echo path did not, so the delay is still
            // the best available estimate and is the one the tracker will
            // re-confirm or replace.
            self.anchor = None;
            // The acquisition is told rather than left to infer the seam from a
            // jump in the block base, because a discontinuity declared while
            // the near stream sits exactly AT the frontier (a capture restart
            // with no far-side gap) moves that base by nothing at all, and its
            // partial evidence straddles the seam just the same.
            if let Some(acquirer) = &mut self.acquirer {
                acquirer.declare_discontinuity();
            }
            // A declaration is the ground truth the inference exists to
            // approximate, so it RESEEDS the learned model rather than being
            // measured by it: the anchor is about to be rebuilt, the lead
            // restarts at zero, and the envelope learned against the old anchor
            // describes a stream that no longer exists. This is also what stops
            // a declared seam from being counted twice. The block that carries
            // a declaration never reaches the inference below, so a host that
            // declares a loss AND presents the large lead that loss produces
            // gets exactly one seam, and it is the host-declared one.
            self.lead.reseed();
            debug!(
                lost_samples = lost_samples.unwrap_or(0),
                lost_reported = lost_samples.is_some(),
                delay_offset_samples = self.delay_offset,
                "host declared a capture discontinuity; re-anchoring and keeping the \
                 standing delay as a prior"
            );
        }

        // The frontier LEAD: how far the far-end feed has run ahead of where
        // the standing alignment says the near stream's next sample sits.
        //
        //     lead = reference_frontier - expected_reference_frontier
        //
        // See [`LeadModel`].
        if let Some(anchor) = self.anchor {
            let lead = self
                .ring
                .next_abs()
                .saturating_sub(anchor + self.near_processed);
            let inferred = self.lead.observe(lead, near.len() as u64);
            // The one case that does not wait on the learned model. Past the
            // ring's retained depth the standing alignment cannot be served by
            // any sample the engine still holds, so every read starves and
            // holding the alignment is not a judgement call: there is nothing
            // left to hold it against. Not a threshold on what the lead MEANS,
            // which is the model's job, but the structural point past which the
            // question stops being answerable.
            let unservable = lead >= self.ring.capacity() as u64;
            if inferred || unservable {
                self.reference_reanchors = self.reference_reanchors.saturating_add(1);
                warn!(
                    lead_samples = lead,
                    learned_bound = self.lead.bound().unwrap_or(0),
                    unservable = unservable,
                    "inferred a capture discontinuity from a sustained step outside this \
                     caller's learned frontier-lead envelope; re-anchoring"
                );
                self.anchor = None;
                // The same seam a declaration produces, because it is the same
                // physical event and differs only in who noticed it: the
                // straddling evidence in both search stages is discarded and
                // the standing delay is kept as a prior for the tracker to
                // re-confirm. Told explicitly rather than left to the block-base
                // jump so an inferred seam and a declared one take identical
                // paths; `begin_block` seeing the same jump below is idempotent.
                if let Some(acquirer) = &mut self.acquirer {
                    acquirer.declare_discontinuity();
                }
                self.lead.reseed();
            }
        }

        let anchor = match self.anchor {
            Some(anchor) => anchor,
            None => {
                let anchor = self.ring.next_abs();
                self.anchor = Some(anchor);
                self.near_processed = 0;
                debug!(
                    anchor = anchor,
                    delay_offset_samples = self.delay_offset,
                    "anchored the near-end stream to the reference frontier"
                );
                anchor
            }
        };

        // Sanitize the near-end block (non-finite -> 0.0) before the canceller
        // sees it: a non-finite sample reaching an adaptive update would poison
        // the learned state permanently.
        sanitize_into(near, &mut self.near_scratch);

        // Feed the automatic estimator, if one is running. Frames are cut on the
        // near-end sample count, so where the caller's chunk boundaries fall
        // never changes when a frame completes or what it contains.
        if let Some(acquirer) = &mut self.acquirer {
            // The far-absolute index the next near sample carries. Both stages
            // cut their framing on this absolute grid, and a discontinuity here
            // is how the acquisition learns the engine re-anchored.
            acquirer.begin_block(anchor + self.near_processed);
            let window_len = acquirer.fine_window_len();
            for &sample in self.near_scratch.iter() {
                if !acquirer.push_near(sample) {
                    continue;
                }
                // A fine frame just completed. The acquirer owns the search
                // origin, so it reports where its far window starts; reference
                // the window asks for that was never fed is rendered as silence,
                // and where that silence falls decides whether the frame is
                // usable.
                let window_start = acquirer.fine_window_start_abs();
                let support = assemble_fine_window(
                    &self.ring,
                    &mut self.estimator_scratch,
                    window_start,
                    window_len,
                );
                let outcome = acquirer.observe(&self.estimator_scratch, support);
                // A relocation cannot re-correlate the buffered frame itself:
                // the rescan needs a fresh far window at the NEW origin, which
                // only the ring here can assemble. When the frame just relocated
                // away from held a candidate that would otherwise promote an
                // edge-pinned value from the abandoned range, re-read the window
                // at the new origin the acquirer has already moved to and let
                // the acquisition rescan the frame against it instead.
                let action = if outcome.rescan_pending {
                    let window_start = acquirer.fine_window_start_abs();
                    let support = assemble_fine_window(
                        &self.ring,
                        &mut self.estimator_scratch,
                        window_start,
                        window_len,
                    );
                    acquirer.rescan(&self.estimator_scratch, support)
                } else {
                    outcome.action
                };
                match action {
                    // A trusted promotion. The alignment the canceller's
                    // coefficients were learned against has changed, so it is
                    // reset to re-converge from the corrected alignment. The
                    // exception is a reacquisition that re-confirmed the
                    // standing alignment: the trigger was spurious, and the
                    // converged filter is kept along with the old offset.
                    Some(AcquireAction::Promote(delay)) => {
                        // A decision of the acquisition's own, which is what
                        // the declaration run is counting the absence of.
                        self.declarations_without_decision = 0;
                        let unchanged = self.delay_known
                            && self.delay_offset.abs_diff(delay as u64) <= self.relock_keep;
                        #[cfg(feature = "tracing")]
                        let previous = self.delay_offset;
                        // The offset is adopted either way. The acquisition has
                        // already taken `delay` as its own alignment, and every
                        // later tracking cycle is measured against THAT value,
                        // so an engine that kept the old one would describe a
                        // different alignment from the estimator that is
                        // steering it; on a static delay the tracker holds and
                        // the two never reconverge. Only the canceller RESET is
                        // conditional, which is the whole point of the keep
                        // band: a re-confirmation inside it is not worth
                        // throwing a converged filter away for.
                        self.delay_offset = delay as u64;
                        if unchanged {
                            debug!(
                                delay_samples = delay,
                                previous_offset = previous,
                                "reacquisition re-confirmed the standing alignment; \
                                 adopted the offset and kept the canceller state"
                            );
                        } else {
                            self.canceller.reset();
                            // The canceller just discarded its partial block
                            // carry, so those in-flight near samples will never
                            // be emitted. The graded gate's pending mirror tracks
                            // that same carry, so it drops them too and stays in
                            // lockstep; the ramp's current gain is untouched,
                            // because the filter reset is not an output-fade
                            // event. At this point in the block the mirror still
                            // holds only the previous call's leftover (this
                            // call's near samples are pushed after the acquirer
                            // loop), which is exactly the carry that was cleared.
                            if let OutputBlend::Graded(gate) = &mut self.output_blend {
                                gate.pending_near.clear();
                                gate.pending_gain.clear();
                            }
                            debug!(
                                delay_samples = delay,
                                status = ?acquirer.estimate().status,
                                "delay acquisition promoted a trusted lock; adopted the \
                                 offset and reset the canceller"
                            );
                        }
                        self.delay_known = true;
                    }
                    // A tracking move. The learned filter state is still
                    // mostly right for a small shift, and the canceller's own
                    // shadow adaptation absorbs it, so nothing is reset.
                    Some(AcquireAction::Track(delay)) => {
                        self.declarations_without_decision = 0;
                        self.delay_offset = delay as u64;
                        debug!(
                            delay_samples = delay,
                            "delay tracker moved the alignment; canceller state kept"
                        );
                    }
                    None => {}
                }
            }
        }

        // Pull the aligned far-end block. Near-end index n reads far-end absolute
        // index (anchor + n - delay_offset); a signed index below the far origin,
        // an index dropped from the ring, or an index not yet fed is rendered as
        // a starved (silent) sample.
        //
        // While no alignment is active the read is PARKED instead: with the
        // offset still at its unlocked zero, every aligned index sits at or
        // beyond the reference frontier, so the engine renders silence by
        // design and counts the span as parked rather than starved. Starvation
        // is thereby a transport report (an active alignment whose reference
        // was dropped or never fed), not an artefact of searching.
        self.far_scratch.clear();
        self.far_scratch.reserve(near.len());
        if self.delay_known {
            let start = anchor as i64 - self.delay_offset as i64 + self.near_processed as i64;
            let mut starved = 0_u64;
            for i in 0..near.len() as i64 {
                let far_index = start + i;
                let sample = if far_index < 0 {
                    None
                } else {
                    self.ring.get(far_index as u64)
                };
                match sample {
                    Some(value) => self.far_scratch.push(value),
                    None => {
                        self.far_scratch.push(0.0);
                        starved += 1;
                    }
                }
            }
            self.reference_starved += starved;
        } else {
            self.far_scratch.resize(near.len(), 0.0);
            self.acquisition_parked += near.len() as u64;
        }
        self.near_processed += near.len() as u64;

        // The engine upholds the trait's equal-length aligned-block precondition.
        debug_assert_eq!(self.near_scratch.len(), self.far_scratch.len());

        // The output-transition policy acts here, on the EMITTED audio only. It
        // is the last thing the engine does and it reads no delay decision it
        // did not already make: it gates purely on the status the acquisition
        // just published. `Reacquiring` is the one state whose correction the
        // engine distrusts, and it is reachable only after a trusted lock, so
        // initial acquisition never enters it and the ramp holds full correction
        // through startup.
        //
        // `reacquiring` is derived only when a graded blend is active, so the
        // preserve path pays nothing for it. The status read borrows the engine
        // immutably and completes before the mutable blend borrow begins.
        let graded = matches!(self.output_blend, OutputBlend::Graded(_));
        let reacquiring =
            graded && matches!(self.delay_estimate().status, DelayStatus::Reacquiring);

        match &mut self.output_blend {
            // Deliver the correction unchanged: the exact call the engine made
            // before the policy existed, so PreserveCorrection is byte-identical
            // to the pre-policy engine on every fixture, reacquisitions included.
            OutputBlend::Preserve => {
                self.canceller
                    .process(&self.near_scratch, &self.far_scratch, out)
            }
            OutputBlend::Graded(gate) => {
                // Extend the pending mirror with this block's near samples and
                // the per-sample gain each will be emitted at. Tau delivers the
                // n-th output sample as the cancelled n-th near sample, so the
                // gain a sample blends at is the gain in force when it was
                // captured; the ramp advances in near order, which is emit
                // order.
                gate.pending_near.reserve(self.near_scratch.len());
                gate.pending_gain.reserve(self.near_scratch.len());
                for &sample in &self.near_scratch {
                    let g = gate.step(reacquiring);
                    gate.pending_near.push(sample);
                    gate.pending_gain.push(g);
                }

                // Run the canceller, then blend exactly the samples it emitted.
                // A framed canceller appends whole blocks, so it emits the front
                // `emitted` of the pending mirror, in order, and the mirror stays
                // in lockstep with its internal carry (a time-domain canceller
                // emits every sample immediately, which this same accounting
                // handles with `emitted == pending length`).
                let before = out.len();
                let result = self
                    .canceller
                    .process(&self.near_scratch, &self.far_scratch, out);
                let emitted = out.len() - before;
                debug_assert!(
                    emitted <= gate.pending_near.len(),
                    "canceller emitted more samples than the engine has fed since the last flush"
                );
                blend_emitted(
                    &mut out[before..before + emitted],
                    &gate.pending_near[..emitted],
                    &gate.pending_gain[..emitted],
                );
                gate.pending_near.drain(..emitted);
                gate.pending_gain.drain(..emitted);
                result
            }
        }
    }

    /// Drains the end-of-stream carry into `out`.
    ///
    /// Call once at close, after the final [`Aec::process`].
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AecError> {
        let before = out.len();
        let result = self.canceller.flush(out);
        let emitted = out.len() - before;
        // The flushed tail is the canceller's remaining carry, which is exactly
        // the samples still pending in the gate mirror. Blend them at their
        // already-computed gains: the flush advances no ramp (these samples were
        // captured during `process` and carry the gain in force then), so the
        // last flushed sample continues smoothly from the last emitted one and
        // introduces no final jump.
        if let OutputBlend::Graded(gate) = &mut self.output_blend {
            debug_assert_eq!(
                emitted,
                gate.pending_near.len(),
                "flush must emit exactly the samples still pending in the gate mirror"
            );
            let n = emitted.min(gate.pending_near.len());
            blend_emitted(
                &mut out[before..before + n],
                &gate.pending_near[..n],
                &gate.pending_gain[..n],
            );
            gate.pending_near.drain(..n);
            gate.pending_gain.drain(..n);
        }
        debug!(tail = out.len() - before, "flushed AEC engine tail");
        result
    }

    /// Returns the selected canceller's constant algorithmic latency in samples
    /// at the configured rate.
    ///
    /// # Output alignment
    ///
    /// This value is not an alignment correction, and callers must not shift the
    /// output by it. [`Aec::process`] emits a stream that is sample-index aligned
    /// with the near-end input at lag zero: the cancelled counterpart of near-end
    /// sample `n` is emitted at output position `n`, same index, no offset. The
    /// engine holds this across the whole stream, with [`Aec::flush`] delivering
    /// the final framing carry. Shifting the returned output by
    /// `latency_samples()` would misalign it against the near-end capture.
    ///
    /// What the value is for is a caller's own latency budget: it is the constant
    /// algorithmic delay the canceller introduces between feeding a near-end
    /// sample and its cancelled counterpart becoming available, so a caller
    /// accounting for end-to-end latency can include it.
    pub fn latency_samples(&self) -> usize {
        self.canceller.latency_samples()
    }

    /// Clears all streaming state, keeping the configuration and the reference
    /// ring's capacity. The next stream re-anchors and re-converges from scratch.
    pub fn reset(&mut self) {
        self.ring.clear();
        self.anchor = None;
        self.near_processed = 0;
        self.reference_starved = 0;
        self.acquisition_parked = 0;
        self.reference_reanchors = 0;
        self.lead.reseed();
        self.declared_continuity = CaptureContinuity::Continuous;
        self.capture_discontinuities = 0;
        self.capture_samples_lost = 0;
        self.declarations_without_decision = 0;
        self.declarations_without_decision_max = 0;
        if let Some(acquirer) = &mut self.acquirer {
            acquirer.reset();
            self.delay_offset = 0;
            self.delay_known = false;
        }
        self.estimator_scratch.clear();
        self.near_scratch.clear();
        self.far_scratch.clear();
        self.reference_scratch.clear();
        // The whole-stream restart returns the output-transition gate to full
        // correction and drops any pending mirror, so the next stream begins at
        // gain 1.0 exactly as construction does.
        if let OutputBlend::Graded(gate) = &mut self.output_blend {
            gate.reset();
        }
        self.canceller.reset();
        debug!(model = ?self.config.model, "reset AEC engine state");
    }

    /// Snapshots the engine's observable state: the selected canceller's metrics,
    /// the delay estimate, and the reference-transport counters.
    ///
    /// The delay estimate is the active alignment offset, whether that came from
    /// a supplied hint or from the automatic estimator, and is [`None`] while an
    /// estimator is still searching.
    pub fn metrics(&self) -> AecMetrics {
        AecMetrics {
            canceller: self.canceller.metrics(),
            delay_samples: if self.delay_known {
                Some(self.delay_offset as usize)
            } else {
                None
            },
            reference_starved: self.reference_starved,
            acquisition_parked: self.acquisition_parked,
            reference_dropped: self.ring.dropped(),
            reference_reanchors: self.reference_reanchors,
            capture_discontinuities: self.capture_discontinuities,
            capture_samples_lost: self.capture_samples_lost,
            capture_declaration_pending: matches!(
                self.declared_continuity,
                CaptureContinuity::Discontinuity { .. }
            ),
            capture_declarations_without_decision: self.declarations_without_decision,
            capture_declarations_without_decision_max: self.declarations_without_decision_max,
            delay: self.delay_estimate(),
        }
    }

    /// The delay acquisition's snapshot, synthesized for the hinted path where
    /// no search of either stage is constructed or run.
    fn delay_estimate(&self) -> DelayEstimate {
        match &self.acquirer {
            Some(acquirer) => acquirer.estimate(),
            None => DelayEstimate {
                status: DelayStatus::Locked(DelayLockSource::Hint),
                delay_samples: Some(self.delay_offset as usize),
                fine_search_start_samples: self.delay_offset as usize,
                fine_search_end_samples: self.delay_offset as usize,
                coarse_ceiling_samples: 0,
                coarse_region_samples: None,
                coarse_bin_samples: 0,
                coarse_correlation: 0.0,
                beyond_ceiling: false,
                coarse_frames: 0,
                fine_frames: 0,
                fine_frames_skipped: 0,
                relocated: false,
                fine_scans: 0,
                fine_last_ratio: 0.0,
                fine_last_delay_samples: None,
                fine_last_origin_samples: 0,
                fine_last_peak_interior: false,
                tracking_moves: 0,
                reacquisitions: 0,
                last_reacquire_trigger: None,
                coarse_rearms: 0,
                coarse_regions_rejected: 0,
                // The hinted path runs no coarse re-verification, so the
                // last-resort stand-down can never engage.
                coarse_last_resort_exhausted: false,
                // The hinted path runs no tracker, so there is nothing that
                // could contradict itself.
                tracking_contradiction_run: 0,
                tracking_contradiction_run_max: 0,
            },
        }
    }
}

/// A snapshot of the [`Aec`] engine's observable state.
///
/// Composes the selected canceller's [`CancellerMetrics`] with the delay
/// estimate and the reference-transport counters the engine owns. Metadata only,
/// never sample data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AecMetrics {
    /// The selected canceller's own metrics.
    pub canceller: CancellerMetrics,
    /// The active alignment offset in samples: a supplied delay hint, or the
    /// automatic estimator's locked estimate. [`None`] while the estimator is
    /// still searching and no hint was supplied. While the acquisition is
    /// [`DelayStatus::Reacquiring`], this keeps reporting the previously
    /// trusted offset, which the engine holds until a new lock is promoted.
    pub delay_samples: Option<usize>,
    /// Total near-end samples rendered silent because the aligned far-end sample
    /// was unavailable (dropped from the ring or not yet fed) while an
    /// alignment (a hint or a promoted lock) was ACTIVE. A transport report:
    /// samples processed before any alignment existed are counted under
    /// [`AecMetrics::acquisition_parked`] instead.
    pub reference_starved: u64,
    /// Total near-end samples processed while no alignment was active. The
    /// engine parks the far-end read during acquisition and renders silence by
    /// design; this is the searching span, not a transport failure.
    pub acquisition_parked: u64,
    /// Total far-end samples overwritten in the reference ring once it was full.
    /// A sliding window of finite depth does this on every push for the whole
    /// life of any stream longer than that depth, so a large count here is the
    /// normal steady state and not a fault on its own. The fault to watch for is
    /// [`AecMetrics::reference_reanchors`].
    pub reference_dropped: u64,
    /// Times the engine INFERRED a capture discontinuity from the caller's
    /// frontier lead and re-anchored for it.
    ///
    /// The fallback for a host that does not use
    /// [`Aec::declare_capture_continuity`]. The engine learns the caller's own
    /// resting lead and chunking oscillation, and infers a lost capture span
    /// only from a sudden step outside that learned envelope which persists
    /// across more than one `process` boundary. A caller that feeds and
    /// processes at the same rate holds this at zero, whatever chunk sizes each
    /// side uses, and so does a caller legitimately buffering a long way ahead
    /// of the engine.
    ///
    /// A non-zero count is a real transport report: the alignment was rebuilt
    /// from the reference frontier, the delay acquisition was told of the seam,
    /// and the standing delay was kept only as a prior to be re-confirmed.
    ///
    /// INFERRED, not host-declared. A seam the host reported through
    /// [`Aec::declare_capture_continuity`] is a different fact with a different
    /// source of truth, and is counted under
    /// [`AecMetrics::capture_discontinuities`] instead. A host declaration
    /// takes precedence and reseeds the inference, so a declared seam never
    /// also appears here.
    pub reference_reanchors: u64,
    /// Times a host-declared capture discontinuity was applied: the alignment
    /// was re-anchored onto the reference frontier, the acquisition's
    /// cross-seam evidence was discarded, and the standing delay was kept as a
    /// prior for the tracker to re-confirm.
    ///
    /// Zero for every caller that never calls
    /// [`Aec::declare_capture_continuity`], which is what leaves that caller's
    /// behaviour exactly as it was before the call existed.
    pub capture_discontinuities: u64,
    /// Total near-end samples the host reported lost across those
    /// discontinuities.
    ///
    /// INFORMATIONAL, and nothing else. The re-anchor rebuilds the alignment
    /// from the reference frontier, which needs no count:
    /// `Discontinuity { lost_samples: Some(n) }` and
    /// `Discontinuity { lost_samples: None }` produce the same re-anchor, the
    /// same seam, and the same recovery for every `n`. The only observable
    /// difference between them is this counter. A host that knows only THAT a
    /// hole exists is therefore served exactly as well as one that can size it.
    ///
    /// Accumulated saturating, so a stream that runs indefinitely against a
    /// host reporting large losses cannot wrap or panic it. The value is
    /// telemetry, and a saturated telemetry counter is visible in the number
    /// itself.
    pub capture_samples_lost: u64,
    /// Whether a host declaration is latched and awaiting the next
    /// [`Aec::process`] to consume it.
    ///
    /// True only between a [`Aec::declare_capture_continuity`] call carrying a
    /// [`CaptureContinuity::Discontinuity`] and the `process` that applies it,
    /// so a host sampling metrics on its own thread normally sees `false`. Its
    /// use is diagnostic: a host that declares and never processes can see that
    /// its declarations are latching rather than disappearing.
    pub capture_declaration_pending: bool,
    /// Applied host declarations that arrived with NO alignment standing and
    /// no decision reached since the last one.
    ///
    /// Each applied declaration discards the partial evidence both search
    /// stages were accumulating, so a host that declares on every block rather
    /// than on the event never banks enough of it to decide anything, and the
    /// alignment silently never converges. This counter is what makes that
    /// visible: it climbs, and a climbing run is the signature of declaring too
    /// often.
    ///
    /// A standing alignment resets it, so declaring real seams against a
    /// healthy lock never accumulates a run: a settled tracker can idle for
    /// seconds at a time without acting, and counting that as starvation would
    /// make this climb on the hosts doing it right. Zero throughout on the
    /// hinted path, where the caller supplied the alignment and there is no
    /// search for a declaration to be starving.
    pub capture_declarations_without_decision: u64,
    /// The high-water mark of
    /// [`AecMetrics::capture_declarations_without_decision`], which survives
    /// the run that produced it and is therefore the field a post-hoc log can
    /// be read for.
    pub capture_declarations_without_decision_max: u64,
    /// The delay acquisition's full state: what is being searched, and what has
    /// been found. [`AecMetrics::delay_samples`] is this estimate's own
    /// `delay_samples`, kept as a top-level field for source compatibility.
    ///
    /// Its `status` and `delay_samples` are a stable contract; most of its other
    /// fields are diagnostic telemetry that may evolve. See [`DelayEstimate`]'s
    /// stability note.
    pub delay: DelayEstimate,
}

/// Assembles the far-end window one fine frame consumes, filling `scratch` with
/// exactly `window_len` samples starting at absolute index `window_start` in the
/// reference ring.
///
/// An index the ring cannot supply is rendered as silence and counted in the
/// returned [`WindowSupport`]: a leading deficit (the window reaching back
/// before the stream start, so `window_start` is negative) is `missing_head`,
/// and a trailing one (reaching past the reference frontier) is `missing_tail`.
/// Where the silence falls is what decides whether the frame is usable, so the
/// two are kept apart. Factored out of [`Aec::process`] so the frame relocation
/// leaves behind can be re-assembled at the new origin for a rescan with the
/// identical logic.
fn assemble_fine_window(
    ring: &ReferenceRing,
    scratch: &mut Vec<f32>,
    window_start: i64,
    window_len: usize,
) -> WindowSupport {
    scratch.clear();
    scratch.reserve(window_len);
    let mut support = WindowSupport::default();
    let mut seen_signal = false;
    for offset in 0..window_len as i64 {
        let index = window_start + offset;
        let value = if index < 0 {
            None
        } else {
            ring.get(index as u64)
        };
        match value {
            Some(sample) => {
                seen_signal = true;
                scratch.push(sample);
            }
            None => {
                if seen_signal {
                    support.missing_tail += 1;
                } else {
                    support.missing_head += 1;
                }
                scratch.push(0.0);
            }
        }
    }
    debug_assert_eq!(scratch.len(), window_len);
    support
}

/// Converts a duration in milliseconds to a sample count at `sample_rate`,
/// truncating toward zero.
fn ms_to_samples(ms: u16, sample_rate: u32) -> u64 {
    (ms as u64 * sample_rate as u64) / 1000
}

/// Converts a fade duration in milliseconds to a sample count at `sample_rate`,
/// truncating toward zero. Separate from [`ms_to_samples`] because a fade
/// length is `u32`: a transition ramp may legitimately be set longer than any
/// echo delay, which the `u16` delay quantities never are.
fn fade_ms_to_samples(ms: u32, sample_rate: u32) -> u64 {
    (ms as u64 * sample_rate as u64) / 1000
}

/// Blends each emitted correction sample toward its lag-0-aligned near sample
/// at the per-sample gain: `out = gain * correction + (1 - gain) * near`, a
/// linear (not equal-power) crossfade. The correction and the capture are
/// coherent versions of the same microphone stream, so an equal-power law would
/// raise the amplitude wherever they correlate; the linear law holds it.
///
/// A gain of exactly `1.0` leaves the sample bit-for-bit as the canceller wrote
/// it. That short-circuit is load-bearing: it is what makes the graded default
/// byte-identical to [`OutputTransitionPolicy::PreserveCorrection`] on every
/// sample outside a reacquisition, where a naive `1.0 * c + 0.0 * n` would flip
/// a `-0.0` correction to `+0.0`.
fn blend_emitted(emitted: &mut [f32], near: &[f32], gain: &[f32]) {
    debug_assert_eq!(emitted.len(), near.len());
    debug_assert_eq!(emitted.len(), gain.len());
    for ((sample, &near_sample), &gain) in emitted.iter_mut().zip(near).zip(gain) {
        if gain != 1.0 {
            *sample = gain * *sample + (1.0 - gain) * near_sample;
        }
    }
}

/// Derives the reference ring depth from the configuration: the maximum echo
/// delay plus the filter tail plus [`RING_SLACK_SECONDS`] of slack, at the
/// configured rate.
fn derive_ring_capacity(config: &AecConfig) -> usize {
    let span_ms = config.max_echo_delay_ms as u64 + config.tail_ms as u64;
    let span_samples = (span_ms * config.sample_rate as u64) / 1000;
    let slack_samples = RING_SLACK_SECONDS * config.sample_rate as u64;
    (span_samples + slack_samples).max(1) as usize
}

/// Copies `src` into `dst`, replacing each non-finite sample (`NaN`, infinities)
/// with `0.0`. `dst` is cleared first. Replacing each non-finite sample keeps the
/// canceller's recursive state finite and bounds the damage to the single
/// offending sample.
fn sanitize_into(src: &[f32], dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(src.len());
    dst.extend(src.iter().map(|&s| if s.is_finite() { s } else { 0.0 }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AecModel;

    #[test]
    fn sanitize_replaces_non_finite_and_passes_finite_through() {
        let mut dst = Vec::new();
        sanitize_into(
            &[1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0, 0.0],
            &mut dst,
        );
        assert_eq!(dst, vec![1.0, 0.0, 0.0, 0.0, -2.0, 0.0]);
    }

    #[test]
    fn ms_to_samples_truncates_toward_zero() {
        assert_eq!(ms_to_samples(200, 16000), 3200);
        assert_eq!(ms_to_samples(16, 16000), 256);
        assert_eq!(ms_to_samples(1, 16000), 16);
    }

    #[test]
    fn ring_capacity_covers_delay_tail_and_slack() {
        let capacity = derive_ring_capacity(&AecConfig::default());
        // (250 + 200) ms + 4 s at 16 kHz = 7200 + 64000.
        assert_eq!(capacity, 71200);
    }

    #[test]
    fn construction_accepts_defaults() {
        let aec = Aec::new(AecConfig::default()).expect("default config is valid");
        // The default model is the framed frequency-domain canceller, whose
        // framing latency is one block.
        assert_eq!(aec.latency_samples(), crate::tau::BLOCK);
        let metrics = aec.metrics();
        assert_eq!(metrics.delay_samples, None);
        assert_eq!(metrics.reference_starved, 0);
        assert_eq!(metrics.reference_dropped, 0);
        assert_eq!(metrics.canceller, CancellerMetrics::default());
    }

    #[test]
    fn construction_rejects_out_of_range_sample_rate() {
        let config = AecConfig {
            sample_rate: 4000,
            ..Default::default()
        };
        assert!(matches!(
            Aec::new(config),
            Err(AecError::SampleRateOutOfRange { requested: 4000 })
        ));
    }

    #[test]
    fn construction_rejects_out_of_range_tail() {
        let config = AecConfig {
            tail_ms: 8,
            ..Default::default()
        };
        assert!(matches!(
            Aec::new(config),
            Err(AecError::TailOutOfRange { requested_ms: 8 })
        ));
    }

    #[test]
    fn construction_rejects_out_of_range_echo_delay() {
        let config = AecConfig {
            max_echo_delay_ms: 5,
            ..Default::default()
        };
        assert!(matches!(
            Aec::new(config),
            Err(AecError::EchoDelayOutOfRange { requested_ms: 5 })
        ));
    }

    /// Selecting the public model now yields a working canceller: a full block
    /// in produces a full block out, with no error path left for a validly
    /// named model.
    #[test]
    fn process_with_the_public_model_produces_output() {
        let config = AecConfig {
            delay_hint_ms: Some(16),
            ..Default::default()
        };
        assert_eq!(config.model, AecModel::Tau);
        let mut aec = Aec::new(config).unwrap();
        aec.feed_reference(&[0.1; 256]);
        let mut out = Vec::new();
        aec.process(&[0.2; 256], &mut out)
            .expect("the public model resolves to a working canceller");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn unaligned_processing_parks_rather_than_starves() {
        // With no hint and no lock there is no alignment yet: the whole block
        // is a search state, counted as parked, and starvation stays a pure
        // transport report.
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let mut out = Vec::new();
        let _ = aec.process(&[0.0; 128], &mut out);
        let metrics = aec.metrics();
        assert_eq!(metrics.acquisition_parked, 128);
        assert_eq!(metrics.reference_starved, 0);
    }

    #[test]
    fn default_offset_parks_at_the_frontier() {
        // With no delay hint the offset is zero and the near-end reads would
        // sit at the reference frontier; the engine parks them instead.
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        aec.feed_reference(&[0.5; 256]);
        let mut out = Vec::new();
        let _ = aec.process(&[0.5; 256], &mut out);
        let metrics = aec.metrics();
        assert_eq!(metrics.acquisition_parked, 256);
        assert_eq!(metrics.reference_starved, 0);
    }

    #[test]
    fn a_hinted_engine_counts_starvation_not_parking() {
        // A hint is an active alignment from the first sample, so unavailable
        // reference is genuine starvation and nothing is parked.
        let config = AecConfig {
            delay_hint_ms: Some(100),
            ..Default::default()
        };
        let mut aec = Aec::new(config).unwrap();
        let mut out = Vec::new();
        let _ = aec.process(&[0.5; 256], &mut out);
        let metrics = aec.metrics();
        assert_eq!(metrics.acquisition_parked, 0);
        assert_eq!(metrics.reference_starved, 256);
    }

    #[test]
    fn delay_hint_shifts_reads_into_the_retained_window() {
        // A 16 ms hint at 16 kHz is a 256-sample offset, exactly the frontier
        // anchor, so the near-end reads land on the fed reference and none starve.
        let config = AecConfig {
            delay_hint_ms: Some(16),
            ..Default::default()
        };
        let mut aec = Aec::new(config).unwrap();
        aec.feed_reference(&[0.5; 256]);
        let mut out = Vec::new();
        let _ = aec.process(&[0.5; 256], &mut out);
        assert_eq!(aec.metrics().reference_starved, 0);
    }

    /// A hinted engine at the lowest rate the configuration allows, so the
    /// derived ring is small enough to overflow inside a test.
    fn small_ring_config(hint_ms: u16) -> AecConfig {
        AecConfig {
            sample_rate: 8000,
            delay_hint_ms: Some(hint_ms),
            ..Default::default()
        }
    }

    #[test]
    fn reference_overflow_counts_drops_without_disturbing_the_alignment() {
        // Overwriting the oldest sample is what a full ring does on every push.
        // A consumer that keeps up is not affected by it, however long the
        // stream runs, so the anchor it was given must survive the whole of it.
        let config = small_ring_config(100);
        let capacity = derive_ring_capacity(&config);
        let mut aec = Aec::new(config).unwrap();
        let mut out = Vec::new();

        aec.feed_reference(&[0.25; 256]);
        aec.process(&[0.1; 256], &mut out).unwrap();
        let anchored = aec.anchor;
        assert!(anchored.is_some(), "the first process anchors the stream");

        // Twice the ring's whole depth, in matched turns.
        for _ in 0..(capacity / 256 * 2) {
            aec.feed_reference(&[0.25; 256]);
            aec.process(&[0.1; 256], &mut out).unwrap();
        }

        let metrics = aec.metrics();
        assert!(
            metrics.reference_dropped > capacity as u64 / 2,
            "the ring must have overflowed for this to be testing anything"
        );
        assert_eq!(
            aec.anchor, anchored,
            "a consumer that keeps up keeps its anchor"
        );
        assert_eq!(metrics.reference_reanchors, 0);
    }

    /// One matched turn at the 256-sample cadence the continuity cases use.
    /// The standing lead is exactly zero from the second turn onward, because
    /// the anchor was taken at the frontier and both counters advance together.
    fn matched_turns(aec: &mut Aec, turns: usize, out: &mut Vec<f32>) {
        for _ in 0..turns {
            aec.feed_reference(&[0.25; 256]);
            aec.process(&[0.1; 256], out).unwrap();
        }
    }

    /// Matched turns enough to establish the lead baseline with room to spare.
    const BASELINE_TURNS: usize = LEAD_BASELINE_BLOCKS as usize + 8;

    #[test]
    fn a_stalled_consumer_re_anchors_and_is_counted() {
        // A capture stall loses near samples the far stream did not lose, so
        // the near block that arrives next belongs later in the far stream than
        // the standing alignment says, and unlike a caller running momentarily
        // late it never catches up. The step is therefore a new plateau outside
        // everything this caller has ever done, and the model infers the loss
        // once the plateau has survived a second process boundary.
        let config = small_ring_config(100);
        let tail = ms_to_samples(config.tail_ms, config.sample_rate);
        let mut aec = Aec::new(config).unwrap();
        let mut out = Vec::new();

        matched_turns(&mut aec, BASELINE_TURNS, &mut out);
        let anchored = aec.anchor;
        assert!(anchored.is_some());
        assert_eq!(aec.metrics().reference_reanchors, 0);

        // The renderer keeps feeding through a stall of several tails.
        aec.feed_reference(&vec![0.25; tail as usize * 4]);
        aec.process(&[0.1; 256], &mut out).unwrap();
        assert_eq!(
            aec.anchor, anchored,
            "one block is a spike, not a plateau: the model holds"
        );
        assert_eq!(aec.metrics().reference_reanchors, 0);

        // The step is still there, and still exactly where it was, block after
        // block. That is what a lost span does; a caller merely running ahead
        // eats one block of its lead per call and this stays flat.
        let required = aec.lead.required_run();
        assert!(required >= LEAD_STEP_BLOCKS);
        for _ in 1..required {
            assert_eq!(aec.anchor, anchored, "not yet a believed plateau");
            aec.feed_reference(&[0.25; 256]);
            aec.process(&[0.1; 256], &mut out).unwrap();
        }
        assert_ne!(aec.anchor, anchored, "the stale alignment must be rebuilt");
        assert_eq!(aec.metrics().reference_reanchors, 1);

        // Back in step behind the rebuilt anchor: one stall, one re-anchor.
        let rebuilt = aec.anchor;
        matched_turns(&mut aec, 8, &mut out);
        assert_eq!(aec.anchor, rebuilt);
        assert_eq!(aec.metrics().reference_reanchors, 1);
    }

    #[test]
    fn the_re_anchor_boundary_is_learned_from_the_callers_own_lead_envelope() {
        // A caller that feeds one block and processes one block rests at a lead
        // of exactly zero and never moves, so its learned envelope has no span
        // at all and the margin falls back to the floor the model always keeps:
        // this caller's own block size, the granularity it delivers audio in.
        // The boundary is therefore 256 here because the CALLER is 256, not
        // because any constant in the crate says so.
        const BLOCK: u64 = 256;
        for (extra, expect_reanchor) in [(BLOCK, false), (BLOCK + 1, true)] {
            let mut aec = Aec::new(small_ring_config(100)).unwrap();
            let mut out = Vec::new();
            matched_turns(&mut aec, BASELINE_TURNS, &mut out);
            let anchored = aec.anchor;

            // A plateau, held for the run this caller's own oscillation asks
            // for: one block would be a spike.
            aec.feed_reference(&vec![0.25; 256 + extra as usize]);
            aec.process(&[0.1; 256], &mut out).unwrap();
            for _ in 1..LEAD_STEP_BLOCKS {
                aec.feed_reference(&[0.25; 256]);
                aec.process(&[0.1; 256], &mut out).unwrap();
            }

            assert_eq!(
                aec.anchor != anchored,
                expect_reanchor,
                "at a sustained lead of {extra} samples against a learned bound of {}",
                aec.lead.bound().unwrap_or(0)
            );
        }
    }

    /// The margin's block-size floor ages on the same rotation as the envelope
    /// it floors.
    ///
    /// One oversized block is legitimate (a host recovering from a scheduling
    /// stall hands over a long span at once) and is tolerated when it arrives.
    /// What is pinned here is the mechanism.
    #[test]
    fn the_learned_block_size_ages_out_of_the_margin() {
        const OVERSIZED: usize = 8192;
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turns(&mut aec, BASELINE_TURNS, &mut out);
        assert_eq!(aec.lead.block(), 256, "the granularity so far");

        // The oversized block, with the reference to match it. Nothing is lost,
        // and the step it puts in the lead is at most the block itself, so the
        // floor it sets covers it: a caller changing block size is tolerated by
        // construction.
        aec.feed_reference(&[0.25; OVERSIZED]);
        aec.process(&[0.1; OVERSIZED], &mut out).unwrap();
        assert_eq!(
            aec.metrics().reference_reanchors,
            0,
            "an oversized block is not a capture loss"
        );
        assert_eq!(
            aec.lead.block(),
            OVERSIZED as u64,
            "and it IS the caller's granularity while it is recent"
        );

        // Back to the ordinary cadence for long enough that both buckets rotate
        // past it.
        matched_turns(&mut aec, LEAD_WINDOW_BLOCKS as usize * 2, &mut out);
        assert_eq!(
            aec.lead.block(),
            256,
            "and it must be forgotten once it is not"
        );
        assert_eq!(
            aec.metrics().reference_reanchors,
            0,
            "with no seam forged anywhere along the way"
        );
    }

    /// Part four of the trigger, on its own. A lead that overshoots for a
    /// single block and comes back is the caller's transport being momentarily
    /// late, and a late caller catches up; a lost capture span cannot.
    #[test]
    fn a_lead_spike_that_comes_back_is_not_a_discontinuity() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turns(&mut aec, BASELINE_TURNS, &mut out);
        let anchored = aec.anchor;

        // A whole second of reference arrives early, and the caller then feeds
        // nothing until the near stream has caught back up to it.
        aec.feed_reference(&vec![0.25; 8000]);
        aec.process(&[0.1; 256], &mut out).unwrap();
        for _ in 0..(8000 / 256) {
            aec.process(&[0.1; 256], &mut out).unwrap();
        }
        matched_turns(&mut aec, 8, &mut out);

        assert_eq!(aec.anchor, anchored, "the caller was early, not short");
        assert_eq!(aec.metrics().reference_reanchors, 0);
    }

    /// Part one of the trigger. Before a baseline exists the model has seen
    /// nothing of this caller and refuses to infer, which is a deliberate
    /// warm-up hole: a loss inside the first [`LEAD_BASELINE_BLOCKS`] blocks is
    /// not inferred, and the alternative would be a model guessing at a caller
    /// it has never watched.
    #[test]
    fn a_step_before_the_baseline_exists_is_not_inferred() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turns(&mut aec, 4, &mut out);
        let anchored = aec.anchor;

        aec.feed_reference(&vec![0.25; 4000]);
        for _ in 0..4 {
            aec.feed_reference(&[0.25; 256]);
            aec.process(&[0.1; 256], &mut out).unwrap();
        }

        assert_eq!(aec.anchor, anchored);
        assert_eq!(aec.metrics().reference_reanchors, 0);
    }

    /// The structural backstop, which is not the learned model and does not
    /// wait for it. Past the ring's retained depth the standing alignment
    /// cannot be served by any sample the engine still holds, so every read
    /// starves; there is no judgement left to make.
    #[test]
    fn a_lead_past_the_rings_depth_re_anchors_even_with_no_baseline() {
        let config = small_ring_config(100);
        let capacity = derive_ring_capacity(&config);
        let mut aec = Aec::new(config).unwrap();
        let mut out = Vec::new();

        aec.feed_reference(&[0.25; 256]);
        aec.process(&[0.1; 256], &mut out).unwrap();
        let anchored = aec.anchor;

        aec.feed_reference(&vec![0.25; capacity + 256]);
        aec.process(&[0.1; 256], &mut out).unwrap();

        assert_ne!(aec.anchor, anchored);
        assert_eq!(aec.metrics().reference_reanchors, 1);
    }

    /// The lead is a transport quantity, so a reference stall (the far feed
    /// pausing while the near stream keeps going) drives it toward zero, never
    /// past the bound. A stall is not a timeline skip and must not forge a
    /// seam.
    #[test]
    fn a_reference_stall_with_no_timeline_skip_forges_no_seam() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turns(&mut aec, BASELINE_TURNS, &mut out);
        let anchored = aec.anchor;

        // The far feed stops outright for a second while capture continues.
        for _ in 0..(8000 / 256) {
            aec.process(&[0.1; 256], &mut out).unwrap();
        }
        // It resumes, and catches back up in one burst.
        aec.feed_reference(&vec![0.25; 8000]);
        matched_turns(&mut aec, 16, &mut out);

        assert_eq!(aec.anchor, anchored, "no near sample was ever lost");
        assert_eq!(aec.metrics().reference_reanchors, 0);
    }

    #[test]
    fn reset_clears_streaming_state() {
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        aec.feed_reference(&[0.5; 512]);
        let mut out = Vec::new();
        let _ = aec.process(&[0.5; 256], &mut out);
        aec.reset();
        let metrics = aec.metrics();
        assert_eq!(metrics.reference_starved, 0);
        assert_eq!(metrics.reference_dropped, 0);
        assert_eq!(aec.anchor, None);
        assert_eq!(aec.metrics().delay_samples, None);
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

    /// A far-end single-talk pair whose echo path begins exactly `bulk` samples
    /// after the far-end sample that caused it: broadband noise under a
    /// syllabic envelope, which is enough structure for the acquisition to
    /// promote a lock, owned here so this test needs no fixture and no data.
    fn delayed_pair(len: usize, bulk: usize) -> (Vec<f32>, Vec<f32>) {
        let mut carrier = Lcg(0x1234_5678);
        let mut shape = Lcg(0x00C0_FFEE);

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

        let taps = [(0_usize, 0.5_f32), (80, -0.25), (240, 0.12), (410, -0.06)];
        let mic = (0..len)
            .map(|i| {
                let mut echo = 0.0_f32;
                for &(tap, coeff) in &taps {
                    let lag = bulk + tap;
                    if i >= lag {
                        echo += coeff * far[i - lag];
                    }
                }
                0.5 * echo
            })
            .collect();
        (far, mic)
    }

    /// A caller whose reference chunk does not divide its near block still
    /// feeds and reads at the same rate, so it is not behind and nothing about
    /// its alignment is stale.
    #[test]
    fn a_mismatched_reference_chunk_never_makes_a_healthy_lock_suspect() {
        const TURN: usize = 256;
        // 160 is the pathological chunk: the reference frontier coincides with
        // a near-block boundary only every 1280 samples. The clip runs past the
        // ring's depth.
        const FAR_CHUNK: usize = 160;
        let rate = AecConfig::default().sample_rate as usize;
        let clip = rate * 7;
        let (far, mic) = delayed_pair(clip, rate / 10);

        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let mut out = Vec::new();
        let mut fed = 0usize;
        let mut near = 0usize;
        while near + TURN <= clip {
            while fed < near + TURN {
                let end = (fed + FAR_CHUNK).min(clip);
                aec.feed_reference(&far[fed..end]);
                fed = end;
            }
            aec.process(&mic[near..near + TURN], &mut out).unwrap();
            near += TURN;
        }

        let metrics = aec.metrics();
        assert!(
            metrics.reference_dropped > 0,
            "the clip must outrun the ring for this to be testing anything"
        );
        assert_eq!(
            metrics.reference_reanchors, 0,
            "a caller that keeps pace has not fallen behind, whatever its chunk size"
        );
        assert!(
            matches!(metrics.delay.status, DelayStatus::Locked(_)),
            "the acquisition must have promoted a lock: {:?}",
            metrics.delay.status
        );

        let acquirer = aec
            .acquirer
            .as_ref()
            .expect("no hint, so an estimator runs");
        assert_eq!(
            acquirer.tracker_suspect(),
            Some(false),
            "no re-anchor happened, so nothing reached the tracker as a discontinuity"
        );
    }

    // ---- Capture continuity ------------------------------------------------

    /// Drives one matched turn: feed a block, process a block. The standing
    /// frontier lag stays exactly zero this way, which is what makes the
    /// declared-discontinuity cases below test the declaration and not a lag
    /// the automatic rule would have caught on its own.
    fn matched_turn(aec: &mut Aec, far: &[f32], mic: &[f32], out: &mut Vec<f32>) {
        aec.feed_reference(far);
        aec.process(mic, out).expect("process succeeds");
    }

    /// Declaring continuity is a no-op, so a host that declares unconditionally
    /// on every callback pays nothing for the blocks where nothing happened.
    #[test]
    fn declaring_continuity_changes_nothing() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        let anchored = aec.anchor;

        aec.declare_capture_continuity(CaptureContinuity::Continuous);
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        assert_eq!(aec.anchor, anchored, "no re-anchor");
        let metrics = aec.metrics();
        assert_eq!(metrics.capture_discontinuities, 0);
        assert_eq!(metrics.capture_samples_lost, 0);
        assert_eq!(metrics.reference_reanchors, 0);
    }

    /// The whole point of the explicit path. With the near stream sitting
    /// exactly at the reference frontier there is no lag for the automatic rule
    /// to measure and nothing for it to infer, and it is right about that: the
    /// samples the capture device dropped left no trace in either stream. Only
    /// the host knows, and saying so re-anchors.
    #[test]
    fn a_declared_discontinuity_re_anchors_with_no_frontier_lag_at_all() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        for _ in 0..4 {
            matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        }
        let anchored = aec.anchor;
        assert_eq!(
            aec.ring
                .next_abs()
                .saturating_sub(aec.anchor.unwrap() + aec.near_processed),
            0,
            "the case is only meaningful at the standing lag of zero a matched \
             caller holds, where the automatic rule is nowhere near firing"
        );

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(800),
        });
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        assert_ne!(aec.anchor, anchored, "the alignment must be rebuilt");
        assert_eq!(aec.near_processed, 256, "counted from the new anchor");
        let metrics = aec.metrics();
        assert_eq!(metrics.capture_discontinuities, 1);
        assert_eq!(metrics.capture_samples_lost, 800);
        assert_eq!(
            metrics.reference_reanchors, 0,
            "the declaration supersedes the automatic rule; it does not borrow \
             its counter"
        );
    }

    /// A declaration is latched until a `process` consumes it, so a host that
    /// declares a discontinuity and then reports continuity before the next
    /// block still gets the re-anchor. Losing a real discontinuity costs the
    /// cancellation outright; a spurious one costs a re-anchor the engine
    /// performs at every stream start anyway.
    #[test]
    fn a_declared_discontinuity_survives_a_later_continuous_declaration() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        let anchored = aec.anchor;

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity { lost_samples: None });
        aec.declare_capture_continuity(CaptureContinuity::Continuous);
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        assert_ne!(aec.anchor, anchored);
        assert_eq!(aec.metrics().capture_discontinuities, 1);
        assert_eq!(
            aec.metrics().capture_samples_lost,
            0,
            "a host that supplied no count contributes nothing"
        );
    }

    /// Two declarations in one gap describe one seam in the near stream, so
    /// they latch into one re-anchor, and the losses they reported add.
    #[test]
    fn two_declarations_before_one_process_latch_into_one_re_anchor() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(300),
        });
        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(120),
        });
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        let metrics = aec.metrics();
        assert_eq!(
            metrics.capture_discontinuities, 1,
            "one seam, one re-anchor"
        );
        assert_eq!(metrics.capture_samples_lost, 420, "the reported losses add");
    }

    /// The declaration is consumed by the `process` it applies to and does not
    /// linger into the next one.
    #[test]
    fn a_declaration_applies_to_exactly_one_process() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity { lost_samples: None });
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        let rebuilt = aec.anchor;
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        assert_eq!(aec.anchor, rebuilt, "the second block re-anchors nothing");
        assert_eq!(aec.metrics().capture_discontinuities, 1);
    }

    /// A hinted engine runs no acquisition at all, and must still take the
    /// declaration: a capture device drops samples whether or not the caller
    /// measured the delay itself.
    #[test]
    fn a_hinted_engine_takes_a_declared_discontinuity() {
        let config = AecConfig {
            delay_hint_ms: Some(16),
            ..Default::default()
        };
        let mut aec = Aec::new(config).unwrap();
        assert!(aec.acquirer.is_none(), "a hint constructs no acquisition");
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        let anchored = aec.anchor;

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(64),
        });
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        assert_ne!(aec.anchor, anchored);
        assert_eq!(aec.metrics().capture_discontinuities, 1);
        assert_eq!(
            aec.metrics().delay_samples,
            Some(256),
            "the hint is the caller's own measurement and survives the seam"
        );
    }

    /// A pending declaration is streaming state, so the whole-stream reset
    /// clears it along with everything else.
    #[test]
    fn reset_clears_a_pending_declaration_and_its_counters() {
        let mut aec = Aec::new(small_ring_config(100)).unwrap();
        let mut out = Vec::new();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(64),
        });
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);
        assert_eq!(aec.metrics().capture_discontinuities, 1);

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(64),
        });
        aec.reset();
        matched_turn(&mut aec, &[0.25; 256], &[0.1; 256], &mut out);

        let metrics = aec.metrics();
        assert_eq!(metrics.capture_discontinuities, 0);
        assert_eq!(metrics.capture_samples_lost, 0);
    }

    /// The acquisition is told about a declared seam directly, and this is the
    /// test that proves the telling is load-bearing rather than decorative.
    ///
    /// At a standing lag of zero the re-anchor moves the block base by nothing
    /// the acquisition can see: its far-absolute cursor lands exactly where it
    /// expected the next block to start. The inferred path therefore has no
    /// jump to detect, while the evidence it is holding straddles the seam just
    /// the same. The lock survives as a PRIOR (the delay is unchanged) but is
    /// now suspect, which halves the rope its reacquisition trigger is given.
    #[test]
    fn a_declared_discontinuity_makes_the_acquisition_suspect_its_lock() {
        const TURN: usize = 256;
        let rate = AecConfig::default().sample_rate as usize;
        let clip = rate * 7;
        let (far, mic) = delayed_pair(clip, rate / 10);

        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let mut out = Vec::new();
        let mut cursor = 0usize;
        // One turn short of the clip: the last one is spent after the
        // declaration below.
        while cursor + 2 * TURN <= clip {
            matched_turn(
                &mut aec,
                &far[cursor..cursor + TURN],
                &mic[cursor..cursor + TURN],
                &mut out,
            );
            cursor += TURN;
        }

        let locked = aec.metrics().delay_samples;
        assert!(
            matches!(aec.metrics().delay.status, DelayStatus::Locked(_)),
            "the case needs a standing lock: {:?}",
            aec.metrics().delay.status
        );
        assert_eq!(
            aec.acquirer_mut().unwrap().tracker_suspect(),
            Some(false),
            "a healthy matched stream is not suspect"
        );
        // What the acquisition will be handed as its next block base if the
        // alignment is rebuilt here. It is exactly the base it already expects,
        // because the engine passes `anchor + near_processed` and the rebuilt
        // anchor lands on that same index.
        let expected_base = aec.anchor.unwrap() + aec.near_processed;

        aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
            lost_samples: Some(800),
        });
        matched_turn(
            &mut aec,
            &far[cursor..cursor + TURN],
            &mic[cursor..cursor + TURN],
            &mut out,
        );

        assert_eq!(
            aec.anchor,
            Some(expected_base),
            "the rebuilt anchor lands on the very index the acquisition already \
             expected, so the block base did not move and the inferred path had \
             no jump to detect: only the declaration could have told it"
        );
        assert_eq!(
            aec.acquirer_mut().unwrap().tracker_suspect(),
            Some(true),
            "the acquisition must learn of a seam it could not have inferred"
        );
        assert_eq!(
            aec.metrics().delay_samples,
            locked,
            "the standing delay is kept as a prior, not discarded: the timeline \
             moved, the echo path did not"
        );
        assert_eq!(aec.metrics().reference_reanchors, 0);
        assert_eq!(aec.metrics().capture_discontinuities, 1);
    }

    /// With nothing processed there is no partial block to drain, so flush
    /// appends nothing and leaves the caller's buffer untouched.
    #[test]
    fn flush_with_no_carry_appends_nothing() {
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let mut out = vec![1.0, 2.0];
        aec.flush(&mut out).expect("flush succeeds with no carry");
        assert_eq!(out, vec![1.0, 2.0]);
    }

    // ---- Output-transition policy (GRADED-REACQ) ---------------------------
    //
    // The gate math is proven in isolation (fade-out and fade-in sample counts,
    // ramp reversal, the [0, 1] invariant under status flaps); the engine tests
    // then prove the wiring: the default is byte-identical to PreserveCorrection
    // everywhere the status never reaches Reacquiring, and it fades only there.

    #[test]
    fn graded_gate_holds_full_correction_when_not_reacquiring() {
        // The target while not reacquiring is full correction, and the gain is
        // pinned to exactly 1.0 by the clamp, which is what lets the blend leave
        // the correction bit-for-bit untouched.
        let mut gate = GradedGate::new(1_600, 3_200);
        for _ in 0..5_000 {
            assert_eq!(gate.step(false), 1.0);
        }
    }

    #[test]
    fn graded_gate_fade_out_reaches_passthrough_after_exactly_fade_out_samples() {
        // 100 ms at 16 kHz. The ramp decreases strictly and reaches exactly 0.0
        // on the 1600th reacquiring sample, then holds.
        let mut gate = GradedGate::new(1_600, 3_200);
        let mut prev = 1.0_f32;
        for k in 1..1_600 {
            let g = gate.step(true);
            assert!(
                g < prev,
                "fade-out must decrease at step {k}: {g} !< {prev}"
            );
            assert!(
                g > 0.0,
                "fade-out must not reach passthrough early (step {k})"
            );
            prev = g;
        }
        assert_eq!(
            gate.step(true),
            0.0,
            "fade-out reaches passthrough at exactly fade_out_samples"
        );
        for _ in 0..500 {
            assert_eq!(
                gate.step(true),
                0.0,
                "and holds passthrough while reacquiring"
            );
        }
    }

    #[test]
    fn graded_gate_fade_in_reaches_full_correction_after_exactly_fade_in_samples() {
        // 200 ms at 16 kHz. From passthrough, the ramp increases strictly and
        // reaches exactly 1.0 on the 3200th non-reacquiring sample, then holds.
        let mut gate = GradedGate::new(1_600, 3_200);
        for _ in 0..1_600 {
            gate.step(true);
        }
        assert_eq!(gate.gain, 0.0, "must be at passthrough before the fade-in");
        let mut prev = 0.0_f32;
        for k in 1..3_200 {
            let g = gate.step(false);
            assert!(g > prev, "fade-in must increase at step {k}: {g} !> {prev}");
            assert!(
                g < 1.0,
                "fade-in must not reach full correction early (step {k})"
            );
            prev = g;
        }
        assert_eq!(
            gate.step(false),
            1.0,
            "fade-in reaches full correction at exactly fade_in_samples"
        );
        for _ in 0..500 {
            assert_eq!(gate.step(false), 1.0, "and holds full correction");
        }
    }

    #[test]
    fn graded_gate_reverses_from_the_current_gain_without_stepping() {
        // The anti-flap property. A re-lock halfway through the fade-out
        // reverses toward full correction FROM the current gain, by one up-step;
        // it does not jump to 0.0 or 1.0.
        let mut gate = GradedGate::new(1_600, 3_200);
        for _ in 0..800 {
            gate.step(true);
        }
        let mid = gate.gain;
        assert!(mid > 0.0 && mid < 1.0, "must be mid-fade, got {mid}");
        let after = gate.step(false);
        assert!(
            after > mid,
            "reversal rises from the current gain, not from 0"
        );
        assert!(after < 1.0, "reversal must not jump to full correction");
        assert_eq!(
            after,
            (mid + gate.up_step).min(1.0),
            "reversal continues from the current gain by one up-step"
        );

        // Symmetric: re-entering a reacquisition mid-fade-in reverses downward
        // from the current gain by one down-step.
        for _ in 0..200 {
            gate.step(false);
        }
        let up = gate.gain;
        let down = gate.step(true);
        assert!(
            down < up && down > 0.0,
            "reversal falls from the current gain"
        );
        assert_eq!(down, (up - gate.down_step).max(0.0));
    }

    #[test]
    fn graded_gate_stays_in_range_and_continuous_under_status_flaps() {
        // However the status flaps, the gain never leaves [0, 1] and never moves
        // by more than one step per sample, so the emitted signal is continuous.
        let mut gate = GradedGate::new(1_600, 3_200);
        let max_step = gate.down_step.max(gate.up_step);
        let mut prev = gate.gain;
        let mut state = 0x9E37_79B9_u32;
        for _ in 0..200_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let reacquiring = (state >> 16) & 1 == 0;
            let g = gate.step(reacquiring);
            assert!((0.0..=1.0).contains(&g), "gain left [0, 1]: {g}");
            assert!(
                (g - prev).abs() <= max_step + f32::EPSILON,
                "gain moved more than one step: {prev} -> {g}"
            );
            prev = g;
        }
    }

    // ---- Engine wiring ------------------------------------------------------

    /// The default engine config, but preserving the correction throughout: the
    /// pre-policy behavior, used as the byte-identity reference.
    fn preserve_config() -> AecConfig {
        AecConfig {
            output_transition: OutputTransitionPolicy::PreserveCorrection,
            ..AecConfig::default()
        }
    }

    fn assert_bits_identical(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len(), "output lengths differ");
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "sample {i} differs: {g} vs {e}");
        }
    }

    const POLICY_TURN: usize = 256;

    /// What the driver does to the stream once a lock stands.
    #[derive(Clone, Copy)]
    enum Seam {
        /// Nothing: converge and keep streaming.
        None,
        /// Force a spurious reacquisition at the given absolute turn index,
        /// emulating a trigger on a static delay.
        ForceReacquire { turn: usize },
        /// Lose `lag` near samples the far stream did not, and declare it.
        DeclaredLoss { lag: usize },
        /// Lose `lag` near samples the far stream did not and say nothing, so
        /// the engine infers the seam from the frontier lead.
        InferredLoss { lag: usize },
    }

    /// One engine run over a far/near pair at matched 256-sample turns.
    struct PolicyRun {
        out: Vec<f32>,
        /// Whether each processed turn published `Reacquiring`.
        reacquiring_per_turn: Vec<bool>,
        /// The delay status just before the seam was applied.
        status_before_seam: DelayStatus,
        /// Whether the run re-anchored (declared or inferred).
        reanchored: bool,
    }

    /// Drives `far`/`near` through a fresh engine at matched 256-sample turns,
    /// applying `seam` once after an eight-second convergence window. Feed and
    /// process are equal-sized, so the standing frontier lag is exactly zero and
    /// a declared or inferred loss is exactly the capture loss and nothing else.
    fn drive_policy(config: AecConfig, far: &[f32], near: &[f32], seam: Seam) -> PolicyRun {
        let rate = config.sample_rate as usize;
        let mut aec = Aec::new(config).expect("configuration is valid");
        let mut out = Vec::new();
        let mut reacq = Vec::new();
        let total = far.len().min(near.len());
        let converge = (rate * 8).min(total.saturating_sub(POLICY_TURN));
        let mut cursor = 0usize;
        let mut turn = 0usize;

        let force_turn = if let Seam::ForceReacquire { turn } = seam {
            Some(turn)
        } else {
            None
        };

        while cursor + POLICY_TURN <= converge {
            aec.feed_reference(&far[cursor..cursor + POLICY_TURN]);
            if force_turn == Some(turn) {
                aec.acquirer_mut()
                    .expect("no hint, so an estimator runs")
                    .force_reacquire();
            }
            aec.process(&near[cursor..cursor + POLICY_TURN], &mut out)
                .expect("process succeeds");
            reacq.push(matches!(
                aec.metrics().delay.status,
                DelayStatus::Reacquiring
            ));
            cursor += POLICY_TURN;
            turn += 1;
        }

        let status_before_seam = aec.metrics().delay.status;

        if let Seam::DeclaredLoss { lag } | Seam::InferredLoss { lag } = seam {
            let lag = lag.min(total - cursor);
            aec.feed_reference(&far[cursor..cursor + lag]);
            cursor += lag;
            if let Seam::DeclaredLoss { .. } = seam {
                aec.declare_capture_continuity(CaptureContinuity::Discontinuity {
                    lost_samples: Some(lag as u64),
                });
            }
        }

        while cursor + POLICY_TURN <= total {
            aec.feed_reference(&far[cursor..cursor + POLICY_TURN]);
            if force_turn == Some(turn) {
                aec.acquirer_mut()
                    .expect("no hint, so an estimator runs")
                    .force_reacquire();
            }
            aec.process(&near[cursor..cursor + POLICY_TURN], &mut out)
                .expect("process succeeds");
            reacq.push(matches!(
                aec.metrics().delay.status,
                DelayStatus::Reacquiring
            ));
            cursor += POLICY_TURN;
            turn += 1;
        }
        aec.flush(&mut out).expect("flush succeeds");

        let m = aec.metrics();
        PolicyRun {
            out,
            reacquiring_per_turn: reacq,
            status_before_seam,
            reanchored: m.reference_reanchors + m.capture_discontinuities > 0,
        }
    }

    #[test]
    fn the_graded_default_is_byte_identical_to_preserve_without_a_reacquisition() {
        // The bit-identity crux. With no reacquisition the gate never leaves
        // full correction, so the graded default is byte-identical to
        // PreserveCorrection over the whole stream: initial acquisition, the
        // lock, and steady state.
        let (far, near) = delayed_pair(16_000 * 10, 1_600);
        let graded = drive_policy(AecConfig::default(), &far, &near, Seam::None);
        let preserve = drive_policy(preserve_config(), &far, &near, Seam::None);
        assert!(
            matches!(graded.status_before_seam, DelayStatus::Locked(_)),
            "the pair must lock, or the test proves nothing: {:?}",
            graded.status_before_seam
        );
        assert!(
            graded.reacquiring_per_turn.iter().all(|&r| !r),
            "a static pair must never reacquire, which is WHY the outputs match"
        );
        assert_bits_identical(&graded.out, &preserve.out);
    }

    #[test]
    fn the_default_policy_leaves_startup_output_unchanged() {
        // The policy is Reacquiring-only, and initial acquisition never enters
        // Reacquiring, so the first two seconds are byte-identical under the
        // default. Startup is untouched.
        let (far, near) = delayed_pair(16_000 * 3, 1_600);
        let cut = 16_000 * 2;
        let graded = drive_policy(AecConfig::default(), &far[..cut], &near[..cut], Seam::None);
        let preserve = drive_policy(preserve_config(), &far[..cut], &near[..cut], Seam::None);
        assert!(
            graded.reacquiring_per_turn.iter().all(|&r| !r),
            "startup must never reacquire"
        );
        assert_bits_identical(&graded.out, &preserve.out);
    }

    #[test]
    fn a_reacquisition_fades_the_correction_toward_the_capture() {
        // The policy fires ONLY in Reacquiring. A forced reacquisition is applied
        // identically to a graded engine and a preserve engine, so their delay
        // decisions and canceller I/O are bit-identical and the ONLY difference
        // is the graded blend. Before the reacquisition the two are
        // byte-identical; during it the graded output fades toward the untouched
        // capture, staying a convex blend of the correction and the capture.
        let (far, near) = delayed_pair(16_000 * 12, 1_600);
        let force_turn = 505;
        let graded = drive_policy(
            AecConfig::default(),
            &far,
            &near,
            Seam::ForceReacquire { turn: force_turn },
        );
        let preserve = drive_policy(
            preserve_config(),
            &far,
            &near,
            Seam::ForceReacquire { turn: force_turn },
        );

        assert!(
            matches!(graded.status_before_seam, DelayStatus::Locked(_)),
            "the forced case is only meaningful once a lock stands: {:?}",
            graded.status_before_seam
        );
        assert_eq!(graded.out.len(), preserve.out.len());
        assert!(
            graded.reacquiring_per_turn.iter().any(|&r| r),
            "the forced trigger must publish Reacquiring"
        );

        let boundary = force_turn * POLICY_TURN;
        // Before the reacquisition (leaving a one-block margin for the status to
        // take effect): byte-identical.
        let safe = boundary.saturating_sub(POLICY_TURN);
        for (k, (&g, &p)) in graded.out[..safe]
            .iter()
            .zip(&preserve.out[..safe])
            .enumerate()
        {
            assert_eq!(
                g.to_bits(),
                p.to_bits(),
                "byte-identical before the reacquisition at sample {k}"
            );
        }
        // During and after: the fade engaged (a substantial span changed), every
        // sample is finite, and each graded sample is a convex blend of the
        // correction (the preserve output) and the lag-0 capture.
        let end = graded.out.len();
        let mut changed = 0usize;
        for (offset, ((&g, &corrected), &capture)) in graded.out[boundary..end]
            .iter()
            .zip(&preserve.out[boundary..end])
            .zip(&near[boundary..end])
            .enumerate()
        {
            let k = boundary + offset;
            assert!(g.is_finite(), "graded output must stay finite at {k}");
            let lo = corrected.min(capture) - 1e-6;
            let hi = corrected.max(capture) + 1e-6;
            assert!(
                g >= lo && g <= hi,
                "graded {g} must be a convex blend of correction {corrected} and \
                 capture {capture} at sample {k}"
            );
            if g.to_bits() != corrected.to_bits() {
                changed += 1;
            }
        }
        assert!(
            changed > 1_000,
            "the reacquisition must fade a substantial span; changed {changed} samples"
        );
    }

    #[test]
    fn a_declared_capture_loss_stays_locked_and_leaves_output_unchanged() {
        // A declared capture loss re-anchors the alignment, but the status stays
        // Locked (the re-anchor is itself the repair), so the gate holds full
        // correction and the graded output is byte-identical to Preserve.
        let (far, near) = delayed_pair(16_000 * 12, 1_600);
        let graded = drive_policy(
            AecConfig::default(),
            &far,
            &near,
            Seam::DeclaredLoss { lag: 1_600 },
        );
        let preserve = drive_policy(
            preserve_config(),
            &far,
            &near,
            Seam::DeclaredLoss { lag: 1_600 },
        );
        assert!(
            graded.reanchored,
            "the declared loss must actually re-anchor"
        );
        assert!(
            graded.reacquiring_per_turn.iter().all(|&r| !r),
            "a declared re-anchor must not enter Reacquiring"
        );
        assert_bits_identical(&graded.out, &preserve.out);
    }

    #[test]
    fn an_inferred_capture_loss_stays_locked_and_leaves_output_unchanged() {
        // The same as the declared case but the host says nothing, so the engine
        // infers the seam from the frontier lead. Still a re-anchor while Locked,
        // still byte-identical output.
        let (far, near) = delayed_pair(16_000 * 12, 1_600);
        let graded = drive_policy(
            AecConfig::default(),
            &far,
            &near,
            Seam::InferredLoss { lag: 6_000 },
        );
        let preserve = drive_policy(
            preserve_config(),
            &far,
            &near,
            Seam::InferredLoss { lag: 6_000 },
        );
        assert!(
            graded.reanchored,
            "the inferred loss must actually re-anchor"
        );
        assert!(
            graded.reacquiring_per_turn.iter().all(|&r| !r),
            "an inferred re-anchor must not enter Reacquiring"
        );
        assert_bits_identical(&graded.out, &preserve.out);
    }

    #[test]
    fn reset_restores_full_correction_and_clears_the_pending_mirror() {
        // A reset is the whole-stream restart: it returns the gate to full
        // correction and drops any pending samples, so a fade in progress does
        // not bleed into the next stream.
        let (far, near) = delayed_pair(16_000 * 10, 1_600);
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor + POLICY_TURN <= 16_000 * 8 {
            aec.feed_reference(&far[cursor..cursor + POLICY_TURN]);
            aec.process(&near[cursor..cursor + POLICY_TURN], &mut out)
                .unwrap();
            cursor += POLICY_TURN;
        }
        assert!(matches!(aec.metrics().delay.status, DelayStatus::Locked(_)));

        aec.acquirer_mut().unwrap().force_reacquire();
        aec.feed_reference(&far[cursor..cursor + POLICY_TURN]);
        aec.process(&near[cursor..cursor + POLICY_TURN], &mut out)
            .unwrap();
        cursor += POLICY_TURN;
        let gain_after_reacquire = match &aec.output_blend {
            OutputBlend::Graded(gate) => gate.gain,
            OutputBlend::Preserve => panic!("the default policy is graded"),
        };
        assert!(
            gain_after_reacquire < 1.0,
            "the reacquisition must have started fading: gain {gain_after_reacquire}"
        );
        // A few more turns so pending samples exist mid-fade.
        for _ in 0..3 {
            aec.feed_reference(&far[cursor..cursor + POLICY_TURN]);
            aec.process(&near[cursor..cursor + POLICY_TURN], &mut out)
                .unwrap();
            cursor += POLICY_TURN;
        }

        aec.reset();
        match &aec.output_blend {
            OutputBlend::Graded(gate) => {
                assert_eq!(gate.gain, 1.0, "reset restores full correction");
                assert!(
                    gate.pending_near.is_empty(),
                    "reset clears the pending mirror"
                );
                assert!(
                    gate.pending_gain.is_empty(),
                    "reset clears the pending gains"
                );
            }
            OutputBlend::Preserve => panic!("the default policy is graded"),
        }
    }

    #[test]
    fn a_flush_mid_fade_drains_the_blend_neutrally() {
        // Turns that do not divide the block leave the canceller holding a
        // partial block, and a mid-stream promote resets it, so this exercises
        // both the flush-through-blend path and the reset/pending-mirror
        // coupling (the flush debug-assert that emitted == pending is the
        // regression guard). The blend is length-neutral and finiteness-neutral:
        // a graded run matches a preserve run in length, and every sample stays
        // finite. Because the flush advances no ramp, it adds no final jump.
        const T: usize = 200;
        let (far, near) = delayed_pair(16_000 * 9, 1_600);
        let run = |config: AecConfig| -> Vec<f32> {
            let mut aec = Aec::new(config).unwrap();
            let mut out = Vec::new();
            let mut cursor = 0usize;
            let mut forced = false;
            while cursor + T <= 16_000 * 9 {
                aec.feed_reference(&far[cursor..cursor + T]);
                if !forced && cursor >= 16_000 * 8 {
                    aec.acquirer_mut()
                        .expect("no hint, so an estimator runs")
                        .force_reacquire();
                    forced = true;
                }
                aec.process(&near[cursor..cursor + T], &mut out).unwrap();
                cursor += T;
            }
            aec.flush(&mut out).unwrap();
            out
        };
        let graded = run(AecConfig::default());
        let preserve = run(preserve_config());
        assert_eq!(
            graded.len(),
            preserve.len(),
            "the blend changes no emitted length"
        );
        assert!(
            graded.iter().all(|s| s.is_finite()),
            "every emitted sample stays finite across the flush"
        );
    }

    // ---- Config edge cases: the fade durations are public input --------------
    //
    // `OutputTransitionPolicy::GradedReacquisition::{fade_out_ms, fade_in_ms}`
    // are `u32` fields a caller sets directly, so the ms-to-samples conversion
    // and the gate must stay well-defined for zero, for the whole `u32` range,
    // and at every supported rate: no division by zero, no non-finite gain, no
    // gain that leaves `[0, 1]`. The chosen zero-duration SEMANTICS is an
    // explicit immediate (one-sample) transition, not a config-validation error:
    // a zero fade collapses the linear ramp to its hardest cut. This is the
    // documented behavior of `GradedGate::new` (the `.max(1)` on the fade sample
    // count), and it keeps a caller who asks for an instant transition served
    // rather than rejected. Every non-zero-duration behavior is unchanged.

    #[test]
    fn fade_ms_to_samples_does_not_overflow_at_the_u32_ceiling() {
        // The product is `ms(u32) * sample_rate(u32)` computed in `u64`. The
        // widest case the engine can present is the maximum `u32` fade at the
        // maximum supported rate; it must not overflow `u64` (a debug build
        // panics on overflow, so a green test here is the proof).
        let samples = fade_ms_to_samples(u32::MAX, 48_000);
        assert_eq!(samples, u32::MAX as u64 * 48_000 / 1_000);
        assert!(samples < u64::MAX, "the conversion stays well inside u64");
        // Truncation toward zero at a rate that does not divide evenly.
        assert_eq!(fade_ms_to_samples(1, 44_100), 44);
        assert_eq!(fade_ms_to_samples(100, 44_100), 4_410);
        assert_eq!(fade_ms_to_samples(1, 8_000), 8);
    }

    #[test]
    fn zero_fade_out_is_an_immediate_one_sample_cut_to_passthrough() {
        // fade_out_ms = 0 -> zero fade samples -> the `.max(1)` makes the
        // down-step exactly 1.0, so the first reacquiring sample steps straight
        // to passthrough with no division by zero and no non-finite gain.
        let mut gate = GradedGate::new(0, 3_200);
        assert!(gate.down_step.is_finite());
        assert_eq!(gate.down_step, 1.0, "a zero fade-out is a one-sample cut");
        assert_eq!(
            gate.step(true),
            0.0,
            "reaches passthrough on the first sample"
        );
        assert_eq!(gate.step(true), 0.0, "and holds it");
    }

    #[test]
    fn zero_fade_in_is_an_immediate_one_sample_return_to_full_correction() {
        let mut gate = GradedGate::new(1_600, 0);
        assert!(gate.up_step.is_finite());
        assert_eq!(gate.up_step, 1.0, "a zero fade-in is a one-sample restore");
        // Drive to passthrough, then a single non-reacquiring sample restores it.
        for _ in 0..1_600 {
            gate.step(true);
        }
        assert_eq!(gate.gain, 0.0, "at passthrough before the restore");
        assert_eq!(
            gate.step(false),
            1.0,
            "restores full correction on one sample"
        );
        assert_eq!(gate.step(false), 1.0, "and holds it");
    }

    #[test]
    fn both_fades_zero_stay_finite_and_in_range_under_status_flaps() {
        // Both zero: every step is a one-sample hard switch. The gain must stay
        // exactly on the endpoints, never leave [0, 1], and never go non-finite,
        // however the status flaps.
        let mut gate = GradedGate::new(0, 0);
        let mut state = 0x1234_5678_u32;
        for _ in 0..50_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let reacquiring = (state >> 17) & 1 == 0;
            let g = gate.step(reacquiring);
            assert!(g.is_finite(), "gain must stay finite");
            assert!(
                g == 0.0 || g == 1.0,
                "a zero fade only ever sits on an endpoint: {g}"
            );
            assert_eq!(g, if reacquiring { 0.0 } else { 1.0 });
        }
    }

    #[test]
    fn very_large_fades_stay_finite_and_in_range_and_never_go_non_finite() {
        // A `u32::MAX` fade at 48 kHz is about 2.06e11 samples; its per-sample
        // step is about 4.85e-12, which is below f32 resolution near 1.0, so the
        // gain holds at exactly 1.0 (full correction). That is the correct limit
        // of an absurdly slow fade, not a stuck-gain BUG: the gain is finite and
        // in range, no division by zero occurred, and the policy simply
        // degenerates to holding the correction, exactly as PreserveCorrection
        // would. A merely large fade (100 s) does resolve and progresses.
        let mut huge = GradedGate::new(
            fade_ms_to_samples(u32::MAX, 48_000),
            fade_ms_to_samples(u32::MAX, 48_000),
        );
        assert!(huge.down_step.is_finite() && huge.down_step > 0.0);
        assert!(huge.up_step.is_finite() && huge.up_step > 0.0);
        for _ in 0..10_000 {
            let g = huge.step(true);
            assert!(g.is_finite() && (0.0..=1.0).contains(&g));
        }

        // A large-but-resolvable fade (100 s at 16 kHz) makes real progress
        // toward passthrough without ever leaving the range.
        let mut big = GradedGate::new(fade_ms_to_samples(100_000, 16_000), 3_200);
        assert!(big.down_step > 0.0 && big.down_step < 1.0);
        let start = big.gain;
        for _ in 0..10_000 {
            let g = big.step(true);
            assert!(g.is_finite() && (0.0..=1.0).contains(&g));
        }
        assert!(big.gain < start, "a resolvable large fade must progress");
    }

    #[test]
    fn fade_endpoints_are_pinned_exactly_regardless_of_block_alignment() {
        // The ramp advances per sample, so a fade whose sample count does not
        // divide the driving block size still lands EXACTLY on 0.0 and 1.0 by the
        // clamp, never overshooting into a negative or above-one gain. 777 and
        // 513 are coprime with the usual 256 block, so the endpoints never fall
        // on a block boundary. The `1.0 / n` step is not exact in f32, so the
        // endpoint is reached within a sample of `n` (once the running sum
        // crosses the clamp), NOT necessarily on exactly the n-th sample; the
        // load-bearing property is that the clamp pins it to exactly 0.0 / 1.0
        // and holds it there, never a value just past the endpoint.
        let mut gate = GradedGate::new(777, 513);
        for k in 1..=800 {
            let g = gate.step(true);
            assert!(
                (0.0..=1.0).contains(&g),
                "gain left [0, 1] at step {k}: {g}"
            );
        }
        assert_eq!(
            gate.gain, 0.0,
            "clamped to exactly passthrough by sample 800"
        );
        for _ in 0..500 {
            assert_eq!(gate.step(true), 0.0, "and pinned there, never below 0.0");
        }
        for k in 1..=550 {
            let g = gate.step(false);
            assert!(
                (0.0..=1.0).contains(&g),
                "gain left [0, 1] at step {k}: {g}"
            );
        }
        assert_eq!(
            gate.gain, 1.0,
            "clamped to exactly full correction by sample 550"
        );
        for _ in 0..500 {
            assert_eq!(gate.step(false), 1.0, "and pinned there, never above 1.0");
        }
        // Reversing from a fractional mid-point stays exactly in range and still
        // lands exactly on full correction.
        for _ in 0..300 {
            gate.step(true);
        }
        let mid = gate.gain;
        assert!(mid > 0.0 && mid < 1.0, "must be mid-fade, got {mid}");
        for _ in 0..2_000 {
            let g = gate.step(false);
            assert!((0.0..=1.0).contains(&g));
        }
        assert_eq!(
            gate.gain, 1.0,
            "reversal still lands exactly on full correction"
        );
    }

    #[test]
    fn engine_accepts_zero_and_max_fades_and_stays_finite_at_every_supported_rate() {
        // The engine must construct and run for the zero and `u32::MAX` fade
        // durations at every supported rate, producing only finite output. The
        // fade fields are never rejected: `Aec::new` validates the rate, tail,
        // and delay windows, and the fade durations are always well-defined.
        for &rate in &[8_000_u32, 16_000, 22_050, 44_100, 48_000] {
            for policy in [
                OutputTransitionPolicy::GradedReacquisition {
                    fade_out_ms: 0,
                    fade_in_ms: 0,
                },
                OutputTransitionPolicy::GradedReacquisition {
                    fade_out_ms: u32::MAX,
                    fade_in_ms: u32::MAX,
                },
            ] {
                let config = AecConfig {
                    sample_rate: rate,
                    output_transition: policy,
                    ..AecConfig::default()
                };
                let mut aec = Aec::new(config).expect("fade durations are never rejected");
                let block = rate as usize / 100; // 10 ms blocks
                let far = vec![0.1_f32; block * 40];
                let near = vec![0.05_f32; block * 40];
                let mut out = Vec::new();
                let mut cursor = 0;
                while cursor + block <= near.len() {
                    aec.feed_reference(&far[cursor..cursor + block]);
                    aec.process(&near[cursor..cursor + block], &mut out)
                        .expect("process succeeds");
                    cursor += block;
                }
                aec.flush(&mut out).expect("flush succeeds");
                assert!(
                    out.iter().all(|s| s.is_finite()),
                    "rate {rate}, policy {policy:?}: every sample must be finite"
                );
            }
        }
    }
}
