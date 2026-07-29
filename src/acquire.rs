//! The coarse-to-fine delay acquisition and the lock's afterlife: the global
//! scan, the sample-granular fine estimator, the promotion gate that decides
//! when a candidate becomes the stream's alignment, the local tracker that
//! follows the alignment once it exists, and the reacquisition triggers that
//! send a lock gone bad back to the global search.
//!
//! # Why a gate exists at all
//!
//! The fine estimator scans a fixed-width range and reports where the
//! correlation peaked. A peak in the interior of that range is a maximum of the
//! function; a peak pinned at either END of the range is not, and is what a
//! delay OUTSIDE the range produces. The gate refuses to adopt an edge-pinned
//! peak without corroboration. The correlation index maps to delay as
//! `origin + span - index`, so the far end of the range maps to a short delay
//! and the near end to the deepest delay the range can express.
//!
//! # What the gate requires
//!
//! A candidate is promoted to a trusted lock on one of three grounds:
//!
//! - the coarse global scan independently found a region that AGREES with it;
//! - no region is available at all, and the fine peak stands clear of both ends
//!   of its range and clears a confidence bar set higher than the corroborated
//!   one;
//! - the fine search never produced an agreeing peak inside a region the coarse
//!   scan is confident about, in which case the region itself is adopted, backed
//!   off toward the early side. Before this last resort fires, the scan is
//!   re-armed and must find the region a SECOND time, independently.
//!
//! Agreement is asymmetric: the consumer is a filter that models only causal
//! lags, so every tolerance and every rounding in this module is wider or biased
//! toward early.
//!
//! # After the lock
//!
//! [`AcquisitionState::Locked`] is no longer terminal. The fine estimator keeps
//! running, re-centred on a local window around the adopted delay, with its
//! accumulator restarted every [`TRACK_CYCLE_FRAMES`] scored frames so each
//! short cycle yields an independent local estimate. [`Tracker`] consumes that
//! cycle stream and decides per cycle whether to hold the alignment, move it
//! (a delay that drifts or steps within local reach is followed with no global
//! rescan and no canceller reset), or fire a [`ReacquireTrigger`], on which the
//! whole acquisition re-enters the global search while the engine keeps the
//! stale offset until something better is promoted. The coarse scan's near
//! chain is stopped while a lock stands; its far envelope keeps building.
//!
//! A stream that never locks is not abandoned either: when the coarse scan
//! gives up but confident far-end excitation keeps arriving, the scan is
//! re-armed.

use crate::coarse::{CoarseRegion, CoarseScan};
use crate::delay::{
    DelayEstimate, DelayEstimator, DelayLockSource, DelayStatus, FineLock, FineObservation,
    ReacquireTrigger, WindowSupport, LOCK_MARGIN_SAMPLES,
};
use crate::track::{CycleOutcome, TrackVerdict, Tracker, TRACK_CYCLE_FRAMES};

/// How far a fine peak must stand from either end of its search range before it
/// can be adopted without corroboration, in milliseconds.
///
/// A genuine delay shorter than this is not banned, only refused the
/// uncorroborated path: it is still reached through a corroborating coarse
/// region.
const FINE_EDGE_GUARD_MS: usize = 8;

/// The confidence a fine peak must reach to be adopted with no corroboration.
///
/// Uncorroborated promotion must cost strictly more evidence than corroborated
/// promotion.
const LOCAL_ONLY_CONFIDENCE: f64 = 20.0;

/// How much LATER than the coarse region a fine value may sit and still be
/// called agreement, in milliseconds. The tight side.
const AGREE_LATE_MS: usize = 32;

/// How much EARLIER than the coarse region a fine value may sit and still be
/// called agreement, in milliseconds. The wide side.
const AGREE_EARLY_MS: usize = 128;

/// Where the relocated fine window sits around a coarse region, as a fraction of
/// the window's own width placed BEFORE the region.
const RELOCATE_LEAD_NUM: usize = 3;
/// Denominator of [`RELOCATE_LEAD_NUM`].
const RELOCATE_LEAD_DEN: usize = 4;

/// Extra backoff applied when a coarse region is adopted as the alignment
/// outright, in milliseconds. Spent out of the filter tail on the side the
/// filter can absorb.
const COARSE_ADOPT_BACKOFF_MS: usize = 64;

/// How many times the fine search may be re-centred per acquisition round.
///
/// A reacquisition or a re-arm starts a fresh round with a fresh budget.
const MAX_RELOCATIONS: u32 = 1;

/// Fine frames observed with a region available before the region is adopted on
/// its own.
const COARSE_ONLY_PATIENCE_FRAMES: u32 = 4;

/// How closely an independently re-found coarse region must agree with the one
/// awaiting last-resort adoption, in milliseconds.
///
/// Re-finding the region inside this band is confirmation; re-finding it outside
/// is contradiction.
const REVERIFY_AGREE_MS: usize = 30;

/// How many times the last-resort re-verification may refuse a region before it
/// stands down for the acquisition, instead of re-arming the coarse scan yet
/// again.
///
/// The re-verification demands that a region found once be found a SECOND time,
/// independently, before it is adopted (see
/// [`try_coarse_only`](DelayAcquirer::try_coarse_only)). When the budget is
/// spent the last resort stands down and the coarse scan goes quiescent rather
/// than re-arming again; a genuinely past-ceiling delay is still refused rather
/// than adopted as a shallow alias.
const MAX_REVERIFY_REJECTS: u32 = 8;

/// Scored fine frames after the coarse scan gives up before it is re-armed.
///
/// Counted in SCORED frames, so this is a measure of far-end excitation, not
/// wall time: a stream that goes quiet after the scan gives up never re-arms it.
const REARM_SCORED_FRAMES: u32 = 16;

/// How many times the coarse re-arm interval may double after successive
/// re-arms that never produced a lock.
///
/// The interval doubles after each unsuccessful re-arm; the FIRST re-arm keeps
/// its original interval.
const REARM_BACKOFF_MAX: u32 = 16;

/// Consecutive settled tracking cycles before the tracking search idles.
///
/// A cycle is settled when the tracker is quiescent (see
/// [`Tracker::quiescent`]) AND the cycle produced a confident interior estimate
/// within [`TRACK_STABLE_MS`] of the standing alignment.
const TRACK_STABLE_CYCLES: u32 = 4;

/// How far a cycle estimate may sit from the alignment, AND from the previous
/// cycle's estimate, and still count the cycle as settled, in milliseconds.
///
/// Both comparisons are required. Agreement with the ALIGNMENT establishes there
/// is no standing offset; agreement with the PREVIOUS cycle's estimate
/// establishes the delay is not walking.
const TRACK_STABLE_MS: usize = 4;

/// The deepest the tracking idle ramps, in whole tracking cycles.
///
/// Bounds what the idle costs in detection latency: a delay that moves during
/// an idle is observed up to this many cycles late. The ramp doubles so a
/// short-lived lock never reaches the deep steps.
const TRACK_IDLE_MAX_CYCLES: u32 = 4;

/// How far off-centre the tracked delay may sit in its local window before the
/// window is re-centred on it, as a reciprocal fraction of the window span.
///
/// Re-centring discards the cycle's accumulation, so it is done on drift that
/// threatens the margin for following further movement, not on every move.
const TRACK_RECENTRE_DEN: usize = 4;

/// What the acquisition asks the engine to do with the alignment.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AcquireAction {
    /// A trusted lock was promoted. The engine adopts the offset and resets
    /// the canceller, whose coefficients were learned against the alignment
    /// that just changed; if the offset is essentially unchanged (a
    /// reacquisition that re-confirmed the standing lock), it keeps both.
    Promote(usize),
    /// The tracker moved the alignment. The engine adopts the offset WITHOUT
    /// resetting the canceller: the movement is small by construction, the
    /// learned filter state is still mostly right, and the canceller's own
    /// shadow adaptation absorbs the shift.
    Track(usize),
}

/// What one [`observe`](DelayAcquirer::observe) call produced.
///
/// A relocation cannot be carried out inside `observe`: re-correlating the
/// buffered frame against the new search range needs a fresh far window, and
/// only the engine holds the reference ring to assemble it. So a relocation
/// that leaves behind an abandoned-range candidate which would otherwise
/// promote is reported through [`rescan_pending`](ObserveOutcome::rescan_pending),
/// and the engine answers it by assembling the new window and calling
/// [`rescan`](DelayAcquirer::rescan).
pub(crate) struct ObserveOutcome {
    /// The alignment action this frame produced, if any. Always [`None`] when
    /// [`rescan_pending`](ObserveOutcome::rescan_pending) is set.
    pub(crate) action: Option<AcquireAction>,
    /// Whether the engine must assemble a far window at the just-relocated
    /// origin and call [`rescan`](DelayAcquirer::rescan).
    pub(crate) rescan_pending: bool,
}

impl ObserveOutcome {
    /// An outcome carrying an alignment action (or none) and no rescan.
    fn action(action: Option<AcquireAction>) -> ObserveOutcome {
        ObserveOutcome {
            action,
            rescan_pending: false,
        }
    }

    /// An outcome asking the engine to rescan the just-relocated frame.
    fn rescan() -> ObserveOutcome {
        ObserveOutcome {
            action: None,
            rescan_pending: true,
        }
    }
}

/// The outcome of a [`try_relocate`](DelayAcquirer::try_relocate) attempt.
enum Relocation {
    /// The fine search was not re-centred; the caller falls through to the
    /// promotion ladder exactly as before a relocation was ever attempted.
    NotRelocated,
    /// The fine search was re-centred, and the candidate measured against the
    /// abandoned range would NOT have promoted, so nothing more is owed this
    /// frame: the ordinary path would have produced no action here either.
    Relocated,
    /// The fine search was re-centred, and the candidate measured against the
    /// abandoned range WOULD have promoted through the corroborated arm, which
    /// applies no interior test. The engine must rescan the buffered frame
    /// against the new range and promote the correct interior candidate instead
    /// of the abandoned edge-pinned one.
    RelocatedRescan,
}

/// The acquisition state machine's states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcquisitionState {
    /// Both searches running, nothing promoted, fine search at its original
    /// origin.
    Searching,
    /// The fine search has been re-centred on a coarse region and is
    /// re-accumulating there.
    Relocated,
    /// A trusted alignment has been promoted and the tracker is following it.
    Locked,
}

/// The coarse-to-fine delay acquirer. See the module documentation.
pub(crate) struct DelayAcquirer {
    /// The cheap global scan.
    coarse: CoarseScan,
    /// The sample-granular local estimator.
    fine: DelayEstimator,

    /// The configured rate, kept for constructing per-lock trackers.
    sample_rate: u32,
    /// The fine search span in samples, cached.
    fine_span: usize,
    /// [`FINE_EDGE_GUARD_MS`] in samples.
    edge_guard: usize,
    /// [`AGREE_LATE_MS`] in samples.
    agree_late: usize,
    /// [`AGREE_EARLY_MS`] in samples.
    agree_early: usize,
    /// [`COARSE_ADOPT_BACKOFF_MS`] in samples.
    adopt_backoff: usize,
    /// [`REVERIFY_AGREE_MS`] in samples.
    reverify_agree: usize,

    /// The far-absolute index the next near sample carries. A discontinuity here
    /// means the engine re-anchored.
    next_near_abs: u64,
    /// Whether `next_near_abs` has been set at least once.
    anchored: bool,

    state: AcquisitionState,
    relocations: u32,
    /// Fine frames observed while a region was available. Drives the last-resort
    /// path; zeroed by a relocation.
    frames_with_region: u32,
    /// The region awaiting independent re-confirmation before the last resort
    /// may adopt it. While it stands, the re-armed scan is looking for it.
    pending_region: Option<CoarseRegion>,
    /// Regions that contradicted their re-confirmation and were refused.
    regions_rejected: u32,
    /// Scored fine frames since the coarse scan gave up, driving the re-arm.
    scored_since_exhausted: u32,
    /// The multiplier on the coarse re-arm interval, doubling after each
    /// unsuccessful re-arm. One until the first re-arm has been spent, so the
    /// first re-arm fires on exactly the frame it always did.
    rearm_backoff: u32,
    /// Times the coarse scan was re-armed after giving up.
    coarse_rearms: u32,

    /// The per-lock tracker, present exactly while a lock stands.
    tracker: Option<Tracker>,
    /// Scored fine frames in the current tracking cycle.
    cycle_scored: u32,
    /// The current tracking cycle's confident candidate, if one has appeared.
    cycle_candidate: Option<FineLock>,
    /// [`TRACK_STABLE_MS`] in samples.
    track_stable: usize,
    /// Consecutive settled tracking cycles.
    stable_cycles: u32,
    /// The previous observed cycle's confident interior estimate, for the
    /// not-walking half of the settled test.
    last_cycle_delay: Option<usize>,
    /// The current idle depth in tracking cycles, doubling while the lock stays
    /// settled and dropping to zero the moment it does not.
    idle_depth: u32,
    /// Near samples still to be withheld from the fine estimator. Withheld
    /// samples complete no frame, so no transform runs and no cycle concludes;
    /// the far-absolute grid advances regardless, which is what keeps a skip
    /// from being mistaken for a stream discontinuity.
    idle_samples: usize,
    /// Alignment updates applied by the tracker after the initial lock.
    tracking_moves: u32,
    /// Whether a reacquisition trigger has fired and no new lock has been
    /// promoted since.
    reacquiring: bool,
    /// Times a reacquisition trigger fired.
    reacquisitions: u32,
    /// The most recent reacquisition's trigger.
    last_trigger: Option<ReacquireTrigger>,
    /// The longest contradiction run seen since construction or reset.
    ///
    /// Held here rather than on the [`Tracker`] because a reacquisition
    /// replaces the tracker outright, and a high-water mark that resets with it
    /// would erase the very episode worth reporting.
    contradiction_run_max: u32,

    locked: Option<usize>,
    source: Option<DelayLockSource>,
}

impl DelayAcquirer {
    /// Constructs the acquisition for a fine window of `max_delay_ms` and a
    /// coarse ceiling of `ceiling_ms`, both at `sample_rate`.
    pub(crate) fn new(sample_rate: u32, max_delay_ms: u16, ceiling_ms: u16) -> DelayAcquirer {
        let fine = DelayEstimator::new(sample_rate, max_delay_ms);
        let fine_span = fine.max_delay();
        let per_ms = |ms: usize| (ms * sample_rate as usize) / 1000;
        DelayAcquirer {
            coarse: CoarseScan::new(sample_rate, ceiling_ms),
            fine,
            sample_rate,
            fine_span,
            edge_guard: per_ms(FINE_EDGE_GUARD_MS).max(1),
            agree_late: per_ms(AGREE_LATE_MS).max(1),
            agree_early: per_ms(AGREE_EARLY_MS).max(1),
            adopt_backoff: per_ms(COARSE_ADOPT_BACKOFF_MS),
            reverify_agree: per_ms(REVERIFY_AGREE_MS).max(1),
            next_near_abs: 0,
            anchored: false,
            state: AcquisitionState::Searching,
            relocations: 0,
            frames_with_region: 0,
            pending_region: None,
            regions_rejected: 0,
            scored_since_exhausted: 0,
            rearm_backoff: 1,
            coarse_rearms: 0,
            tracker: None,
            cycle_scored: 0,
            cycle_candidate: None,
            track_stable: per_ms(TRACK_STABLE_MS).max(1),
            stable_cycles: 0,
            last_cycle_delay: None,
            idle_depth: 0,
            idle_samples: 0,
            tracking_moves: 0,
            reacquiring: false,
            reacquisitions: 0,
            last_trigger: None,
            contradiction_run_max: 0,
            locked: None,
            source: None,
        }
    }

    /// The number of far-end samples one fine frame's window holds. Constant for
    /// the life of the acquirer: relocation moves the window, never resizes it.
    pub(crate) fn fine_window_len(&self) -> usize {
        self.fine.window_len()
    }

    /// The far-absolute index the current fine frame's far window starts at.
    ///
    /// Signed, because a window early in a stream legitimately reaches back past
    /// the first reference sample; the engine renders those indices as silence
    /// and reports them as missing head support.
    pub(crate) fn fine_window_start_abs(&self) -> i64 {
        self.next_near_abs as i64
            - self.fine.frame_len() as i64
            - self.fine.window_back_off() as i64
    }

    /// Declares that the next near sample carries far-absolute index `base`.
    ///
    /// Called once per [`Aec::process`](crate::Aec::process) block, before any
    /// sample is pushed. A `base` that does not continue the previous block
    /// means the engine re-anchored, which is how a seam the host never
    /// declared is INFERRED; the seam itself is handled in
    /// [`seam`](DelayAcquirer::seam).
    pub(crate) fn begin_block(&mut self, base: u64) {
        if self.anchored && base != self.next_near_abs {
            self.seam();
        }
        self.next_near_abs = base;
        self.anchored = true;
    }

    /// Declares a discontinuity the engine was TOLD about rather than inferred:
    /// the host reported that its capture stream lost samples, and the engine
    /// re-anchored for it.
    ///
    /// Does exactly what an inferred seam does, deliberately: the two describe
    /// the same physical event and differ only in who noticed it. Called
    /// directly rather than left to
    /// [`begin_block`](DelayAcquirer::begin_block) because a discontinuity
    /// declared while the near stream sits exactly at the reference frontier
    /// moves the block base by nothing at all, leaving no jump to infer the
    /// seam from while the evidence straddles it just the same.
    pub(crate) fn declare_discontinuity(&mut self) {
        self.seam();
    }

    /// Discards the evidence that straddles a seam in the near stream.
    ///
    /// Every partial frame in both stages spans the seam and would be
    /// correlated across a break in the very stream it measures, so both are
    /// dropped. A standing lock is put under suspicion rather than dropped: a
    /// seam usually preserves the physical delay, and the tracker either
    /// re-confirms the lock on fresh post-seam evidence or escalates to
    /// [`ReacquireTrigger::Discontinuity`]. Idempotent, so an inferred seam
    /// landing on the same block as a declared one costs nothing.
    fn seam(&mut self) {
        self.fine.restart();
        self.coarse.discard_near();
        self.cycle_scored = 0;
        self.cycle_candidate = None;
        // A re-anchor is exactly the evidence that the lock may no longer
        // describe the stream, so the tracking search returns to full
        // cadence for it rather than finishing an idle granted by
        // pre-seam evidence.
        self.unsettle();
        if let Some(tracker) = &mut self.tracker {
            tracker.on_discontinuity();
        }
    }

    /// Appends far-end reference samples, exactly as the reference ring received
    /// them and immediately after it received them, so the two absolute grids
    /// cannot drift.
    pub(crate) fn push_far(&mut self, reference: &[f32]) {
        self.coarse.push_far(reference);
    }

    /// Appends one near-end sample, returning whether that completed a FINE
    /// frame, which the caller must then supply a far window for.
    ///
    /// The coarse stage advances inside this call and needs no cooperation from
    /// the caller: it owns its own decimated far history. While a lock stands,
    /// only the fine stage runs (it is the tracker's instrument); the coarse
    /// near chain is stopped, which is what keeps the steady-state cost of a
    /// locked stream to one fine frame per cycle.
    /// While a settled lock idles, the sample is withheld from the fine
    /// estimator so no frame completes and no transform runs. The withholding
    /// happens HERE and not in the engine's loop, and `next_near_abs` advances
    /// either way, because that counter is the acquisition's far-absolute grid
    /// cursor: it is what
    /// [`begin_block`](DelayAcquirer::begin_block) compares the engine's block
    /// base against to detect a re-anchor, and what
    /// [`fine_window_start_abs`](DelayAcquirer::fine_window_start_abs) derives
    /// the far window from. Stalling it would make every idle look like a
    /// stream discontinuity, which puts the lock under suspicion and halves the
    /// threshold its reacquisition trigger fires at: a changed decision, not a
    /// cheaper one.
    pub(crate) fn push_near(&mut self, sample: f32) -> bool {
        if self.state != AcquisitionState::Locked
            && self.coarse.push_near(sample, self.next_near_abs)
        {
            self.coarse.observe();
        }
        let complete = if self.idle_samples > 0 {
            self.idle_samples -= 1;
            false
        } else {
            self.fine.push_near(sample)
        };
        self.next_near_abs += 1;
        complete
    }

    /// Scores the completed fine frame and returns the alignment action, if
    /// this frame produced one.
    ///
    /// [`AcquireAction::Promote`] is returned when a trusted lock is promoted,
    /// which is the only thing that resets the canceller.
    /// [`AcquireAction::Track`] is returned when the tracker moves a standing
    /// alignment, which deliberately does not.
    pub(crate) fn observe(&mut self, far_window: &[f32], support: WindowSupport) -> ObserveOutcome {
        let obs = self.fine.observe(far_window, support);
        if self.state == AcquisitionState::Locked {
            return ObserveOutcome::action(self.observe_tracking(obs));
        }

        let lock = obs.candidate;
        let region = self.coarse.region();
        if region.is_some() {
            self.frames_with_region = self.frames_with_region.saturating_add(1);
        }
        self.maybe_rearm_after_give_up(obs.scored);

        match self.try_relocate(lock, region) {
            // The frame just relocated away from held a candidate that would
            // have promoted an edge-pinned value from the abandoned range. The
            // engine rescans it against the new range instead; nothing else
            // runs this frame.
            Relocation::RelocatedRescan => return ObserveOutcome::rescan(),
            // A relocation the abandoned candidate would not have promoted
            // through: the ordinary path below would have produced no action
            // this frame either, so return none and leave it to new evidence.
            Relocation::Relocated => return ObserveOutcome::action(None),
            // No relocation: fall through to the promotion ladder unchanged.
            Relocation::NotRelocated => {}
        }
        if let Some(delay) = self.try_promote(lock, region) {
            return ObserveOutcome::action(Some(AcquireAction::Promote(delay)));
        }
        ObserveOutcome::action(self.try_coarse_only(region).map(AcquireAction::Promote))
    }

    /// Re-correlates the frame just relocated away from against the NEW fine
    /// search range, and promotes it only if it is a valid INTERIOR candidate
    /// that agrees with the corroborating region.
    ///
    /// Called by the engine after [`observe`](DelayAcquirer::observe) reported
    /// [`ObserveOutcome::rescan_pending`], with `far_window` assembled at the
    /// origin the relocation moved to. This is what recovers the correct
    /// alignment on the exact frame the abandoned-range candidate would
    /// otherwise have promoted an edge-pinned value from: the buffered frame is
    /// re-used rather than discarded, so the correctness is bought without the
    /// fine-frame of acquisition latency that discarding it would cost. The
    /// interior test the corroborated ([`DelayLockSource::GlobalAgreement`])
    /// arm lacks is applied here, so an edge-pinned rescan peak is refused
    /// exactly as the abandoned-range one now is.
    pub(crate) fn rescan(
        &mut self,
        far_window: &[f32],
        support: WindowSupport,
    ) -> Option<AcquireAction> {
        let obs = self.fine.rescan(far_window, support);
        let region = self.coarse.region().or(self.pending_region)?;
        let lock = obs.candidate?;
        if self.peak_interior(lock.peak_at) && self.agrees(lock.delay, region.delay) {
            Some(AcquireAction::Promote(
                self.promote(lock.delay, DelayLockSource::GlobalAgreement),
            ))
        } else {
            None
        }
    }

    /// One tracking step: banks the frame into the current cycle and, when the
    /// cycle concludes, asks the tracker for its verdict and carries it out.
    fn observe_tracking(&mut self, obs: FineObservation) -> Option<AcquireAction> {
        if obs.scored {
            self.cycle_scored += 1;
            if let Some(candidate) = obs.candidate {
                self.cycle_candidate = Some(candidate);
            }
        }
        if self.cycle_scored < TRACK_CYCLE_FRAMES {
            return None;
        }

        let outcome = match self.cycle_candidate {
            Some(candidate) => CycleOutcome::Candidate {
                delay: candidate.delay,
                interior: self.peak_interior(candidate.peak_at),
            },
            None => CycleOutcome::Unconfident,
        };
        self.cycle_scored = 0;
        self.cycle_candidate = None;

        let alignment = self.locked.unwrap_or(0);
        let tracker = self.tracker.as_mut()?;
        let verdict = tracker.on_cycle(alignment, outcome);
        let quiescent = tracker.quiescent();
        // Purely observational: the run this cycle left standing, banked so a
        // contradiction that never escalates is still visible afterwards. No
        // decision reads it.
        let contradiction_run = tracker.jump_cycles();
        self.contradiction_run_max = self.contradiction_run_max.max(contradiction_run);
        match verdict {
            TrackVerdict::Hold => {
                self.fine.restart();
                self.settle(alignment, outcome, quiescent);
                None
            }
            TrackVerdict::Move(delay) => {
                self.unsettle();
                self.locked = Some(delay);
                self.tracking_moves = self.tracking_moves.saturating_add(1);
                self.recentre_tracking_window(delay);
                Some(AcquireAction::Track(delay))
            }
            TrackVerdict::Reacquire(trigger) => {
                self.enter_reacquire(trigger);
                None
            }
        }
    }

    /// Scores a held cycle for settledness and, once enough of them have
    /// agreed in a row, idles the tracking search for a bounded span.
    ///
    /// The two conditions are independent and both are required. The tracker
    /// being quiescent says no partial evidence is banked, so no cycle that
    /// goes unobserved could have completed an escalation. The estimate sitting
    /// within [`TRACK_STABLE_MS`] of the alignment says the delay is not
    /// walking, which the tracker's own dead bands do not establish: a drifting
    /// delay holds inside the early band for a long time while its estimates
    /// march steadily away. Idling on the first without the second would
    /// observe a drift crossing its band late.
    fn settle(&mut self, alignment: usize, outcome: CycleOutcome, quiescent: bool) {
        let estimate = match outcome {
            CycleOutcome::Candidate {
                delay,
                interior: true,
            } => Some(delay),
            _ => None,
        };
        let previous = self.last_cycle_delay.replace(match estimate {
            Some(delay) => delay,
            // An unconfident or edge-pinned cycle carries no delay comparable
            // to the next one, so it breaks the chain rather than seeding it.
            None => {
                self.unsettle();
                return;
            }
        });
        let settled = quiescent
            && estimate.is_some_and(|delay| {
                delay.abs_diff(alignment) <= self.track_stable
                    && previous.is_some_and(|prev| delay.abs_diff(prev) <= self.track_stable)
            });
        if !settled {
            self.stable_cycles = 0;
            self.idle_depth = 0;
            self.idle_samples = 0;
            return;
        }
        self.stable_cycles = self.stable_cycles.saturating_add(1);
        if self.stable_cycles < TRACK_STABLE_CYCLES {
            return;
        }
        self.idle_depth = (self.idle_depth * 2).clamp(1, TRACK_IDLE_MAX_CYCLES);
        self.idle_samples =
            self.idle_depth as usize * TRACK_CYCLE_FRAMES as usize * self.fine.frame_len();
    }

    /// Returns the tracking search to full cadence at once, and discards any
    /// idle already granted. Every departure from a settled lock runs through
    /// here: an unsettled cycle, a move, a discontinuity, a reacquisition.
    fn unsettle(&mut self) {
        self.stable_cycles = 0;
        self.idle_depth = 0;
        self.idle_samples = 0;
        // Evidence from before a seam, a move or a fresh lock is not comparable
        // to evidence after it, so the not-walking chain starts over too.
        self.last_cycle_delay = None;
    }

    /// Whether a raw fine peak index stands clear of both ends of the range.
    fn peak_interior(&self, peak_at: usize) -> bool {
        peak_at >= self.edge_guard && peak_at + self.edge_guard <= self.fine_span
    }

    /// The tracking window origin for a given delay: the delay centred, capped
    /// at the coarse ceiling so successive moves cannot walk the window past
    /// the depth the reference ring provably retains.
    fn tracking_origin(&self, delay: usize) -> usize {
        delay
            .min(self.coarse.ceiling_samples())
            .saturating_sub(self.fine_span / 2)
    }

    /// Re-centres the tracking window when the tracked delay has drifted far
    /// enough off-centre to threaten the margin for following further
    /// movement; otherwise just restarts the cycle accumulator.
    fn recentre_tracking_window(&mut self, delay: usize) {
        let (low, _) = self.fine.search_range();
        let centre = low + self.fine_span / 2;
        if delay.abs_diff(centre) > self.fine_span / TRACK_RECENTRE_DEN {
            self.fine.relocate(self.tracking_origin(delay));
        } else {
            self.fine.restart();
        }
    }

    /// Leaves the lock for a fresh global search, keeping the stale value on
    /// report so the engine can hold its offset until something better exists.
    fn enter_reacquire(&mut self, trigger: ReacquireTrigger) {
        self.state = AcquisitionState::Searching;
        self.reacquiring = true;
        self.reacquisitions = self.reacquisitions.saturating_add(1);
        self.last_trigger = Some(trigger);
        self.tracker = None;
        self.cycle_scored = 0;
        self.cycle_candidate = None;
        self.unsettle();
        self.coarse.rearm();
        self.fine.relocate(0);
        self.relocations = 0;
        self.frames_with_region = 0;
        self.pending_region = None;
        self.scored_since_exhausted = 0;
        // A reacquisition is a fresh acquisition, so the give-up retry starts
        // over at its original interval rather than inheriting the back-off a
        // previous unsuccessful round accumulated.
        self.rearm_backoff = 1;
    }

    /// Re-arms the coarse scan when it has given up but confident far-end
    /// excitation keeps arriving, so an echo path that appears late in a
    /// stream is still acquired. Excitation is measured in SCORED fine frames:
    /// a stream that goes quiet re-arms nothing.
    fn maybe_rearm_after_give_up(&mut self, scored: bool) {
        if !self.coarse.exhausted() {
            return;
        }
        if scored {
            self.scored_since_exhausted += 1;
        }
        // The pending region's lifetime is deliberately NOT on the cost
        // schedule. It is the only thing standing between an arriving fine
        // candidate and the uncorroborated promotion path: while it is Some,
        // `try_promote` corroborates against it instead of applying the raised
        // local-evidence bar, so stretching its life would refuse promotions
        // that fire today and admit weak ones that do not. It therefore still
        // expires on exactly the frame it always did, and only the coarse
        // re-arm below backs off.
        if self.scored_since_exhausted == REARM_SCORED_FRAMES {
            self.pending_region = None;
        }
        if self.scored_since_exhausted < REARM_SCORED_FRAMES * self.rearm_backoff {
            return;
        }
        self.scored_since_exhausted = 0;
        self.rearm_backoff = (self.rearm_backoff * 2).min(REARM_BACKOFF_MAX);
        self.coarse.rearm();
        self.coarse_rearms = self.coarse_rearms.saturating_add(1);
        // A fresh acquisition round: the relocation budget and the last-resort
        // bookkeeping start over. The fine search returns to the global origin
        // only if a relocation had moved it, so a slowly accumulating
        // uncorroborated case at the original origin keeps its evidence.
        if self.fine.search_range().0 != 0 {
            self.fine.relocate(0);
        }
        self.relocations = 0;
        self.frames_with_region = 0;
        self.pending_region = None;
        self.state = AcquisitionState::Searching;
    }

    /// Re-centres the fine search when the coarse region lies outside the range
    /// the fine search covers, or when the fine peak flatly disagrees with it,
    /// and reports whether the frame it left behind must be rescanned.
    ///
    /// A relocation abandons the range this frame's candidate was measured
    /// against. When that candidate would have promoted through the
    /// corroborated arm (the arm applies no interior test, so an edge-pinned
    /// peak agreeing with the region promotes there uninspected), the frame is
    /// rescanned against the new range rather than discarded, which is what
    /// recovers the correct interior alignment at no acquisition-latency cost.
    /// Otherwise the ordinary path would have produced nothing this frame, so
    /// the relocation is silent.
    fn try_relocate(&mut self, lock: Option<FineLock>, region: Option<CoarseRegion>) -> Relocation {
        if self.state != AcquisitionState::Searching || self.relocations >= MAX_RELOCATIONS {
            return Relocation::NotRelocated;
        }
        let Some(region) = region else {
            return Relocation::NotRelocated;
        };
        let (low, high) = self.fine.search_range();
        let inside =
            region.delay >= low + self.edge_guard && region.delay + self.edge_guard <= high;
        let disagrees = lock.is_some_and(|l| !self.agrees(l.delay, region.delay));
        if inside && !disagrees {
            return Relocation::NotRelocated;
        }

        // Whether the candidate measured against the range now being abandoned
        // would have promoted through `try_promote`'s corroborated arm. That
        // arm agrees a fine value with the region and adopts it with no interior
        // test, so an edge-pinned peak that happens to agree promotes there: the
        // exact hazard the rescan replaces. On this frame `region.or(pending)`
        // is this same `region`, so the condition matches `try_promote`'s.
        let would_promote_abandoned = lock.is_some_and(|l| self.agrees(l.delay, region.delay));

        // Place the window mostly BEFORE the region: the fine peak is expected
        // earlier than the coarse envelope centroid, and running off the early
        // end costs taps while running off the late end costs coverage.
        let lead = self.fine_span * RELOCATE_LEAD_NUM / RELOCATE_LEAD_DEN;
        self.fine.relocate(region.delay.saturating_sub(lead));
        self.state = AcquisitionState::Relocated;
        self.relocations += 1;
        self.frames_with_region = 0;

        if would_promote_abandoned {
            Relocation::RelocatedRescan
        } else {
            Relocation::Relocated
        }
    }

    /// Promotes this frame's fine candidate when the coarse scan corroborates
    /// it, or when nothing is available to corroborate it and it is strong and
    /// clearly interior.
    ///
    /// A region awaiting re-confirmation still corroborates: the fine stage
    /// agreeing with it is exactly the independent second finding the
    /// re-confirmation exists to demand.
    fn try_promote(
        &mut self,
        lock: Option<FineLock>,
        region: Option<CoarseRegion>,
    ) -> Option<usize> {
        let lock = lock?;
        match region.or(self.pending_region) {
            Some(region) if self.agrees(lock.delay, region.delay) => {
                Some(self.promote(lock.delay, DelayLockSource::GlobalAgreement))
            }
            // A region exists and this candidate disagrees with it: keep
            // accumulating. Relocation has already had its chance, and the
            // last-resort path will take the region if the fine never agrees.
            Some(_) => None,
            None => {
                if self.peak_interior(lock.peak_at) && lock.ratio >= LOCAL_ONLY_CONFIDENCE {
                    Some(self.promote(lock.delay, DelayLockSource::LocalEvidence))
                } else {
                    None
                }
            }
        }
    }

    /// Adopts the coarse region itself once the fine search has had a fair
    /// chance and has not agreed with it, and the re-armed scan has found the
    /// region a second time, independently.
    fn try_coarse_only(&mut self, region: Option<CoarseRegion>) -> Option<usize> {
        if let Some(prev) = self.pending_region {
            // Re-verification is running. Nothing happens until the re-armed
            // scan confirms a region of its own.
            let region = region?;
            if region.delay.abs_diff(prev.delay) <= self.reverify_agree {
                // Found again where it was found before: a real region. One
                // coarse bin of quantization, the standard safety margin, and
                // the centroid-to-onset backoff, all spent early.
                let delay = region
                    .delay
                    .saturating_sub(self.coarse.bin_samples())
                    .saturating_sub(LOCK_MARGIN_SAMPLES)
                    .saturating_sub(self.adopt_backoff);
                return Some(self.promote(delay, DelayLockSource::CoarseRegion));
            }
            // Found somewhere unrelated: the previous finding was not a
            // region, it was structure that does not reproduce. Refuse it, and
            // while the rejection budget still has room demand the same of the
            // next one by re-arming for another independent look. Once the
            // budget is spent the regions are wander, not a path: give up on the
            // pending region and STOP re-arming, so the coarse scan is not pinned
            // at full duty re-confirming spurious regions for the rest of the
            // stream. See [`MAX_REVERIFY_REJECTS`].
            self.regions_rejected = self.regions_rejected.saturating_add(1);
            self.pending_region = (self.regions_rejected < MAX_REVERIFY_REJECTS).then_some(region);
            if self.pending_region.is_some() {
                self.coarse.rearm();
            }
            return None;
        }
        // The last resort only opens while the re-verification budget has room.
        // Once it is spent for this acquisition it stays shut, so the first-entry
        // path below cannot re-open the very spin the reject arm just closed:
        // `coarse.region()` still returns the last refused region, so without
        // this guard the patience gate would set it pending and re-arm again.
        //
        // STAND-DOWN: after eight (`MAX_REVERIFY_REJECTS`) failed coarse-only
        // re-verifications the coarse-only last-resort path stands down for the
        // rest of the stream, until the AEC is reset (`reset` below zeroes
        // `regions_rejected`). This is intended and acceptable: normal
        // corroborated acquisition (the `try_promote` ladder in `observe`, run
        // before this last resort) remains available throughout, so a genuine
        // reproducible path still locks the ordinary way after stand-down. The
        // state is observable through the public `AecMetrics`: see
        // `DelayEstimate::coarse_last_resort_exhausted` and the count it derives
        // from, `coarse_regions_rejected`.
        if self.regions_rejected >= MAX_REVERIFY_REJECTS {
            return None;
        }
        if self.frames_with_region < COARSE_ONLY_PATIENCE_FRAMES {
            return None;
        }
        let region = region?;
        self.pending_region = Some(region);
        self.frames_with_region = 0;
        self.coarse.rearm();
        None
    }

    /// Whether a fine value and a coarse region describe the same delay, under
    /// the asymmetric tolerance the causal filter downstream requires.
    fn agrees(&self, fine: usize, coarse: usize) -> bool {
        if fine >= coarse {
            fine - coarse <= self.agree_late
        } else {
            coarse - fine <= self.agree_early
        }
    }

    /// Records a trusted lock and starts tracking it.
    fn promote(&mut self, delay: usize, source: DelayLockSource) -> usize {
        self.locked = Some(delay);
        self.source = Some(source);
        self.state = AcquisitionState::Locked;
        self.reacquiring = false;
        self.pending_region = None;
        self.scored_since_exhausted = 0;
        self.rearm_backoff = 1;
        self.tracker = Some(Tracker::new(self.sample_rate));
        self.cycle_scored = 0;
        self.cycle_candidate = None;
        // A freshly promoted lock has no settled history: the tracking search
        // runs at full cadence until it earns an idle.
        self.unsettle();
        self.fine.relocate(self.tracking_origin(delay));
        delay
    }

    /// The acquisition's observable state.
    pub(crate) fn estimate(&self) -> DelayEstimate {
        let (low, high) = self.fine.search_range();
        let scan_peak = self.fine.last_scan_peak_at();
        DelayEstimate {
            status: match (self.state, self.source, self.reacquiring) {
                (AcquisitionState::Locked, Some(source), _) => DelayStatus::Locked(source),
                (_, _, true) => DelayStatus::Reacquiring,
                (AcquisitionState::Relocated, _, _) => DelayStatus::Relocated,
                _ => DelayStatus::Searching,
            },
            delay_samples: self.locked,
            fine_search_start_samples: low,
            fine_search_end_samples: high,
            coarse_ceiling_samples: self.coarse.ceiling_samples(),
            coarse_region_samples: self.coarse.region().map(|r| r.delay),
            coarse_bin_samples: self.coarse.bin_samples(),
            coarse_correlation: self.coarse.correlation(),
            beyond_ceiling: self.coarse.beyond_ceiling(),
            coarse_frames: self.coarse.frames(),
            fine_frames: self.fine.frames(),
            fine_frames_skipped: self.fine.skipped(),
            relocated: self.relocations > 0,
            fine_scans: self.fine.scans(),
            fine_last_ratio: self.fine.last_scan_ratio(),
            fine_last_delay_samples: scan_peak.map(|peak_at| {
                (self.fine.last_scan_origin() + self.fine_span - peak_at)
                    .saturating_sub(LOCK_MARGIN_SAMPLES)
            }),
            fine_last_origin_samples: self.fine.last_scan_origin(),
            fine_last_peak_interior: scan_peak.map(|p| self.peak_interior(p)).unwrap_or(false),
            tracking_moves: self.tracking_moves,
            reacquisitions: self.reacquisitions,
            last_reacquire_trigger: self.last_trigger,
            coarse_rearms: self.coarse_rearms,
            coarse_regions_rejected: self.regions_rejected,
            coarse_last_resort_exhausted: self.regions_rejected >= MAX_REVERIFY_REJECTS,
            tracking_contradiction_run: self
                .tracker
                .as_ref()
                .map(|tracker| tracker.jump_cycles())
                .unwrap_or(0),
            tracking_contradiction_run_max: self.contradiction_run_max,
        }
    }

    /// Whether the standing lock is under post-discontinuity suspicion, and
    /// [`None`] when no lock stands.
    ///
    /// Test-only, and the only honest way to read the state a forged
    /// discontinuity used to leave a healthy stream in: suspicion is not
    /// reported through [`DelayEstimate`], because until a reacquisition
    /// actually fires it has changed nothing a caller can act on.
    #[cfg(test)]
    pub(crate) fn tracker_suspect(&self) -> Option<bool> {
        self.tracker.as_ref().map(|tracker| tracker.suspect())
    }

    /// Sends a standing lock back to the global search, emulating a spurious
    /// trigger.
    ///
    /// Test-only. Which clip fires a real trigger, and when, is a property of
    /// the audio; the re-promotion that FOLLOWS a trigger is what the keep band
    /// governs, and reaching it deterministically is the only way to test that
    /// band on a static delay.
    #[cfg(test)]
    pub(crate) fn force_reacquire(&mut self) {
        self.enter_reacquire(ReacquireTrigger::ConfidenceLost);
    }

    /// Whether the coarse global scan is actively working this instant: its
    /// near chain is running and neither a region nor a give-up has stopped it.
    ///
    /// Test-only, and the honest instrument for coarse-scan DUTY. A scan that
    /// keeps confirming regions the re-verification then refuses is re-armed
    /// every cycle, so it never stops working and this reads true frame after
    /// frame; a scan that has locked, given up, or stood down reads false.
    /// Sampling it per block over a long stream is how the wandering-region
    /// spin is measured against the bounded cadence that replaces it.
    #[cfg(test)]
    pub(crate) fn coarse_active(&self) -> bool {
        !self.coarse.finished()
    }

    /// Clears the acquisition to its just-constructed state.
    pub(crate) fn reset(&mut self) {
        self.coarse.reset();
        self.fine.reset();
        self.next_near_abs = 0;
        self.anchored = false;
        self.state = AcquisitionState::Searching;
        self.relocations = 0;
        self.frames_with_region = 0;
        self.pending_region = None;
        self.regions_rejected = 0;
        self.scored_since_exhausted = 0;
        self.rearm_backoff = 1;
        self.coarse_rearms = 0;
        self.tracker = None;
        self.cycle_scored = 0;
        self.cycle_candidate = None;
        self.unsettle();
        self.tracking_moves = 0;
        self.reacquiring = false;
        self.reacquisitions = 0;
        self.last_trigger = None;
        self.contradiction_run_max = 0;
        self.locked = None;
        self.source = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AecConfig;
    use crate::engine::Aec;

    /// One near-end block, the cadence every example and the benchmark use.
    const TURN: usize = 256;
    const RATE: u32 = 16000;

    /// A deterministic linear congruential generator, integer state mapped to
    /// `f32`. The same generator the crate's other fixtures use, so a synthetic
    /// pair here is comparable to the ones the quality suite pins.
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

    /// The alignment offset the engine must find, in ENGINE terms.
    ///
    /// The engine anchors the near-end stream at the reference frontier, and the
    /// standard cadence feeds one block of reference before the first block of
    /// capture, so near-end sample `n` sits at far-absolute `TURN + n`. The
    /// offset that puts the echo path's onset under the filter's first tap is
    /// therefore the synthesized bulk delay plus that lead. Getting this wrong
    /// is the easiest way to mis-score an otherwise correct acquisition, so it
    /// is derived here once and used everywhere.
    fn onset(bulk: usize) -> usize {
        bulk + TURN + ECHO_IR[0].0
    }

    /// A speech-shaped far-end signal: broadband noise under a syllabic
    /// amplitude envelope, with segment lengths and levels both drawn from the
    /// generator so no period can manufacture a correlation peak of its own.
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

    /// A far-end single-talk pair whose bulk delay follows `bulk_at(i)`: the
    /// echo path onset for mic sample `i` sits `bulk_at(i)` samples behind it.
    fn moving_pair(len: usize, bulk_at: impl Fn(usize) -> usize) -> (Vec<f32>, Vec<f32>) {
        let far = speech_like(len, 0x1234_5678, 0x00C0_FFEE);
        let mut floor = Lcg(0x0F10_0F10);
        let mic: Vec<f32> = (0..len)
            .map(|i| {
                let bulk = bulk_at(i);
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

    /// A far-end single-talk pair with a fixed bulk delay.
    fn delayed_pair(len: usize, bulk: usize) -> (Vec<f32>, Vec<f32>) {
        moving_pair(len, |_| bulk)
    }

    /// Runs the engine over a pair at the standard cadence and reports the final
    /// metrics plus the near-end sample at which a lock was promoted.
    fn run(config: AecConfig, far: &[f32], near: &[f32]) -> (crate::AecMetrics, Option<usize>) {
        let mut aec = Aec::new(config).expect("configuration is valid");
        let mut out = Vec::with_capacity(near.len() + TURN);
        let mut far_chunks = far.chunks(TURN);
        let mut locked_at = None;
        let mut processed = 0usize;
        for near_chunk in near.chunks(TURN) {
            if let Some(far_chunk) = far_chunks.next() {
                aec.feed_reference(far_chunk);
            }
            aec.process(near_chunk, &mut out).expect("process succeeds");
            processed += near_chunk.len();
            if locked_at.is_none() && aec.metrics().delay_samples.is_some() {
                locked_at = Some(processed);
            }
        }
        aec.flush(&mut out).expect("flush succeeds");
        (aec.metrics(), locked_at)
    }

    fn estimator_config() -> AecConfig {
        AecConfig {
            sample_rate: RATE,
            ..AecConfig::default()
        }
    }

    /// What one run of [`drive_reverify_spin`] observed.
    struct SpinStats {
        /// Regions the re-verification refused (`coarse_regions_rejected`).
        rejects: u32,
        /// The derived stand-down flag (`coarse_last_resort_exhausted`).
        exhausted: bool,
        /// Coarse re-arms charged to the never-lock give-up back-off.
        give_up_rearms: u32,
        /// The final adopted alignment, if the stream ever locked.
        locked: Option<usize>,
        /// Per-eighth of the run: (first frame, last frame, active fraction,
        /// rejects so far), for reading duty over time.
        windows: Vec<(usize, usize, f64, u32)>,
    }

    /// Drives a [`DelayAcquirer`] directly with a wandering-region stream and
    /// measures the coarse-scan duty and the re-verification counters over time.
    ///
    /// The stream is a shared syllabic envelope replayed into both stages: the
    /// far side carries the envelope, the near side carries the same envelope
    /// delayed by the current wander value, so the coarse magnitude-envelope
    /// correlation finds a clear region at that delay. The delay is stepped to a
    /// fresh, well-separated value every time the coarse scan re-arms, so each
    /// re-verification finds the region somewhere new and refuses it: the
    /// confident-but-wandering shallow regions the module doc names, made
    /// deterministic. The fine stage is fed a silent far window on purpose, the
    /// stand-in for a stream whose sample-accurate path the fine estimator
    /// cannot find (a nonlinear or past-window echo), so the last-resort coarse
    /// path is the only promotion route and the re-verification loop is what
    /// governs the scan's duty.
    ///
    /// `hold_after` models a stream that wanders for a while and then SETTLES:
    /// once that many regions have been refused the delay stops stepping, so the
    /// next re-armed scan re-confirms the same region, the re-verification agrees,
    /// and the last resort adopts it. It is the converging acquirer against which
    /// the budget must not fire prematurely; [`None`] never settles.
    fn drive_reverify_spin(sample_budget: usize, hold_after: Option<u32>) -> SpinStats {
        // Wander values, every pair well beyond the re-verification agreement
        // band (30 ms = 480 samples at 16 kHz), cycling so no two consecutive
        // confirmations can ever agree.
        const WANDER: [usize; 6] = [640, 1920, 3200, 1280, 2560, 3840]; // 40..240 ms
        let mut acq = DelayAcquirer::new(RATE, 250, 1000);
        let window_len = acq.fine_window_len();
        let silent_far = vec![0.0f32; window_len];

        // A deterministic syllabic envelope, rich enough that the coarse peak is
        // a clear winner. Sized a block past the sample budget so the delayed
        // read never runs off the end.
        let env_len = sample_budget + 2 * TURN;
        let mut shape = Lcg(0x00C0_FFEE);
        let mut env = vec![0.0f32; env_len];
        let mut i = 0usize;
        while i < env_len {
            let seg = 300 + (shape.next_unit() * 1500.0) as usize;
            let level = if shape.next_unit() < 0.12 {
                0.05
            } else {
                0.2 + 0.8 * shape.next_unit()
            };
            for _ in 0..seg {
                if i < env_len {
                    env[i] = level;
                    i += 1;
                }
            }
        }

        let mut wi = 0usize;
        let mut d = WANDER[wi];
        let mut far_frontier = 0usize;
        let mut near_abs = 0u64;
        let mut prev_coarse_frames = 0u32;

        let mut fine_frames = 0usize;
        let mut windows: Vec<(usize, usize, f64, u32)> = Vec::new();
        let window_span = (sample_budget / 8).max(1) as u64;
        let mut next_window_at = window_span;
        let mut win_start = 0usize;
        let mut win_active = 0usize;

        while near_abs < sample_budget as u64 {
            // One block of far, then one block of near, near lagging by a block.
            let far_block: Vec<f32> = (far_frontier..far_frontier + TURN)
                .map(|m| env[m])
                .collect();
            acq.push_far(&far_block);
            far_frontier += TURN;

            acq.begin_block(near_abs);
            for _ in 0..TURN {
                let a = near_abs as usize;
                let sample = if a >= d { env[a - d] } else { 0.0 };
                let complete = acq.push_near(sample);
                near_abs += 1;
                if !complete {
                    if near_abs >= next_window_at {
                        let span = fine_frames - win_start;
                        windows.push((
                            win_start,
                            fine_frames,
                            if span == 0 {
                                0.0
                            } else {
                                win_active as f64 / span as f64
                            },
                            acq.estimate().coarse_regions_rejected,
                        ));
                        win_start = fine_frames;
                        win_active = 0;
                        next_window_at += window_span;
                    }
                    continue;
                }
                acq.observe(&silent_far, WindowSupport::default());
                let est = acq.estimate();

                fine_frames += 1;
                if acq.coarse_active() {
                    win_active += 1;
                }

                // A coarse frame count that dropped is a re-arm: step the wander
                // so the next confirmation lands somewhere new, unless the stream
                // has settled (held after enough refusals), in which case the
                // delay stays put so the region reproduces and the last resort
                // adopts it.
                let settled = hold_after.is_some_and(|k| est.coarse_regions_rejected >= k);
                if est.coarse_frames < prev_coarse_frames && !settled {
                    wi = (wi + 1) % WANDER.len();
                    d = WANDER[wi];
                }
                prev_coarse_frames = est.coarse_frames;
                if est.delay_samples.is_some() {
                    // The stream locked: report it and stop.
                    return finish_spin_stats(&acq, windows);
                }

                if near_abs >= next_window_at {
                    let span = fine_frames - win_start;
                    windows.push((
                        win_start,
                        fine_frames,
                        if span == 0 {
                            0.0
                        } else {
                            win_active as f64 / span as f64
                        },
                        est.coarse_regions_rejected,
                    ));
                    win_start = fine_frames;
                    win_active = 0;
                    next_window_at += window_span;
                }
            }
        }

        finish_spin_stats(&acq, windows)
    }

    /// Snapshots a [`SpinStats`] from the acquirer's current estimate.
    fn finish_spin_stats(acq: &DelayAcquirer, windows: Vec<(usize, usize, f64, u32)>) -> SpinStats {
        let est = acq.estimate();
        SpinStats {
            rejects: est.coarse_regions_rejected,
            exhausted: est.coarse_last_resort_exhausted,
            give_up_rearms: est.coarse_rearms,
            locked: est.delay_samples,
            windows,
        }
    }

    /// Asserts an alignment is causally safe against a known onset: at or
    /// before it, and no more than a filter tail early.
    fn assert_causally_safe(delay: usize, onset: usize, context: &str) {
        let tail = (AecConfig::default().tail_ms as usize * RATE as usize) / 1000;
        assert!(
            delay <= onset,
            "{context}: alignment {delay} is later than the path onset {onset}"
        );
        assert!(
            onset - delay < tail,
            "{context}: alignment {delay} is more than a tail early of onset {onset}"
        );
    }

    /// A delay far outside the fine window, which the fine search alone cannot
    /// see, is acquired through the coarse region.
    ///
    /// 400 ms is past the 250 ms fine window and inside the 1000 ms ceiling.
    #[test]
    fn acquisition_reaches_a_delay_beyond_the_fine_window() {
        let bulk = 6400; // 400 ms at 16 kHz.
        let (far, near) = delayed_pair(160_000, bulk);
        let (metrics, locked_at) = run(estimator_config(), &far, &near);

        let delay = metrics
            .delay_samples
            .expect("a 400 ms delay must be acquired through the coarse region");
        assert_causally_safe(delay, onset(bulk), "coverage acquisition");
        assert!(metrics.delay.relocated, "the fine search must have moved");
        assert!(
            matches!(metrics.delay.status, DelayStatus::Locked(_)),
            "status must be Locked, got {:?}",
            metrics.delay.status
        );
        let locked_at = locked_at.expect("a lock implies a lock sample");
        assert!(
            locked_at <= 5 * RATE as usize,
            "acquisition took {locked_at} samples, beyond the 5 s budget"
        );
    }

    /// A delay past the ceiling is reported as a coverage failure, not
    /// mistaken for a shallow one. Specifically, neither boundary signature
    /// appears: no lock at 0 and none at the deep end of the fine window.
    #[test]
    fn a_delay_beyond_the_ceiling_is_reported_not_faked() {
        let bulk = 24_000; // 1500 ms at 16 kHz, past the 1000 ms ceiling.
        let (far, near) = delayed_pair(200_000, bulk);
        let (metrics, _) = run(estimator_config(), &far, &near);

        if let Some(delay) = metrics.delay_samples {
            // Whatever it did, it must not have manufactured either edge value.
            let deep = metrics.delay.fine_search_end_samples;
            assert!(
                delay > (RATE as usize * FINE_EDGE_GUARD_MS) / 1000,
                "locked at {delay}, the shallow-edge artefact"
            );
            assert!(
                delay + (RATE as usize * FINE_EDGE_GUARD_MS) / 1000 < deep,
                "locked at {delay}, pinned against the deep search edge {deep}"
            );
        }
    }

    /// The narrowed-ceiling sharp edge, guarded: a delay past a DELIBERATELY
    /// narrowed ceiling must be refused, not aliased onto a spurious shallow
    /// region by the last-resort path. The re-verification requirement is what
    /// refuses it.
    #[test]
    fn a_narrowed_ceiling_refuses_rather_than_adopting_an_alias() {
        let bulk = 7200; // 450 ms at 16 kHz, past a 250 ms ceiling.
        let (far, near) = delayed_pair(240_000, bulk);
        let config = AecConfig {
            sample_rate: RATE,
            max_search_delay_ms: 250,
            ..AecConfig::default()
        };
        let (metrics, _) = run(config, &far, &near);

        assert_eq!(
            metrics.delay_samples, None,
            "a past-ceiling delay must be refused at a narrowed ceiling, \
             not adopted from an aliased region (rejected {} regions)",
            metrics.delay.coarse_regions_rejected
        );
    }

    /// A stream that keeps confirming confident-but-wandering shallow regions
    /// that never reproduce must not pin the coarse scan at full duty re-arming
    /// forever. The re-verification refuses each region, and once the rejection
    /// budget is spent the last resort stands down: the scan goes quiescent
    /// instead of re-arming again, and nothing is ever promoted.
    ///
    /// Both assertions below fail on an unbounded loop and hold on the bounded
    /// one. The never-lock give-up back-off must not engage at all: it governs a
    /// scan that gave up WITHOUT finding a region, a state this stream never
    /// enters, so a nonzero give-up re-arm count here would be the two paths
    /// fighting.
    #[test]
    fn the_reverification_loop_stands_down_after_the_reject_budget() {
        let stats = drive_reverify_spin(40 * RATE as usize, None); // 40 s.

        assert_eq!(
            stats.rejects, MAX_REVERIFY_REJECTS,
            "the wandering stream must reject exactly the budget and then stop, \
             got {} rejections",
            stats.rejects
        );
        assert!(
            stats.exhausted,
            "reaching the reject budget must surface the derived stand-down flag \
             coarse_last_resort_exhausted"
        );
        assert_eq!(
            stats.locked, None,
            "a stream of wander that never reproduces a region must never be \
             promoted, got {:?}",
            stats.locked
        );
        assert_eq!(
            stats.give_up_rearms, 0,
            "the never-lock give-up back-off must not engage on a scan that keeps \
             finding regions: {} give-up re-arms means the two paths are fighting",
            stats.give_up_rearms
        );
        // The duty drops from pinned-high to quiescent once the budget is spent:
        // the spin is not merely counted out, the coarse scan actually stops
        // working. Read from the run's own windows, first versus last.
        let first = stats.windows.first().expect("at least one window");
        let last = stats.windows.last().expect("at least one window");
        assert!(
            first.2 > 0.8,
            "the coarse duty must start pinned high (the spin), got {:.2}",
            first.2
        );
        assert!(
            last.2 < 0.1,
            "the coarse duty must fall to quiescent after the budget is spent, \
             got {:.2} (a new spin, not a stand-down)",
            last.2
        );
    }

    /// The budget must not fire on a stream that genuinely takes a while to lock.
    /// An acquirer that wanders through several refusals and then SETTLES, so its
    /// region reproduces and the last resort adopts it, still locks: the budget
    /// sits above the refusals this converging stream spends before it settles.
    /// Only a stream that never converges is cut off.
    #[test]
    fn a_slow_converging_acquirer_still_locks_under_the_reject_budget() {
        // Settle only after seven refusals: one short of the budget, the worst
        // case that must still lock.
        let stats = drive_reverify_spin(40 * RATE as usize, Some(MAX_REVERIFY_REJECTS - 1));

        assert!(
            stats.locked.is_some(),
            "a converging acquirer that settles within the budget must still \
             lock, got no lock after {} rejections",
            stats.rejects
        );
        assert!(
            stats.rejects < MAX_REVERIFY_REJECTS,
            "the converging stream must lock before spending the budget, \
             got {} rejections",
            stats.rejects
        );
        assert_eq!(
            stats.give_up_rearms, 0,
            "the give-up back-off must not engage while the scan keeps finding \
             regions, got {}",
            stats.give_up_rearms
        );
    }

    /// The control: a delay well inside the fine window still acquires, quickly,
    /// on local evidence, is not disturbed by the coarse stage, and then holds:
    /// a static delay produces no tracking moves and no reacquisitions.
    #[test]
    fn a_short_delay_still_acquires_on_local_evidence_and_holds() {
        let bulk = 1600; // 100 ms at 16 kHz, comfortably inside the window.
        let (far, near) = delayed_pair(160_000, bulk);
        let (metrics, locked_at) = run(estimator_config(), &far, &near);

        let delay = metrics
            .delay_samples
            .expect("an in-window delay must still be acquired");
        assert!(delay <= onset(bulk), "alignment {delay} is late");
        assert!(onset(bulk) - delay < 800, "alignment {delay} is far early");
        let locked_at = locked_at.expect("a lock implies a lock sample");
        assert!(
            locked_at <= 2 * RATE as usize,
            "an in-window delay took {locked_at} samples to acquire"
        );
        assert_eq!(
            metrics.delay.tracking_moves, 0,
            "a static delay must not be chased"
        );
        assert_eq!(
            metrics.delay.reacquisitions, 0,
            "a good lock must never reacquire"
        );
    }

    /// Tracking follows a step change that stays inside the local window,
    /// without any reacquisition: the with-movement case in miniature.
    #[test]
    fn tracking_follows_an_in_window_delay_step() {
        let first = 1600; // 100 ms.
        let second = 2880; // 180 ms, an 80 ms step, inside the local window.
        let step_at = 160_000; // 10 s in.
        let (far, near) = moving_pair(320_000, |i| if i < step_at { first } else { second });
        let (metrics, _) = run(estimator_config(), &far, &near);

        let delay = metrics
            .delay_samples
            .expect("the stream must stay locked across an in-window step");
        assert_causally_safe(delay, onset(second), "post-step alignment");
        assert!(
            metrics.delay.tracking_moves >= 1,
            "the step must be followed by the tracker, not ignored"
        );
        assert_eq!(
            metrics.delay.reacquisitions, 0,
            "an in-window step must be tracked locally, not reacquired \
             (last trigger {:?})",
            metrics.delay.last_reacquire_trigger
        );
    }

    /// A delay jump that leaves the local window entirely fires a
    /// reacquisition trigger, and the global search then locks the new delay.
    #[test]
    fn an_out_of_window_jump_reacquires_and_relocks() {
        let first = 1600; // 100 ms.
        let second = 9600; // 600 ms: far outside the local tracking window.
        let step_at = 160_000; // 10 s in.
        let (far, near) = moving_pair(480_000, |i| if i < step_at { first } else { second });
        let (metrics, _) = run(estimator_config(), &far, &near);

        assert!(
            metrics.delay.reacquisitions >= 1,
            "a jump beyond local reach must fire a reacquisition trigger"
        );
        assert!(
            metrics.delay.last_reacquire_trigger.is_some(),
            "the trigger must be reported"
        );
        let delay = metrics
            .delay_samples
            .expect("the stream must re-lock after the jump");
        assert_causally_safe(delay, onset(second), "post-jump alignment");
        assert!(
            matches!(metrics.delay.status, DelayStatus::Locked(_)),
            "the stream must end re-locked, got {:?}",
            metrics.delay.status
        );
    }

    /// A long span with no far-end energy is not evidence against the lock:
    /// unscored frames advance no tracking cycle, so the lock is held straight
    /// through and no trigger fires.
    #[test]
    fn a_long_near_end_only_span_holds_the_lock() {
        let bulk = 1600;
        let len = 400_000; // 25 s.
        let gap = 128_000..288_000; // far silent from 8 s to 18 s.
        let mut far = speech_like(len, 0x1234_5678, 0x00C0_FFEE);
        for i in gap.clone() {
            far[i] = 0.0;
        }
        let mut talker = Lcg(0x7E57_7A1E);
        let mut shape = Lcg(0x0B0B_0B0B);
        let mut floor = Lcg(0x0F10_0F10);
        let mut level = 0.0_f32;
        let mut remaining = 0usize;
        let near: Vec<f32> = (0..len)
            .map(|i| {
                let mut echo = 0.0_f32;
                for &(tap, coeff) in &ECHO_IR {
                    let lag = bulk + tap;
                    if i >= lag {
                        echo += coeff * far[i - lag];
                    }
                }
                if remaining == 0 {
                    remaining = 400 + (shape.next_unit() * 2400.0) as usize;
                    level = if shape.next_unit() < 0.4 {
                        0.0
                    } else {
                        0.1 + 0.3 * shape.next_unit()
                    };
                }
                remaining -= 1;
                // The near end keeps talking through the far-end gap.
                let voice = if gap.contains(&i) {
                    level * talker.next_f32()
                } else {
                    0.0
                };
                ECHO_GAIN * echo + voice + NOISE_FLOOR * floor.next_f32()
            })
            .collect();

        let (metrics, locked_at) = run(estimator_config(), &far, &near);
        let delay = metrics
            .delay_samples
            .expect("the stream must lock before the gap");
        let locked_at = locked_at.expect("a lock implies a lock sample");
        assert!(
            locked_at < 128_000,
            "the fixture assumes a pre-gap lock, got {locked_at}"
        );
        assert_causally_safe(delay, onset(bulk), "post-gap alignment");
        assert_eq!(
            metrics.delay.reacquisitions, 0,
            "a near-end-only span must not fire a trigger (last {:?})",
            metrics.delay.last_reacquire_trigger
        );
    }

    /// Double-talk from the very start does not defeat acquisition: the far
    /// end is still correlated with its echo underneath the near talker.
    #[test]
    fn double_talk_during_acquisition_still_locks() {
        let bulk = 6400; // 400 ms: the coverage path, under double-talk.
        let (far, echo_only) = delayed_pair(240_000, bulk);
        let talker = speech_like(240_000, 0x7E57_7A1E, 0x0B0B_0B0B);
        let near: Vec<f32> = echo_only
            .iter()
            .zip(talker.iter())
            .map(|(&e, &t)| e + 0.35 * t)
            .collect();
        let (metrics, _) = run(estimator_config(), &far, &near);

        let delay = metrics
            .delay_samples
            .expect("double-talk must not defeat acquisition");
        assert_causally_safe(delay, onset(bulk), "double-talk acquisition");
        assert_eq!(
            metrics.delay.reacquisitions, 0,
            "a good lock under double-talk must hold"
        );
    }

    /// A hint constructs no search at all, so neither stage can run, and the
    /// reported status names the hint as the source.
    #[test]
    fn a_hint_runs_no_search_of_either_stage() {
        let (far, near) = delayed_pair(48_000, 1600);
        let config = AecConfig {
            sample_rate: RATE,
            delay_hint_ms: Some(100),
            ..AecConfig::default()
        };
        let (metrics, _) = run(config, &far, &near);

        assert_eq!(
            metrics.delay.status,
            DelayStatus::Locked(DelayLockSource::Hint)
        );
        assert_eq!(metrics.delay.coarse_frames, 0);
        assert_eq!(metrics.delay.fine_frames, 0);
        assert!(!metrics.delay.relocated);
        assert_eq!(metrics.delay.tracking_moves, 0);
        assert_eq!(metrics.delay.reacquisitions, 0);
    }

    /// An uncorrelated pair must not be talked into a lock by either stage.
    #[test]
    fn an_uncorrelated_pair_is_declined_by_both_stages() {
        let (far, _) = delayed_pair(160_000, 0);
        let mut other = Lcg(0xDEAD_BEEF);
        let near: Vec<f32> = (0..160_000).map(|_| 0.1 * other.next_f32()).collect();
        let (metrics, _) = run(estimator_config(), &far, &near);

        assert_eq!(
            metrics.delay_samples, None,
            "locked onto an uncorrelated pair"
        );
        assert_eq!(metrics.delay.status, DelayStatus::Searching);
        assert!(!metrics.delay.relocated);
    }

    /// The configuration surface: a ceiling below the fine window is refused,
    /// and one above the supported bound is refused, both before any audio.
    #[test]
    fn an_out_of_range_search_ceiling_is_rejected() {
        let too_small = AecConfig {
            max_echo_delay_ms: 250,
            max_search_delay_ms: 100,
            ..AecConfig::default()
        };
        assert!(matches!(
            Aec::new(too_small),
            Err(crate::AecError::SearchDelayOutOfRange {
                requested_ms: 100,
                fine_window_ms: 250
            })
        ));
        let too_large = AecConfig {
            max_search_delay_ms: 2001,
            ..AecConfig::default()
        };
        assert!(matches!(
            Aec::new(too_large),
            Err(crate::AecError::SearchDelayOutOfRange { .. })
        ));
    }

    /// Reset returns the acquisition to its just-constructed state, so the next
    /// stream searches afresh rather than inheriting a stale region.
    #[test]
    fn reset_clears_the_acquisition() {
        let (far, near) = delayed_pair(160_000, 6400);
        let mut aec = Aec::new(estimator_config()).expect("configuration is valid");
        let mut out = Vec::new();
        let mut far_chunks = far.chunks(TURN);
        for near_chunk in near.chunks(TURN) {
            if let Some(far_chunk) = far_chunks.next() {
                aec.feed_reference(far_chunk);
            }
            aec.process(near_chunk, &mut out).expect("process succeeds");
        }
        assert!(aec.metrics().delay_samples.is_some());

        aec.reset();
        let after = aec.metrics();
        assert_eq!(after.delay_samples, None);
        assert_eq!(after.delay.status, DelayStatus::Searching);
        assert_eq!(after.delay.coarse_frames, 0);
        assert_eq!(after.delay.fine_frames, 0);
        assert_eq!(after.delay.coarse_region_samples, None);
        assert!(!after.delay.relocated);
        assert_eq!(after.delay.tracking_moves, 0);
        assert_eq!(after.delay.reacquisitions, 0);
        assert_eq!(after.delay.fine_scans, 0);
        assert_eq!(after.acquisition_parked, 0);
    }

    /// Two identical runs produce identical acquisitions, sample for sample,
    /// tracking counters included.
    #[test]
    fn acquisition_is_deterministic_across_runs() {
        let (far, near) = delayed_pair(160_000, 6400);
        let first = run(estimator_config(), &far, &near);
        let second = run(estimator_config(), &far, &near);
        assert_eq!(first.0.delay_samples, second.0.delay_samples);
        assert_eq!(first.0.delay, second.0.delay);
        assert_eq!(first.1, second.1);
    }

    /// The framing is cut on absolute sample count, so the acquisition does not
    /// depend on how the caller chunks the stream.
    #[test]
    fn acquisition_is_invariant_to_caller_chunking() {
        let (far, near) = delayed_pair(160_000, 6400);
        let (reference, _) = run(estimator_config(), &far, &near);

        let mut aec = Aec::new(estimator_config()).expect("configuration is valid");
        let mut out = Vec::new();
        // A deliberately ragged cadence over the same audio. The reference is
        // still fed one TURN ahead of the capture, so the engine-domain delay is
        // the same as under the uniform cadence and the two runs are comparable;
        // only where the caller's chunk boundaries fall differs.
        let sizes = [37usize, 512, 1, 913, 256, 128];
        let mut cursor = 0usize;
        let mut fed = 0usize;
        let mut step = 0usize;
        while cursor < near.len() {
            let take = sizes[step % sizes.len()].min(near.len() - cursor);
            // Keep the reference exactly one TURN ahead of the capture, as the
            // uniform run does, so both runs anchor at the same offset.
            let feed_to = (cursor + TURN).min(far.len());
            if feed_to > fed {
                aec.feed_reference(&far[fed..feed_to]);
                fed = feed_to;
            }
            aec.process(&near[cursor..cursor + take], &mut out)
                .expect("process succeeds");
            cursor += take;
            step += 1;
        }
        aec.flush(&mut out).expect("flush succeeds");
        assert_eq!(aec.metrics().delay_samples, reference.delay_samples);
    }

    /// The contradiction metric must read zero on a stream that never
    /// contradicts itself, however long it runs.
    ///
    /// A held lock re-confirms its alignment every cycle, and re-confirmation
    /// clears the run, so a healthy matched stream reports zero for both the
    /// live run and the high-water mark. A metric that drifts upward on healthy
    /// audio would report every stream as suspect and be worthless as a signal.
    #[test]
    fn a_healthy_stream_reports_no_estimator_contradiction() {
        let (far, near) = delayed_pair(RATE as usize * 12, 1200);
        let (metrics, locked_at) = run(estimator_config(), &far, &near);

        assert!(locked_at.is_some(), "the clip must lock");
        assert_eq!(
            metrics.delay.tracking_contradiction_run, 0,
            "a re-confirmed lock leaves no standing contradiction"
        );
        assert_eq!(
            metrics.delay.tracking_contradiction_run_max, 0,
            "and none was ever banked"
        );
    }

    /// An estimator that offers the alignment incompatible answers must be
    /// visible at RUNTIME, not only from a test accessor.
    ///
    /// A contradiction that cannot escalate is otherwise silent. The high-water
    /// mark is what survives the re-confirmation that clears the live run, so an
    /// episode that climbed and then subsided is still reportable afterwards.
    #[test]
    fn a_contradicting_estimator_is_reported_through_the_public_estimate() {
        let mut acquirer = DelayAcquirer::new(RATE, 250, 1000);
        let per_ms = |ms: usize| (ms * RATE as usize) / 1000;
        acquirer.promote(per_ms(200), DelayLockSource::LocalEvidence);
        assert_eq!(acquirer.estimate().tracking_contradiction_run, 0);

        // Two candidates far enough apart to be incompatible, both clear of the
        // range edges so nothing is refused for being edge-pinned. Alternating
        // them is the estimator disagreeing with ITSELF rather than with the
        // alignment, which is exactly what the run counts.
        let candidate = |delay: usize| FineObservation {
            scored: true,
            candidate: Some(FineLock {
                delay,
                peak_at: acquirer_span_midpoint(),
                ratio: 30.0,
            }),
        };
        for cycle in 0..3 {
            let delay = if cycle % 2 == 0 {
                per_ms(160)
            } else {
                per_ms(120)
            };
            for _ in 0..TRACK_CYCLE_FRAMES {
                acquirer.observe_tracking(candidate(delay));
            }
        }

        let estimate = acquirer.estimate();
        assert!(
            estimate.tracking_contradiction_run > 0,
            "the standing contradiction run must be visible at runtime"
        );
        assert!(
            estimate.tracking_contradiction_run_max >= estimate.tracking_contradiction_run,
            "the high-water mark covers the standing run"
        );

        // The estimator now settles. The live run clears, and the banked one
        // does not: that difference is the whole point of the high-water mark.
        for _ in 0..(TRACK_CYCLE_FRAMES * 2) {
            acquirer.observe_tracking(candidate(per_ms(160)));
        }
        let settled = acquirer.estimate();
        assert_eq!(
            settled.tracking_contradiction_run, 0,
            "settling clears the standing run"
        );
        assert!(
            settled.tracking_contradiction_run_max > 0,
            "but the episode stays reportable after the fact"
        );
    }

    /// A peak comfortably inside the fine range, so nothing under test is
    /// refused for sitting on an edge.
    fn acquirer_span_midpoint() -> usize {
        (RATE as usize * 250) / 1000 / 2
    }

    /// A reacquisition that re-promotes inside the keep band must leave the
    /// engine's offset and the acquisition's alignment equal.
    ///
    /// [`DelayEstimate::delay_samples`] is documented as always equal to
    /// [`AecMetrics::delay_samples`]. The keep band exists so a spurious
    /// trigger does not throw away a converged filter, and it therefore holds
    /// the engine's offset across the re-promotion. The acquisition adopts the
    /// newly promoted value unconditionally, and every later tracking cycle is
    /// measured against THAT value, so if the engine does not adopt it too the
    /// two describe different alignments. On a static delay nothing ever moves
    /// them back together.
    #[test]
    fn a_keep_band_re_promotion_leaves_both_alignments_equal() {
        const BULK: usize = 1200;
        // A step SMALLER than the keep band (8 ms is 128 samples at 16 kHz), so
        // the re-promotion that follows the trigger lands inside the band and
        // the engine keeps its offset while the acquisition takes the new one.
        const STEP: usize = 60;
        let len = RATE as usize * 12;
        let step_at = len / 2;
        let (far, near) = moving_pair(len, |i| if i < step_at { BULK } else { BULK + STEP });
        let mut aec = Aec::new(estimator_config()).expect("configuration is valid");
        let mut out = Vec::with_capacity(near.len() + TURN);
        let mut far_chunks = far.chunks(TURN);

        let mut forced = false;
        let mut checked_after = false;
        let mut processed = 0usize;
        for near_chunk in near.chunks(TURN) {
            if let Some(far_chunk) = far_chunks.next() {
                aec.feed_reference(far_chunk);
            }
            aec.process(near_chunk, &mut out).expect("process succeeds");
            processed += near_chunk.len();

            let metrics = aec.metrics();
            // Send the settled lock back to the global search once the step has
            // arrived, emulating a spurious trigger at the worst moment: the
            // re-promotion then lands a step away from the standing offset, and
            // a step this small falls inside the keep band.
            if !forced
                && processed > step_at
                && matches!(metrics.delay.status, DelayStatus::Locked(_))
            {
                aec.acquirer_mut()
                    .expect("estimator is running")
                    .force_reacquire();
                forced = true;
                continue;
            }
            // The first frame after the re-promotion is where the two can part.
            if forced && matches!(metrics.delay.status, DelayStatus::Locked(_)) {
                assert_eq!(
                    metrics.delay_samples, metrics.delay.delay_samples,
                    "the engine's offset and the acquisition's alignment must \
                     describe the same delay after a keep-band re-promotion"
                );
                checked_after = true;
            }
        }
        aec.flush(&mut out).expect("flush succeeds");

        assert!(
            forced,
            "the clip must reach a lock for the test to mean anything"
        );
        assert!(
            checked_after,
            "the clip must re-promote after the forced reacquisition"
        );
    }
}
