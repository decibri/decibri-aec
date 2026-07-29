//! The delay tracker: the decision logic that follows a locked delay and
//! decides when the lock has gone bad.
//!
//! # What tracking is
//!
//! Acquisition ([`crate::acquire`]) answers "where is the echo" once, from
//! nothing. Tracking answers a much smaller question continuously: "is the
//! alignment still right, and if the delay moved, where did it move to". The
//! fine estimator keeps running after the lock, re-centred on a local window
//! around the adopted delay, and its accumulator is restarted every few frames
//! so each short cycle produces an independent local estimate. This module
//! consumes that cycle stream and holds the hysteresis: it decides whether to
//! hold the alignment, move it, or declare the lock lost.
//!
//! # Why hysteresis
//!
//! Every move requires two consecutive cycles that agree with each other, and
//! small disagreements inside the dead bands are deliberately ignored. The dead
//! bands are asymmetric, tight on the late side and wide on the early side.
//!
//! # Why the triggers are estimator-side
//!
//! Every reacquisition trigger here is computed from the tracker's own
//! evidence: where the correlation peaked, whether it cleared the gate, and
//! whether consecutive estimates cohere.
//!
//! # Determinism
//!
//! The tracker is pure integer bookkeeping over the cycle stream: no floats,
//! no containers, no time. The same cycles produce the same verdicts on every
//! run and every platform.

use crate::delay::ReacquireTrigger;

/// Scored fine frames per tracking decision cycle.
pub(crate) const TRACK_CYCLE_FRAMES: u32 = 2;

/// How far EARLIER than the alignment a confirmed estimate must sit before the
/// alignment moves, in milliseconds.
const MOVE_LATE_MS: usize = 4;

/// How far LATER than the alignment a confirmed estimate must sit before the
/// alignment moves, in milliseconds.
const MOVE_EARLY_MS: usize = 48;

/// How closely two consecutive cycle estimates must agree to confirm a move,
/// in milliseconds.
const MOVE_AGREE_MS: usize = 24;

/// How far apart two consecutive confident estimates must sit to be counted
/// as a jump between unrelated delays, in milliseconds.
const JUMP_APART_MS: usize = 100;

/// Consecutive confident cycles with an edge-pinned peak before
/// [`ReacquireTrigger::TrackingEdge`] fires.
const EDGE_CYCLES: u32 = 3;

/// Consecutive contradicting cycles before
/// [`ReacquireTrigger::EstimatorJumping`] fires.
///
/// A cycle contradicts when it disagrees with the previous confident estimate
/// by more than [`JUMP_APART_MS`], or when it replaces an estimate that was
/// awaiting confirmation instead of confirming it.
const JUMP_CYCLES: u32 = 4;

/// Extra lead spent when a move corrects in the EARLIER direction, in
/// milliseconds.
const MOVE_LATE_LEAD_MS: usize = 8;

/// Consecutive scored-but-unconfident cycles before
/// [`ReacquireTrigger::ConfidenceLost`] fires.
const UNCONFIDENT_CYCLES: u32 = 8;

/// Consecutive scored-but-unconfident cycles before a SUSPECT lock (one that
/// just survived a stream discontinuity) is declared lost.
const SUSPECT_UNCONFIDENT_CYCLES: u32 = 4;

/// What one concluded tracking cycle observed.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CycleOutcome {
    /// The cycle's scan produced a confident candidate.
    Candidate {
        /// The candidate delay in samples, margin applied.
        delay: usize,
        /// Whether the peak stood clear of both ends of the scanned range. An
        /// edge-pinned peak is not evidence of a delay AT the edge; it is
        /// evidence the maximum may lie outside the range.
        interior: bool,
    },
    /// The cycle scored frames but its scan never cleared the confidence gate.
    Unconfident,
}

/// The tracker's decision for one concluded cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrackVerdict {
    /// The alignment stands.
    Hold,
    /// The alignment moves to the confirmed estimate. The engine applies this
    /// without resetting the canceller.
    Move(usize),
    /// The lock has gone bad; re-enter the global search.
    Reacquire(ReacquireTrigger),
}

/// The per-lock tracking state. Constructed at promotion, discarded at
/// reacquisition. See the module documentation.
pub(crate) struct Tracker {
    /// [`MOVE_LATE_MS`] in samples.
    late_band: i64,
    /// [`MOVE_EARLY_MS`] in samples.
    early_band: i64,
    /// [`MOVE_AGREE_MS`] in samples.
    move_agree: i64,
    /// [`JUMP_APART_MS`] in samples.
    jump_apart: i64,
    /// [`MOVE_LATE_LEAD_MS`] in samples.
    late_lead: usize,

    /// An out-of-band estimate awaiting its confirming cycle.
    pending_move: Option<usize>,
    /// The previous cycle's confident interior estimate, for the jump test.
    prev_candidate: Option<usize>,
    /// Consecutive confident cycles with an edge-pinned peak.
    edge_cycles: u32,
    /// Consecutive confident cycles that contradicted the standing evidence,
    /// either by jumping between unrelated delays or by replacing an estimate
    /// that was awaiting confirmation instead of confirming it.
    jump_cycles: u32,
    /// Consecutive scored cycles that never cleared the confidence gate.
    unconfident_cycles: u32,
    /// Whether the lock is under suspicion after a stream discontinuity.
    suspect: bool,
}

impl Tracker {
    /// Constructs the tracker for a freshly promoted lock at `sample_rate`.
    pub(crate) fn new(sample_rate: u32) -> Tracker {
        let per_ms = |ms: usize| ((ms * sample_rate as usize) / 1000).max(1) as i64;
        Tracker {
            late_band: per_ms(MOVE_LATE_MS),
            early_band: per_ms(MOVE_EARLY_MS),
            move_agree: per_ms(MOVE_AGREE_MS),
            jump_apart: per_ms(JUMP_APART_MS),
            late_lead: per_ms(MOVE_LATE_LEAD_MS) as usize,
            pending_move: None,
            prev_candidate: None,
            edge_cycles: 0,
            jump_cycles: 0,
            unconfident_cycles: 0,
            suspect: false,
        }
    }

    /// Declares a stream discontinuity: the lock survives but is suspect, the
    /// lost-lock threshold tightens, and evidence from before the seam is
    /// discarded because it is not comparable across it.
    pub(crate) fn on_discontinuity(&mut self) {
        self.suspect = true;
        self.pending_move = None;
        self.prev_candidate = None;
        self.edge_cycles = 0;
        self.jump_cycles = 0;
        self.unconfident_cycles = 0;
    }

    /// Whether the lock is currently under post-discontinuity suspicion.
    #[cfg(test)]
    pub(crate) fn suspect(&self) -> bool {
        self.suspect
    }

    /// The standing run of contradicting cycles: how many consecutive decision
    /// cycles the estimator has disagreed with ITSELF about where the delay is.
    ///
    /// Reported at runtime through
    /// [`DelayEstimate::tracking_contradiction_run`](crate::DelayEstimate::tracking_contradiction_run).
    /// Zero on a healthy stream, because a lock that keeps being re-confirmed
    /// clears the run every cycle.
    pub(crate) fn jump_cycles(&self) -> u32 {
        self.jump_cycles
    }

    /// Whether the tracker carries no unfinished business: nothing awaiting a
    /// confirming cycle, every escalation counter at zero, and no standing
    /// suspicion.
    ///
    /// This is the cadence predicate.
    pub(crate) fn quiescent(&self) -> bool {
        self.pending_move.is_none()
            && self.edge_cycles == 0
            && self.jump_cycles == 0
            && self.unconfident_cycles == 0
            && !self.suspect
    }

    /// Judges one concluded cycle against the current `alignment` and returns
    /// the verdict.
    pub(crate) fn on_cycle(&mut self, alignment: usize, outcome: CycleOutcome) -> TrackVerdict {
        match outcome {
            CycleOutcome::Unconfident => {
                self.unconfident_cycles = self.unconfident_cycles.saturating_add(1);
                let limit = if self.suspect {
                    SUSPECT_UNCONFIDENT_CYCLES
                } else {
                    UNCONFIDENT_CYCLES
                };
                if self.unconfident_cycles >= limit {
                    let trigger = if self.suspect {
                        ReacquireTrigger::Discontinuity
                    } else {
                        ReacquireTrigger::ConfidenceLost
                    };
                    return TrackVerdict::Reacquire(trigger);
                }
                TrackVerdict::Hold
            }
            CycleOutcome::Candidate { delay, interior } => {
                self.unconfident_cycles = 0;
                if !interior {
                    // An edge-pinned peak carries no adoptable delay, and it
                    // is not comparable to interior estimates, so it feeds
                    // only its own counter.
                    self.pending_move = None;
                    self.prev_candidate = None;
                    self.edge_cycles = self.edge_cycles.saturating_add(1);
                    if self.edge_cycles >= EDGE_CYCLES {
                        return TrackVerdict::Reacquire(ReacquireTrigger::TrackingEdge);
                    }
                    return TrackVerdict::Hold;
                }
                self.edge_cycles = 0;

                // The jump test compares consecutive confident estimates to
                // each other, not to the alignment.
                let had_prev = self.prev_candidate.is_some();
                let jumped = match self.prev_candidate {
                    Some(prev) => (delay as i64 - prev as i64).abs() > self.jump_apart,
                    None => false,
                };
                self.prev_candidate = Some(delay);
                let mut contradicted = jumped;

                let drift = delay as i64 - alignment as i64;
                if -drift > self.late_band || drift > self.early_band {
                    // Outside the dead bands: a move is warranted, once a
                    // second cycle confirms it.
                    if let Some(pending) = self.pending_move {
                        if (delay as i64 - pending as i64).abs() <= self.move_agree {
                            self.pending_move = None;
                            // Agreement resets the contradiction run.
                            self.jump_cycles = 0;
                            // A confirmed move lifts suspicion.
                            self.suspect = false;
                            // A correction toward earlier lands with a lead.
                            let target = if delay < alignment {
                                delay.saturating_sub(self.late_lead)
                            } else {
                                delay
                            };
                            return TrackVerdict::Move(target);
                        }
                        // The pending estimate was REPLACED rather than
                        // confirmed, which counts as a contradiction.
                        contradicted = true;
                    }
                    self.pending_move = Some(delay);
                } else {
                    // Inside the dead bands: the alignment is re-confirmed.
                    self.pending_move = None;
                    self.suspect = false;
                }

                // One escalation step per contradicting cycle, however many
                // of the two tests saw it, and the run resets the moment a
                // cycle stops contradicting. A lock that is merely being held
                // re-confirms every cycle and so never escalates.
                if contradicted {
                    self.jump_cycles = self.jump_cycles.saturating_add(1);
                    if self.jump_cycles >= JUMP_CYCLES {
                        return TrackVerdict::Reacquire(ReacquireTrigger::EstimatorJumping);
                    }
                } else if had_prev {
                    self.jump_cycles = 0;
                }
                TrackVerdict::Hold
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16000;

    fn per_ms(ms: usize) -> usize {
        (ms * RATE as usize) / 1000
    }

    fn interior(delay: usize) -> CycleOutcome {
        CycleOutcome::Candidate {
            delay,
            interior: true,
        }
    }

    fn edge(delay: usize) -> CycleOutcome {
        CycleOutcome::Candidate {
            delay,
            interior: false,
        }
    }

    /// Estimates inside the dead bands never move the alignment, however many
    /// arrive: a stable lock is held, not chased.
    #[test]
    fn estimates_inside_the_dead_bands_hold() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        for wobble in [0i64, 30, -30, 60, 0, -60, 700, 0] {
            let delay = (alignment as i64 + wobble) as usize;
            assert_eq!(
                tracker.on_cycle(alignment, interior(delay)),
                TrackVerdict::Hold
            );
        }
    }

    /// A late alignment (estimate earlier than it) moves after exactly two
    /// agreeing cycles, and the move lands a lead earlier than the estimate.
    #[test]
    fn a_late_alignment_moves_on_the_confirming_cycle() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        let moved = alignment - per_ms(10);
        assert_eq!(
            tracker.on_cycle(alignment, interior(moved)),
            TrackVerdict::Hold
        );
        assert_eq!(
            tracker.on_cycle(alignment, interior(moved + 8)),
            TrackVerdict::Move(moved + 8 - per_ms(MOVE_LATE_LEAD_MS))
        );
    }

    /// A deepening delay moves only past the wide early band.
    #[test]
    fn a_deepening_delay_moves_only_past_the_early_band() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        let inside = alignment + per_ms(MOVE_EARLY_MS) - 8;
        assert_eq!(
            tracker.on_cycle(alignment, interior(inside)),
            TrackVerdict::Hold
        );
        assert_eq!(
            tracker.on_cycle(alignment, interior(inside)),
            TrackVerdict::Hold
        );
        let beyond = alignment + per_ms(MOVE_EARLY_MS + 20);
        assert_eq!(
            tracker.on_cycle(alignment, interior(beyond)),
            TrackVerdict::Hold
        );
        assert_eq!(
            tracker.on_cycle(alignment, interior(beyond)),
            TrackVerdict::Move(beyond)
        );
    }

    /// Two alternating unrelated peaks can never confirm a move: the pending
    /// estimate is replaced each cycle and the jump counter rises instead,
    /// ending in reacquisition rather than oscillation. The tracker is done
    /// once it fires; the acquirer discards it there.
    #[test]
    fn alternating_peaks_reacquire_instead_of_oscillating() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        let a = per_ms(220);
        let b = per_ms(60);
        let mut fired = None;
        for i in 0..8 {
            let delay = if i % 2 == 0 { a } else { b };
            match tracker.on_cycle(alignment, interior(delay)) {
                TrackVerdict::Hold => {}
                TrackVerdict::Move(to) => {
                    panic!("an alternating pair must never move the alignment (to {to})")
                }
                TrackVerdict::Reacquire(trigger) => {
                    fired = Some(trigger);
                    break;
                }
            }
        }
        assert_eq!(
            fired,
            Some(ReacquireTrigger::EstimatorJumping),
            "sustained jumping must end in reacquisition"
        );
    }

    /// Two alternating interior peaks that sit closer together than the jump
    /// distance but further apart than the confirmation tolerance. Neither peak
    /// can confirm the other, so no move ever lands, and the separation is
    /// small enough that the jump test calls them the same delay, so no counter
    /// rises either. The escalation must be bounded.
    ///
    /// The separation here is strictly inside the band between
    /// [`MOVE_AGREE_MS`] and [`JUMP_APART_MS`] that
    /// `alternating_peaks_reacquire_instead_of_oscillating` leaves uncovered.
    #[test]
    fn alternating_interior_peaks_inside_the_jump_distance_reacquire() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(200);
        let a = per_ms(160);
        let b = per_ms(120);
        let apart = (a as i64 - b as i64).abs();
        assert!(
            apart > per_ms(MOVE_AGREE_MS) as i64 && apart <= per_ms(JUMP_APART_MS) as i64,
            "the reproduction must sit inside the band this test exists to \
             cover: {apart} samples apart"
        );

        let mut fired = None;
        let mut cycles: u32 = 0;
        for i in 0..1_000 {
            let delay = if i % 2 == 0 { a } else { b };
            cycles += 1;
            match tracker.on_cycle(alignment, interior(delay)) {
                TrackVerdict::Hold => {}
                TrackVerdict::Move(to) => {
                    panic!("two peaks {apart} samples apart must never confirm a move (to {to})")
                }
                TrackVerdict::Reacquire(trigger) => {
                    fired = Some(trigger);
                    break;
                }
            }
        }
        assert_eq!(
            fired,
            Some(ReacquireTrigger::EstimatorJumping),
            "an estimator that will not agree with itself must end in \
             reacquisition, not hold the alignment forever"
        );
        assert_eq!(
            cycles,
            JUMP_CYCLES + 1,
            "the bound is one cycle to bank the first estimate plus one per \
             contradiction"
        );
    }

    /// The contradiction counter must not fire on a lock that is merely being
    /// held. A stable delay re-confirms the alignment every cycle, and
    /// re-confirmation clears the contradiction count, so an arbitrarily long
    /// hold never escalates however much the estimator jitters inside the dead
    /// bands.
    #[test]
    fn a_stable_lock_never_escalates_however_long_it_is_held() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        // All inside the dead bands: 12.5 ms early at the widest, 3 ms late.
        let jitter = [0i64, 60, -40, 120, -30, 200, 0, -50];
        for cycle in 0..1_000 {
            let delay = (alignment as i64 + jitter[cycle % jitter.len()]) as usize;
            assert_eq!(
                tracker.on_cycle(alignment, interior(delay)),
                TrackVerdict::Hold,
                "a stable lock must hold at cycle {cycle}"
            );
        }
        assert!(
            tracker.quiescent(),
            "a lock that only ever held banks no partial evidence"
        );
    }

    /// A contradiction run that RESOLVES still moves. An estimator that
    /// dithers between two delays and then settles on one confirms the move
    /// on the settling pair, and the count it banked while dithering is
    /// cleared rather than carried forward into the next run.
    #[test]
    fn a_resolved_contradiction_run_confirms_the_move_and_clears_the_count() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(200);
        let a = per_ms(160);
        let b = per_ms(120);

        assert_eq!(tracker.on_cycle(alignment, interior(a)), TrackVerdict::Hold);
        assert_eq!(tracker.on_cycle(alignment, interior(b)), TrackVerdict::Hold);
        assert_eq!(tracker.on_cycle(alignment, interior(a)), TrackVerdict::Hold);
        assert_eq!(
            tracker.jump_cycles(),
            2,
            "two replaced estimates are two contradictions"
        );

        assert_eq!(
            tracker.on_cycle(alignment, interior(a)),
            TrackVerdict::Move(a - per_ms(MOVE_LATE_LEAD_MS)),
            "an estimator that settles must still be followed"
        );
        assert_eq!(
            tracker.jump_cycles(),
            0,
            "a confirmed move is the estimator agreeing with itself, which is \
             the opposite of what the counter measures"
        );
    }

    /// Edge-pinned confident peaks reacquire after the configured run, and an
    /// interior peak in between clears the run.
    #[test]
    fn edge_pinned_cycles_reacquire_and_interior_clears_them() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        assert_eq!(
            tracker.on_cycle(alignment, edge(per_ms(240))),
            TrackVerdict::Hold
        );
        assert_eq!(
            tracker.on_cycle(alignment, interior(alignment)),
            TrackVerdict::Hold
        );
        for _ in 0..EDGE_CYCLES - 1 {
            assert_eq!(
                tracker.on_cycle(alignment, edge(per_ms(240))),
                TrackVerdict::Hold
            );
        }
        assert_eq!(
            tracker.on_cycle(alignment, edge(per_ms(240))),
            TrackVerdict::Reacquire(ReacquireTrigger::TrackingEdge)
        );
    }

    /// Sustained unconfident cycles reacquire, and a single confident cycle
    /// anywhere in the run restarts the count.
    #[test]
    fn sustained_unconfidence_reacquires_and_confidence_restarts_the_count() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        for _ in 0..UNCONFIDENT_CYCLES - 1 {
            assert_eq!(
                tracker.on_cycle(alignment, CycleOutcome::Unconfident),
                TrackVerdict::Hold
            );
        }
        assert_eq!(
            tracker.on_cycle(alignment, interior(alignment)),
            TrackVerdict::Hold
        );
        for _ in 0..UNCONFIDENT_CYCLES - 1 {
            assert_eq!(
                tracker.on_cycle(alignment, CycleOutcome::Unconfident),
                TrackVerdict::Hold
            );
        }
        assert_eq!(
            tracker.on_cycle(alignment, CycleOutcome::Unconfident),
            TrackVerdict::Reacquire(ReacquireTrigger::ConfidenceLost)
        );
    }

    /// After a discontinuity the lock is suspect: the unconfident threshold
    /// halves and the trigger names the discontinuity, while an in-band
    /// confident cycle clears the suspicion entirely.
    #[test]
    fn a_discontinuity_tightens_the_leash_until_reconfirmed() {
        let mut tracker = Tracker::new(RATE);
        let alignment = per_ms(100);
        tracker.on_discontinuity();
        assert!(tracker.suspect());
        for _ in 0..SUSPECT_UNCONFIDENT_CYCLES - 1 {
            assert_eq!(
                tracker.on_cycle(alignment, CycleOutcome::Unconfident),
                TrackVerdict::Hold
            );
        }
        assert_eq!(
            tracker.on_cycle(alignment, CycleOutcome::Unconfident),
            TrackVerdict::Reacquire(ReacquireTrigger::Discontinuity)
        );

        let mut tracker = Tracker::new(RATE);
        tracker.on_discontinuity();
        assert_eq!(
            tracker.on_cycle(alignment, interior(alignment)),
            TrackVerdict::Hold
        );
        assert!(!tracker.suspect(), "re-confirmation must lift suspicion");
        for _ in 0..UNCONFIDENT_CYCLES - 1 {
            assert_eq!(
                tracker.on_cycle(alignment, CycleOutcome::Unconfident),
                TrackVerdict::Hold
            );
        }
        assert_eq!(
            tracker.on_cycle(alignment, CycleOutcome::Unconfident),
            TrackVerdict::Reacquire(ReacquireTrigger::ConfidenceLost),
            "a cleared suspicion must restore the standing threshold"
        );
    }

    /// A drifting delay is followed: consecutive estimates that agree within
    /// the confirmation tolerance keep stepping the alignment even though the
    /// delay never stands still.
    #[test]
    fn a_drifting_delay_is_followed_in_steps() {
        let mut tracker = Tracker::new(RATE);
        let mut alignment = per_ms(100);
        // The delay walks earlier by 8 ms per cycle.
        let mut truth = alignment as i64;
        let mut moves = 0;
        for _ in 0..12 {
            truth -= per_ms(8) as i64;
            let verdict = tracker.on_cycle(alignment, interior(truth as usize));
            if let TrackVerdict::Move(to) = verdict {
                alignment = to;
                moves += 1;
            }
        }
        assert!(moves >= 4, "a steady drift must keep confirming: {moves}");
        let lag = alignment as i64 - truth;
        assert!(
            lag <= per_ms(24) as i64,
            "the alignment must stay within one confirmation interval of a \
             drifting delay, lagged {lag} samples"
        );
    }
}
