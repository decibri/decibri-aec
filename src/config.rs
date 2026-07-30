//! Configuration for the [`Aec`](crate::Aec) engine: [`AecConfig`], the
//! [`AecModel`] selector and its string parse, and the [`Suppression`]
//! setting.

use std::str::FromStr;

use crate::error::AecError;

/// Echo canceller model selector.
///
/// A closed, `#[non_exhaustive]` set: today the only value is [`AecModel::Tau`].
/// Naming the model rather than taking a bool keeps adding further models a
/// non-breaking widening (a new variant), and keeps the caller on record about
/// which model, and which license, they invoked. The model runs on the CPU and
/// bundles no file; a later model that loads weights takes them by path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AecModel {
    /// Tau: the default production echo canceller. Cancels the loudspeaker echo
    /// from the near-end capture and applies the configured residual
    /// [`Suppression`]. It is selected by name so a future model is added
    /// without a breaking change.
    Tau,
}

impl AecModel {
    /// The lowercase string names of the publicly selectable models, in
    /// declaration order.
    ///
    /// This is the single source of the available-model list formatted into
    /// the [`AecError::UnknownModel`] message, so the binding-facing error can
    /// only ever name models a caller is allowed to select. A crate-internal
    /// reference implementation is deliberately absent from this list and has
    /// no name here to leak.
    pub const PUBLIC_MODEL_NAMES: &'static [&'static str] = &["tau"];

    /// The model's lowercase string name: the same name [`AecModel::from_str`]
    /// parses, so a selector round-trips through its string form.
    pub fn as_str(self) -> &'static str {
        match self {
            AecModel::Tau => "tau",
        }
    }
}

impl Default for AecModel {
    /// [`AecModel::Tau`], the default production canceller.
    fn default() -> Self {
        AecModel::Tau
    }
}

impl FromStr for AecModel {
    type Err = AecError;

    /// Parses a lowercase model name into the selector.
    ///
    /// This is the crate-owned string boundary the decibri bindings inherit, so
    /// the model string is parsed in one place. The match is exact and
    /// case-sensitive.
    ///
    /// An unknown name, a typo like `"tao"` included, returns
    /// [`AecError::UnknownModel`], whose message names the received string and
    /// lists the publicly selectable models from
    /// [`AecModel::PUBLIC_MODEL_NAMES`].
    fn from_str(s: &str) -> Result<AecModel, AecError> {
        match s {
            "tau" => Ok(AecModel::Tau),
            other => Err(AecError::UnknownModel {
                requested: other.to_string(),
            }),
        }
    }
}

/// Residual echo suppression setting.
///
/// A closed, `#[non_exhaustive]` set that is intentionally designed to grow.
/// Suppression is a deterministic post-filter that attenuates the residual echo
/// a linear canceller leaves behind; it is separate from the cancellation
/// itself. Naming the setting rather than taking a bool keeps adding a further
/// level (a more aggressive one, say) a non-breaking widening, the way
/// [`AecModel`] grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Suppression {
    /// No residual suppression: the canceller output is delivered as-is.
    ///
    /// Combined with [`OutputTransitionPolicy::PreserveCorrection`] this yields
    /// the un-suppressed linear canceller output: no residual post-filter and no
    /// transition blend toward the microphone, in every delay state. That is the
    /// configuration a downstream residual stage composes on top of.
    Off,
    /// Conservative residual suppression: a bounded post-filter that reduces
    /// residual echo while keeping the near-end voice essentially intact. The
    /// default, because it is what makes the classical canceller usable for
    /// barge-in.
    Conservative,
}

impl Default for Suppression {
    /// [`Suppression::Conservative`].
    fn default() -> Self {
        Suppression::Conservative
    }
}

/// How the engine renders its output while a previously trusted delay
/// alignment is being reacquired
/// ([`DelayStatus::Reacquiring`](crate::DelayStatus::Reacquiring)).
///
/// A closed, `#[non_exhaustive]` set. During a reacquisition the standing
/// alignment no longer describes the stream, so the correction the canceller
/// produces against it is untrustworthy and can actively add echo energy. This
/// setting decides whether the engine keeps emitting that correction or fades
/// to the untouched near-end capture until a fresh lock is promoted. It governs
/// the EMITTED AUDIO only, and only during a reacquisition: every other state,
/// initial acquisition included, is delivered exactly as it was before this
/// setting existed.
///
/// It does not touch a single delay DECISION, the adaptive filter lifecycle, or
/// the reference alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputTransitionPolicy {
    /// Emit the canceller's correction throughout, including while the
    /// alignment is being reacquired. This is the behavior of the engine before
    /// the setting existed, kept for comparison and for a consumer that drives
    /// its own transition policy downstream.
    ///
    /// Combined with [`Suppression::Off`] the engine emits the un-suppressed
    /// linear canceller output with no transition blending toward the near-end
    /// capture, in every delay state. That is the configuration a downstream
    /// residual stage composes on top of.
    PreserveCorrection,
    /// While the alignment is trusted (locked, and throughout initial
    /// acquisition), emit the correction unchanged. On entering a reacquisition,
    /// fade the emitted signal from the correction toward the untouched near-end
    /// capture over `fade_out_ms`; hold the untouched capture while the
    /// reacquisition runs; on re-lock, fade back to the correction over
    /// `fade_in_ms`.
    ///
    /// The fade is linear and always moves from the CURRENT level, so a status
    /// change mid-fade reverses smoothly rather than stepping the signal.
    ///
    /// # Recommended range
    ///
    /// Both durations are a short perceptual transition, and a practical range
    /// is roughly 20 ms to 2000 ms each; the default (100 ms fade-out,
    /// 200 ms fade-in) sits inside it. `0` is valid and collapses the ramp to a
    /// one-sample hard cut. Every value up the whole `u32` range is accepted and
    /// stays well-defined (finite gain, always within `[0, 1]`), but two limits
    /// are worth knowing:
    ///
    /// - Values above a few seconds are legal and safe but make an audibly slow
    ///   transition that no longer reads as a graded reacquisition.
    /// - An EXTREMELY large value (roughly tens of minutes and up) makes the
    ///   per-sample step fall below f32 resolution near a gain of 1.0, so the
    ///   gain never moves off full correction and the policy degenerates to
    ///   holding the correction, identical to
    ///   [`OutputTransitionPolicy::PreserveCorrection`]. This is safe and
    ///   deterministic, not a stuck-gain fault, but it is almost certainly not
    ///   what a caller reaching for a fade intended, hence the recommended
    ///   ceiling above.
    GradedReacquisition {
        /// Milliseconds to fade from full correction toward the untouched
        /// capture on entering a reacquisition. Recommended range roughly 20 to
        /// 2000; see the type-level note. Default 100.
        fade_out_ms: u32,
        /// Milliseconds to fade from the untouched capture back to full
        /// correction on re-lock. Recommended range roughly 20 to 2000; see the
        /// type-level note. Default 200.
        fade_in_ms: u32,
    },
}

impl Default for OutputTransitionPolicy {
    /// [`OutputTransitionPolicy::GradedReacquisition`] with a 100 ms fade-out
    /// and a 200 ms fade-in: fade quickly off a correction that has gone
    /// untrustworthy, restore it more gently once a fresh lock is trusted.
    fn default() -> Self {
        OutputTransitionPolicy::GradedReacquisition {
            fade_out_ms: 100,
            fade_in_ms: 200,
        }
    }
}

/// Configuration for the [`Aec`](crate::Aec) engine.
///
/// `#[non_exhaustive]`: construct it with [`AecConfig::default`] and then assign
/// the public fields you need. Direct struct-literal construction from another
/// crate is intentionally not supported, so adding a field later stays backward
/// compatible. [`Aec::new`](crate::Aec::new) validates the fields and rejects an
/// out-of-range value with [`AecError`](crate::AecError).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AecConfig {
    /// Sample rate in Hz, shared by the near-end and far-end streams. Range:
    /// 8000 to 48000. Default: 16000. A rate outside the range is rejected at
    /// construction rather than producing a canceller that cannot work.
    pub sample_rate: u32,

    /// The canceller model to run. Default: [`AecModel::Tau`].
    pub model: AecModel,

    /// Filter tail length in milliseconds: how much of the room's echo decay the
    /// canceller models. Range: 16 to 500. Default: 200.
    pub tail_ms: u16,

    /// Maximum echo delay in milliseconds: the upper bound of the delay search
    /// window between a far-end reference sample and its echo in the near-end
    /// capture. A different quantity from the tail. Range: 10 to 1000. Default:
    /// 250.
    pub max_echo_delay_ms: u16,

    /// Upper bound of the COARSE global delay search in milliseconds.
    ///
    /// A different quantity from [`AecConfig::max_echo_delay_ms`], which bounds
    /// the sample-accurate fine search. The coarse scan finds the REGION the
    /// echo lives in, out to this ceiling, at a resolution of one millisecond;
    /// the fine search is then re-centred on that region and locks
    /// sample-accurately inside it. Raising this covers Bluetooth and
    /// speakerphone transport delays without widening, and without slowing, the
    /// fine search.
    ///
    /// It expands render-history ALIGNMENT only. How much of the room's echo
    /// decay the canceller models is still [`AecConfig::tail_ms`], and this
    /// setting does not lengthen the adaptive filter by one tap.
    ///
    /// Range: [`AecConfig::max_echo_delay_ms`] to 2000. Default: 1000.
    pub max_search_delay_ms: u16,

    /// Optional caller-supplied delay hint in milliseconds, used to seed the
    /// delay search instead of estimating it. `None` (the default) runs the
    /// estimator with no seed. A hint outside the search window is clamped, not
    /// rejected.
    ///
    /// # The hint is measured from the reference frontier, not from an absolute
    /// timeline
    ///
    /// The offset the hint seeds is measured from the far-end reference frontier
    /// as the caller's own feeding establishes it: how far BACK from the newest
    /// fed reference sample the echo of the block now being processed sits. It
    /// is not an absolute platform figure. Two callers with the same physical
    /// echo need different hints if they interleave
    /// [`feed_reference`](crate::Aec::feed_reference) and
    /// [`process`](crate::Aec::process) differently, because the frontier sits
    /// somewhere different when the block is processed. A caller whose renderer
    /// buffers ahead has a frontier that far ahead, and the hint has to include
    /// that lead. Feeding the block's reference and then processing the block
    /// keeps the lead at one block; keeping the reference N blocks ahead adds N
    /// blocks to the offset.
    ///
    /// So a measured platform latency is the right hint only for a caller who
    /// feeds exactly in step with processing. A hint taken from platform
    /// latency while the reference runs ahead is short by that lead.
    ///
    /// # A hint that is too long is not recoverable
    ///
    /// The two directions of error are not equivalent, and the difference
    /// matters more than the size:
    ///
    /// - SHORT of the frontier-relative offset: the remainder is inside the
    ///   modelled tail and adaptation absorbs it, at the cost of some of the
    ///   [`tail_ms`](AecConfig::tail_ms) budget and some convergence time.
    ///   Short by more than the tail cancels nothing.
    /// - LONGER than the frontier-relative offset: the aligned reference is
    ///   older than the echo, which a causal filter cannot model, and NOTHING
    ///   is cancelled for as long as the hint stands. No error is returned and
    ///   the delay reads as locked.
    ///
    /// A caller unsure of its own lead should therefore err short, or supply no
    /// hint at all and let the estimator find the offset.
    pub delay_hint_ms: Option<u16>,

    /// Residual echo suppression setting. Default: [`Suppression::Conservative`].
    pub suppression: Suppression,

    /// How the engine renders output while a previously trusted delay alignment
    /// is being reacquired. Default:
    /// [`OutputTransitionPolicy::GradedReacquisition`] with a 100 ms fade-out
    /// and a 200 ms fade-in. It changes the emitted audio only during a
    /// reacquisition; every other state, initial acquisition included, is
    /// byte-identical to the engine before the setting existed.
    pub output_transition: OutputTransitionPolicy,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            model: AecModel::default(),
            tail_ms: 200,
            max_echo_delay_ms: 250,
            max_search_delay_ms: 1000,
            delay_hint_ms: None,
            suppression: Suppression::default(),
            output_transition: OutputTransitionPolicy::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_documented_values() {
        let config = AecConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.model, AecModel::Tau);
        assert_eq!(config.tail_ms, 200);
        assert_eq!(config.max_echo_delay_ms, 250);
        assert_eq!(config.max_search_delay_ms, 1000);
        assert_eq!(config.delay_hint_ms, None);
        assert_eq!(config.suppression, Suppression::Conservative);
        assert_eq!(
            config.output_transition,
            OutputTransitionPolicy::GradedReacquisition {
                fade_out_ms: 100,
                fade_in_ms: 200,
            }
        );
    }

    #[test]
    fn output_transition_default_is_graded_reacquisition_100_200() {
        assert_eq!(
            OutputTransitionPolicy::default(),
            OutputTransitionPolicy::GradedReacquisition {
                fade_out_ms: 100,
                fade_in_ms: 200,
            }
        );
    }

    #[test]
    fn model_default_is_tau() {
        assert_eq!(AecModel::default(), AecModel::Tau);
    }

    #[test]
    fn model_parses_its_public_names() {
        assert_eq!("tau".parse::<AecModel>().unwrap(), AecModel::Tau);
    }

    /// Locks the public-name list, the parse, and `as_str` to each other: every
    /// listed name must parse, and the parsed model must render back to the
    /// same name. A new public model that misses one of the three fails here.
    #[test]
    fn public_model_names_round_trip_through_the_parse() {
        for &name in AecModel::PUBLIC_MODEL_NAMES {
            let model = name
                .parse::<AecModel>()
                .unwrap_or_else(|_| panic!("public model name '{name}' must parse"));
            assert_eq!(model.as_str(), name);
        }
    }

    #[test]
    fn unknown_model_error_names_the_received_string_and_lists_tau_only() {
        let err = "tao".parse::<AecModel>().unwrap_err();
        assert!(matches!(
            &err,
            AecError::UnknownModel { requested } if requested == "tao"
        ));
        assert_eq!(err.to_string(), "model must be one of: 'tau'; got 'tao'");
    }

    /// The internal reference canceller must be unreachable by string: its
    /// name neither parses nor appears in the available-model list.
    #[test]
    fn internal_reference_name_is_not_a_public_model() {
        let err = "rho".parse::<AecModel>().unwrap_err();
        assert_eq!(err.to_string(), "model must be one of: 'tau'; got 'rho'");
        assert!(!AecModel::PUBLIC_MODEL_NAMES.contains(&"rho"));
    }

    #[test]
    fn case_and_whitespace_variants_do_not_parse() {
        assert!("Tau".parse::<AecModel>().is_err());
        assert!("TAU".parse::<AecModel>().is_err());
        assert!(" tau".parse::<AecModel>().is_err());
        assert!("".parse::<AecModel>().is_err());
    }

    #[test]
    fn suppression_default_is_conservative() {
        assert_eq!(Suppression::default(), Suppression::Conservative);
    }
}
