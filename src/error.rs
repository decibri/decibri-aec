//! The [`AecError`] type returned by the crate's fallible operations.

use thiserror::Error;

use crate::config::AecModel;

/// Errors produced by the echo canceller.
///
/// This enum is `#[non_exhaustive]`: consumers pattern-matching on it must
/// include a `_ =>` catch-all arm so future variant additions are not
/// source-breaking.
///
/// The out-of-range variants carry the offending value as a structured payload
/// that is never formatted into the `Display` string, so the message text stays
/// stable across releases. [`AecError::UnknownModel`] is the deliberate
/// exception: it is the binding-facing typo catch, so echoing the received
/// string and the available-model list into the message is its whole job,
/// mirroring how the wider decibri bindings report an unknown closed-set
/// selector value.
///
/// Every publicly selectable model is compiled into the library and constructed
/// with the engine, so a validly named model can never be unavailable at
/// runtime. The only model error is therefore the string-boundary one, and
/// every remaining variant is a configuration error raised at construction:
/// once [`Aec::new`](crate::Aec::new) has returned, the classical cancellers do
/// not fail.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AecError {
    /// The configured sample rate was outside the supported range. The
    /// supported range is 8000 to 48000 Hz inclusive; a rate outside it fails
    /// at construction rather than producing a canceller that cannot work.
    #[error("sample rate must be between 8000 and 48000 Hz")]
    SampleRateOutOfRange {
        /// The rejected sample rate, in Hz.
        requested: u32,
    },

    /// The configured maximum echo delay was outside the supported range. The
    /// supported range is 10 to 1000 milliseconds inclusive.
    #[error("maximum echo delay must be between 10 and 1000 milliseconds")]
    EchoDelayOutOfRange {
        /// The rejected maximum echo delay, in milliseconds.
        requested_ms: u16,
    },

    /// The configured coarse search ceiling was outside the supported range.
    ///
    /// It must be at least the fine search window and at most 2000
    /// milliseconds.
    #[error(
        "coarse search ceiling must be between the fine search window \
         ({fine_window_ms} ms) and 2000 milliseconds"
    )]
    SearchDelayOutOfRange {
        /// The rejected ceiling, in milliseconds.
        requested_ms: u16,
        /// The fine search window it must not undercut, in milliseconds.
        fine_window_ms: u16,
    },

    /// The configured filter tail was outside the supported range. The
    /// supported range is 16 to 500 milliseconds inclusive.
    #[error("filter tail must be between 16 and 500 milliseconds")]
    TailOutOfRange {
        /// The rejected filter tail length, in milliseconds.
        requested_ms: u16,
    },

    /// The model string did not name any publicly selectable model.
    ///
    /// This is the boundary error behind [`AecModel`]'s
    /// [`FromStr`](std::str::FromStr) parse: Python and Node callers pass the
    /// model as a string, so a typo like `"tao"` surfaces here with a message
    /// that names the received string and lists the available models. The
    /// available list is built from [`AecModel::PUBLIC_MODEL_NAMES`], so it can
    /// only ever name models a caller is allowed to select; a crate-internal
    /// reference implementation never appears in it.
    #[error("model must be one of: {}; got '{requested}'", available_model_names())]
    UnknownModel {
        /// The received string that named no publicly selectable model.
        requested: String,
    },
}

/// Formats the public model-name list for the [`AecError::UnknownModel`]
/// message: each name quoted, comma separated, in declaration order. Built
/// from [`AecModel::PUBLIC_MODEL_NAMES`] so the binding-facing text can only
/// ever name publicly selectable models.
fn available_model_names() -> String {
    AecModel::PUBLIC_MODEL_NAMES
        .iter()
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ")
}
