//! Scoped face recognition: register embeddings globally or under a
//! specific event, and recognize against either.
//!
//! ```text
//! register(person, embedding, Global)        -> recognized under ANY context
//! register(person, embedding, Event("expo"))  -> recognized only under
//!                                                 RecognitionContext::Event("expo")
//! ```
//!
//! See [`scope`] for the full semantics and [`store::VectorStore`] for the
//! Kiddo-backed implementation.

pub mod persistence;
pub mod scope;
pub mod store;

pub use scope::{EventId, PersonId, RecognitionContext, RegistrationScope};
pub use store::{MatchOrigin, MatchResult, Record, RecordId, VectorStore};
