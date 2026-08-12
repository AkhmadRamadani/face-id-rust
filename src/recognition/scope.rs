//! Identity and scope types for the registry.
//!
//! **Scoping model:** a face can be registered [`RegistrationScope::Global`]
//! (recognized everywhere) or under a specific [`RegistrationScope::Event`]
//! (recognized only when the recognizer is given that same event as its
//! [`RecognitionContext`]). When recognizing *within* an event context, both
//! the global registry and that event's registry are searched together — a
//! globally-registered person is always found; someone registered only for
//! event A is invisible while recognizing under event B.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

impl From<&str> for EventId {
    fn from(s: &str) -> Self {
        EventId(s.to_string())
    }
}

impl From<String> for EventId {
    fn from(s: String) -> Self {
        EventId(s)
    }
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PersonId(pub String);

impl From<&str> for PersonId {
    fn from(s: &str) -> Self {
        PersonId(s.to_string())
    }
}

impl From<String> for PersonId {
    fn from(s: String) -> Self {
        PersonId(s)
    }
}

impl std::fmt::Display for PersonId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Where a single registration (one embedding sample) lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegistrationScope {
    /// Recognized regardless of event context.
    Global,
    /// Recognized only when the recognizer's [`RecognitionContext`] names
    /// this same event.
    Event(EventId),
}

impl std::fmt::Display for RegistrationScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistrationScope::Global => write!(f, "global"),
            RegistrationScope::Event(id) => write!(f, "event:{id}"),
        }
    }
}

/// Which registrations to search when recognizing a probe face.
///
/// The global registry is *always* included; naming an event additionally
/// searches that event's registry. There is deliberately no "search all
/// events" mode — cross-event visibility has to be opted into per person by
/// registering them globally, not granted implicitly by the query.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RecognitionContext {
    #[default]
    GlobalOnly,
    Event(EventId),
}

impl RecognitionContext {
    pub fn event(&self) -> Option<&EventId> {
        match self {
            RecognitionContext::Event(id) => Some(id),
            RecognitionContext::GlobalOnly => None,
        }
    }
}
