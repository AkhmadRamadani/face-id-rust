//! The face registry itself: one global [`MutableKdTree`] plus one per
//! event, all searched together according to a [`RecognitionContext`].
//!
//! **Memory note:** each registered embedding is currently stored twice —
//! once inside the relevant Kiddo tree (which needs the raw point to support
//! `remove`), and once in `records` (needed to answer `remove` and to
//! rebuild trees on load). At 512 dims that's ~2KB/record x2. For registries
//! in the thousands this is a non-issue; if you're pushing into the
//! hundreds of thousands, consider periodically freezing settled scopes
//! (e.g. a finished event) into an [`kiddo::ImmutableKdTree`] + on-disk rkyv
//! snapshot and dropping them from live `VectorStore`, per Kiddo's own
//! guidance that heavy-churn `MutableKdTree` workloads benefit from
//! periodic rebuilds.

use std::collections::HashMap;
use std::num::NonZero;
use std::time::{SystemTime, UNIX_EPOCH};

use kiddo::{MutableKdTree, SquaredEuclidean};
use serde::{Deserialize, Serialize};

use crate::error::{FaceError, Result};
use crate::types::{squared_dist_to_cosine, Embedding, EMBED_DIM};

use super::scope::{EventId, PersonId, RecognitionContext, RegistrationScope};

pub type RecordId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub person_id: PersonId,
    pub scope: RegistrationScope,
    pub embedding: Embedding,
    /// Free-form note (e.g. a display name or photo reference) — not used
    /// for matching, just carried through to [`MatchResult`].
    pub label: Option<String>,
    pub registered_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchOrigin {
    Global,
    Event(EventId),
}

impl std::fmt::Display for MatchOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchOrigin::Global => write!(f, "global"),
            MatchOrigin::Event(id) => write!(f, "event:{id}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub record_id: RecordId,
    pub person_id: PersonId,
    pub origin: MatchOrigin,
    /// Cosine similarity in `[-1, 1]` (real face pairs land in `[0, 1]`).
    pub similarity: f32,
    pub label: Option<String>,
}

type Tree = MutableKdTree<f32, EMBED_DIM>;

#[derive(Default)]
pub struct VectorStore {
    global_tree: Tree,
    event_trees: HashMap<EventId, Tree>,
    records: HashMap<RecordId, Record>,
    next_id: RecordId,
}

impl VectorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Registers `embedding` under `person_id` in `scope`, returning the new
    /// registration's id. A person can hold multiple registrations — e.g.
    /// several enrollment photos, or both a global and an event-scoped
    /// sample — each gets its own [`RecordId`] and can be removed
    /// independently via [`Self::unregister`].
    pub fn register(
        &mut self,
        person_id: PersonId,
        embedding: Embedding,
        scope: RegistrationScope,
        label: Option<String>,
    ) -> Result<RecordId> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(FaceError::StoreExhausted)?;

        let point = *embedding.as_array();
        match &scope {
            RegistrationScope::Global => {
                self.global_tree
                    .add(&point, id)
                    .map_err(|e| FaceError::Construction(e.to_string()))?;
            }
            RegistrationScope::Event(event_id) => {
                self.event_trees
                    .entry(event_id.clone())
                    .or_default()
                    .add(&point, id)
                    .map_err(|e| FaceError::Construction(e.to_string()))?;
            }
        }

        self.records.insert(
            id,
            Record {
                id,
                person_id,
                scope,
                embedding,
                label,
                registered_at_unix_ms: now_unix_ms(),
            },
        );
        Ok(id)
    }

    /// Removes a single registration. Other registrations for the same
    /// person (other scopes, or other samples in the same scope) are
    /// untouched.
    pub fn unregister(&mut self, record_id: RecordId) -> Result<()> {
        let record = self
            .records
            .remove(&record_id)
            .ok_or(FaceError::RecordNotFound(record_id))?;
        let point = *record.embedding.as_array();
        match &record.scope {
            RegistrationScope::Global => self.global_tree.remove(&point, record_id),
            RegistrationScope::Event(event_id) => {
                if let Some(tree) = self.event_trees.get_mut(event_id) {
                    tree.remove(&point, record_id);
                }
            }
        }
        Ok(())
    }

    /// Removes every registration for `person_id`, across every scope.
    /// Returns how many registrations were removed.
    pub fn unregister_person(&mut self, person_id: &PersonId) -> usize {
        let ids: Vec<RecordId> = self
            .records
            .values()
            .filter(|r| &r.person_id == person_id)
            .map(|r| r.id)
            .collect();
        let n = ids.len();
        for id in ids {
            let _ = self.unregister(id);
        }
        n
    }

    /// Finds the single best match for `probe`: the global registry is
    /// always searched, and (if `context` names one) an event registry is
    /// searched alongside it on equal footing — whichever candidate is
    /// closer wins, regardless of which registry it came from. Returns
    /// `None` if the best candidate doesn't clear `similarity_threshold`
    /// (or if the registry is empty).
    pub fn identify(
        &self,
        probe: &Embedding,
        context: &RecognitionContext,
        similarity_threshold: f32,
    ) -> Option<MatchResult> {
        let mut best: Option<(RecordId, f32, MatchOrigin)> = None;

        if let Some((id, sim)) = nearest_in(&self.global_tree, probe) {
            best = Some((id, sim, MatchOrigin::Global));
        }

        if let Some(event_id) = context.event() {
            if let Some(tree) = self.event_trees.get(event_id) {
                if let Some((id, sim)) = nearest_in(tree, probe) {
                    let better = best.as_ref().map(|(_, b, _)| sim > *b).unwrap_or(true);
                    if better {
                        best = Some((id, sim, MatchOrigin::Event(event_id.clone())));
                    }
                }
            }
        }

        let (id, similarity, origin) = best?;
        if similarity < similarity_threshold {
            return None;
        }
        let record = self.records.get(&id)?;
        Some(MatchResult {
            record_id: id,
            person_id: record.person_id.clone(),
            origin,
            similarity,
            label: record.label.clone(),
        })
    }

    /// Like [`Self::identify`], but returns up to `k` candidates (merged
    /// from global + the context's event, sorted best-first, *not*
    /// threshold-filtered) — useful for a manual-review queue on
    /// borderline matches instead of a hard accept/reject.
    pub fn identify_top_k(&self, probe: &Embedding, context: &RecognitionContext, k: usize) -> Vec<MatchResult> {
        let mut all: Vec<(RecordId, f32, MatchOrigin)> = Vec::new();
        if let Some(hits) = nearest_n_in(&self.global_tree, probe, k) {
            all.extend(hits.into_iter().map(|(id, sim)| (id, sim, MatchOrigin::Global)));
        }
        if let Some(event_id) = context.event() {
            if let Some(tree) = self.event_trees.get(event_id) {
                if let Some(hits) = nearest_n_in(tree, probe, k) {
                    all.extend(
                        hits.into_iter()
                            .map(|(id, sim)| (id, sim, MatchOrigin::Event(event_id.clone()))),
                    );
                }
            }
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all.into_iter()
            .filter_map(|(id, similarity, origin)| {
                let record = self.records.get(&id)?;
                Some(MatchResult {
                    record_id: id,
                    person_id: record.person_id.clone(),
                    origin,
                    similarity,
                    label: record.label.clone(),
                })
            })
            .collect()
    }

    pub fn record(&self, id: RecordId) -> Option<&Record> {
        self.records.get(&id)
    }

    pub fn records_for_person(&self, person_id: &PersonId) -> Vec<&Record> {
        self.records.values().filter(|r| &r.person_id == person_id).collect()
    }

    pub fn all_records(&self) -> impl Iterator<Item = &Record> {
        self.records.values()
    }

    /// Rebuilds a store from a flat list of records (used by
    /// [`super::persistence`] on load). Skips — rather than fails on — a
    /// record whose tree insertion errors, since a single corrupt point
    /// shouldn't prevent the rest of the registry from loading; count the
    /// return value against `records.len()` if you want to detect that.
    pub(crate) fn rebuild_from_records(records: Vec<Record>) -> Self {
        let mut store = Self::new();
        let mut max_id = 0u32;
        for record in records {
            max_id = max_id.max(record.id);
            let point = *record.embedding.as_array();
            let insert = match &record.scope {
                RegistrationScope::Global => store.global_tree.add(&point, record.id),
                RegistrationScope::Event(event_id) => {
                    store.event_trees.entry(event_id.clone()).or_default().add(&point, record.id)
                }
            };
            if insert.is_ok() {
                store.records.insert(record.id, record);
            }
        }
        store.next_id = max_id.saturating_add(1);
        store
    }
}

fn nearest_in(tree: &Tree, probe: &Embedding) -> Option<(RecordId, f32)> {
    if tree.size() == 0 {
        return None;
    }
    let hit = tree.query(probe.as_array()).nearest_one::<SquaredEuclidean<f32>>().execute();
    Some((hit.item, squared_dist_to_cosine(hit.distance)))
}

fn nearest_n_in(tree: &Tree, probe: &Embedding, k: usize) -> Option<Vec<(RecordId, f32)>> {
    if tree.size() == 0 || k == 0 {
        return None;
    }
    let k = NonZero::new(k.min(tree.size()))?;
    let hits = tree.query(probe.as_array()).nearest_n::<SquaredEuclidean<f32>>(k).execute();
    Some(hits.into_iter().map(|h| (h.item, squared_dist_to_cosine(h.distance))).collect())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
