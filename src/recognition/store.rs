//! The face registry itself: one global [`Voyager`] HNSW index plus one per
//! event, all searched together according to a [`RecognitionContext`].
//!
//! **Memory note:** each registered embedding is stored in the Voyager HNSW index
//! and in `records` (needed for payload queries, persistence, and tombstone filtering
//! on unregister).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use voyager_rs::Voyager;

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

pub struct VoyagerIndex {
    inner: Voyager<EMBED_DIM>,
    count: usize,
}

impl Default for VoyagerIndex {
    fn default() -> Self {
        Self {
            inner: Voyager::new(),
            count: 0,
        }
    }
}

unsafe impl Send for VoyagerIndex {}
unsafe impl Sync for VoyagerIndex {}

#[derive(Default)]
pub struct VectorStore {
    global_tree: VoyagerIndex,
    event_trees: HashMap<EventId, VoyagerIndex>,
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
                self.global_tree.inner.add_item(point, Some(id));
                self.global_tree.count += 1;
            }
            RegistrationScope::Event(event_id) => {
                let tree = self.event_trees.entry(event_id.clone()).or_default();
                tree.inner.add_item(point, Some(id));
                tree.count += 1;
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
        match &record.scope {
            RegistrationScope::Global => {
                if self.global_tree.count > 0 {
                    self.global_tree.count -= 1;
                }
            }
            RegistrationScope::Event(event_id) => {
                if let Some(tree) = self.event_trees.get_mut(event_id) {
                    if tree.count > 0 {
                        tree.count -= 1;
                    }
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

        if let Some((id, sim)) = nearest_in(&self.global_tree, probe, &self.records) {
            best = Some((id, sim, MatchOrigin::Global));
        }

        if let Some(event_id) = context.event() {
            if let Some(tree) = self.event_trees.get(event_id) {
                if let Some((id, sim)) = nearest_in(tree, probe, &self.records) {
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
        if let Some(hits) = nearest_n_in(&self.global_tree, probe, k, &self.records) {
            all.extend(hits.into_iter().map(|(id, sim)| (id, sim, MatchOrigin::Global)));
        }
        if let Some(event_id) = context.event() {
            if let Some(tree) = self.event_trees.get(event_id) {
                if let Some(hits) = nearest_n_in(tree, probe, k, &self.records) {
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

    /// Inserts a pre-existing record directly into the store and index.
    pub fn insert_record(&mut self, record: Record) {
        self.next_id = self.next_id.max(record.id.saturating_add(1));
        let point = *record.embedding.as_array();
        match &record.scope {
            RegistrationScope::Global => {
                self.global_tree.inner.add_item(point, Some(record.id));
                self.global_tree.count += 1;
            }
            RegistrationScope::Event(event_id) => {
                let tree = self.event_trees.entry(event_id.clone()).or_default();
                tree.inner.add_item(point, Some(record.id));
                tree.count += 1;
            }
        }
        self.records.insert(record.id, record);
    }

    /// Rebuilds a store from a flat list of records.
    pub fn rebuild_from_records(records: Vec<Record>) -> Self {
        let mut store = Self::new();
        for record in records {
            store.insert_record(record);
        }
        store
    }
}

fn nearest_in(tree: &VoyagerIndex, probe: &Embedding, records: &HashMap<RecordId, Record>) -> Option<(RecordId, f32)> {
    if tree.count == 0 {
        return None;
    }
    let k = (tree.count.min(10)) as i32;
    let (ids, distances) = tree.inner.query(*probe.as_array(), k, None);
    for (id_idx, dist) in ids.into_iter().zip(distances.into_iter()) {
        let rec_id = id_idx as RecordId;
        if records.contains_key(&rec_id) {
            return Some((rec_id, squared_dist_to_cosine(dist)));
        }
    }
    None
}

fn nearest_n_in(
    tree: &VoyagerIndex,
    probe: &Embedding,
    k: usize,
    records: &HashMap<RecordId, Record>,
) -> Option<Vec<(RecordId, f32)>> {
    if tree.count == 0 || k == 0 {
        return None;
    }
    let fetch_k = (k * 2).max(tree.count).min(100) as i32;
    let (ids, distances) = tree.inner.query(*probe.as_array(), fetch_k, None);
    let mut hits = Vec::new();
    for (id_idx, dist) in ids.into_iter().zip(distances.into_iter()) {
        let rec_id = id_idx as RecordId;
        if records.contains_key(&rec_id) {
            hits.push((rec_id, squared_dist_to_cosine(dist)));
            if hits.len() == k {
                break;
            }
        }
    }
    if hits.is_empty() {
        None
    } else {
        Some(hits)
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_embedding(val: f32) -> Embedding {
        let mut arr = [0.0f32; EMBED_DIM];
        arr[0] = val;
        let mut emb = Embedding(arr);
        emb.normalize();
        emb
    }

    #[test]
    fn test_vector_store_voyager() {
        let mut store = VectorStore::new();

        let alice_emb = dummy_embedding(1.0);
        let bob_emb = dummy_embedding(-1.0);

        let alice_id = store
            .register(
                PersonId("alice".into()),
                alice_emb.clone(),
                RegistrationScope::Global,
                Some("Alice".into()),
            )
            .unwrap();

        let bob_id = store
            .register(
                PersonId("bob".into()),
                bob_emb.clone(),
                RegistrationScope::Event(EventId("expo".into())),
                Some("Bob".into()),
            )
            .unwrap();

        assert_eq!(store.len(), 2);

        // Identify global
        let match_global = store
            .identify(&alice_emb, &RecognitionContext::GlobalOnly, 0.5)
            .unwrap();
        assert_eq!(match_global.record_id, alice_id);
        assert_eq!(match_global.person_id.0, "alice");

        // Identify event
        let match_event = store
            .identify(&bob_emb, &RecognitionContext::Event(EventId("expo".into())), 0.5)
            .unwrap();
        assert_eq!(match_event.record_id, bob_id);
        assert_eq!(match_event.person_id.0, "bob");
    }
}
