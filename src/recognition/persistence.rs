//! Registry persistence: a flat JSON snapshot of every [`Record`], rebuilt
//! into fresh Kiddo trees on load.
//!
//! JSON (not a binary format like `bincode`) is the deliberate choice here:
//! registries in this crate's target size range (thousands to low tens of
//! thousands of faces) are small enough that the size difference doesn't
//! matter, and a human/diff-readable, cross-language-friendly format is more
//! useful when this store is one piece of a larger system (e.g. inspected
//! from a Dart/Flutter side, or diffed in version control during testing).
//! Swap in `bincode` here if you're at a scale where that tradeoff flips.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::store::{Record, VectorStore};
use crate::error::Result;

const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Snapshot {
    format_version: u32,
    records: Vec<Record>,
}

/// Writes every registration in `store` to `path` as a JSON snapshot.
pub fn save(store: &VectorStore, path: impl AsRef<Path>) -> Result<()> {
    let snapshot = Snapshot {
        format_version: FORMAT_VERSION,
        records: store.all_records().cloned().collect(),
    };
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &snapshot)?;
    Ok(())
}

/// Loads a snapshot written by [`save`], rebuilding the global tree and
/// every event tree from the flat record list (an O(n log n) rebuild —
/// fine at realistic registry sizes; see the module docs on `store.rs` for
/// the tradeoff this implies at very large scale).
pub fn load(path: impl AsRef<Path>) -> Result<VectorStore> {
    let file = std::fs::File::open(path)?;
    let snapshot: Snapshot = serde_json::from_reader(file)?;
    Ok(VectorStore::rebuild_from_records(snapshot.records))
}
