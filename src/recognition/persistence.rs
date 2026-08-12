//! Registry persistence: Append-Only JSON Lines (`.jsonl`) format.
//!
//! Each face record is saved as a single JSON object on its own line.
//! Enrolling a new face appends a single line to disk in O(1) time without
//! rewriting existing records. On server startup, lines are streamed into
//! the Voyager HNSW index.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::store::{Record, VectorStore};
use crate::error::Result;

#[derive(Serialize, Deserialize)]
struct Snapshot {
    format_version: u32,
    records: Vec<Record>,
}

/// Appends a single record to `path` as a JSON line (O(1) disk write).
pub fn append_record(record: &Record, path: impl AsRef<Path>) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(record)?;
    writeln!(file, "{line}")?;
    Ok(())
}

/// Writes all records in `store` to `path` as JSON Lines (full rewrite/compaction).
pub fn save_all(store: &VectorStore, path: impl AsRef<Path>) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    for record in store.all_records() {
        let line = serde_json::to_string(record)?;
        writeln!(file, "{line}")?;
    }
    Ok(())
}

/// Alias for `save_all`.
pub fn save(store: &VectorStore, path: impl AsRef<Path>) -> Result<()> {
    save_all(store, path)
}

/// Loads a store from `path`. Streams JSON Lines line-by-line into Voyager.
/// If `path` is an old JSON array snapshot file, automatically falls back to parsing it.
pub fn load(path: impl AsRef<Path>) -> Result<VectorStore> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(VectorStore::new());
    }

    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut store = VectorStore::new();

    let mut first_line = String::new();
    if reader.read_line(&mut first_line)? > 0 {
        let trimmed = first_line.trim();
        if trimmed.starts_with('{') {
            if let Ok(rec) = serde_json::from_str::<Record>(trimmed) {
                store.insert_record(rec);
                for line in reader.lines() {
                    let line = line?;
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(rec) = serde_json::from_str::<Record>(trimmed) {
                            store.insert_record(rec);
                        }
                    }
                }
            } else if let Ok(snapshot) = serde_json::from_str::<Snapshot>(trimmed) {
                for rec in snapshot.records {
                    store.insert_record(rec);
                }
            }
        }
    }

    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognition::{PersonId, RegistrationScope};
    use crate::types::{Embedding, EMBED_DIM};

    #[test]
    fn test_jsonl_append_and_load() -> Result<()> {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_registry.jsonl");
        if file_path.exists() {
            let _ = std::fs::remove_file(&file_path);
        }

        let mut emb_alice = Embedding([0.0f32; EMBED_DIM]);
        emb_alice.0[0] = 1.0;
        emb_alice.normalize();

        let mut emb_bob = Embedding([0.0f32; EMBED_DIM]);
        emb_bob.0[1] = 1.0;
        emb_bob.normalize();

        let rec1 = Record {
            id: 1,
            person_id: PersonId("alice".into()),
            scope: RegistrationScope::Global,
            embedding: emb_alice.clone(),
            label: Some("Alice".into()),
            registered_at_unix_ms: 1000,
        };

        let rec2 = Record {
            id: 2,
            person_id: PersonId("bob".into()),
            scope: RegistrationScope::Event("expo-2026".into()),
            embedding: emb_bob.clone(),
            label: Some("Bob".into()),
            registered_at_unix_ms: 2000,
        };

        append_record(&rec1, &file_path)?;
        append_record(&rec2, &file_path)?;

        let store = load(&file_path)?;
        assert_eq!(store.len(), 2);

        // Verify global search matches Alice
        let match_alice = store.identify(&emb_alice, &crate::recognition::RecognitionContext::GlobalOnly, 0.5);
        assert!(match_alice.is_some());
        assert_eq!(match_alice.unwrap().person_id.0, "alice");

        // Verify event search matches Bob
        let match_bob = store.identify(&emb_bob, &crate::recognition::RecognitionContext::Event("expo-2026".into()), 0.5);
        assert!(match_bob.is_some());
        assert_eq!(match_bob.unwrap().person_id.0, "bob");

        let _ = std::fs::remove_file(file_path);
        Ok(())
    }
}
