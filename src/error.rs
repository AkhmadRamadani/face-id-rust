//! Centralized error type. Every fallible operation in this crate returns
//! [`FaceError`] so callers only need to match on one type regardless of which
//! backend (LiteRT / ORT) or stage (detect / align / spoof-check / embed /
//! recognize) produced the failure.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, FaceError>;

#[derive(Debug, thiserror::Error)]
pub enum FaceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to decode/encode image: {0}")]
    Image(#[from] image::ImageError),

    #[error("failed to (de)serialize registry snapshot: {0}")]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "litert-runtime")]
    #[error("LiteRT runtime error: {0}")]
    LiteRt(#[from] litert::Error),

    #[cfg(feature = "ort-runtime")]
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("model at {path:?} has {actual:?} input/output tensors, but this crate only \
             drives Float32 I/O — export/convert the model to float32, or add a dedicated \
             quantized-I/O code path")]
    UnsupportedElementType {
        path: PathBuf,
        actual: String,
    },

    #[error("model at {path:?} declares {declared} input tensor(s), but this wrapper was \
             asked to feed {given}")]
    InputCountMismatch {
        path: PathBuf,
        declared: usize,
        given: usize,
    },

    #[error("tensor buffer for '{name}' expects {expected} elements, got {got}")]
    ShapeMismatch {
        name: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("embedding has {got} dimensions, expected {expected} (see EMBED_DIM in types.rs — \
             it must match your embedding model's output size)")]
    InvalidEmbeddingDim { expected: usize, got: usize },

    #[error("no face detected in the input image")]
    NoFaceDetected,

    #[error("{count} faces detected but this operation requires exactly one \
             (crop tighter, or use `detect_all` + pick a face yourself)")]
    AmbiguousFaceCount { count: usize },

    #[error("liveness check failed: live-score {live_score:.3} is below the \
             {threshold:.3} threshold — looks like a photo/replay attack, not a live face")]
    SpoofDetected { live_score: f32, threshold: f32 },

    #[error("person '{person_id}' is not registered in scope {scope}")]
    NotRegistered { person_id: String, scope: String },

    #[error("no registration record with id {0}")]
    RecordNotFound(u32),

    #[error("record {record_id} belongs to scope {actual}, not the requested scope {requested}")]
    ScopeMismatch {
        record_id: u32,
        requested: String,
        actual: String,
    },

    #[error("registry is full: exhausted the u32 record-id space")]
    StoreExhausted,

    #[error("k-d tree construction error: {0}")]
    Construction(String),

    #[error("invalid configuration: {0}")]
    Config(String),
}
