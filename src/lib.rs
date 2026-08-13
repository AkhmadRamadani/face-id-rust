//! # faceid
//!
//! On-device face recognition: [`FaceDetector`] (YuNet) and
//! [`LivenessDetector`] (MiniFASNetV2 anti-spoofing) run on the LiteRT GPU
//! delegate; [`Embedder`] is a pluggable trait with an ONNX Runtime backend
//! ([`OrtEmbedder`]) and a LiteRT backend ([`LiteRtEmbedder`]); recognition
//! is backed by Spotify's [`voyager`] HNSW index, scoped globally or per-event (see
//! [`recognition`]).
//!
//! ## Pipeline
//!
//! ```text
//! image -> detect (YuNet)        -> DetectedFace { bbox, landmarks, score }
//!       -> liveness (optional)   -> reject presentation attacks
//!       -> align (5-pt warp)     -> AlignedFace (canonical pose, fixed size)
//!       -> embed (ORT or LiteRT) -> Embedding ([f32; EMBED_DIM], L2-normalized)
//!       -> register / identify  -> VectorStore (global + per-event Voyager trees)
//! ```
//!
//! [`FacePipeline`] drives all of this for you; the individual stages
//! ([`detector`], [`antispoof`], [`align`], [`embedder`], [`recognition`])
//! are also public if you need to compose them differently (e.g. running
//! detection on every frame of a video stream but only embedding once every
//! few seconds).
//!
//! ## Memory-usage notes
//!
//! - Embeddings are `[f32; EMBED_DIM]`, stored inline (no heap indirection
//!   per embedding), and compared with squared-Euclidean distance on
//!   L2-normalized vectors instead of a custom cosine metric — see
//!   [`types::squared_dist_to_cosine`].
//! - No `ndarray` dependency: tensors are built and read as flat `Vec<f32>` +
//!   shape tuples on both the `ort` and `litert` sides.
//! - [`config::MemoryProfile`] trades ORT's memory arena / thread count for
//!   a smaller, more predictable resident footprint when you need it.
//! - See `recognition::store` for the one real memory tradeoff worth
//!   knowing about at large registry sizes (each embedding is currently
//!   held both in the Kiddo tree and in the registry's record table).

pub mod align;
pub mod config;
pub mod embedder;
pub mod error;
pub mod geometry;
pub mod recognition;
pub mod tensor_prep;
pub mod types;

#[cfg(feature = "litert-runtime")]
pub mod antispoof;
#[cfg(feature = "litert-runtime")]
pub mod blazeface;
#[cfg(feature = "litert-runtime")]
pub mod detector;
#[cfg(feature = "litert-runtime")]
mod litert_backend;
// The orchestration layer hard-depends on `FaceDetector`/`LivenessDetector`
// (detection and anti-spoofing are LiteRT-only per this crate's design — see
// the module docs on `detector`/`antispoof`), so it only makes sense — and
// only compiles — with `litert-runtime` enabled. Building with only
// `ort-runtime` still gets you `Embedder`/`OrtEmbedder` directly; you'd just
// drive detect/align/embed/recognize yourself instead of through
// `FacePipeline`.
#[cfg(feature = "litert-runtime")]
pub mod landmarker;
#[cfg(feature = "litert-runtime")]
pub mod pipeline;
#[cfg(feature = "litert-runtime")]
pub mod server;

pub use error::{FaceError, Result};
pub use recognition::{
    EventId, MatchOrigin, MatchResult, PersonId, RecognitionContext, RegistrationScope, VectorStore,
};
pub use types::{AlignedFace, BBox, DetectedFace, Embedding, Landmarks5, LivenessResult, EMBED_DIM};

pub use embedder::{Embedder, EmbedderConfig};

#[cfg(feature = "litert-runtime")]
pub use antispoof::{AntiSpoofConfig, LivenessDetector};
#[cfg(feature = "litert-runtime")]
pub use blazeface::{BlazeFaceConfig, BlazeFaceDetector};
#[cfg(feature = "litert-runtime")]
pub use detector::{DetectorConfig, FaceDetector};
#[cfg(feature = "litert-runtime")]
pub use landmarker::{FACEMESH_FACE_OVAL, FaceLandmarker};
#[cfg(feature = "litert-runtime")]
pub use pipeline::{load_image, load_image_from_bytes, FacePipeline, PipelineOptions, PipelinePool};
#[cfg(feature = "litert-runtime")]
pub use embedder::litert_embedder::LiteRtEmbedder;
#[cfg(feature = "ort-runtime")]
pub use embedder::ort_embedder::OrtEmbedder;

/// Re-exported so callers can pick GPU/CPU/NPU without a direct `litert`
/// dependency of their own.
#[cfg(feature = "litert-runtime")]
pub use litert::Accelerators;
