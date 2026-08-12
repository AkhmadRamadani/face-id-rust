//! Pluggable embedding backend: run either an ONNX model (via `ort`) or a
//! `.tflite` model (via `litert`) to turn an [`AlignedFace`] into an
//! [`Embedding`]. Both backends implement the same [`Embedder`] trait so the
//! rest of the pipeline doesn't care which one you picked.

use crate::error::Result;
use crate::tensor_prep::Normalization;
use crate::types::{AlignedFace, Embedding};

#[cfg(feature = "litert-runtime")]
pub mod litert_embedder;
#[cfg(feature = "ort-runtime")]
pub mod ort_embedder;

/// Preprocessing parameters for the embedding model.
///
/// `input_size` is the square resolution the model expects an [`AlignedFace`]
/// to already be warped to before embedding.
///
/// Set `input_size` to `None` (the default) to let the backend derive it
/// automatically from the model's declared input shape at load time — this
/// works for any well-formed export (FaceNet 160, ArcFace 112, etc.) without
/// you having to know the size in advance.
///
/// Set `input_size` to `Some(n)` to assert that the model must use that
/// specific size; the loader will fail fast if the model disagrees.
///
/// `normalization` defaults to the common ArcFace / FaceNet / MobileFaceNet
/// convention: RGB, `(x/255 - 0.5) / 0.5`.
#[derive(Debug, Clone, Copy)]
pub struct EmbedderConfig {
    /// Square input resolution. `None` = infer from the model's signature.
    pub input_size: Option<u32>,
    pub normalization: Normalization,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            input_size: None,
            normalization: Normalization::arcface_rgb(),
        }
    }
}

pub trait Embedder: Send {
    /// The square input resolution this embedder expects an [`AlignedFace`]
    /// to already be warped to (see `align_face` in `crate::align`). The
    /// pipeline reads this instead of requiring every caller to separately
    /// track and pass an alignment size that has to match the model.
    fn input_size(&self) -> u32;

    /// Runs the embedding model on an already-aligned face and returns an
    /// L2-normalized [`Embedding`] of length [`crate::types::EMBED_DIM`].
    fn embed(&mut self, face: &AlignedFace) -> Result<Embedding>;

    /// 1:1 verification between two aligned faces, as a cosine similarity in
    /// `[-1, 1]` (in practice usually `[0, 1]` for real face pairs). The
    /// default implementation just embeds both and compares; override only
    /// if a backend can share work between the two forward passes.
    fn verify(&mut self, a: &AlignedFace, b: &AlignedFace) -> Result<f32> {
        let ea = self.embed(a)?;
        let eb = self.embed(b)?;
        Ok(ea.cosine_similarity(&eb))
    }
}

impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn input_size(&self) -> u32 {
        (**self).input_size()
    }

    fn embed(&mut self, face: &AlignedFace) -> Result<Embedding> {
        (**self).embed(face)
    }

    fn verify(&mut self, a: &AlignedFace, b: &AlignedFace) -> Result<f32> {
        (**self).verify(a, b)
    }
}
