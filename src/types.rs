//! Core value types shared by every stage of the pipeline.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Dimensionality of the face embedding vector.
///
/// This has to be a compile-time constant: Kiddo's `KdTree` takes the point
/// dimension as a `const K: usize` generic parameter, and `Embedding` below
/// stores its data inline as `[f32; EMBED_DIM]` (no heap allocation per
/// embedding — this matters once you have tens of thousands of registrations
/// resident in memory).
///
/// **Change this to match your embedding model's output size and recompile.**
/// Common values: 128 (FaceNet), 256/512 (ArcFace, MobileFaceNet, FaceLiVT).
pub const EMBED_DIM: usize = 192;

/// An axis-aligned bounding box in pixel coordinates of the *original*,
/// un-letterboxed image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Intersection-over-union against another box. Used by NMS.
    pub fn iou(&self, other: &BBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Expands the box by `factor` around its center (e.g. 2.7x for the
    /// anti-spoofing crop convention). Does not clip to image bounds.
    pub fn scaled_about_center(&self, factor: f32) -> BBox {
        let cx = (self.x1 + self.x2) / 2.0;
        let cy = (self.y1 + self.y2) / 2.0;
        let hw = self.width() * factor / 2.0;
        let hh = self.height() * factor / 2.0;
        BBox {
            x1: cx - hw,
            y1: cy - hh,
            x2: cx + hw,
            y2: cy + hh,
        }
    }
}

/// A 2D point in pixel coordinates.
pub type Point2 = [f32; 2];

/// The 5 landmarks YuNet emits, in its native order: left eye, right eye,
/// nose tip, left mouth corner, right mouth corner. "Left"/"right" are from
/// the subject's own perspective (i.e. mirrored vs. what you see on screen).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmarks5 {
    pub left_eye: Point2,
    pub right_eye: Point2,
    pub nose: Point2,
    pub mouth_left: Point2,
    pub mouth_right: Point2,
}

impl Landmarks5 {
    pub fn as_array(&self) -> [Point2; 5] {
        [
            self.left_eye,
            self.right_eye,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        ]
    }
}

/// One detected face, in original-image pixel coordinates.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: BBox,
    pub score: f32,
    pub landmarks: Landmarks5,
}

/// A face crop that has been similarity-warped into the canonical pose the
/// embedding model expects (square, landmarks aligned to a fixed template).
#[derive(Debug, Clone)]
pub struct AlignedFace {
    /// Interleaved RGB8, row-major, length == width * height * 3.
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Outcome of a liveness (anti-spoofing) check.
#[derive(Debug, Clone, Copy)]
pub struct LivenessResult {
    /// Softmax probability of the "live" class. Higher = more likely real.
    pub live_score: f32,
    pub is_live: bool,
}

/// A face embedding: `EMBED_DIM` floats stored inline (no heap indirection),
/// which also happens to be exactly the shape Kiddo wants for a tree point.
///
/// Uses `serde-big-array`'s `BigArray` for (de)serialization: serde's own
/// derive only implements `Serialize`/`Deserialize` for arrays up to length
/// 32 (a fixed set of macro-generated impls, not truly const-generic), which
/// `EMBED_DIM` will essentially always exceed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(#[serde(with = "BigArray")] pub [f32; EMBED_DIM]);

impl Embedding {
    /// Builds an embedding from a model's raw output, L2-normalizing it.
    ///
    /// Normalizing here — once, at the source — is what lets the recognizer
    /// use plain squared-Euclidean distance and still get cosine-similarity
    /// ranking (`||a-b||^2 = 2 - 2*cos(a,b)` for unit vectors), so Kiddo
    /// doesn't need a custom distance metric.
    pub fn from_raw(values: &[f32]) -> crate::error::Result<Self> {
        if values.len() != EMBED_DIM {
            return Err(crate::error::FaceError::InvalidEmbeddingDim {
                expected: EMBED_DIM,
                got: values.len(),
            });
        }
        let mut arr = [0f32; EMBED_DIM];
        arr.copy_from_slice(values);
        let mut emb = Embedding(arr);
        emb.normalize();
        Ok(emb)
    }

    pub fn normalize(&mut self) {
        let norm: f32 = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in self.0.iter_mut() {
                *v /= norm;
            }
        }
    }

    #[inline]
    pub fn as_array(&self) -> &[f32; EMBED_DIM] {
        &self.0
    }

    /// Cosine similarity, valid for any two embeddings regardless of whether
    /// they came from `from_raw` (both sides are re-derived from the dot
    /// product + norms rather than assuming pre-normalization).
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = other.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na <= 1e-12 || nb <= 1e-12 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

/// Converts a Kiddo squared-Euclidean distance between two *unit* vectors
/// back into cosine similarity: `d2 = 2 - 2*cos` => `cos = 1 - d2/2`.
#[inline]
pub fn squared_dist_to_cosine(squared_distance: f32) -> f32 {
    (1.0 - squared_distance / 2.0).clamp(-1.0, 1.0)
}
