//! Combines the detector's 5 landmarks with the similarity-transform math in
//! `geometry.rs` to produce a canonically-posed [`AlignedFace`] ready for the
//! embedder. This is the "straighten the face" step: it corrects in-plane
//! rotation, scale, and translation so the embedding model always sees eyes
//! and mouth in roughly the same place regardless of head pose in the
//! original photo.

use image::RgbImage;

use crate::geometry::{arcface_reference_scaled, warp_align, SimilarityTransform};
use crate::types::{AlignedFace, Landmarks5};

/// Aligns `image` using `landmarks`, warping into an `out_size`x`out_size`
/// crop matching the ArcFace-style reference template (see
/// [`crate::geometry::ARCFACE_REFERENCE_112`]).
///
/// `out_size` should match your embedding model's expected input resolution
/// (112 is the most common; some models use 96, 120, or 128).
pub fn align_face(image: &RgbImage, landmarks: &Landmarks5, out_size: u32) -> AlignedFace {
    let reference = arcface_reference_scaled(out_size);
    let transform = SimilarityTransform::estimate(&landmarks.as_array(), &reference);
    let warped = warp_align(image, &transform, out_size);
    AlignedFace {
        rgb: warped.into_raw(),
        width: out_size,
        height: out_size,
    }
}
