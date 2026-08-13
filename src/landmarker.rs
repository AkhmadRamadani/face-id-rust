//! MediaPipe Face Landmarker (468 3D landmarks), running on LiteRT GPU/CPU.
//!
//! Input: [1, 192, 192, 3] float32 [0..1] NHWC.
//! Output: [1, 1, 1, 1404] float32 -> 468 3D landmarks (x, y, z) normalized to [0..192].

use image::RgbImage;
use litert::Accelerators;

use crate::error::{FaceError, Result};
use crate::litert_backend::LiteRtModel;
use crate::tensor_prep::{image_to_nhwc, Normalization};

/// Canonical 36 MediaPipe Face Oval contour indices (tracing exact skin boundary of forehead, cheekbones, jawline, and chin).
pub const FACEMESH_FACE_OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

/// MediaPipe Left Eye contour ring indices.
pub const LEFT_EYE_INDICES: [usize; 8] = [33, 133, 160, 159, 158, 144, 145, 153];
/// MediaPipe Right Eye contour ring indices.
pub const RIGHT_EYE_INDICES: [usize; 8] = [362, 263, 387, 386, 385, 373, 374, 380];
/// MediaPipe Nose Tip index.
pub const NOSE_INDEX: usize = 1;
/// MediaPipe Left Mouth corner index.
pub const MOUTH_LEFT_INDEX: usize = 61;
/// MediaPipe Right Mouth corner index.
pub const MOUTH_RIGHT_INDEX: usize = 291;

pub struct FaceLandmarker {
    model: LiteRtModel,
}

impl FaceLandmarker {
    /// Loads the MediaPipe `face_landmark.tflite` model.
    pub fn load(path: impl AsRef<std::path::Path>, accelerators: Accelerators) -> Result<Self> {
        let model = LiteRtModel::load(path, accelerators)?;
        if model.input_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "mediapipe_landmarker_inputs",
                expected: 1,
                got: model.input_count(),
            });
        }
        Ok(Self { model })
    }

    /// Predicts 468 3D landmarks `[x, y, z]` for a face crop (x, y normalized to 0..192 space).
    pub fn predict(&mut self, face_crop: &RgbImage) -> Result<Vec<[f32; 3]>> {
        let resized = image::imageops::resize(
            face_crop,
            192,
            192,
            image::imageops::FilterType::Triangle,
        );

        let norm = Normalization::zero_to_one_rgb();
        let tensor = image_to_nhwc(&resized, &norm);

        let outputs = self.model.run_f32(&[&tensor])?;

        let landmark_output = outputs
            .iter()
            .find(|o| o.len() == 1404)
            .ok_or_else(|| FaceError::ShapeMismatch {
                name: "mediapipe_landmarks_output",
                expected: 1404,
                got: outputs.iter().map(|o| o.len()).max().unwrap_or(0),
            })?;

        let mut landmarks = Vec::with_capacity(468);
        for i in 0..468 {
            landmarks.push([
                landmark_output[i * 3],
                landmark_output[i * 3 + 1],
                landmark_output[i * 3 + 2],
            ]);
        }

        Ok(landmarks)
    }

    /// Extracts sub-pixel 5-point landmarks (Left Eye, Right Eye, Nose, Mouth Left, Mouth Right)
    /// from 468 MediaPipe 3D landmarks, mapped back to target image coordinate space.
    pub fn extract_landmarks5(
        mediapipe_468: &[[f32; 3]],
        crop_offset_x: f32,
        crop_offset_y: f32,
        crop_w: f32,
        crop_h: f32,
    ) -> crate::types::Landmarks5 {
        let map_pt = |idx: usize| -> [f32; 2] {
            let p = mediapipe_468[idx];
            [
                (p[0] / 192.0) * crop_w + crop_offset_x,
                (p[1] / 192.0) * crop_h + crop_offset_y,
            ]
        };

        let average_ring = |indices: &[usize]| -> [f32; 2] {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for &idx in indices {
                let p = map_pt(idx);
                sum_x += p[0];
                sum_y += p[1];
            }
            let n = indices.len() as f32;
            [sum_x / n, sum_y / n]
        };

        crate::types::Landmarks5 {
            left_eye: average_ring(&LEFT_EYE_INDICES),
            right_eye: average_ring(&RIGHT_EYE_INDICES),
            nose: map_pt(NOSE_INDEX),
            mouth_left: map_pt(MOUTH_LEFT_INDEX),
            mouth_right: map_pt(MOUTH_RIGHT_INDEX),
        }
    }
}
