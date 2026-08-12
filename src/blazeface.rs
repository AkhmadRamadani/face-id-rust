//! MediaPipe BlazeFace Detector, running on LiteRT GPU/CPU.
//!
//! Models supported:
//! - `blazeface.tflite` (Short-range, 128x128 input, 896 anchors)
//! - `blazeface_full_range.tflite` (Full-range, 192x192 input, 2304 anchors)

use image::RgbImage;
use litert::Accelerators;

use crate::error::{FaceError, Result};
use crate::litert_backend::LiteRtModel;
use crate::tensor_prep::{image_to_nhwc, Normalization};
use crate::types::{BBox, DetectedFace, Landmarks5};

/// Configuration for the BlazeFace detector.
#[derive(Debug, Clone)]
pub struct BlazeFaceConfig {
    pub score_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for BlazeFaceConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            iou_threshold: 0.3,
        }
    }
}

pub struct BlazeFaceDetector {
    model: LiteRtModel,
    config: BlazeFaceConfig,
    input_w: u32,
    input_h: u32,
    anchors: Vec<[f32; 2]>,
}

impl BlazeFaceDetector {
    /// Loads the BlazeFace model (`blazeface.tflite` or `blazeface_full_range.tflite`).
    pub fn load(
        path: impl AsRef<std::path::Path>,
        accelerators: Accelerators,
        config: BlazeFaceConfig,
    ) -> Result<Self> {
        let mut model = LiteRtModel::load(path, accelerators)?;
        if model.input_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "blazeface_input_count",
                expected: 1,
                got: model.input_count(),
            });
        }

        let (input_w, input_h, anchors) = {
            if let Ok(outs) = model.run_f32(&[&vec![0.0; 1 * 192 * 192 * 3]]) {
                let max_anchors = outs.iter().map(|o| o.len()).max().unwrap_or(0);
                if max_anchors == 2304 * 16 || max_anchors == 2304 {
                    (192, 192, generate_blazeface_full_range_anchors())
                } else {
                    (128, 128, generate_blazeface_short_range_anchors())
                }
            } else {
                (128, 128, generate_blazeface_short_range_anchors())
            }
        };

        Ok(Self {
            model,
            config,
            input_w,
            input_h,
            anchors,
        })
    }

    /// Detects all faces in `image`, returning bounding boxes and 5-point facial landmarks.
    pub fn detect_all(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>> {
        let orig_w = image.width() as f32;
        let orig_h = image.height() as f32;

        let resized = image::imageops::resize(
            image,
            self.input_w,
            self.input_h,
            image::imageops::FilterType::Triangle,
        );

        let norm = Normalization::arcface_rgb();
        let tensor = image_to_nhwc(&resized, &norm);

        let outputs = self.model.run_f32(&[&tensor])?;

        let regressors = outputs
            .iter()
            .find(|o| o.len() % 16 == 0 && !o.is_empty())
            .ok_or_else(|| FaceError::ShapeMismatch {
                name: "blazeface_regressors",
                expected: 16,
                got: outputs.iter().map(|o| o.len()).max().unwrap_or(0),
            })?;

        let num_anchors = regressors.len() / 16;

        let classificators = outputs
            .iter()
            .find(|o| o.len() == num_anchors)
            .ok_or_else(|| FaceError::ShapeMismatch {
                name: "blazeface_classificators",
                expected: num_anchors,
                got: outputs.iter().map(|o| o.len()).min().unwrap_or(0),
            })?;

        if self.anchors.len() != num_anchors {
            self.anchors = generate_anchors_for_count(num_anchors, self.input_w);
        }

        let mut candidates = Vec::new();
        let input_w_f = self.input_w as f32;
        let input_h_f = self.input_h as f32;

        for i in 0..num_anchors {
            let score_logit = classificators[i];
            let score = 1.0 / (1.0 + (-score_logit.clamp(-100.0, 100.0)).exp());

            if score < self.config.score_threshold {
                continue;
            }

            let reg_base = i * 16;
            let anc_cx = self.anchors[i][0];
            let anc_cy = self.anchors[i][1];

            let cx = (regressors[reg_base] + anc_cx) / input_w_f;
            let cy = (regressors[reg_base + 1] + anc_cy) / input_h_f;
            let w = regressors[reg_base + 2] / input_w_f;
            let h = regressors[reg_base + 3] / input_h_f;

            let x1 = (cx - w / 2.0) * orig_w;
            let y1 = (cy - h / 2.0) * orig_h;
            let x2 = (cx + w / 2.0) * orig_w;
            let y2 = (cy + h / 2.0) * orig_h;

            let decode_kpt = |k: usize| -> [f32; 2] {
                let kx = regressors[reg_base + 4 + k * 2];
                let ky = regressors[reg_base + 4 + k * 2 + 1];
                [
                    ((kx + anc_cx) / input_w_f) * orig_w,
                    ((ky + anc_cy) / input_h_f) * orig_h,
                ]
            };

            let left_eye = decode_kpt(0);
            let right_eye = decode_kpt(1);
            let nose = decode_kpt(2);
            let mouth_center = decode_kpt(3);

            // Compute mouth left and mouth right from mouth center + eye spread vector
            let eye_dx = right_eye[0] - left_eye[0];
            let eye_dy = right_eye[1] - left_eye[1];

            let mouth_left = [
                mouth_center[0] - 0.25 * eye_dx,
                mouth_center[1] - 0.25 * eye_dy,
            ];
            let mouth_right = [
                mouth_center[0] + 0.25 * eye_dx,
                mouth_center[1] + 0.25 * eye_dy,
            ];

            candidates.push(DetectedFace {
                bbox: BBox { x1, y1, x2, y2 },
                score,
                landmarks: Landmarks5 {
                    left_eye,
                    right_eye,
                    nose,
                    mouth_left,
                    mouth_right,
                },
            });
        }

        // Apply Non-Maximum Suppression (NMS)
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut keep: Vec<DetectedFace> = Vec::new();
        for cand in candidates {
            let mut overlap = false;
            for kept in &keep {
                if iou(&cand.bbox, &kept.bbox) > self.config.iou_threshold {
                    overlap = true;
                    break;
                }
            }
            if !overlap {
                keep.push(cand);
            }
        }

        Ok(keep)
    }
}

fn generate_anchors_for_count(count: usize, _input_size: u32) -> Vec<[f32; 2]> {
    if count == 896 {
        generate_blazeface_short_range_anchors()
    } else if count == 2304 {
        generate_blazeface_full_range_anchors()
    } else {
        let side = (count as f32).sqrt() as usize;
        let mut anchors = Vec::with_capacity(count);
        for y in 0..side {
            for x in 0..side {
                let cx = (x as f32 + 0.5) * 128.0 / (side as f32);
                let cy = (y as f32 + 0.5) * 128.0 / (side as f32);
                anchors.push([cx, cy]);
            }
        }
        while anchors.len() < count {
            anchors.push([64.0, 64.0]);
        }
        anchors
    }
}

fn generate_blazeface_short_range_anchors() -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(896);
    let specs = [
        (16, 16, 8, 2),  // 16x16 grid, 2 anchors per cell -> 512
        (8, 8, 16, 6),   // 8x8 grid, 6 anchors per cell -> 384
    ];
    for &(grid_w, grid_h, stride, count) in &specs {
        for y in 0..grid_h {
            for x in 0..grid_w {
                let cx = (x as f32 + 0.5) * (stride as f32);
                let cy = (y as f32 + 0.5) * (stride as f32);
                for _ in 0..count {
                    anchors.push([cx, cy]);
                }
            }
        }
    }
    anchors
}

fn generate_blazeface_full_range_anchors() -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(2304);
    let specs = [
        (24, 24, 8, 2),  // 24x24 grid, 2 anchors per cell -> 1152
        (12, 12, 16, 8), // 12x12 grid, 8 anchors per cell -> 1152
    ];
    for &(grid_w, grid_h, stride, count) in &specs {
        for y in 0..grid_h {
            for x in 0..grid_w {
                let cx = (x as f32 + 0.5) * (stride as f32);
                let cy = (y as f32 + 0.5) * (stride as f32);
                for _ in 0..count {
                    anchors.push([cx, cy]);
                }
            }
        }
    }
    anchors
}

fn iou(a: &BBox, b: &BBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = a.width() * a.height();
    let area_b = b.width() * b.height();
    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
