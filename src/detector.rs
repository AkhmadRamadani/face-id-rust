//! YuNet face detector (boxes + 5 landmarks), running on the LiteRT
//! `CompiledModel` GPU delegate.
//!
//! I/O contract (from the model's published spec — verified against the
//! model's own declared signature shapes at load time, see [`FaceDetector::load`]):
//! - Input: `[1,3,640,640]` NCHW, BGR, raw `0..255` (no normalization).
//! - Output: 12 tensors in a fixed order — `cls`, `obj`, `bbox`, `kps`, each
//!   repeated for strides `{8,16,32}` — anchor-free, decoded on the host.

use image::RgbImage;
use litert::Accelerators;

use crate::error::{FaceError, Result};
use crate::geometry::letterbox;
use crate::litert_backend::LiteRtModel;
use crate::tensor_prep::{image_to_nchw, Normalization};
use crate::types::{BBox, DetectedFace, Landmarks5};

const STRIDES: [u32; 3] = [8, 16, 32];

#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// Square input size the model was exported for. 640 for the published
    /// YuNet LiteRT export; change only if you re-export at a different size.
    pub input_size: u32,
    /// Minimum `cls * obj` score to keep a candidate box, pre-NMS.
    pub score_threshold: f32,
    /// IoU threshold for greedy NMS.
    pub nms_iou_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            score_threshold: 0.6,
            nms_iou_threshold: 0.45,
        }
    }
}

pub struct FaceDetector {
    model: LiteRtModel,
    config: DetectorConfig,
}

impl FaceDetector {
    /// Loads the YuNet `.tflite` export and compiles it for `accelerators`.
    /// Pass `Accelerators::GPU` for the "fully on GPU, no fallback" mode the
    /// upstream benchmark describes; union with `Accelerators::CPU` if you'd
    /// rather degrade to CPU than fail to load on unsupported hardware.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        accelerators: Accelerators,
        config: DetectorConfig,
    ) -> Result<Self> {
        let model = LiteRtModel::load(path, accelerators)?;

        // Fail loudly at load time rather than mysteriously mis-decoding
        // every detection if someone points this at a differently-exported
        // YuNet variant (e.g. one that fuses strides or drops obj/cls), or
        // at a model exported for a different input resolution than
        // `config.input_size`.
        if model.input_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "yunet_signature_inputs",
                expected: 1,
                got: model.input_count(),
            });
        }
        let expected_input_elements = 3 * (config.input_size as usize) * (config.input_size as usize);
        if model.input_shape(0).num_elements() != expected_input_elements {
            return Err(FaceError::ShapeMismatch {
                name: "yunet_input_elements",
                expected: expected_input_elements,
                got: model.input_shape(0).num_elements(),
            });
        }
        if model.output_count() != 12 {
            return Err(FaceError::ShapeMismatch {
                name: "yunet_signature_outputs",
                expected: 12,
                got: model.output_count(),
            });
        }

        Ok(Self { model, config })
    }

    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// `true` if the compiled model placed every op on the requested
    /// accelerator with no per-op CPU fallback.
    pub fn is_fully_accelerated(&self) -> Result<bool> {
        self.model.is_fully_accelerated()
    }

    /// Detects all faces above `config.score_threshold`, in original-image
    /// pixel coordinates, sorted by descending score.
    pub fn detect_all(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>> {
        let size = self.config.input_size;
        let (letterboxed, transform) = letterbox(image, size);
        let tensor = image_to_nchw(&letterboxed, &Normalization::raw_bgr());

        let outputs = self.model.run_f32(&[&tensor])?;
        if outputs.len() != 12 {
            return Err(FaceError::ShapeMismatch {
                name: "yunet_outputs",
                expected: 12,
                got: outputs.len(),
            });
        }

        let mut candidates: Vec<DetectedFace> = Vec::new();

        for (li, &s) in STRIDES.iter().enumerate() {
            let cls = &outputs[li];
            let obj = &outputs[3 + li];
            let bbox = &outputs[6 + li];
            let kps = &outputs[9 + li];

            let fw = size / s;
            let n = (fw * fw) as usize;
            if cls.len() != n || obj.len() != n || bbox.len() != n * 4 || kps.len() != n * 10 {
                return Err(FaceError::ShapeMismatch {
                    name: "yunet_stride_output",
                    expected: n,
                    got: cls.len(),
                });
            }

            for i in 0..n {
                let score = cls[i] * obj[i];
                if score <= self.config.score_threshold {
                    continue;
                }
                let px = (i as u32 % fw) as f32 * s as f32;
                let py = (i as u32 / fw) as f32 * s as f32;
                let cx = bbox[i * 4] * s as f32 + px;
                let cy = bbox[i * 4 + 1] * s as f32 + py;
                let w = bbox[i * 4 + 2].exp() * s as f32;
                let h = bbox[i * 4 + 3].exp() * s as f32;

                let mut pts = [[0f32; 2]; 5];
                for j in 0..5 {
                    pts[j] = [
                        kps[i * 10 + 2 * j] * s as f32 + px,
                        kps[i * 10 + 2 * j + 1] * s as f32 + py,
                    ];
                }

                let bbox_lb = BBox {
                    x1: cx - w / 2.0,
                    y1: cy - h / 2.0,
                    x2: cx + w / 2.0,
                    y2: cy + h / 2.0,
                };

                // Map bbox + landmarks from letterboxed space back to the
                // original image's pixel coordinates.
                let top_left = transform.to_original([bbox_lb.x1, bbox_lb.y1]);
                let bottom_right = transform.to_original([bbox_lb.x2, bbox_lb.y2]);
                let bbox_orig = BBox {
                    x1: top_left[0],
                    y1: top_left[1],
                    x2: bottom_right[0],
                    y2: bottom_right[1],
                };
                let lm_orig: Vec<[f32; 2]> = pts.iter().map(|&p| transform.to_original(p)).collect();

                candidates.push(DetectedFace {
                    bbox: bbox_orig,
                    score,
                    landmarks: Landmarks5 {
                        left_eye: lm_orig[0],
                        right_eye: lm_orig[1],
                        nose: lm_orig[2],
                        mouth_left: lm_orig[3],
                        mouth_right: lm_orig[4],
                    },
                });
            }
        }

        Ok(greedy_nms(candidates, self.config.nms_iou_threshold))
    }

    /// Convenience: highest-scoring face only, or `None` if none cleared the
    /// score threshold.
    pub fn detect_best(&mut self, image: &RgbImage) -> Result<Option<DetectedFace>> {
        let mut faces = self.detect_all(image)?;
        faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(faces.into_iter().next())
    }

    /// Strict variant for enrollment flows: succeeds only if exactly one
    /// face is present in the frame, so you can't accidentally register the
    /// wrong person out of a crowd shot.
    pub fn detect_exactly_one(&mut self, image: &RgbImage) -> Result<DetectedFace> {
        let faces = self.detect_all(image)?;
        match faces.len() {
            0 => Err(FaceError::NoFaceDetected),
            1 => Ok(faces.into_iter().next().unwrap()),
            n => Err(FaceError::AmbiguousFaceCount { count: n }),
        }
    }
}

fn greedy_nms(mut dets: Vec<DetectedFace>, iou_threshold: f32) -> Vec<DetectedFace> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<DetectedFace> = Vec::with_capacity(dets.len());
    for d in dets {
        if kept.iter().all(|k| d.bbox.iou(&k.bbox) < iou_threshold) {
            kept.push(d);
        }
    }
    kept
}
