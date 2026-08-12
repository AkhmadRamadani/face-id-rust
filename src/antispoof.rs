//! Silent-Face-Anti-Spoofing (MiniFASNetV2) liveness check, running on the
//! LiteRT `CompiledModel` GPU delegate with no CPU fallback by default —
//! see the doc comment on [`LivenessDetector::load`] for why that default
//! is deliberate rather than an oversight.
//!
//! I/O contract:
//! - Input: `[1,3,80,80]` NCHW, BGR, `x/255` — a face crop ~2.7x the face
//!   box, centered.
//! - Output: `[1,3]` softmax; class 1 = live, classes 0 & 2 = spoof
//!   (print / replay). Live score = `output[1]`.

use image::RgbImage;
use litert::Accelerators;

use crate::error::{FaceError, Result};
use crate::litert_backend::LiteRtModel;
use crate::tensor_prep::{image_to_nchw, Normalization};
use crate::types::{BBox, LivenessResult};

#[derive(Debug, Clone, Copy)]
pub struct AntiSpoofConfig {
    pub input_size: u32,
    /// How much wider than the tight face box to crop before feeding the
    /// model — the published weights were trained on ~2.7x crops.
    pub crop_scale: f32,
    /// Minimum `output[1]` (live-class softmax probability) to accept as
    /// live. The model's own decision rule is `argmax == 1`, which is
    /// equivalent to a threshold of 1/3; most production e-KYC deployments
    /// want more margin than that, hence a separate, higher default here.
    pub live_threshold: f32,
}

impl Default for AntiSpoofConfig {
    fn default() -> Self {
        Self {
            input_size: 80,
            crop_scale: 2.7,
            live_threshold: 0.6,
        }
    }
}

pub struct LivenessDetector {
    model: LiteRtModel,
    config: AntiSpoofConfig,
}

impl LivenessDetector {
    /// Loads the MiniFASNetV2 `.tflite` export.
    ///
    /// Defaults to `Accelerators::GPU` alone (see the module's design note):
    /// anti-spoofing exists to catch fraud, so a silent fallback to a
    /// numerically-different CPU path is a worse failure mode than a loud
    /// startup error on unsupported hardware. If you need a fallback for
    /// device coverage, pass `Accelerators::GPU | Accelerators::CPU`
    /// explicitly and log `is_fully_accelerated()` so you can monitor how
    /// often it's actually taken.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        accelerators: Accelerators,
        config: AntiSpoofConfig,
    ) -> Result<Self> {
        let model = LiteRtModel::load(path, accelerators)?;
        if model.input_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "minifasnet_signature_inputs",
                expected: 1,
                got: model.input_count(),
            });
        }
        let expected_input_elements = 3 * (config.input_size as usize) * (config.input_size as usize);
        if model.input_shape(0).num_elements() != expected_input_elements {
            return Err(FaceError::ShapeMismatch {
                name: "minifasnet_input_elements",
                expected: expected_input_elements,
                got: model.input_shape(0).num_elements(),
            });
        }
        if model.output_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "minifasnet_signature_outputs",
                expected: 1,
                got: model.output_count(),
            });
        }
        Ok(Self { model, config })
    }

    pub fn is_fully_accelerated(&self) -> Result<bool> {
        self.model.is_fully_accelerated()
    }

    /// Runs the liveness check on `image` at `face_bbox` (a tight face box,
    /// e.g. straight from the detector — this function applies the 2.7x
    /// expansion itself).
    pub fn check(&mut self, image: &RgbImage, face_bbox: &BBox) -> Result<LivenessResult> {
        let expanded = face_bbox.scaled_about_center(self.config.crop_scale);
        let crop = crop_and_resize(image, &expanded, self.config.input_size);
        let tensor = image_to_nchw(&crop, &Normalization::zero_to_one_bgr());

        let outputs = self.model.run_f32(&[&tensor])?;
        let probs = &outputs[0];
        if probs.len() != 3 {
            return Err(FaceError::ShapeMismatch {
                name: "minifasnet_softmax",
                expected: 3,
                got: probs.len(),
            });
        }

        let live_score = probs[1];
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(LivenessResult {
            live_score,
            is_live: argmax == 1 && live_score >= self.config.live_threshold,
        })
    }

    /// Same as [`Self::check`] but returns [`FaceError::SpoofDetected`]
    /// instead of a `false` result — convenient when you want the pipeline
    /// to short-circuit on a spoof via `?` rather than branching on a bool.
    pub fn require_live(&mut self, image: &RgbImage, face_bbox: &BBox) -> Result<LivenessResult> {
        let result = self.check(image, face_bbox)?;
        if result.is_live {
            Ok(result)
        } else {
            Err(FaceError::SpoofDetected {
                live_score: result.live_score,
                threshold: self.config.live_threshold,
            })
        }
    }
}

fn crop_and_resize(image: &RgbImage, bbox: &BBox, out_size: u32) -> RgbImage {
    let (iw, ih) = image.dimensions();
    if iw == 0 || ih == 0 {
        // Degenerate input; nothing sensible to crop. Callers only ever
        // reach this with a real decoded photo, so this is unreachable in
        // practice — it's here purely so malformed input can't panic.
        return RgbImage::new(out_size, out_size);
    }
    let x1 = bbox.x1.round().clamp(0.0, (iw - 1) as f32) as u32;
    let y1 = bbox.y1.round().clamp(0.0, (ih - 1) as f32) as u32;
    let x2 = bbox.x2.round().clamp((x1 + 1) as f32, iw as f32) as u32;
    let y2 = bbox.y2.round().clamp((y1 + 1) as f32, ih as f32) as u32;
    let cropped = image::imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
    image::imageops::resize(&cropped, out_size, out_size, image::imageops::FilterType::Triangle)
}
