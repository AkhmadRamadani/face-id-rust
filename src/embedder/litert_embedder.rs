//! LiteRT (`.tflite`) embedding backend — the "litert" side of the
//! "onnx/litert embedding" requirement. Use this when your embedding model
//! is exported to `.tflite` instead of ONNX (e.g. converted the same way as
//! the detector/anti-spoof models, via `litert-torch`).

use litert::Accelerators;

use crate::embedder::{Embedder, EmbedderConfig};
use crate::error::{FaceError, Result};
use crate::litert_backend::LiteRtModel;
use crate::tensor_prep::{aligned_face_to_nchw, aligned_face_to_nhwc, TensorLayout};
use crate::types::{AlignedFace, Embedding};

pub struct LiteRtEmbedder {
    model: LiteRtModel,
    config: EmbedderConfig,
    input_size: u32,
    layout: TensorLayout,
}

impl LiteRtEmbedder {
    /// Loads the embedding `.tflite` export.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        accelerators: Accelerators,
        config: EmbedderConfig,
    ) -> Result<Self> {
        let model = LiteRtModel::load(path, accelerators)?;
        if model.input_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "litert_embedder_signature_inputs",
                expected: 1,
                got: model.input_count(),
            });
        }
        let input_shape = model.input_shape(0);
        let layout = if input_shape.dims.len() == 4 && *input_shape.dims.last().unwrap_or(&0) == 3 {
            TensorLayout::Nhwc
        } else {
            TensorLayout::Nchw
        };

        let actual_elements = input_shape
            .dims
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| d as usize)
            .product::<usize>();
        let input_size = match config.input_size {
            Some(expected_size) => {
                let expected_input_elements = 3 * (expected_size as usize) * (expected_size as usize);
                if actual_elements != expected_input_elements {
                    return Err(FaceError::ShapeMismatch {
                        name: "litert_embedder_input_elements",
                        expected: expected_input_elements,
                        got: actual_elements,
                    });
                }
                expected_size
            }
            None => {
                if actual_elements % 3 != 0 {
                    return Err(FaceError::ShapeMismatch {
                        name: "litert_embedder_input_elements",
                        expected: 3,
                        got: actual_elements,
                    });
                }
                let spatial = actual_elements / 3;
                let size = (spatial as f64).sqrt() as u32;
                if (size as usize) * (size as usize) != spatial {
                    return Err(FaceError::ShapeMismatch {
                        name: "litert_embedder_input_elements_square",
                        expected: (size as usize) * (size as usize) * 3,
                        got: actual_elements,
                    });
                }
                size
            }
        };

        if model.output_count() != 1 {
            return Err(FaceError::ShapeMismatch {
                name: "embedder_signature_outputs",
                expected: 1,
                got: model.output_count(),
            });
        }
        let out_elements = model
            .output_shape(0)
            .dims
            .iter()
            .filter(|&&d| d > 0)
            .product::<i32>() as usize;

        if out_elements != crate::types::EMBED_DIM {
            return Err(FaceError::InvalidEmbeddingDim {
                expected: crate::types::EMBED_DIM,
                got: out_elements,
            });
        }
        Ok(Self {
            model,
            config,
            input_size,
            layout,
        })
    }

    pub fn is_fully_accelerated(&self) -> Result<bool> {
        self.model.is_fully_accelerated()
    }
}

impl Embedder for LiteRtEmbedder {
    fn input_size(&self) -> u32 {
        self.input_size
    }

    fn embed(&mut self, face: &AlignedFace) -> Result<Embedding> {
        let data = match self.layout {
            TensorLayout::Nchw => aligned_face_to_nchw(face, &self.config.normalization),
            TensorLayout::Nhwc => aligned_face_to_nhwc(face, &self.config.normalization),
        };
        let outputs = self.model.run_f32(&[&data])?;
        Embedding::from_raw(&outputs[0])
    }
}
