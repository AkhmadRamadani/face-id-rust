//! ONNX Runtime embedding backend, via `ort` 2.x.
//!
//! Works with essentially any face-embedding ONNX export (ArcFace,
//! MobileFaceNet, FaceLiVT, InsightFace's buffalo_l/antelopev2, etc.) — point
//! [`OrtEmbedder::load`] at the `.onnx` file and set [`EmbedderConfig`] to
//! match its expected input size/normalization.

use std::path::Path;

use ort::ep::CPU;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::config::MemoryProfile;
use crate::embedder::{Embedder, EmbedderConfig};
use crate::error::Result;
use crate::tensor_prep::{aligned_face_to_nchw, aligned_face_to_nhwc, TensorLayout};
use crate::types::{AlignedFace, Embedding};

pub struct OrtEmbedder {
    session: Session,
    config: EmbedderConfig,
    input_size: u32,
    layout: TensorLayout,
}

impl OrtEmbedder {
    pub fn load(path: impl AsRef<Path>, config: EmbedderConfig, memory_profile: MemoryProfile) -> Result<Self> {
        let session = build_session(path.as_ref(), memory_profile)?;
        let mut layout = TensorLayout::Nchw;
        let input_size = match config.input_size {
            Some(size) => size,
            None => {
                let mut inferred = None;
                if let Some(input) = session.inputs().first() {
                    if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                        let dims = shape.as_ref();
                        if dims.len() == 4 {
                            if dims[1] == 3 && dims[2] > 0 {
                                inferred = Some(dims[2] as u32);
                                layout = TensorLayout::Nchw;
                            } else if dims[3] == 3 && dims[1] > 0 {
                                inferred = Some(dims[1] as u32);
                                layout = TensorLayout::Nhwc;
                            }
                        }
                    }
                }
                inferred.unwrap_or(112)
            }
        };
        Ok(Self {
            session,
            config,
            input_size,
            layout,
        })
    }
}

fn build_session(path: &Path, profile: MemoryProfile) -> ort::Result<Session> {
    let cpu_ep = CPU::default().with_arena_allocator(profile.use_arena()).build();
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(profile.intra_threads())?
        .with_memory_pattern(profile.use_memory_pattern())?
        .with_execution_providers([cpu_ep])?
        .commit_from_file(path)
}

impl Embedder for OrtEmbedder {
    fn input_size(&self) -> u32 {
        self.input_size
    }

    fn embed(&mut self, face: &AlignedFace) -> Result<Embedding> {
        let (data, shape) = match self.layout {
            TensorLayout::Nchw => (
                aligned_face_to_nchw(face, &self.config.normalization),
                vec![1i64, 3, face.height as i64, face.width as i64],
            ),
            TensorLayout::Nhwc => (
                aligned_face_to_nhwc(face, &self.config.normalization),
                vec![1i64, face.height as i64, face.width as i64, 3],
            ),
        };
        let input = Tensor::from_array((shape, data))?;

        let outputs = self.session.run(ort::inputs![input])?;
        let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        Embedding::from_raw(raw)
    }
}
