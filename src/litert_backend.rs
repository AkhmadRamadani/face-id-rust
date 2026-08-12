//! Thin wrapper around the `litert` crate.
//!
//! `litert` 0.2.x is intentionally low-level: it hands you `Environment`,
//! `Model`, `CompiledModel`, and raw `TensorBuffer`s, but doesn't yet include
//! a `create_input_buffers()`/`create_output_buffers()` convenience (its own
//! docs mark that "coming in a later phase"). This module is that convenience
//! layer, isolated in one place so a future upgrade of `litert` only requires
//! touching this file.
//!
//! Two real constraints of the 0.2.x API shaped this code and are worth
//! knowing about if you touch it:
//!
//! 1. `CompiledModel::new(env, model, &options)` **consumes** `env` by value
//!    and there's no way to get it back. `Environment` also isn't `Clone`.
//!    So a second `Environment` is created purely to allocate `TensorBuffer`s
//!    with — this is the one part of this file that's slightly wasteful, and
//!    the reason each `LiteRtModel` you load carries two `Environment`
//!    instances instead of sharing one.
//! 2. Input/output buffers are allocated **once**, at load time, and reused
//!    for every `run_f32` call by locking/writing/unlocking. Allocating fresh
//!    `TensorBuffer`s per inference would be simpler to write but adds an
//!    alloc/free pair per frame on what is usually a per-frame hot path.

use std::path::{Path, PathBuf};

use litert::{
    Accelerators, CompilationOptions, CompiledModel, ElementType, Environment, Model, TensorBuffer,
    TensorShape,
};

use crate::error::{FaceError, Result};

pub struct LiteRtModel {
    // Kept alive so the managed-host `TensorBuffer`s in `input_buffers` /
    // `output_buffers` remain valid; never read after construction.
    _buffer_env: Environment,
    compiled: CompiledModel,
    input_buffers: Vec<TensorBuffer>,
    output_buffers: Vec<TensorBuffer>,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    path: PathBuf,
}

impl LiteRtModel {
    /// Loads a `.tflite` model and compiles it for `accelerators`.
    ///
    /// Pass `Accelerators::GPU` alone (not unioned with `CPU`) when you need
    /// a hard failure if the GPU delegate isn't available on this device,
    /// rather than a silent, possibly-slower-or-different-numerics fallback.
    /// That matters most for anti-spoofing: a silent fallback there is a
    /// silent change in fraud-detection behavior, which is worse than a
    /// loud startup error.
    pub fn load(path: impl AsRef<Path>, accelerators: Accelerators) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let buffer_env = Environment::new()?;
        let model = Model::from_file(&path)?;
        let sig = model.signature(0)?;

        let input_shapes: Vec<TensorShape> = (0..sig.input_count()?)
            .map(|i| sig.input_shape(i))
            .collect::<litert::Result<_>>()?;
        let output_shapes: Vec<TensorShape> = (0..sig.output_count()?)
            .map(|i| sig.output_shape(i))
            .collect::<litert::Result<_>>()?;

        for shape in input_shapes.iter().chain(output_shapes.iter()) {
            if shape.element_type != ElementType::Float32 {
                return Err(FaceError::UnsupportedElementType {
                    path: path.clone(),
                    actual: format!("{:?}", shape.element_type),
                });
            }
        }

        let sanitize_shape = |shape: &TensorShape| -> TensorShape {
            TensorShape {
                element_type: shape.element_type,
                dims: shape.dims.iter().map(|&d| if d <= 0 { 1 } else { d }).collect(),
            }
        };

        let input_buffers = input_shapes
            .iter()
            .map(|s| TensorBuffer::managed_host(&buffer_env, &sanitize_shape(s)))
            .collect::<litert::Result<Vec<_>>>()?;
        let output_buffers = output_shapes
            .iter()
            .map(|s| TensorBuffer::managed_host(&buffer_env, &sanitize_shape(s)))
            .collect::<litert::Result<Vec<_>>>()?;

        let compile_env = Environment::new()?;
        let options = CompilationOptions::new()?.with_accelerators(accelerators)?;
        let compiled = CompiledModel::new(compile_env, model, &options)?;

        Ok(Self {
            _buffer_env: buffer_env,
            compiled,
            input_buffers,
            output_buffers,
            input_shapes,
            output_shapes,
            path,
        })
    }

    /// `true` if every op landed on the requested accelerator (no silent
    /// per-op CPU fallback). Worth logging once at startup per model.
    pub fn is_fully_accelerated(&self) -> Result<bool> {
        Ok(self.compiled.is_fully_accelerated()?)
    }

    pub fn input_shape(&self, i: usize) -> &TensorShape {
        &self.input_shapes[i]
    }

    pub fn output_shape(&self, i: usize) -> &TensorShape {
        &self.output_shapes[i]
    }

    pub fn input_count(&self) -> usize {
        self.input_shapes.len()
    }

    pub fn output_count(&self) -> usize {
        self.output_shapes.len()
    }

    /// Writes `inputs` into the pre-allocated input buffers (each slice must
    /// exactly match that input's declared element count), runs inference,
    /// and returns each output buffer's contents as an owned `Vec<f32>`, in
    /// declaration order.
    pub fn run_f32(&mut self, inputs: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if inputs.len() != self.input_buffers.len() {
            return Err(FaceError::InputCountMismatch {
                path: self.path.clone(),
                declared: self.input_buffers.len(),
                given: inputs.len(),
            });
        }

        for (buf, data) in self.input_buffers.iter_mut().zip(inputs.iter()) {
            let mut guard = buf.lock_for_write::<f32>()?;
            if guard.len() != data.len() {
                return Err(FaceError::ShapeMismatch {
                    name: "input",
                    expected: guard.len(),
                    got: data.len(),
                });
            }
            guard.copy_from_slice(data);
        }

        self.compiled.run(&mut self.input_buffers, &mut self.output_buffers)?;

        let mut outputs = Vec::with_capacity(self.output_buffers.len());
        for buf in &self.output_buffers {
            let guard = buf.lock_for_read::<f32>()?;
            outputs.push(guard.to_vec());
        }
        Ok(outputs)
    }
}
