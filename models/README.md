# Models

This crate doesn't bundle any model weights — point the loaders at your own
exports. What's expected, based on the specs this crate was built against:

## 1. Detector — YuNet (`FaceDetector`)

- Source: [ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection), converted to LiteRT.
- Expected I/O: `[1,3,640,640]` float32 NCHW, BGR, raw `0..255` in; 12 output
  tensors (`cls`, `obj`, `bbox`, `kps` x strides `{8,16,32}`) out. `FaceDetector::load`
  checks the signature declares exactly 12 outputs and rejects anything else
  at load time — if you re-export YuNet with a fused/different head, you'll
  need to adjust the decode logic in `src/detector.rs`, not just swap the file.
- License: BSD-3-Clause.

## 2. Anti-spoofing — MiniFASNetV2 (`LivenessDetector`)

- Source: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
  (the `2.7_80x80_MiniFASNetV2` checkpoint), converted to LiteRT.
- Expected I/O: `[1,3,80,80]` float32 NCHW, BGR, `x/255` in; `[1,3]` softmax
  out (index 1 = live).
- License: Apache-2.0.
- Optional at runtime: pass `None` instead of a `LivenessDetector` to
  `FacePipeline::new` to disable liveness checking entirely — nothing else
  in the pipeline depends on it being present.

## 3. Embedder (`OrtEmbedder` / `LiteRtEmbedder`)

Bring your own — this is deliberately not pinned to one architecture.
Anything that takes an aligned face crop and outputs a fixed-length float
vector works, as long as you set [`EmbedderConfig`] to match:

- `input_size`: auto-detected from the model's tensor signature by default (`None`), or set to `Some(size)` to enforce/validate a specific resolution.
- `normalization`: channel order + per-channel mean/std. `EmbedderConfig::default()`
  assumes the common ArcFace-style `(x/255 - 0.5)/0.5`, RGB — check your
  specific export.
- `EMBED_DIM` in `src/types.rs`: must match the model's output dimension
  (default 512). This is a compile-time constant (Kiddo needs it as a const
  generic), so changing it means a rebuild, not a config change.

Known-good options: ArcFace / MobileFaceNet / FaceLiVT ONNX exports,
InsightFace's `buffalo_l` or `antelopev2` recognition heads (exclude the
detector head — this crate's own `FaceDetector` replaces that part), or a
`.tflite` export of any of the above for `LiteRtEmbedder`.

## Alignment reference

`src/geometry.rs`'s `ARCFACE_REFERENCE_112` is the standard 5-point template
(left eye, right eye, nose, mouth-left, mouth-right) most public face
embedding models were trained against. If your embedder was trained with a
noticeably different alignment convention, recognition accuracy will suffer
even though nothing errors — this is the first thing to check if similarity
scores look worse than the model's published benchmarks.
