# faceid

On-device face recognition in Rust: [LiteRT](https://crates.io/crates/litert)
GPU inference for detection (YuNet) and anti-spoofing (MiniFASNetV2), a
pluggable [ort](https://ort.pyke.io)/LiteRT embedding backend, and
[Kiddo](https://crates.io/crates/kiddo)-backed recognition scoped globally or
per-event.

```text
image -> detect (YuNet)        -> DetectedFace { bbox, landmarks, score }
      -> liveness (optional)   -> reject presentation attacks
      -> align (5-pt warp)     -> AlignedFace (canonical pose, fixed size)
      -> embed (ORT or LiteRT) -> Embedding ([f32; EMBED_DIM], L2-normalized)
      -> register / identify  -> VectorStore (global + per-event Kiddo trees)
```

## Quickstart

```rust
use faceid::{
    AntiSpoofConfig, Accelerators, DetectorConfig, EmbedderConfig, FaceDetector,
    FacePipeline, LivenessDetector, OrtEmbedder, PersonId, RecognitionContext,
    RegistrationScope, load_image,
};
use faceid::config::MemoryProfile;

let detector = FaceDetector::load("models/yunet_fp16.tflite", Accelerators::GPU, DetectorConfig::default())?;
let antispoof = Some(LivenessDetector::load("models/silentface.tflite", Accelerators::GPU, AntiSpoofConfig::default())?);
let embedder = OrtEmbedder::load("models/embedder.onnx", EmbedderConfig::default(), MemoryProfile::Balanced)?;

let mut pipeline = FacePipeline::new(detector, antispoof, embedder, /* similarity_threshold */ 0.45);

// Enroll, scoped to one event only.
let photo = load_image("alice.jpg")?;
pipeline.enroll(&photo, PersonId::from("alice"), RegistrationScope::Event("expo-2026".into()), None)?;

// Recognized while checking in at that same event...
let probe = load_image("camera_frame.jpg")?;
let ctx = RecognitionContext::Event("expo-2026".into());
if let Some(m) = pipeline.identify(&probe, &ctx)? {
    println!("welcome back, {} (similarity {:.2})", m.person_id, m.similarity);
}

// ...but invisible under a different event's context, and invisible in
// GlobalOnly mode too, because she was never registered globally.
assert!(pipeline.identify(&probe, &RecognitionContext::GlobalOnly)?.is_none());
```

See `examples/enroll.rs`, `examples/identify.rs`, `examples/verify.rs` for
complete CLI programs, and `models/README.md` for where to get/convert the
three model files.

## HTTP REST API Server (`faceid-server`)

To run as an HTTP REST API server:

```bash
cargo run --bin faceid-server -- \
    --host 127.0.0.1 \
    --port 8080 \
    --detector models/yunet_fp16.tflite \
    --antispoof models/silentface.tflite \
    --embedder models/facenet512.tflite \
    --registry registry.jsonl
```

### Swagger UI & Interactive Docs
- `GET  /swagger-ui` - Interactive OpenAPI Swagger UI documentation
- `GET  /health` - Server & GPU acceleration status
- `POST /api/v1/enroll` - Enroll face (`photo`, `person_id`, `scope`, `label`, `antispoof`)
- `POST /api/v1/identify` - Search face match (`photo`, `event`, `threshold`, `antispoof`)
- `POST /api/v1/verify` - 1:1 photo similarity (`photo_a`, `photo_b`, `antispoof`)
- `GET  /api/v1/registry/stats` - Total registered faces & active scopes
- `DELETE /api/v1/registry/records/:id` - Unregister face record by ID

### Production Docker Deployment

Run with Docker Compose:

```bash
docker compose up -d
```

Or build and run manually with Docker:

```bash
docker build -t faceid-server:latest .

docker run -d \
  --name faceid-server \
  -p 8080:8080 \
  -v faceid-data:/app/data \
  faceid-server:latest
```

### GitHub Actions CI/CD Pipeline

Automated workflow defined in [.github/workflows/ci.yml](file:///Users/akhmadramadani/Downloads/faceid/.github/workflows/ci.yml):
- **Lint & Test**: Runs `cargo check`, `cargo clippy`, and unit tests automatically on every `push` and `pull_request`.
- **Multi-Arch Docker Build**: Builds and pushes `linux/amd64` and `linux/arm64` container images to **GitHub Container Registry (GHCR)** on push to `main` or release tags.

## Global vs. per-event scoping

- `RegistrationScope::Global` — recognized under *any* `RecognitionContext`.
- `RegistrationScope::Event(id)` — recognized only when the recognizer is
  given `RecognitionContext::Event(id)` for that same event.

Recognizing under an event context searches the global registry *and* that
event's registry together, and returns whichever candidate is closer — so a
globally-registered person is always found, and someone enrolled only for
event A stays invisible while recognizing under event B or under
`GlobalOnly`. There's deliberately no "search every event" mode: cross-event
visibility is something you opt a person into (by registering them globally),
not something a query can request.

Implementation: one `kiddo::MutableKdTree` for the global registry, plus one
per event, in a `HashMap<EventId, MutableKdTree<..>>`. A person can hold
multiple registrations (different scopes, or multiple enrollment photos in
the same scope); each gets its own `RecordId` and can be removed
independently.

## Anti-spoofing is optional, at runtime

`FacePipeline::new` takes `Option<LivenessDetector>`. `None` disables the
liveness check entirely — nothing else in the pipeline depends on it. This is
a runtime toggle (not a Cargo feature) so you can, e.g., run enrollment
against known-good ID photos without a liveness check while still requiring
one for live-camera recognition, using the same pipeline type.

When a `LivenessDetector` *is* present, it defaults to `Accelerators::GPU`
alone (no CPU fallback) — see the design note on `LivenessDetector::load` for
why that's deliberate: a silent fallback to a numerically-different CPU path
is a change in fraud-detection behavior you'd want to know about, not paper
over.

## Memory-usage notes

- Embeddings are `[f32; EMBED_DIM]`, stored inline (no heap indirection per
  embedding, no `Vec<f32>` allocation churn), and compared with
  squared-Euclidean distance on L2-normalized vectors instead of a custom
  cosine metric — `||a-b||² = 2 - 2·cos(a,b)` for unit vectors, so plain
  `SquaredEuclidean` gives cosine-similarity ranking for free.
- No `ndarray` dependency anywhere — tensors are built and read as flat
  `Vec<f32>` + shape tuples on both the `ort` and `litert` sides. `ort` is
  also built with `default-features = false` (drops `ndarray`, `tls-native`).
- `image` is built with only `jpeg`/`png` decoders enabled.
- `config::MemoryProfile::LowMemory` trades ORT's memory arena, memory
  pattern optimization, and thread count for a smaller, more predictable
  resident footprint, at some latency cost — see `src/config.rs`.
- The one real tradeoff worth knowing about at large registry sizes: each
  embedding currently lives both inside its Kiddo tree (which needs the raw
  point to support `remove`) and in the registry's record table (needed to
  answer `remove` and to rebuild trees on load) — effectively 2x the raw
  embedding data. Fine into the tens of thousands of registrations; see the
  module docs on `src/recognition/store.rs` for the mitigation path
  (freezing settled scopes into `ImmutableKdTree` + rkyv) if you're going
  well past that.

## Honesty about what's been verified here

Every `litert`/`ort`/`kiddo` API call in this crate was checked against the
actual downloaded crate source (not docs, not memory) — `litert` in
particular is a young crate (0.2.x) with API surface not yet reflected
everywhere it should be in its own docs, so this mattered.

This sandbox only had rustc 1.75 available (`kiddo` needs 1.89, `ort` is
edition 2024 / rustc 1.88+), so a real `cargo build` against the actual
crates wasn't possible here. Two levels of verification stood in for that:

1. **Real compile + test**, no stand-ins: the dependency-light modules
   (`types.rs`, `geometry.rs`, `tensor_prep.rs`, `config.rs`, `align.rs` —
   including the Umeyama 5-point alignment math and its unit tests) only
   depend on `image`/`serde`/`thiserror`, all rustc-1.75-compatible, so
   these were compiled and tested directly.
2. **Full-crate type-check against hand-written stub crates**: for
   `litert`/`ort`/`kiddo`, I wrote minimal local crates exposing the exact
   signatures I'd verified from the real source (bodies are
   `unimplemented!()` — they don't run, they only need to type-check), and
   pointed a copy of this crate at them via `[patch.crates-io]`. `cargo
   check --lib --examples` and `cargo clippy --lib --examples` both came back
   clean against every module, including `detector.rs`, `antispoof.rs`,
   `recognition/store.rs`, `pipeline.rs`, and all three example CLIs.

That's meaningfully stronger than a manual read-through — it validates real
type/borrow-checker correctness, not just "this looks right" — but it's
**not the same as building against the real crates**, since a stub can only
be as correct as my own reading of the source it was copied from. Two real,
non-cosmetic bugs surfaced during this process and are already fixed in this
version:

1. `serde`'s derive only implements `Serialize`/`Deserialize` for arrays up
   to length 32 — not arbitrary const-generic lengths. `Embedding`'s
   `[f32; 512]` needed `serde-big-array`'s `BigArray` helper instead of a
   bare derive.
2. `pipeline.rs` unconditionally imported `FaceDetector`/`LivenessDetector`,
   which are only compiled under the `litert-runtime` feature — a build with
   `--no-default-features --features ort-runtime` would have failed. Fixed
   by gating the `pipeline` module the same way.

**Still worth running `cargo check` yourself** with a current toolchain
before relying on this in production — this response's confidence is
"passed a rigorous type-check against a faithfully-reproduced API surface,"
not "built and ran against the real dependencies."

## Build notes

- `litert` and `ort`'s build scripts download prebuilt native libraries at
  build time — you'll need network access to their respective CDNs the first
  time you build (not needed again after that, once cached).
- Features: `litert-runtime` (detector, anti-spoof, LiteRT embedder backend)
  and `ort-runtime` (ONNX embedder backend) are both on by default;
  `cargo build --no-default-features --features ort-runtime` gets you a
  minimal build with just `Embedder`/`OrtEmbedder` (no detector — you'd
  drive your own).
- `EMBED_DIM` (`src/types.rs`) and the model file paths are the only things
  you should need to change to point this at a different set of models.
