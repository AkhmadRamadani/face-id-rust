//! Registers a face from a photo into the registry, globally or under a
//! specific event.
//!
//! ```text
//! cargo run --example enroll -- \
//!     --detector models/yunet_fp16.tflite \
//!     --antispoof models/silentface.tflite \
//!     --embedder models/embedder.onnx \
//!     --registry registry.json \
//!     --person alice \
//!     --photo alice.jpg \
//!     --scope global
//! ```

use clap::Parser;
use faceid::{
    load_image, recognition::persistence, AntiSpoofConfig, DetectorConfig, EmbedderConfig, FaceDetector,
    FacePipeline, LivenessDetector, OrtEmbedder, PersonId, RegistrationScope, VectorStore,
};
use faceid::config::MemoryProfile;

#[derive(Parser)]
struct Args {
    /// Path to the YuNet detector .tflite export.
    #[arg(long)]
    detector: std::path::PathBuf,
    /// Path to the MiniFASNetV2 anti-spoof .tflite export. Omit to enroll
    /// without a liveness check (e.g. when enrolling from a known-good ID
    /// photo rather than a live camera capture).
    #[arg(long)]
    antispoof: Option<std::path::PathBuf>,
    /// Path to the embedding .onnx export.
    #[arg(long)]
    embedder: std::path::PathBuf,
    /// Registry snapshot to load (if present) and save back to.
    #[arg(long, default_value = "registry.json")]
    registry: std::path::PathBuf,
    /// Stable identifier for the person being enrolled.
    #[arg(long)]
    person: String,
    /// Photo containing exactly one face.
    #[arg(long)]
    photo: std::path::PathBuf,
    /// "global", or an event id such as "expo-2026".
    #[arg(long, default_value = "global")]
    scope: String,
    /// Optional free-form note (e.g. a display name).
    #[arg(long)]
    label: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let detector = FaceDetector::load(&args.detector, faceid::Accelerators::GPU, DetectorConfig::default())?;
    let antispoof = args
        .antispoof
        .as_ref()
        .map(|path| LivenessDetector::load(path, faceid::Accelerators::GPU, AntiSpoofConfig::default()))
        .transpose()?;
    let embedder: Box<dyn faceid::Embedder> = if args.embedder.extension().and_then(|s| s.to_str()) == Some("tflite") {
        #[cfg(feature = "litert-runtime")]
        {
            Box::new(faceid::embedder::litert_embedder::LiteRtEmbedder::load(
                &args.embedder,
                faceid::Accelerators::GPU | faceid::Accelerators::CPU,
                EmbedderConfig::default(),
            )?)
        }
        #[cfg(not(feature = "litert-runtime"))]
        {
            return Err("litert-runtime feature disabled".into());
        }
    } else {
        #[cfg(feature = "ort-runtime")]
        {
            Box::new(OrtEmbedder::load(&args.embedder, EmbedderConfig::default(), MemoryProfile::Balanced)?)
        }
        #[cfg(not(feature = "ort-runtime"))]
        {
            return Err("ort-runtime feature disabled".into());
        }
    };

    let store = if args.registry.exists() {
        persistence::load(&args.registry)?
    } else {
        VectorStore::new()
    };

    let mut pipeline = FacePipeline::new(detector, antispoof, embedder, 0.45).with_store(store);

    let scope = if args.scope == "global" {
        RegistrationScope::Global
    } else {
        RegistrationScope::Event(args.scope.as_str().into())
    };

    let image = load_image(&args.photo)?;
    let record_id = pipeline.enroll(&image, PersonId::from(args.person.as_str()), scope, args.label)?;

    persistence::save(pipeline.store(), &args.registry)?;
    println!(
        "Registered '{}' as record #{record_id} ({} total registrations in the store)",
        args.person,
        pipeline.store().len()
    );
    Ok(())
}
