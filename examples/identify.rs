//! Identifies the best face in a photo against the registry, optionally
//! scoped to an event.
//!
//! ```text
//! cargo run --example identify -- \
//!     --detector models/yunet_fp16.tflite \
//!     --antispoof models/silentface.tflite \
//!     --embedder models/embedder.onnx \
//!     --registry registry.json \
//!     --photo unknown.jpg \
//!     --event expo-2026
//! ```

use clap::Parser;
use faceid::config::MemoryProfile;
use faceid::{
    load_image, recognition::persistence, AntiSpoofConfig, DetectorConfig, EmbedderConfig, FaceDetector,
    FacePipeline, LivenessDetector, OrtEmbedder, RecognitionContext,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    detector: std::path::PathBuf,
    #[arg(long)]
    antispoof: Option<std::path::PathBuf>,
    #[arg(long)]
    embedder: std::path::PathBuf,
    #[arg(long, default_value = "registry.json")]
    registry: std::path::PathBuf,
    #[arg(long)]
    photo: std::path::PathBuf,
    /// Also search this event's registry alongside the global registry.
    /// Omit to search only the global registry.
    #[arg(long)]
    event: Option<String>,
    /// Minimum cosine similarity to accept a match.
    #[arg(long, default_value_t = 0.45)]
    threshold: f32,
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

    let store = persistence::load(&args.registry)?;
    let mut pipeline = FacePipeline::new(detector, antispoof, embedder, args.threshold).with_store(store);

    let context = match args.event {
        Some(event) => RecognitionContext::Event(event.as_str().into()),
        None => RecognitionContext::GlobalOnly,
    };

    let image = load_image(&args.photo)?;
    match pipeline.identify(&image, &context)? {
        Some(m) => println!(
            "Matched '{}' (record #{}, {} scope, similarity {:.3})",
            m.person_id, m.record_id, m.origin, m.similarity
        ),
        None => println!("No match."),
    }
    Ok(())
}
