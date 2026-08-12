//! 1:1 verification between two photos — no registry involved.
//!
//! ```text
//! cargo run --example verify -- \
//!     --detector models/yunet_fp16.tflite \
//!     --embedder models/embedder.onnx \
//!     --photo-a a.jpg --photo-b b.jpg
//! ```

use clap::Parser;
use faceid::config::MemoryProfile;
use faceid::{load_image, DetectorConfig, EmbedderConfig, FaceDetector, FacePipeline, OrtEmbedder};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    detector: std::path::PathBuf,
    #[arg(long)]
    embedder: std::path::PathBuf,
    #[arg(long)]
    photo_a: std::path::PathBuf,
    #[arg(long)]
    photo_b: std::path::PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let detector = FaceDetector::load(&args.detector, faceid::Accelerators::GPU, DetectorConfig::default())?;
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
    // No anti-spoof detector here: verification between two provided photos
    // isn't a live-camera capture, so a liveness check doesn't apply.
    let mut pipeline = FacePipeline::new(detector, None, embedder, 0.45);

    let a = load_image(&args.photo_a)?;
    let b = load_image(&args.photo_b)?;
    let similarity = pipeline.verify_photos(&a, &b)?;
    println!("Cosine similarity: {similarity:.4}");
    Ok(())
}
