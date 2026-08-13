use std::path::PathBuf;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use faceid::{
    config::MemoryProfile, recognition::persistence, AntiSpoofConfig, EmbedderConfig,
    FacePipeline, LivenessDetector, VectorStore,
};
use faceid::server::{create_router, AppState};

#[derive(Parser, Debug)]
#[command(name = "faceid-server", author, version, about = "HTTP REST API Server for Face Recognition")]
struct Args {
    /// Host address to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Path to YuNet / BlazeFace detector .tflite model.
    #[arg(long, default_value = "models/yunet_fp16.tflite")]
    detector: PathBuf,

    /// Path to MediaPipe Face Landmarker .tflite model.
    #[arg(long)]
    landmarker: Option<PathBuf>,

    /// Path to MiniFASNetV2 anti-spoof .tflite model. Omit to disable liveness check.
    #[arg(long)]
    antispoof: Option<PathBuf>,

    /// Path to embedding model (.tflite or .onnx).
    #[arg(long, default_value = "models/facenet512.tflite")]
    embedder: PathBuf,

    /// Path to registry file (.jsonl format).
    #[arg(long, default_value = "registry.jsonl")]
    registry: PathBuf,

    /// Default minimum similarity threshold for recognition.
    #[arg(long, default_value_t = 0.45)]
    threshold: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!("Loading BlazeFaceDetector from {:?}", args.detector);
    let detector = faceid::BlazeFaceDetector::load(
        &args.detector,
        faceid::Accelerators::GPU | faceid::Accelerators::CPU,
        faceid::BlazeFaceConfig::default(),
    )?;

    let landmarker = if let Some(path) = &args.landmarker {
        tracing::info!("Loading FaceLandmarker from {:?}", path);
        Some(faceid::FaceLandmarker::load(
            path,
            faceid::Accelerators::GPU | faceid::Accelerators::CPU,
        )?)
    } else {
        tracing::info!("FaceLandmarker disabled (no --landmarker flag provided)");
        None
    };

    let antispoof = if let Some(path) = &args.antispoof {
        tracing::info!("Loading LivenessDetector from {:?}", path);
        Some(LivenessDetector::load(
            path,
            faceid::Accelerators::GPU | faceid::Accelerators::CPU,
            AntiSpoofConfig::default(),
        )?)
    } else {
        tracing::info!("Anti-spoofing disabled (no --antispoof flag provided)");
        None
    };

    tracing::info!("Loading Embedder from {:?}", args.embedder);
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
            Box::new(faceid::OrtEmbedder::load(&args.embedder, EmbedderConfig::default(), MemoryProfile::Balanced)?)
        }
        #[cfg(not(feature = "ort-runtime"))]
        {
            return Err("ort-runtime feature disabled".into());
        }
    };

    let store = if args.registry.exists() {
        tracing::info!("Loading existing registry from {:?}", args.registry);
        persistence::load(&args.registry)?
    } else {
        tracing::info!("Starting with fresh vector store, will save to {:?}", args.registry);
        VectorStore::new()
    };

    let pipeline = FacePipeline::new(detector, antispoof, embedder, args.threshold)
        .with_landmarker(landmarker)
        .with_store(store);
    let state = AppState::new(pipeline, args.registry.clone(), true);

    let app = create_router(state.clone());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("🚀 FaceID REST API Server listening on http://{}", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;

    Ok(())
}

async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, saving registry before exit...");
    if let Err(e) = state.save_registry().await {
        tracing::error!("Failed to save registry on shutdown: {}", e);
    } else {
        tracing::info!("Registry saved successfully.");
    }
}
