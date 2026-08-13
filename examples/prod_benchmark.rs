//! Production Memory & Performance Benchmark for BlazeFace + MediaPipe Mesh + FaceNet512.
//!
//! ```text
//! cargo run --release --example prod_benchmark
//! ```

use std::path::Path;
use std::time::Instant;

use faceid::{
    load_image, Accelerators, AlignedFace, BlazeFaceConfig, BlazeFaceDetector, Embedder, EmbedderConfig, FaceLandmarker,
};

#[derive(Debug, Clone, Copy)]
struct ProcessStats {
    peak_rss_mb: f64,
    cpu_user_ms: f64,
    cpu_sys_ms: f64,
}

fn get_process_stats() -> ProcessStats {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe {
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
    }
    let rss_bytes = if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64
    } else {
        (usage.ru_maxrss as f64) * 1024.0
    };

    let user_ms = (usage.ru_utime.tv_sec as f64) * 1000.0 + (usage.ru_utime.tv_usec as f64) / 1000.0;
    let sys_ms = (usage.ru_stime.tv_sec as f64) * 1000.0 + (usage.ru_stime.tv_usec as f64) / 1000.0;

    ProcessStats {
        peak_rss_mb: rss_bytes / (1024.0 * 1024.0),
        cpu_user_ms: user_ms,
        cpu_sys_ms: sys_ms,
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s0 = get_process_stats();
    println!("============================================================");
    println!("     PRODUCTION PIPELINE BENCHMARK (MEMORY & LATENCY)       ");
    println!("============================================================");
    println!("1. Process Memory Baseline: {:.2} MB RSS", s0.peak_rss_mb);

    let photo1_path = Path::new("test/1.jpg");
    let photo2_path = Path::new("test/2.jpg");
    let photo3_path = Path::new("test/3.JPG");

    let img1 = load_image(photo1_path)?;
    let img2 = load_image(photo2_path)?;
    let img3 = load_image(photo3_path)?;

    let blazeface_path = Path::new("models/blazeface.tflite");
    let landmarker_path = Path::new("models/face_landmark.tflite");
    let embedder_path = Path::new("models/mobilefacenet.tflite");

    // Stage 1: Load BlazeFace Detector
    println!("\nLoading BlazeFace Detector ({:?})...", blazeface_path);
    let mut detector = BlazeFaceDetector::load(
        blazeface_path,
        Accelerators::CPU,
        BlazeFaceConfig::default(),
    )?;
    let s1 = get_process_stats();
    println!("  -> Memory after loading BlazeFace: {:.2} MB RSS (+{:.2} MB)", s1.peak_rss_mb, s1.peak_rss_mb - s0.peak_rss_mb);

    // Stage 2: Load MediaPipe Face Mesh Landmarker
    println!("\nLoading MediaPipe Face Mesh Landmarker ({:?})...", landmarker_path);
    let mut landmarker = FaceLandmarker::load(
        landmarker_path,
        Accelerators::CPU,
    )?;
    let s2 = get_process_stats();
    println!("  -> Memory after loading MediaPipe Mesh: {:.2} MB RSS (+{:.2} MB)", s2.peak_rss_mb, s2.peak_rss_mb - s1.peak_rss_mb);

    // Stage 3: Load MobileFaceNet Embedder
    println!("\nLoading MobileFaceNet Embedder ({:?})...", embedder_path);
    let mut embedder = faceid::embedder::litert_embedder::LiteRtEmbedder::load(
        embedder_path,
        Accelerators::CPU,
        EmbedderConfig::default(),
    )?;
    let s3 = get_process_stats();
    println!("  -> Memory after loading MobileFaceNet Embedder: {:.2} MB RSS (+{:.2} MB)", s3.peak_rss_mb, s3.peak_rss_mb - s2.peak_rss_mb);

    println!("\n------------------------------------------------------------");
    println!("2. PIPELINE INFERENCE BENCHMARK (50 Iterations)");
    println!("------------------------------------------------------------");

    let iters = 50;

    // Warmup
    let faces = detector.detect_all(&img1)?;
    let face = &faces[0];
    let masked = faceid::align::crop_mediapipe_two_pass_face(
        &img1,
        &face.bbox,
        &face.landmarks,
        &mut landmarker,
        0.05,
    )?;
    let resized_masked = image::imageops::resize(
        &masked,
        112,
        112,
        image::imageops::FilterType::Triangle,
    );
    let aligned = AlignedFace {
        rgb: resized_masked.into_raw(),
        width: 112,
        height: 112,
    };
    let _ = embedder.embed(&aligned)?;

    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let faces = detector.detect_all(&img1)?;
        let face = &faces[0];
        let masked = faceid::align::crop_mediapipe_two_pass_face(
            &img1,
            &face.bbox,
            &face.landmarks,
            &mut landmarker,
            0.05,
        )?;
        let resized_masked = image::imageops::resize(
            &masked,
            112,
            112,
            image::imageops::FilterType::Triangle,
        );
        let aligned = AlignedFace {
            rgb: resized_masked.into_raw(),
            width: 112,
            height: 112,
        };
        let _ = embedder.embed(&aligned)?;
    }
    let total_dur = start.elapsed();
    let c_end = get_process_stats();

    let full_latency = total_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!("\n------------------------------------------------------------");
    println!("3. CROSS-VERIFICATION MATRIX");
    println!("------------------------------------------------------------");

    let test_photos = [
        ("1.jpg", &img1),
        ("2.jpg", &img2),
        ("3.JPG", &img3),
    ];

    let mut embs = Vec::new();
    for (name, img) in &test_photos {
        let faces = detector.detect_all(img)?;
        let face = &faces[0];
        let masked = faceid::align::crop_mediapipe_two_pass_face(
            img,
            &face.bbox,
            &face.landmarks,
            &mut landmarker,
            0.05,
        )?;
        let resized_masked = image::imageops::resize(
            &masked,
            112,
            112,
            image::imageops::FilterType::Triangle,
        );
        let aligned = AlignedFace {
            rgb: resized_masked.into_raw(),
            width: 112,
            height: 112,
        };
        let emb = embedder.embed(&aligned)?;
        embs.push((*name, emb));
    }

    println!("{:>10} | {:>10} | {:>10} | {:>10}", "", embs[0].0, embs[1].0, embs[2].0);
    println!("{:-<50}", "");
    for i in 0..embs.len() {
        print!("{:>10} | ", embs[i].0);
        for j in 0..embs.len() {
            let sim = cosine_similarity(&embs[i].1.0, &embs[j].1.0);
            print!("{:>10.4} | ", sim);
        }
        println!();
    }

    let final_stats = get_process_stats();

    println!("\n============================================================");
    println!("               PRODUCTION BENCHMARK VERDICT                 ");
    println!("============================================================");
    println!("  -> Peak Resident RAM (RSS): {:.2} MB", final_stats.peak_rss_mb);
    println!("  -> Full Pipeline Wall Latency: {:.2} ms / frame ({:.1} FPS)", full_latency, 1000.0 / full_latency);
    println!("  -> CPU Time / Frame: {:.2} ms", cpu_ms);
    println!("============================================================");

    Ok(())
}
