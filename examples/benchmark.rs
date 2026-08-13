//! Benchmark and resource tracking CLI for faceid library (Latency, Peak RSS RAM, CPU User/Sys Time).
//!
//! ```text
//! cargo run --release --example benchmark
//! ```

use std::path::Path;
use std::time::Instant;

use faceid::align::align_face;
use faceid::{
    load_image, Accelerators, AlignedFace, BlazeFaceConfig, BlazeFaceDetector, DetectorConfig,
    Embedder, EmbedderConfig, FaceDetector, FaceLandmarker,
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
    let init_stats = get_process_stats();
    println!("============================================================");
    println!("     FACEID BENCHMARK (SPEED, RAM FOOTPRINT & CPU USAGE)   ");
    println!("============================================================");
    println!("Initial Memory Baseline (Process RSS): {:.2} MB", init_stats.peak_rss_mb);

    let photo1_path = Path::new("test/1.jpg");
    let photo2_path = Path::new("test/2.jpg");
    let photo3_path = Path::new("test/3.JPG");

    let img1 = load_image(photo1_path)?;
    let img2 = load_image(photo2_path)?;
    let img3 = load_image(photo3_path)?;

    let yunet_path = Path::new("models/yunet_fp16.tflite");
    let blazeface_path = Path::new("models/blazeface.tflite");
    let blazeface_full_path = Path::new("models/blazeface_full_range.tflite");
    let landmarker_path = Path::new("models/face_landmark.tflite");
    let embedder_path = Path::new("models/facenet512.tflite");

    println!("\n--- 1. LATENCY, RAM & CPU BENCHMARK (20 Iterations) ---");
    let iters = 20;

    // A. YuNet Detector Benchmark
    println!("\nLoading YuNet Detector ({:?})...", yunet_path);
    let _s_before = get_process_stats();
    let mut yunet = FaceDetector::load(
        yunet_path,
        Accelerators::GPU | Accelerators::CPU,
        DetectorConfig::default(),
    )?;
    let s_after = get_process_stats();

    let _ = yunet.detect_all(&img1)?;
    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = yunet.detect_all(&img1)?;
    }
    let yunet_dur = start.elapsed();
    let c_end = get_process_stats();

    let yunet_latency = yunet_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let yunet_cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!(
        "  -> YuNet Detector: {:.2} ms/frame | CPU Time: {:.2} ms/frame | Peak RSS: {:.2} MB",
        yunet_latency,
        yunet_cpu_ms,
        s_after.peak_rss_mb
    );

    // B. BlazeFace Short-Range Benchmark
    println!("\nLoading BlazeFace Short-Range ({:?})...", blazeface_path);
    let _s_before = get_process_stats();
    let mut blazeface = BlazeFaceDetector::load(
        blazeface_path,
        Accelerators::GPU | Accelerators::CPU,
        BlazeFaceConfig::default(),
    )?;
    let s_after = get_process_stats();

    let _ = blazeface.detect_all(&img1)?;
    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = blazeface.detect_all(&img1)?;
    }
    let blazeface_dur = start.elapsed();
    let c_end = get_process_stats();

    let blazeface_latency = blazeface_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let blazeface_cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!(
        "  -> BlazeFace (Short-Range 128x128): {:.2} ms/frame | CPU Time: {:.2} ms/frame | Peak RSS: {:.2} MB",
        blazeface_latency,
        blazeface_cpu_ms,
        s_after.peak_rss_mb
    );

    // C. BlazeFace Full-Range Benchmark
    println!("\nLoading BlazeFace Full-Range ({:?})...", blazeface_full_path);
    let mut blazeface_full = BlazeFaceDetector::load(
        blazeface_full_path,
        Accelerators::GPU | Accelerators::CPU,
        BlazeFaceConfig::default(),
    )?;

    let _ = blazeface_full.detect_all(&img1)?;
    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = blazeface_full.detect_all(&img1)?;
    }
    let blazeface_full_dur = start.elapsed();
    let c_end = get_process_stats();

    let blazeface_full_latency = blazeface_full_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let blazeface_full_cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!(
        "  -> BlazeFace (Full-Range 192x192): {:.2} ms/frame | CPU Time: {:.2} ms/frame | Peak RSS: {:.2} MB",
        blazeface_full_latency,
        blazeface_full_cpu_ms,
        get_process_stats().peak_rss_mb
    );

    // D. MediaPipe Landmarker Benchmark
    println!("\nLoading MediaPipe Face Landmarker ({:?})...", landmarker_path);
    let mut landmarker = FaceLandmarker::load(
        landmarker_path,
        Accelerators::GPU | Accelerators::CPU,
    )?;

    let faces1 = yunet.detect_all(&img1)?;
    let face1 = &faces1[0];

    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = faceid::align::crop_mediapipe_two_pass_face(
            &img1,
            &face1.bbox,
            &face1.landmarks,
            &mut landmarker,
            0.05,
        )?;
    }
    let mp_dur = start.elapsed();
    let c_end = get_process_stats();

    let mp_latency = mp_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let mp_cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!(
        "  -> MediaPipe Face Mesh Landmarker: {:.2} ms/frame | CPU Time: {:.2} ms/frame | Peak RSS: {:.2} MB",
        mp_latency,
        mp_cpu_ms,
        c_end.peak_rss_mb
    );

    // E. FaceNet512 Embedder Benchmark
    println!("\nLoading FaceNet512 Embedder ({:?})...", embedder_path);
    let mut embedder = faceid::embedder::litert_embedder::LiteRtEmbedder::load(
        embedder_path,
        Accelerators::GPU | Accelerators::CPU,
        EmbedderConfig::default(),
    )?;

    let aligned_crop = align_face(&img1, &face1.landmarks, 160);
    let c_start = get_process_stats();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = embedder.embed(&aligned_crop)?;
    }
    let emb_dur = start.elapsed();
    let c_end = get_process_stats();

    let emb_latency = emb_dur.as_secs_f64() * 1000.0 / (iters as f64);
    let emb_cpu_ms = (c_end.cpu_user_ms + c_end.cpu_sys_ms - (c_start.cpu_user_ms + c_start.cpu_sys_ms)) / (iters as f64);

    println!(
        "  -> FaceNet512 Embedder: {:.2} ms/face | CPU Time: {:.2} ms/face | Peak RSS: {:.2} MB",
        emb_latency,
        emb_cpu_ms,
        c_end.peak_rss_mb
    );

    println!("\n============================================================");
    println!("              2. CROSS-VERIFICATION SIMILARITY MATRIX       ");
    println!("============================================================");

    let test_photos = [
        ("1.jpg", &img1),
        ("2.jpg", &img2),
        ("3.JPG", &img3),
    ];

    // Strategy 1: YuNet Detector + 5-point Canonical Alignment
    println!("\n[Strategy A] YuNet + 5-pt Canonical Umeyama Alignment");
    let mut embs_a = Vec::new();
    for (name, img) in &test_photos {
        let faces = yunet.detect_all(img)?;
        let face = &faces[0];
        let aligned = align_face(img, &face.landmarks, 160);
        let emb = embedder.embed(&aligned)?;
        embs_a.push((*name, emb));
    }
    print_matrix(&embs_a);

    // Strategy 2: YuNet Detector + MediaPipe 2-Pass Face Mesh Masking
    println!("\n[Strategy B] YuNet + MediaPipe 468 3D Mesh Mask Alignment");
    let mut embs_b = Vec::new();
    for (name, img) in &test_photos {
        let faces = yunet.detect_all(img)?;
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
            160,
            160,
            image::imageops::FilterType::Triangle,
        );
        let aligned = AlignedFace {
            rgb: resized_masked.into_raw(),
            width: 160,
            height: 160,
        };
        let emb = embedder.embed(&aligned)?;
        embs_b.push((*name, emb));
    }
    print_matrix(&embs_b);

    // Strategy 3: BlazeFace Detector + MediaPipe 2-Pass Face Mesh Masking
    println!("\n[Strategy C] BlazeFace + MediaPipe 468 3D Mesh Mask Alignment");
    let mut embs_c = Vec::new();
    for (name, img) in &test_photos {
        let faces = blazeface.detect_all(img)?;
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
            160,
            160,
            image::imageops::FilterType::Triangle,
        );
        let aligned = AlignedFace {
            rgb: resized_masked.into_raw(),
            width: 160,
            height: 160,
        };
        let emb = embedder.embed(&aligned)?;
        embs_c.push((*name, emb));
    }
    print_matrix(&embs_c);

    let final_stats = get_process_stats();

    println!("\n============================================================");
    println!("                     RESOURCE USAGE SUMMARY                 ");
    println!("============================================================");
    println!("Total Peak Resident Memory (RSS): {:.2} MB", final_stats.peak_rss_mb);
    println!("Total Process CPU User Time: {:.2} ms", final_stats.cpu_user_ms);
    println!("Total Process CPU Kernel/Sys Time: {:.2} ms", final_stats.cpu_sys_ms);
    println!("Full Pipeline Wall-Clock Latency (BlazeFace + Mesh + FaceNet512): {:.2} ms", blazeface_latency + mp_latency + emb_latency);
    println!("Full Pipeline Wall-Clock Latency (YuNet + Mesh + FaceNet512): {:.2} ms", yunet_latency + mp_latency + emb_latency);

    Ok(())
}

fn print_matrix(items: &[(&str, faceid::Embedding)]) {
    println!("{:>10} | {:>10} | {:>10} | {:>10}", "", items[0].0, items[1].0, items[2].0);
    println!("{:-<50}", "");
    for i in 0..items.len() {
        print!("{:>10} | ", items[i].0);
        for j in 0..items.len() {
            let sim = cosine_similarity(&items[i].1.0, &items[j].1.0);
            print!("{:>10.4} | ", sim);
        }
        println!();
    }
}
