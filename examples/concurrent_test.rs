//! Concurrent stress test for PipelinePool with MobileFaceNet.
//!
//! ```text
//! cargo run --release --example concurrent_test
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use faceid::{load_image, PipelinePool, RecognitionContext, RegistrationScope};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("        CONCURRENT PIPELINE POOL STRESS TEST                ");
    println!("============================================================");

    let blazeface_path = Path::new("models/blazeface.tflite");
    let landmarker_path = Path::new("models/face_landmark.tflite");
    let mobilefacenet_path = Path::new("models/mobilefacenet.tflite");

    let worker_count = 4;
    println!("Initializing PipelinePool with {} concurrent MobileFaceNet workers...", worker_count);
    let pool = PipelinePool::new_mobilefacenet(
        worker_count,
        blazeface_path,
        landmarker_path,
        mobilefacenet_path,
        0.40,
    )?;

    let pool = Arc::new(pool);

    let load_or_create = |path: &str| -> faceid::Result<image::RgbImage> {
        match load_image(path) {
            Ok(img) => Ok(img),
            Err(_) => {
                let mut img = image::RgbImage::new(256, 256);
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    *pixel = image::Rgb([30, 30, 30]);
                    let dx = (x as f32 - 128.0) / 64.0;
                    let dy = (y as f32 - 128.0) / 80.0;
                    if dx * dx + dy * dy <= 1.0 {
                        *pixel = image::Rgb([220, 180, 140]);
                    }
                }
                Ok(img)
            }
        }
    };

    let img1 = load_or_create("test/1.jpg")?;
    let img2 = load_or_create("test/2.jpg")?;
    let img3 = load_or_create("test/3.JPG")?;

    println!("Enrolling initial test subject...");
    let rec_id = pool.enroll(img1.clone(), "person_001", RegistrationScope::Global, Some("Alex".into())).await?;
    println!("  -> Successfully enrolled person_001 with RecordId #{}", rec_id);

    println!("\nSpawning 20 concurrent identification & verification tasks...");
    let start_time = Instant::now();

    let mut handles = Vec::new();
    for i in 0..20 {
        let pool_c = Arc::clone(&pool);
        let img = if i % 3 == 0 {
            img1.clone()
        } else if i % 3 == 1 {
            img2.clone()
        } else {
            img3.clone()
        };

        if i % 2 == 0 {
            // Identify task
            handles.push(tokio::spawn(async move {
                let context = RecognitionContext::default();
                let res = pool_c.identify(img, &context).await;
                (i, "identify", res.is_ok())
            }));
        } else {
            // Verify task
            let img_b = img2.clone();
            handles.push(tokio::spawn(async move {
                let res = pool_c.verify_photos(img, img_b).await;
                (i, "verify", res.is_ok())
            }));
        }
    }

    let mut success_count = 0;
    for h in handles {
        let (id, kind, ok) = h.await?;
        if ok {
            success_count += 1;
        } else {
            eprintln!("Task {} ({}) failed!", id, kind);
        }
    }

    let elapsed = start_time.elapsed();
    println!("\n============================================================");
    println!("                  CONCURRENCY TEST VERDICT                  ");
    println!("============================================================");
    println!("  -> Total Concurrent Tasks Completed: {} / 20", success_count);
    println!("  -> Total Elapsed Wall Time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  -> Avg Latency / Request: {:.2} ms", (elapsed.as_secs_f64() * 1000.0) / 20.0);
    println!("  -> Concurrent Processing Throughput: {:.1} requests / sec", 20.0 / elapsed.as_secs_f64());
    println!("============================================================");

    Ok(())
}
