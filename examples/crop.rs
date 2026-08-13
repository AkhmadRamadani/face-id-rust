//! CLI example to detect, crop, and align faces from images in `test/`.
//!
//! ```text
//! cargo run --example crop -- \
//!     --detector models/yunet_fp16.tflite \
//!     --image test/1.jpg \
//!     --output test/1_aligned.jpg
//! ```

use clap::Parser;
use faceid::align::align_face;
use faceid::pipeline::load_image;
use faceid::{Accelerators, DetectorConfig, FaceDetector};
use image::RgbImage;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to the YuNet detector .tflite export.
    #[arg(long, default_value = "models/yunet_fp16.tflite")]
    detector: PathBuf,

    /// Path to the input photo.
    #[arg(long, default_value = "test/1.jpg")]
    image: PathBuf,

    /// Detector model type: "yunet" or "blazeface".
    #[arg(long, default_value = "yunet")]
    detector_type: String,

    /// Path to save the aligned face crop.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Resolution for aligned crop (default: 112 for embedding models).
    #[arg(long, default_value_t = 112)]
    size: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading image from {:?}...", args.image);
    let img = load_image(&args.image)?;

    let faces = if args.detector_type == "blazeface" {
        println!("Loading BlazeFace detector from {:?}...", args.detector);
        let mut detector = faceid::BlazeFaceDetector::load(
            &args.detector,
            Accelerators::GPU | Accelerators::CPU,
            faceid::BlazeFaceConfig::default(),
        )?;
        detector.detect_all(&img)?
    } else {
        println!("Loading YuNet detector from {:?}...", args.detector);
        let mut detector = FaceDetector::load(
            &args.detector,
            Accelerators::GPU | Accelerators::CPU,
            DetectorConfig::default(),
        )?;
        detector.detect_all(&img)?
    };
    if faces.is_empty() {
        println!("No faces detected in {:?}", args.image);
        return Ok(());
    }

    println!("Detected {} face(s) in {:?}", faces.len(), args.image);

    let stem = args.image.file_stem().and_then(|s| s.to_str()).unwrap_or("face");
    let parent = args.image.parent().unwrap_or_else(|| std::path::Path::new("test"));

    for (i, face) in faces.iter().enumerate() {
        println!(
            "Face #{}: score={:.4}, bbox=({:.1}, {:.1}) - ({:.1}, {:.1})",
            i + 1,
            face.score,
            face.bbox.x1,
            face.bbox.y1,
            face.bbox.x2,
            face.bbox.y2
        );
        println!(
            "  landmarks: left_eye={:?}, right_eye={:?}, nose={:?}, mouth_left={:?}, mouth_right={:?}",
            face.landmarks.left_eye,
            face.landmarks.right_eye,
            face.landmarks.nose,
            face.landmarks.mouth_left,
            face.landmarks.mouth_right
        );

        // 1. Aligned face crop (canonical 5-point Umeyama alignment)
        let aligned = align_face(&img, &face.landmarks, args.size);
        let aligned_img = RgbImage::from_raw(aligned.width, aligned.height, aligned.rgb)
            .ok_or("Failed to construct image from aligned RGB buffer")?;

        let aligned_out_path = args.output.clone().unwrap_or_else(|| {
            parent.join(format!("{}_aligned_{}.jpg", stem, i + 1))
        });
        aligned_img.save(&aligned_out_path)?;
        println!("Saved 5-pt aligned crop to {:?}", aligned_out_path);

        // 1b. Eye-only aligned face crop (2-point eye level rotation alignment to 112x112)
        let eye_aligned = faceid::align::align_face_eyes_only(&img, &face.landmarks, args.size);
        let eye_aligned_img = RgbImage::from_raw(eye_aligned.width, eye_aligned.height, eye_aligned.rgb)
            .ok_or("Failed to construct image from eye aligned RGB buffer")?;
        let eye_aligned_out_path = parent.join(format!("{}_eyes_aligned_{}.jpg", stem, i + 1));
        eye_aligned_img.save(&eye_aligned_out_path)?;
        println!("Saved 2-pt eye-level aligned crop to {:?}", eye_aligned_out_path);

        // 1c. Eye-rotated face crop (rotates full photo so eyes are horizontal, then crops face bbox)
        let eye_rotated_crop = faceid::align::crop_eye_aligned_face(&img, &face.bbox, &face.landmarks, 0.0);
        let eye_rotated_out_path = parent.join(format!("{}_eye_rotated_crop_{}.jpg", stem, i + 1));
        eye_rotated_crop.save(&eye_rotated_out_path)?;
        println!("Saved eye-rotated face bbox crop to {:?}", eye_rotated_out_path);

        // 1d. Masked face crop (black background outside landmark face contour & eyebrow line)
        let masked_crop = faceid::align::mask_and_crop_face(&img, &face.bbox, &face.landmarks, 0.05);
        let masked_out_path = parent.join(format!("{}_masked_crop_{}.jpg", stem, i + 1));
        masked_crop.save(&masked_out_path)?;
        println!("Saved face landmark masked crop to {:?}", masked_out_path);

        // 1e. True MediaPipe 468 3D Mesh Masked Crop (using models/face_landmark.tflite)

        let landmarker_path = std::path::Path::new("models/face_landmark.tflite");
        if landmarker_path.exists() {
            if let Ok(mut landmarker) = faceid::FaceLandmarker::load(landmarker_path, Accelerators::GPU | Accelerators::CPU) {
                if let Ok(mp_masked_crop) = faceid::align::crop_mediapipe_two_pass_face(
                    &img,
                    &face.bbox,
                    &face.landmarks,
                    &mut landmarker,
                    0.05,
                ) {
                    let mp_out_path = parent.join(format!("{}_mediapipe_mesh_crop_{}.jpg", stem, i + 1));
                    mp_masked_crop.save(&mp_out_path)?;
                    println!("Saved 2-pass MediaPipe eye-aligned 468-point 3D Mesh Masked crop to {:?}", mp_out_path);
                }
            }
        }

        // 2. Direct BBox crop (raw crop without alignment transform)
        let x1 = (face.bbox.x1.max(0.0) as u32).min(img.width() - 1);
        let y1 = (face.bbox.y1.max(0.0) as u32).min(img.height() - 1);
        let w = (face.bbox.width() as u32).min(img.width() - x1);
        let h = (face.bbox.height() as u32).min(img.height() - y1);

        if w > 0 && h > 0 {
            let cropped = image::imageops::crop_imm(&img, x1, y1, w, h).to_image();
            let crop_out_path = parent.join(format!("{}_crop_{}.jpg", stem, i + 1));
            cropped.save(&crop_out_path)?;
            println!("Saved bbox crop to {:?}", crop_out_path);
        }
    }

    Ok(())
}
