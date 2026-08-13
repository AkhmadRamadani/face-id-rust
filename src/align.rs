//! Combines the detector's 5 landmarks with the similarity-transform math in
//! `geometry.rs` to produce a canonically-posed [`AlignedFace`] ready for the
//! embedder. This is the "straighten the face" step: it corrects in-plane
//! rotation, scale, and translation so the embedding model always sees eyes
//! and mouth in roughly the same place regardless of head pose in the
//! original photo.

use image::RgbImage;

use crate::geometry::{arcface_reference_scaled, warp_align, SimilarityTransform};
use crate::types::{AlignedFace, Landmarks5};

/// Aligns `image` using `landmarks`, warping into an `out_size`x`out_size`
/// crop matching the ArcFace-style reference template (see
/// [`crate::geometry::ARCFACE_REFERENCE_112`]).
///
/// `out_size` should match your embedding model's expected input resolution
/// (112 is the most common; some models use 96, 120, or 128).
pub fn align_face(image: &RgbImage, landmarks: &Landmarks5, out_size: u32) -> AlignedFace {
    let reference = arcface_reference_scaled(out_size);
    let transform = SimilarityTransform::estimate(&landmarks.as_array(), &reference);
    let warped = warp_align(image, &transform, out_size);
    AlignedFace {
        rgb: warped.into_raw(),
        width: out_size,
        height: out_size,
    }
}

/// Aligns `image` using ONLY the left and right eyes to guarantee 100% level,
/// horizontal eye alignment matching the reference template (as in OpenCV/Dlib eye alignment).
pub fn align_face_eyes_only(image: &RgbImage, landmarks: &Landmarks5, out_size: u32) -> AlignedFace {
    let reference = arcface_reference_scaled(out_size);
    let transform = SimilarityTransform::estimate_eyes_only(
        landmarks.left_eye,
        landmarks.right_eye,
        reference[0],
        reference[1],
    );
    let warped = warp_align(image, &transform, out_size);
    AlignedFace {
        rgb: warped.into_raw(),
        width: out_size,
        height: out_size,
    }
}

/// Rotates `image` around the eye midpoint so the eyes are 100% horizontal,
/// then crops the transformed face bounding box (with optional `margin_factor` e.g. 0.0 for exact bbox, 0.1 for 10% padding).
pub fn crop_eye_aligned_face(
    image: &RgbImage,
    bbox: &crate::types::BBox,
    landmarks: &Landmarks5,
    margin_factor: f32,
) -> RgbImage {
    let lx = landmarks.left_eye[0];
    let ly = landmarks.left_eye[1];
    let rx = landmarks.right_eye[0];
    let ry = landmarks.right_eye[1];

    let dx = rx - lx;
    let dy = ry - ly;
    // Negate the eye line angle so that right eye is brought to the exact same Y height as left eye.
    let angle = -dy.atan2(dx);

    let (sin_a, cos_a) = angle.sin_cos();

    // Eye midpoint (pivot center)
    let cx = (lx + rx) / 2.0;
    let cy = (ly + ry) / 2.0;

    let rotate_point = |px: f32, py: f32| -> [f32; 2] {
        let dx_c = px - cx;
        let dy_c = py - cy;
        [
            dx_c * cos_a - dy_c * sin_a + cx,
            dx_c * sin_a + dy_c * cos_a + cy,
        ]
    };

    let corners = [
        rotate_point(bbox.x1, bbox.y1),
        rotate_point(bbox.x2, bbox.y1),
        rotate_point(bbox.x1, bbox.y2),
        rotate_point(bbox.x2, bbox.y2),
    ];

    let mut min_x = corners[0][0];
    let mut max_x = corners[0][0];
    let mut min_y = corners[0][1];
    let mut max_y = corners[0][1];

    for c in &corners[1..] {
        min_x = min_x.min(c[0]);
        max_x = max_x.max(c[0]);
        min_y = min_y.min(c[1]);
        max_y = max_y.max(c[1]);
    }

    let bw = max_x - min_x;
    let bh = max_y - min_y;

    min_x -= bw * margin_factor;
    max_x += bw * margin_factor;
    min_y -= bh * margin_factor;
    max_y += bh * margin_factor;

    let img_w = image.width() as f32;
    let img_h = image.height() as f32;

    let crop_x1 = (min_x.floor() as i64).clamp(0, image.width() as i64 - 1) as u32;
    let crop_y1 = (min_y.floor() as i64).clamp(0, image.height() as i64 - 1) as u32;
    let crop_x2 = (max_x.ceil() as i64).clamp(crop_x1 as i64 + 1, image.width() as i64) as u32;
    let crop_y2 = (max_y.ceil() as i64).clamp(crop_y1 as i64 + 1, image.height() as i64) as u32;

    let out_w = crop_x2 - crop_x1;
    let out_h = crop_y2 - crop_y1;

    let mut cropped = RgbImage::new(out_w, out_h);

    for y in 0..out_h {
        let py_rot = (crop_y1 + y) as f32 + 0.5;
        for x in 0..out_w {
            let px_rot = (crop_x1 + x) as f32 + 0.5;

            let dx_c = px_rot - cx;
            let dy_c = py_rot - cy;
            // Inverse mapping: R(-angle)
            let src_x = dx_c * cos_a + dy_c * sin_a + cx;
            let src_y = -dx_c * sin_a + dy_c * cos_a + cy;

            if src_x >= 0.0 && src_x < img_w && src_y >= 0.0 && src_y < img_h {
                let sx = (src_x as u32).min(image.width() - 1);
                let sy = (src_y as u32).min(image.height() - 1);
                cropped.put_pixel(x, y, *image.get_pixel(sx, sy));
            }
        }
    }

    cropped
}

fn convex_hull(mut points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    points.sort_by(|a, b| {
        if (a[0] - b[0]).abs() > 1e-5 {
            a[0].partial_cmp(&b[0]).unwrap()
        } else {
            a[1].partial_cmp(&b[1]).unwrap()
        }
    });
    if points.len() <= 3 {
        return points;
    }

    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &points {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in points.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn point_in_polygon(x: f32, y: f32, poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let xi = poly[i][0];
        let yi = poly[i][1];
        let xj = poly[j][0];
        let yj = poly[j][1];

        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Rotates `image` so the eyes are 100% horizontal, applies a smooth 24-point face oval convex hull mask
/// (capped at eyebrow height to black-out background outside the face curve), matching the Flutter FaceProcessingService pipeline.
pub fn mask_and_crop_face(
    image: &RgbImage,
    bbox: &crate::types::BBox,
    landmarks: &Landmarks5,
    margin_factor: f32,
) -> RgbImage {
    let lx = landmarks.left_eye[0];
    let ly = landmarks.left_eye[1];
    let rx = landmarks.right_eye[0];
    let ry = landmarks.right_eye[1];

    let dx = rx - lx;
    let dy = ry - ly;
    let eye_dist = (dx * dx + dy * dy).sqrt();
    let angle = -dy.atan2(dx);

    let (sin_a, cos_a) = angle.sin_cos();

    // Eye midpoint (pivot center)
    let cx = (lx + rx) / 2.0;
    let cy = (ly + ry) / 2.0;

    let rotate_point = |px: f32, py: f32| -> [f32; 2] {
        let dx_c = px - cx;
        let dy_c = py - cy;
        [
            dx_c * cos_a - dy_c * sin_a + cx,
            dx_c * sin_a + dy_c * cos_a + cy,
        ]
    };

    let r_left_eye = rotate_point(lx, ly);

    let corners = [
        rotate_point(bbox.x1, bbox.y1),
        rotate_point(bbox.x2, bbox.y1),
        rotate_point(bbox.x1, bbox.y2),
        rotate_point(bbox.x2, bbox.y2),
    ];

    let min_x = corners.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
    let max_x = corners.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
    let max_y = corners.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max);

    let w_box = max_x - min_x;
    let h_box = max_y - min_y;

    let box_cx = (min_x + max_x) / 2.0;
    let box_cy = (min_y + max_y) / 2.0;

    let eyebrow_y = r_left_eye[1] - 0.25 * eye_dist;

    // Generate 24 smooth ellipse points around face bbox (matching Flutter code)
    let num_pts = 24;
    let mut pts = Vec::with_capacity(num_pts);
    for i in 0..num_pts {
        let rad = i as f32 * 2.0 * std::f32::consts::PI / (num_pts as f32);
        let px = box_cx + (w_box / 2.0) * rad.cos();
        let mut py = box_cy + (h_box / 2.0) * rad.sin();

        // Cap points above eyebrow_y across full face width
        if py < eyebrow_y {
            py = eyebrow_y;
        }
        pts.push([px, py]);
    }

    let hull = convex_hull(pts);

    let img_w = image.width() as f32;
    let img_h = image.height() as f32;

    let min_x_poly = min_x.floor().clamp(0.0, img_w - 1.0) as u32;
    let max_x_poly = max_x.ceil().clamp(min_x_poly as f32 + 1.0, img_w) as u32;
    let min_y_poly = eyebrow_y.floor().clamp(0.0, img_h - 1.0) as u32;
    let max_y_poly = max_y.ceil().clamp(min_y_poly as f32 + 1.0, img_h) as u32;

    let bw = (max_x_poly - min_x_poly) as f32;
    let bh = (max_y_poly - min_y_poly) as f32;

    let pad_x = (bw * margin_factor) as u32;
    let pad_y = (bh * margin_factor) as u32;

    let crop_x1 = min_x_poly.saturating_sub(pad_x);
    let crop_y1 = min_y_poly.saturating_sub(pad_y);
    let crop_x2 = (max_x_poly + pad_x).min(image.width());
    let crop_y2 = (max_y_poly + pad_y).min(image.height());

    let out_w = crop_x2 - crop_x1;
    let out_h = crop_y2 - crop_y1;

    let mut cropped = RgbImage::new(out_w, out_h);

    for y in 0..out_h {
        let py_rot = (crop_y1 + y) as f32 + 0.5;
        for x in 0..out_w {
            let px_rot = (crop_x1 + x) as f32 + 0.5;

            // Check if point is inside the smooth convex hull polygon
            if point_in_polygon(px_rot, py_rot, &hull) {
                let dx_c = px_rot - cx;
                let dy_c = py_rot - cy;
                let src_x = dx_c * cos_a + dy_c * sin_a + cx;
                let src_y = -dx_c * sin_a + dy_c * cos_a + cy;

                if src_x >= 0.0 && src_x < img_w && src_y >= 0.0 && src_y < img_h {
                    let sx = (src_x as u32).min(image.width() - 1);
                    let sy = (src_y as u32).min(image.height() - 1);
                    cropped.put_pixel(x, y, *image.get_pixel(sx, sy));
                }
            }
        }
    }

    cropped
}

/// Uses sub-pixel MediaPipe 468 3D landmarks to perform 100% eye-level alignment,
/// extracts the 36 face oval contour points, caps top at eyebrow height, applies the convex hull mask, and crops tightly.
pub fn mask_and_crop_mediapipe_mesh(
    image: &RgbImage,
    mediapipe_468: &[[f32; 3]],
    crop_offset_x: f32,
    crop_offset_y: f32,
    crop_w: f32,
    crop_h: f32,
    margin_factor: f32,
) -> RgbImage {
    let mp_l5 = crate::landmarker::FaceLandmarker::extract_landmarks5(
        mediapipe_468,
        crop_offset_x,
        crop_offset_y,
        crop_w,
        crop_h,
    );

    let lx = mp_l5.left_eye[0];
    let ly = mp_l5.left_eye[1];
    let rx = mp_l5.right_eye[0];
    let ry = mp_l5.right_eye[1];

    let dx = rx - lx;
    let dy = ry - ly;
    let angle = -dy.atan2(dx);

    let (sin_a, cos_a) = angle.sin_cos();

    // Eye midpoint (pivot center)
    let cx = (lx + rx) / 2.0;
    let cy = (ly + ry) / 2.0;

    let rotate_point = |px: f32, py: f32| -> [f32; 2] {
        let dx_c = px - cx;
        let dy_c = py - cy;
        [
            dx_c * cos_a - dy_c * sin_a + cx,
            dx_c * sin_a + dy_c * cos_a + cy,
        ]
    };

    // Map 36 MediaPipe Face Oval contour points from [0..192] to level-rotated image space
    let mut oval_pts = Vec::with_capacity(36);
    for &idx in &crate::landmarker::FACEMESH_FACE_OVAL {
        let pt = mediapipe_468[idx];
        let px = (pt[0] / 192.0) * crop_w + crop_offset_x;
        let py = (pt[1] / 192.0) * crop_h + crop_offset_y;

        let r_pt = rotate_point(px, py);
        oval_pts.push(r_pt);
    }

    let hull = convex_hull(oval_pts);

    let min_x = hull.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min);
    let max_x = hull.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max);
    let min_y = hull.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
    let max_y = hull.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max);

    let img_w = image.width() as f32;
    let img_h = image.height() as f32;

    let min_x_poly = min_x.floor().clamp(0.0, img_w - 1.0) as u32;
    let max_x_poly = max_x.ceil().clamp(min_x_poly as f32 + 1.0, img_w) as u32;
    let min_y_poly = min_y.floor().clamp(0.0, img_h - 1.0) as u32;
    let max_y_poly = max_y.ceil().clamp(min_y_poly as f32 + 1.0, img_h) as u32;

    let bw = (max_x_poly - min_x_poly) as f32;
    let bh = (max_y_poly - min_y_poly) as f32;

    let pad_x = (bw * margin_factor) as u32;
    let pad_y = (bh * margin_factor) as u32;

    let crop_x1 = min_x_poly.saturating_sub(pad_x);
    let crop_y1 = min_y_poly.saturating_sub(pad_y);
    let crop_x2 = (max_x_poly + pad_x).min(image.width());
    let crop_y2 = (max_y_poly + pad_y).min(image.height());

    let out_w = crop_x2 - crop_x1;
    let out_h = crop_y2 - crop_y1;

    let mut cropped = RgbImage::new(out_w, out_h);

    for y in 0..out_h {
        let py_rot = (crop_y1 + y) as f32 + 0.5;
        for x in 0..out_w {
            let px_rot = (crop_x1 + x) as f32 + 0.5;

            if point_in_polygon(px_rot, py_rot, &hull) {
                let dx_c = px_rot - cx;
                let dy_c = py_rot - cy;
                let src_x = dx_c * cos_a + dy_c * sin_a + cx;
                let src_y = -dx_c * sin_a + dy_c * cos_a + cy;

                if src_x >= 0.0 && src_x < img_w && src_y >= 0.0 && src_y < img_h {
                    let sx = (src_x as u32).min(image.width() - 1);
                    let sy = (src_y as u32).min(image.height() - 1);
                    cropped.put_pixel(x, y, *image.get_pixel(sx, sy));
                }
            }
        }
    }

    cropped
}

/// Performs a 2-pass coarse-to-fine MediaPipe face alignment and contour mask crop:
/// 1. Pass 1: Uses initial YuNet landmarks to level-rotate image so the face is upright.
/// 2. Pass 2: Runs MediaPipe FaceLandmarker on the upright face crop to get 468 sub-pixel 3D landmarks.
/// 3. Pass 3: Applies residual pupil micro-rotation + 36-point MediaPipe contour convex hull mask, returning the final crop.
pub fn crop_mediapipe_two_pass_face(
    image: &RgbImage,
    bbox: &crate::types::BBox,
    initial_landmarks: &Landmarks5,
    landmarker: &mut crate::landmarker::FaceLandmarker,
    margin_factor: f32,
) -> crate::error::Result<RgbImage> {
    // Pass 1: Coarse eye rotation & crop
    let coarse_crop = crop_eye_aligned_face(image, bbox, initial_landmarks, 0.0);

    // Pass 2: MediaPipe 468 landmark inference on upright face
    let mp_468 = landmarker.predict(&coarse_crop)?;

    // Pass 3: MediaPipe fine pupil eye alignment + 36-point contour mask
    let final_crop = mask_and_crop_mediapipe_mesh(
        &coarse_crop,
        &mp_468,
        0.0,
        0.0,
        coarse_crop.width() as f32,
        coarse_crop.height() as f32,
        margin_factor,
    );

    Ok(final_crop)
}
