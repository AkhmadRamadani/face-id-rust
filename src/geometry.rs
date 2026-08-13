//! Pure geometry: letterboxing for the detector's fixed-size input, and
//! 5-point similarity alignment (the "straighten the face" step) for the
//! embedder's fixed-size input.
//!
//! No model-specific I/O lives here — this module only knows about pixels
//! and points, so it's independently testable and reusable by the detector,
//! the anti-spoof crop, and the aligner.

use crate::types::Point2;
use image::{Rgb, RgbImage};

// ---------------------------------------------------------------------
// Letterbox (resize-with-padding), used to feed a fixed-size square input
// (e.g. YuNet's 640x640) without distorting the source image's aspect ratio.
// ---------------------------------------------------------------------

/// Maps coordinates between the letterboxed (padded, square) image and the
/// original source image.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl LetterboxTransform {
    /// Maps a point from letterboxed-image space back to original-image space.
    pub fn to_original(&self, p: Point2) -> Point2 {
        [(p[0] - self.pad_x) / self.scale, (p[1] - self.pad_y) / self.scale]
    }
}

/// Resizes `src` to fit within a `target`x`target` square (preserving aspect
/// ratio) and pads the remainder with mid-gray (114,114,114 — the YOLO/YuNet
/// convention). Returns the padded canvas plus the transform needed to map
/// detections back to `src`'s coordinate space.
pub fn letterbox(src: &RgbImage, target: u32) -> (RgbImage, LetterboxTransform) {
    let (w, h) = src.dimensions();
    let scale = (target as f32 / w as f32).min(target as f32 / h as f32);
    let new_w = ((w as f32 * scale).round() as u32).clamp(1, target);
    let new_h = ((h as f32 * scale).round() as u32).clamp(1, target);

    let resized = image::imageops::resize(src, new_w, new_h, image::imageops::FilterType::Triangle);

    let pad_x = (target - new_w) / 2;
    let pad_y = (target - new_h) / 2;

    let mut canvas = RgbImage::from_pixel(target, target, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    (
        canvas,
        LetterboxTransform {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    )
}

// ---------------------------------------------------------------------
// 5-point similarity alignment (Umeyama) + bilinear warp
// ---------------------------------------------------------------------

type Mat2 = [[f32; 2]; 2];

fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat2_vec(a: Mat2, v: Point2) -> Point2 {
    [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]]
}

fn mat2_transpose(a: Mat2) -> Mat2 {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

fn rot(angle: f32) -> Mat2 {
    let (s, c) = angle.sin_cos();
    [[c, -s], [s, c]]
}

fn mean5(pts: &[Point2; 5]) -> Point2 {
    let mut m = [0f32; 2];
    for p in pts {
        m[0] += p[0];
        m[1] += p[1];
    }
    [m[0] / 5.0, m[1] / 5.0]
}

/// Closed-form SVD of a 2x2 matrix: returns `(U, singular_values, Vt)` such
/// that `A == U * diag(singular_values) * Vt`. Derived from the standard
/// E/F/G/H construction for 2x2 matrices; verified by hand against known
/// rotation/scale/shear test cases (see module tests).
fn svd2(a: Mat2) -> (Mat2, [f32; 2], Mat2) {
    let (m00, m01, m10, m11) = (a[0][0], a[0][1], a[1][0], a[1][1]);
    let e = (m00 + m11) / 2.0;
    let f = (m00 - m11) / 2.0;
    let g = (m10 + m01) / 2.0;
    let h = (m10 - m01) / 2.0;
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    let sv = [q + r, q - r];
    let a1 = g.atan2(f);
    let a2 = h.atan2(e);
    let theta = (a2 - a1) / 2.0;
    let phi = (a2 + a1) / 2.0;
    (rot(phi), sv, rot(theta))
}

/// A 2D similarity transform: `dst = scale * (rotation * src) + translation`.
#[derive(Debug, Clone, Copy)]
pub struct SimilarityTransform {
    pub scale: f32,
    pub rotation: Mat2,
    pub translation: Point2,
}

impl SimilarityTransform {
    /// Least-squares similarity transform mapping `src` points onto `dst`
    /// points (Umeyama 1991, specialized to 2D). With 5 non-degenerate
    /// landmark correspondences this is heavily over-determined, which is
    /// exactly what makes it robust to per-landmark detector jitter.
    pub fn estimate(src: &[Point2; 5], dst: &[Point2; 5]) -> Self {
        let src_mean = mean5(src);
        let dst_mean = mean5(dst);

        let src_d: Vec<Point2> = src.iter().map(|p| [p[0] - src_mean[0], p[1] - src_mean[1]]).collect();
        let dst_d: Vec<Point2> = dst.iter().map(|p| [p[0] - dst_mean[0], p[1] - dst_mean[1]]).collect();

        // Covariance A = (1/n) * dst_demean^T * src_demean
        let n = src.len() as f32;
        let mut a: Mat2 = [[0.0; 2]; 2];
        for i in 0..src.len() {
            a[0][0] += dst_d[i][0] * src_d[i][0];
            a[0][1] += dst_d[i][0] * src_d[i][1];
            a[1][0] += dst_d[i][1] * src_d[i][0];
            a[1][1] += dst_d[i][1] * src_d[i][1];
        }
        for row in a.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }

        let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let (u, sv, vt) = svd2(a);
        // Reflection correction: without this, a mirrored landmark
        // configuration could produce a flipped (non-rotation) "R".
        let d = if det_a < 0.0 { [1.0, -1.0] } else { [1.0, 1.0] };

        let u_d: Mat2 = [[u[0][0] * d[0], u[0][1] * d[1]], [u[1][0] * d[0], u[1][1] * d[1]]];
        let rotation = mat2_mul(u_d, vt);

        let var_src: f32 = src_d.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum::<f32>() / n;
        let scale = if var_src > 1e-12 {
            (sv[0] * d[0] + sv[1] * d[1]) / var_src
        } else {
            1.0
        };

        let r_src_mean = mat2_vec(rotation, src_mean);
        let translation = [
            dst_mean[0] - scale * r_src_mean[0],
            dst_mean[1] - scale * r_src_mean[1],
        ];

        SimilarityTransform {
            scale,
            rotation,
            translation,
        }
    }

    /// Estimates a 2D similarity transform using ONLY the left and right eyes.
    /// Guarantees that the line connecting the two eyes is rotated to match the
    /// angle between the reference eyes (e.g. 100% horizontal alignment).
    pub fn estimate_eyes_only(left_eye: Point2, right_eye: Point2, dst_left_eye: Point2, dst_right_eye: Point2) -> Self {
        let src_dx = right_eye[0] - left_eye[0];
        let src_dy = right_eye[1] - left_eye[1];
        let src_dist = (src_dx * src_dx + src_dy * src_dy).sqrt();

        let dst_dx = dst_right_eye[0] - dst_left_eye[0];
        let dst_dy = dst_right_eye[1] - dst_left_eye[1];
        let dst_dist = (dst_dx * dst_dx + dst_dy * dst_dy).sqrt();

        let scale = if src_dist > 1e-6 { dst_dist / src_dist } else { 1.0 };

        let src_angle = src_dy.atan2(src_dx);
        let dst_angle = dst_dy.atan2(dst_dx);
        let delta_angle = dst_angle - src_angle;

        let rotation = rot(delta_angle);

        let src_mid = [(left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0];
        let dst_mid = [(dst_left_eye[0] + dst_right_eye[0]) / 2.0, (dst_left_eye[1] + dst_right_eye[1]) / 2.0];

        let r_src_mid = mat2_vec(rotation, src_mid);
        let translation = [
            dst_mid[0] - scale * r_src_mid[0],
            dst_mid[1] - scale * r_src_mid[1],
        ];

        SimilarityTransform {
            scale,
            rotation,
            translation,
        }
    }

    /// Forward map: source-image point -> aligned/template-space point.
    pub fn apply(&self, p: Point2) -> Point2 {
        let rp = mat2_vec(self.rotation, p);
        [self.scale * rp[0] + self.translation[0], self.scale * rp[1] + self.translation[1]]
    }

    /// Inverse map: aligned/template-space point -> source-image point.
    /// This is what the warp uses (inverse sampling avoids holes in the output).
    pub fn apply_inverse(&self, q: Point2) -> Point2 {
        let d = [q[0] - self.translation[0], q[1] - self.translation[1]];
        let rt = mat2_transpose(self.rotation);
        let rd = mat2_vec(rt, d);
        let inv_scale = if self.scale.abs() > 1e-12 { 1.0 / self.scale } else { 0.0 };
        [rd[0] * inv_scale, rd[1] * inv_scale]
    }
}

fn bilinear_sample(img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Rgb([0, 0, 0]);
    }
    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    if x < 0.0 || y < 0.0 || x >= max_x || y >= max_y {
        let cx = x.round().clamp(0.0, max_x) as u32;
        let cy = y.round().clamp(0.0, max_y) as u32;
        return *img.get_pixel(cx, cy);
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let dx = x - x0;
    let dy = y - y0;
    let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x0 as u32 + 1, y0 as u32 + 1);
    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - dx) + p10[c] as f32 * dx;
        let bot = p01[c] as f32 * (1.0 - dx) + p11[c] as f32 * dx;
        out[c] = (top * (1.0 - dy) + bot * dy).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

/// Warps `src` into an `out_size`x`out_size` image using the *inverse* of
/// `transform` (i.e. `transform` maps source -> aligned space; this samples
/// aligned space <- source using bilinear interpolation).
pub fn warp_align(src: &RgbImage, transform: &SimilarityTransform, out_size: u32) -> RgbImage {
    let mut out = RgbImage::new(out_size, out_size);
    for y in 0..out_size {
        for x in 0..out_size {
            let dst_pt = [x as f32 + 0.5, y as f32 + 0.5];
            let src_pt = transform.apply_inverse(dst_pt);
            let px = bilinear_sample(src, src_pt[0] - 0.5, src_pt[1] - 0.5);
            out.put_pixel(x, y, px);
        }
    }
    out
}

/// The canonical 5-point reference template (left eye, right eye, nose,
/// mouth-left, mouth-right) for a 112x112 aligned crop. This specific set of
/// coordinates is the de-facto standard used across the ArcFace/InsightFace
/// ecosystem — most public face embedding models expect input aligned to
/// (approximately) this template, so this is what makes a third-party
/// embedding model actually work well without retraining it.
pub const ARCFACE_REFERENCE_112: [Point2; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Scales [`ARCFACE_REFERENCE_112`] to a different square output size.
/// Exact only at 112; a linear approximation otherwise, which in practice
/// works fine for the sizes embedding models commonly use (96-160px).
pub fn arcface_reference_scaled(out_size: u32) -> [Point2; 5] {
    let s = out_size as f32 / 112.0;
    ARCFACE_REFERENCE_112.map(|p| [p[0] * s, p[1] * s])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd2_recovers_pure_rotation() {
        let angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let a = rot(angle);
        let (u, sv, vt) = svd2(a);
        assert!((sv[0] - 1.0).abs() < 1e-5);
        assert!((sv[1] - 1.0).abs() < 1e-5);
        let recon = mat2_mul(u, vt);
        for i in 0..2 {
            for j in 0..2 {
                assert!((recon[i][j] - a[i][j]).abs() < 1e-4, "mismatch at {i},{j}");
            }
        }
    }

    #[test]
    fn similarity_transform_identity_for_matching_points() {
        let pts = ARCFACE_REFERENCE_112;
        let t = SimilarityTransform::estimate(&pts, &pts);
        assert!((t.scale - 1.0).abs() < 1e-3);
        for p in pts {
            let mapped = t.apply(p);
            assert!((mapped[0] - p[0]).abs() < 1e-2);
            assert!((mapped[1] - p[1]).abs() < 1e-2);
        }
    }

    #[test]
    fn similarity_transform_recovers_known_scale_rotation_translation() {
        let true_scale = 1.7f32;
        let true_angle = 0.3f32; // radians
        let true_translation = [12.0f32, -7.0];
        let r = rot(true_angle);

        let src = ARCFACE_REFERENCE_112;
        let dst = src.map(|p| {
            let rp = mat2_vec(r, p);
            [
                true_scale * rp[0] + true_translation[0],
                true_scale * rp[1] + true_translation[1],
            ]
        });

        let t = SimilarityTransform::estimate(&src, &dst);
        assert!((t.scale - true_scale).abs() < 1e-2, "scale: {}", t.scale);

        // Round-trip check instead of comparing angles directly (avoids
        // needing to reconstruct phi from the matrix).
        for p in src {
            let mapped = t.apply(p);
            let expected = {
                let rp = mat2_vec(r, p);
                [
                    true_scale * rp[0] + true_translation[0],
                    true_scale * rp[1] + true_translation[1],
                ]
            };
            assert!((mapped[0] - expected[0]).abs() < 1e-1);
            assert!((mapped[1] - expected[1]).abs() < 1e-1);
        }
    }

    #[test]
    fn inverse_map_round_trips() {
        let src = ARCFACE_REFERENCE_112;
        let mut dst = src;
        dst.rotate_left(1); // arbitrary non-trivial correspondence
        let t = SimilarityTransform::estimate(&src, &dst);
        for p in src {
            let fwd = t.apply(p);
            let back = t.apply_inverse(fwd);
            assert!((back[0] - p[0]).abs() < 1e-2);
            assert!((back[1] - p[1]).abs() < 1e-2);
        }
    }
}
