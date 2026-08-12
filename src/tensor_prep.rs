//! Converts decoded images into the flat, row-major NCHW float32 buffers the
//! models expect. Kept separate from `geometry.rs` (which only deals with
//! coordinates) and from the backend wrappers (which only deal with tensors).

use image::RgbImage;

use crate::types::AlignedFace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    Rgb,
    Bgr,
}

/// Per-channel normalization applied as `(pixel_u8 as f32 - mean) / std`,
/// evaluated in `order`'s channel arrangement.
#[derive(Debug, Clone, Copy)]
pub struct Normalization {
    pub order: ChannelOrder,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    /// `x / 255` with no channel swap — YuNet's convention (BGR, 0-255... but
    /// note YuNet is actually *not* divided at all, see [`Self::scale_only`]).
    pub const fn zero_to_one_bgr() -> Self {
        Normalization {
            order: ChannelOrder::Bgr,
            mean: [0.0, 0.0, 0.0],
            std: [255.0, 255.0, 255.0],
        }
    }

    /// Raw `0..255` float range, BGR order, no scaling at all — what YuNet's
    /// published I/O spec calls for (`BGR, 0-255`, no normalization).
    pub const fn raw_bgr() -> Self {
        Normalization {
            order: ChannelOrder::Bgr,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }

    /// The common ArcFace-style embedder normalization: RGB order,
    /// `(x/255 - 0.5) / 0.5`, i.e. maps `0..255` to `-1..1`.
    pub const fn arcface_rgb() -> Self {
        Normalization {
            order: ChannelOrder::Rgb,
            mean: [127.5, 127.5, 127.5],
            std: [127.5, 127.5, 127.5],
        }
    }
}

/// Converts an [`AlignedFace`]'s raw interleaved RGB bytes into a flat NCHW
/// `[1,3,H,W]` float32 buffer, without cloning into an intermediate
/// `RgbImage` (this runs on every `embed()` call, so it's worth avoiding
/// the extra allocation).
pub fn aligned_face_to_nchw(face: &AlignedFace, norm: &Normalization) -> Vec<f32> {
    let (w, h) = (face.width as usize, face.height as usize);
    let mut out = vec![0f32; 3 * w * h];
    let plane = w * h;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let base = idx * 3;
            let (r, g, b) = (
                face.rgb[base] as f32,
                face.rgb[base + 1] as f32,
                face.rgb[base + 2] as f32,
            );
            let (c0, c1, c2) = match norm.order {
                ChannelOrder::Rgb => (r, g, b),
                ChannelOrder::Bgr => (b, g, r),
            };
            out[idx] = (c0 - norm.mean[0]) / norm.std[0];
            out[plane + idx] = (c1 - norm.mean[1]) / norm.std[1];
            out[2 * plane + idx] = (c2 - norm.mean[2]) / norm.std[2];
        }
    }
    out
}

/// Converts an RGB image into a flat NCHW `[1,3,H,W]` float32 buffer.
pub fn image_to_nchw(img: &RgbImage, norm: &Normalization) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0f32; 3 * w * h];
    let plane = w * h;
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
            let (c0, c1, c2) = match norm.order {
                ChannelOrder::Rgb => (r, g, b),
                ChannelOrder::Bgr => (b, g, r),
            };
            let idx = y * w + x;
            out[idx] = (c0 - norm.mean[0]) / norm.std[0];
            out[plane + idx] = (c1 - norm.mean[1]) / norm.std[1];
            out[2 * plane + idx] = (c2 - norm.mean[2]) / norm.std[2];
        }
    }
    out
}
