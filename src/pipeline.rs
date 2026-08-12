//! Ties the stages together: detect -> (optional) liveness check -> align ->
//! embed -> register/recognize. This is the entry point most applications
//! should use instead of calling `detector`/`antispoof`/`align`/`embedder`
//! directly.

use image::RgbImage;

use crate::align::align_face;
use crate::antispoof::LivenessDetector;
use crate::blazeface::BlazeFaceDetector;
use crate::detector::FaceDetector;
use crate::embedder::Embedder;
use crate::error::Result;
use crate::recognition::{PersonId, RecognitionContext, RecordId, RegistrationScope, VectorStore};
use crate::types::{DetectedFace, LivenessResult};

pub trait Detector {
    fn detect_all(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>>;

    fn detect_exactly_one(&mut self, image: &RgbImage) -> Result<DetectedFace> {
        let faces = self.detect_all(image)?;
        match faces.len() {
            1 => Ok(faces.into_iter().next().unwrap()),
            0 => Err(crate::FaceError::NoFaceDetected),
            n => Err(crate::FaceError::AmbiguousFaceCount { count: n }),
        }
    }

    fn detect_best(&mut self, image: &RgbImage) -> Result<Option<DetectedFace>> {
        let mut faces = self.detect_all(image)?;
        faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(faces.into_iter().next())
    }

    fn is_fully_accelerated(&self) -> Result<bool> {
        Ok(true)
    }
}

impl Detector for BlazeFaceDetector {
    fn detect_all(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>> {
        self.detect_all(image)
    }
}

impl Detector for FaceDetector {
    fn detect_all(&mut self, image: &RgbImage) -> Result<Vec<DetectedFace>> {
        self.detect_all(image)
    }
}

pub struct FacePipeline<E: Embedder, D: Detector = BlazeFaceDetector> {
    detector: D,
    antispoof: Option<LivenessDetector>,
    embedder: E,
    store: VectorStore,
    similarity_threshold: f32,
}

impl<E: Embedder, D: Detector> FacePipeline<E, D> {
    pub fn new(detector: D, antispoof: Option<LivenessDetector>, embedder: E, similarity_threshold: f32) -> Self {
        Self {
            detector,
            antispoof,
            embedder,
            store: VectorStore::new(),
            similarity_threshold,
        }
    }

    pub fn with_store(mut self, store: VectorStore) -> Self {
        self.store = store;
        self
    }

    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut VectorStore {
        &mut self.store
    }

    pub fn detector(&self) -> &D {
        &self.detector
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    pub fn antispoof_enabled(&self) -> bool {
        self.antispoof.is_some()
    }

    /// Enables anti-spoofing using an already-loaded detector, or swaps out
    /// the one in use. Pass `None` to disable it.
    pub fn set_antispoof(&mut self, antispoof: Option<LivenessDetector>) {
        self.antispoof = antispoof;
    }

    fn check_live_if_enabled(&mut self, image: &RgbImage, bbox: &crate::types::BBox) -> Result<Option<LivenessResult>> {
        match self.antispoof.as_mut() {
            Some(spoof) => Ok(Some(spoof.require_live(image, bbox)?)),
            None => Ok(None),
        }
    }

    /// Full enrollment flow: requires exactly one face in `image` (so you
    /// can't silently enroll the wrong person out of a group photo),
    /// optionally checks liveness, aligns, embeds, and registers under
    /// `scope`.
    pub fn enroll(
        &mut self,
        image: &RgbImage,
        person_id: impl Into<PersonId>,
        scope: RegistrationScope,
        label: Option<String>,
    ) -> Result<RecordId> {
        self.enroll_opts(image, person_id, scope, label, true)
    }

    /// Enroll with explicit control over whether liveness/anti-spoof check is performed.
    pub fn enroll_opts(
        &mut self,
        image: &RgbImage,
        person_id: impl Into<PersonId>,
        scope: RegistrationScope,
        label: Option<String>,
        check_liveness: bool,
    ) -> Result<RecordId> {
        let face = self.detector.detect_exactly_one(image)?;
        if check_liveness {
            self.check_live_if_enabled(image, &face.bbox)?;
        }
        let aligned = align_face(image, &face.landmarks, self.embedder.input_size());
        let embedding = self.embedder.embed(&aligned)?;
        self.store.register(person_id.into(), embedding, scope, label)
    }

    /// Full identification flow: detects the best face in `image`,
    /// optionally checks liveness, aligns, embeds, and searches the
    /// registry under `context`. Returns `Ok(None)` if no face was found or
    /// no registered person cleared the similarity threshold — a "no match"
    /// is not an error.
    pub fn identify(
        &mut self,
        image: &RgbImage,
        context: &RecognitionContext,
    ) -> Result<Option<crate::recognition::MatchResult>> {
        self.identify_opts(image, context, true)
    }

    /// Identify with explicit control over whether liveness/anti-spoof check is performed.
    pub fn identify_opts(
        &mut self,
        image: &RgbImage,
        context: &RecognitionContext,
        check_liveness: bool,
    ) -> Result<Option<crate::recognition::MatchResult>> {
        let face = match self.detector.detect_best(image)? {
            Some(f) => f,
            None => return Ok(None),
        };
        if check_liveness {
            self.check_live_if_enabled(image, &face.bbox)?;
        }
        let aligned = align_face(image, &face.landmarks, self.embedder.input_size());
        let embedding = self.embedder.embed(&aligned)?;
        Ok(self.store.identify(&embedding, context, self.similarity_threshold))
    }

    /// 1:1 verification between two photos (each expected to contain
    /// exactly one face). Returns a cosine similarity — does not touch the
    /// registry at all.
    pub fn verify_photos(&mut self, image_a: &RgbImage, image_b: &RgbImage) -> Result<f32> {
        self.verify_photos_opts(image_a, image_b, true)
    }

    /// 1:1 verification with explicit control over whether liveness/anti-spoof check is performed.
    pub fn verify_photos_opts(
        &mut self,
        image_a: &RgbImage,
        image_b: &RgbImage,
        check_liveness: bool,
    ) -> Result<f32> {
        let face_a = self.detector.detect_exactly_one(image_a)?;
        let face_b = self.detector.detect_exactly_one(image_b)?;
        if check_liveness {
            self.check_live_if_enabled(image_a, &face_a.bbox)?;
            self.check_live_if_enabled(image_b, &face_b.bbox)?;
        }

        let size = self.embedder.input_size();
        let aligned_a = align_face(image_a, &face_a.landmarks, size);
        let aligned_b = align_face(image_b, &face_b.landmarks, size);
        self.embedder.verify(&aligned_a, &aligned_b)
    }
}

fn auto_orient(img: image::DynamicImage, orientation: u32) -> image::DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

/// Loads a decodable image file (jpeg/png) from disk as RGB8, automatically
/// applying EXIF orientation transforms (e.g. portrait photos taken on mobile cameras).
pub fn load_image(path: impl AsRef<std::path::Path>) -> Result<RgbImage> {
    let path_ref = path.as_ref();
    let file = std::fs::File::open(path_ref)?;
    let mut reader = std::io::BufReader::new(file);

    let orientation = exif::Reader::new()
        .read_from_container(&mut reader)
        .ok()
        .and_then(|exif| exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY).cloned())
        .and_then(|field| field.value.get_uint(0))
        .unwrap_or(1);

    let raw_img = image::open(path_ref)?;
    let oriented_img = auto_orient(raw_img, orientation);
    Ok(oriented_img.into_rgb8())
}

/// Decodes an image from in-memory raw bytes (e.g. from an HTTP multipart payload) as RGB8,
/// automatically applying EXIF orientation transforms.
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<RgbImage> {
    let mut cursor = std::io::Cursor::new(bytes);
    let orientation = exif::Reader::new()
        .read_from_container(&mut cursor)
        .ok()
        .and_then(|exif| exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY).cloned())
        .and_then(|field| field.value.get_uint(0))
        .unwrap_or(1);

    let raw_img = image::load_from_memory(bytes)?;
    let oriented_img = auto_orient(raw_img, orientation);
    Ok(oriented_img.into_rgb8())
}
