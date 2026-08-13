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

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::thread;
use crate::types::AlignedFace;
use crate::landmarker::FaceLandmarker;

pub struct PipelineTask {
    pub kind: TaskKind,
    pub tx: tokio::sync::oneshot::Sender<Result<TaskResult>>,
}

pub enum TaskKind {
    Enroll {
        image: RgbImage,
        person_id: PersonId,
        scope: RegistrationScope,
        label: Option<String>,
        check_liveness: bool,
    },
    Identify {
        image: RgbImage,
        context: RecognitionContext,
        check_liveness: bool,
    },
    Verify {
        image_a: RgbImage,
        image_b: RgbImage,
        check_liveness: bool,
    },
}

pub enum TaskResult {
    Enroll(RecordId),
    Identify(Option<crate::recognition::MatchResult>),
    Verify(f32),
}

/// A high-concurrency worker pool holding multiple lightweight MobileFaceNet pipeline workers.
///
/// Each worker thread owns its own LiteRT models (BlazeFace + MediaPipe Mesh + MobileFaceNet),
/// allowing simultaneous parallel inference across CPU cores while sharing a single thread-safe `VectorStore`.
pub struct PipelinePool {
    sender: std::sync::mpsc::Sender<PipelineTask>,
    store: Arc<RwLock<VectorStore>>,
}

impl PipelinePool {
    pub fn new_mobilefacenet(
        worker_count: usize,
        blazeface_path: impl AsRef<Path>,
        landmarker_path: impl AsRef<Path>,
        mobilefacenet_path: impl AsRef<Path>,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<PipelineTask>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let store = Arc::new(RwLock::new(VectorStore::new()));

        let blazeface_path = blazeface_path.as_ref().to_path_buf();
        let landmarker_path = landmarker_path.as_ref().to_path_buf();
        let mobilefacenet_path = mobilefacenet_path.as_ref().to_path_buf();

        for _ in 0..worker_count {
            let rx = Arc::clone(&rx);
            let store = Arc::clone(&store);
            let blazeface_path = blazeface_path.clone();
            let landmarker_path = landmarker_path.clone();
            let mobilefacenet_path = mobilefacenet_path.clone();

            thread::spawn(move || {
                let mut detector = match BlazeFaceDetector::load(
                    &blazeface_path,
                    litert::Accelerators::CPU,
                    crate::BlazeFaceConfig::default(),
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Failed to load BlazeFace worker: {:?}", e);
                        return;
                    }
                };

                let mut landmarker = match FaceLandmarker::load(
                    &landmarker_path,
                    litert::Accelerators::CPU,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("Failed to load FaceLandmarker worker: {:?}", e);
                        return;
                    }
                };

                let mut embedder = match crate::embedder::litert_embedder::LiteRtEmbedder::load(
                    &mobilefacenet_path,
                    litert::Accelerators::CPU,
                    crate::EmbedderConfig::default(),
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("Failed to load MobileFaceNet worker: {:?}", e);
                        return;
                    }
                };

                loop {
                    let task = {
                        let lock = match rx.lock() {
                            Ok(guard) => guard,
                            Err(_) => break,
                        };
                        match lock.recv() {
                            Ok(t) => t,
                            Err(_) => break,
                        }
                    };

                    match task.kind {
                        TaskKind::Enroll { image, person_id, scope, label, check_liveness: _ } => {
                            let res = (|| -> Result<RecordId> {
                                let face = detector.detect_exactly_one(&image)?;
                                let masked = crate::align::crop_mediapipe_two_pass_face(
                                    &image,
                                    &face.bbox,
                                    &face.landmarks,
                                    &mut landmarker,
                                    0.05,
                                )?;
                                let resized = image::imageops::resize(
                                    &masked,
                                    112,
                                    112,
                                    image::imageops::FilterType::Triangle,
                                );
                                let aligned = AlignedFace {
                                    rgb: resized.into_raw(),
                                    width: 112,
                                    height: 112,
                                };
                                let embedding = embedder.embed(&aligned)?;
                                let mut store_guard = store.write().unwrap();
                                store_guard.register(person_id, embedding, scope, label)
                            })();
                            let _ = task.tx.send(res.map(TaskResult::Enroll));
                        }
                        TaskKind::Identify { image, context, check_liveness: _ } => {
                            let res = (|| -> Result<Option<crate::recognition::MatchResult>> {
                                let face = match detector.detect_best(&image)? {
                                    Some(f) => f,
                                    None => return Ok(None),
                                };
                                let masked = crate::align::crop_mediapipe_two_pass_face(
                                    &image,
                                    &face.bbox,
                                    &face.landmarks,
                                    &mut landmarker,
                                    0.05,
                                )?;
                                let resized = image::imageops::resize(
                                    &masked,
                                    112,
                                    112,
                                    image::imageops::FilterType::Triangle,
                                );
                                let aligned = AlignedFace {
                                    rgb: resized.into_raw(),
                                    width: 112,
                                    height: 112,
                                };
                                let embedding = embedder.embed(&aligned)?;
                                let store_guard = store.read().unwrap();
                                Ok(store_guard.identify(&embedding, &context, similarity_threshold))
                            })();
                            let _ = task.tx.send(res.map(TaskResult::Identify));
                        }
                        TaskKind::Verify { image_a, image_b, check_liveness: _ } => {
                            let res = (|| -> Result<f32> {
                                let face_a = detector.detect_exactly_one(&image_a)?;
                                let face_b = detector.detect_exactly_one(&image_b)?;

                                let masked_a = crate::align::crop_mediapipe_two_pass_face(
                                    &image_a,
                                    &face_a.bbox,
                                    &face_a.landmarks,
                                    &mut landmarker,
                                    0.05,
                                )?;
                                let masked_b = crate::align::crop_mediapipe_two_pass_face(
                                    &image_b,
                                    &face_b.bbox,
                                    &face_b.landmarks,
                                    &mut landmarker,
                                    0.05,
                                )?;

                                let resized_a = image::imageops::resize(
                                    &masked_a,
                                    112,
                                    112,
                                    image::imageops::FilterType::Triangle,
                                );
                                let resized_b = image::imageops::resize(
                                    &masked_b,
                                    112,
                                    112,
                                    image::imageops::FilterType::Triangle,
                                );

                                let aligned_a = AlignedFace {
                                    rgb: resized_a.into_raw(),
                                    width: 112,
                                    height: 112,
                                };
                                let aligned_b = AlignedFace {
                                    rgb: resized_b.into_raw(),
                                    width: 112,
                                    height: 112,
                                };

                                embedder.verify(&aligned_a, &aligned_b)
                            })();
                            let _ = task.tx.send(res.map(TaskResult::Verify));
                        }
                    }
                }
            });
        }

        Ok(Self { sender: tx, store })
    }

    pub fn store(&self) -> &Arc<RwLock<VectorStore>> {
        &self.store
    }

    pub async fn enroll(
        &self,
        image: RgbImage,
        person_id: impl Into<PersonId>,
        scope: RegistrationScope,
        label: Option<String>,
    ) -> Result<RecordId> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = PipelineTask {
            kind: TaskKind::Enroll {
                image,
                person_id: person_id.into(),
                scope,
                label,
                check_liveness: false,
            },
            tx,
        };
        self.sender.send(task).map_err(|_| crate::FaceError::Config("Worker pool channel closed".into()))?;
        match rx.await.map_err(|_| crate::FaceError::Config("Worker task canceled".into()))?? {
            TaskResult::Enroll(id) => Ok(id),
            _ => unreachable!(),
        }
    }

    pub async fn identify(
        &self,
        image: RgbImage,
        context: &RecognitionContext,
    ) -> Result<Option<crate::recognition::MatchResult>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = PipelineTask {
            kind: TaskKind::Identify {
                image,
                context: context.clone(),
                check_liveness: false,
            },
            tx,
        };
        self.sender.send(task).map_err(|_| crate::FaceError::Config("Worker pool channel closed".into()))?;
        match rx.await.map_err(|_| crate::FaceError::Config("Worker task canceled".into()))?? {
            TaskResult::Identify(mat) => Ok(mat),
            _ => unreachable!(),
        }
    }

    pub async fn verify_photos(
        &self,
        image_a: RgbImage,
        image_b: RgbImage,
    ) -> Result<f32> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = PipelineTask {
            kind: TaskKind::Verify {
                image_a,
                image_b,
                check_liveness: false,
            },
            tx,
        };
        self.sender.send(task).map_err(|_| crate::FaceError::Config("Worker pool channel closed".into()))?;
        match rx.await.map_err(|_| crate::FaceError::Config("Worker task canceled".into()))?? {
            TaskResult::Verify(score) => Ok(score),
            _ => unreachable!(),
        }
    }
}

