use axum::extract::{Multipart, Path, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::load_image_from_bytes;
use crate::pipeline::Detector;
use crate::recognition::{PersonId, RecognitionContext, RecordId, RegistrationScope};
use crate::server::error::AppError;
use crate::server::state::AppState;

use utoipa::{IntoParams, ToSchema};

use axum::response::Html;

pub async fn root_index() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceID REST API Server</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; margin-top: 0; }
        .endpoint { background: #0f172a; border: 1px solid #334155; border-radius: 8px; margin: 1rem 0; padding: 1rem; }
        .method { font-weight: bold; padding: 2px 8px; border-radius: 4px; margin-right: 8px; font-size: 0.85rem; }
        .get { background: #10b981; color: #000; }
        .post { background: #6366f1; color: #fff; }
        .delete { background: #ef4444; color: #fff; }
        a { color: #38bdf8; text-decoration: none; }
        a:hover { text-decoration: underline; }
        code { background: #334155; padding: 2px 6px; border-radius: 4px; color: #f43f5e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 FaceID REST API Server</h1>
        <p>On-device face recognition, anti-spoofing & vector search service.</p>
        <hr style="border-color: #334155;">
        <h2>Interactive Documentation & UI</h2>
        <div class="endpoint">
            <span class="method get">GET</span> <a href="/swagger-ui">/swagger-ui</a>
            <p><strong>Interactive Swagger UI API Documentation & Explorer</strong></p>
        </div>
        <h2>Available API Endpoints</h2>
        <div class="endpoint">
            <span class="method get">GET</span> <a href="/health">/health</a>
            <p>Check server health, GPU acceleration & model input status.</p>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span> <a href="/api/v1/registry/stats">/api/v1/registry/stats</a>
            <p>Get registry statistics & registered face records.</p>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/enroll</code>
            <p>Enroll a face. Multipart fields: <code>photo</code> (file), <code>person_id</code>, <code>scope</code>, <code>label</code>.</p>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/identify</code>
            <p>Identify face in image. Multipart fields: <code>photo</code> (file), <code>event</code>, <code>threshold</code>.</p>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/verify</code>
            <p>1:1 face comparison. Multipart fields: <code>photo_a</code> (file), <code>photo_b</code> (file).</p>
        </div>
        <div class="endpoint">
            <span class="method delete">DELETE</span> <code>/api/v1/registry/records/:record_id</code>
            <p>Unregister a face record by numeric ID.</p>
        </div>
    </div>
</body>
</html>"#)
}

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: &'static str,
    pub is_detector_accelerated: bool,
    pub is_antispoof_enabled: bool,
    pub embedder_input_size: u32,
}

#[utoipa::path(
    get,
    path = "/health",
    tag = "Diagnostics",
    responses(
        (status = 200, description = "Server health status and acceleration metrics", body = HealthResponse)
    )
)]
pub async fn health_check(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, AppError> {
    let guard = state.inner.lock().await;
    let is_detector_accelerated = guard.pipeline.detector().is_fully_accelerated().unwrap_or(false);
    let is_antispoof_enabled = guard.pipeline.antispoof_enabled();
    let embedder_input_size = guard.pipeline.embedder().input_size();

    Ok(Json(HealthResponse {
        status: "ok",
        is_detector_accelerated,
        is_antispoof_enabled,
        embedder_input_size,
    }))
}

fn parse_bool_str(s: &str) -> bool {
    !matches!(s.trim().to_lowercase().as_str(), "false" | "0" | "inactive" | "no" | "off")
}

#[derive(Debug, Clone, Deserialize, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct PipelineQueryParams {
    /// Enable face detection ("true"/"false", default: true)
    pub detect: Option<bool>,
    /// Alias for detect
    pub detect_face: Option<bool>,
    /// Enable anti-spoofing check ("true"/"false", default: true)
    pub antispoof: Option<bool>,
    /// Alias for antispoof
    pub liveness: Option<bool>,
    /// Enable 36-point MediaPipe face oval contour background mask ("true"/"false", default: true)
    pub mask: Option<bool>,
    /// Alias for mask
    pub apply_mask: Option<bool>,
}

impl PipelineQueryParams {
    pub fn get_detect(&self) -> Option<bool> {
        self.detect.or(self.detect_face)
    }

    pub fn get_antispoof(&self) -> Option<bool> {
        self.antispoof.or(self.liveness)
    }

    pub fn get_mask(&self) -> Option<bool> {
        self.mask.or(self.apply_mask)
    }
}

#[derive(ToSchema)]
pub struct EnrollRequest {
    /// JPEG/PNG image file containing exactly one face
    #[schema(value_type = String, format = Binary)]
    pub photo: String,
    /// Unique identifier for the person (e.g. "alice")
    pub person_id: String,
    /// Scope: "global" or an event ID (default: "global")
    pub scope: Option<String>,
    /// Free-form display name or note
    pub label: Option<String>,
    /// Enable face detection ("true"/"false", default: true)
    pub detect_face: Option<bool>,
    /// Enable anti-spoofing check ("true"/"false", default: true)
    pub antispoof: Option<bool>,
    /// Enable 36-point MediaPipe face oval contour background mask ("true"/"false", default: true)
    pub mask: Option<bool>,
}

#[derive(Serialize, ToSchema)]
pub struct EnrollResponse {
    pub record_id: u32,
    pub person_id: String,
    pub scope: String,
    pub total_registrations: usize,
}

#[utoipa::path(
    post,
    path = "/api/v1/enroll",
    tag = "Face Recognition",
    params(
        PipelineQueryParams
    ),
    request_body(content = EnrollRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Successfully enrolled face record", body = EnrollResponse),
        (status = 400, description = "No face or multiple faces detected in image"),
        (status = 422, description = "Presentation attack / spoof detected")
    )
)]
pub async fn enroll_face(
    State(state): State<AppState>,
    Query(query): Query<PipelineQueryParams>,
    mut multipart: Multipart,
) -> Result<Json<EnrollResponse>, AppError> {
    let mut photo_bytes: Option<Vec<u8>> = None;
    let mut person_id_raw: Option<String> = None;
    let mut scope_raw: String = "global".to_string();
    let mut label: Option<String> = None;
    let mut detect_face: bool = query.get_detect().unwrap_or(true);
    let mut check_liveness: bool = query.get_antispoof().unwrap_or(true);
    let mut apply_mask: bool = query.get_mask().unwrap_or(true);

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        crate::FaceError::Config(format!("Failed to parse multipart field: {e}"))
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "photo" => {
                let bytes = field.bytes().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read photo field: {e}"))
                })?;
                photo_bytes = Some(bytes.to_vec());
            }
            "person_id" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read person_id field: {e}"))
                })?;
                person_id_raw = Some(text);
            }
            "scope" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read scope field: {e}"))
                })?;
                scope_raw = text;
            }
            "label" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read label field: {e}"))
                })?;
                label = Some(text);
            }
            "detect" | "detect_face" | "check_detect" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read detect_face field: {e}"))
                })?;
                detect_face = parse_bool_str(&text);
            }
            "antispoof" | "liveness" | "check_liveness" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read antispoof field: {e}"))
                })?;
                check_liveness = parse_bool_str(&text);
            }
            "mask" | "apply_mask" | "use_mask" | "mask_face" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read mask field: {e}"))
                })?;
                apply_mask = parse_bool_str(&text);
            }
            _ => {}
        }
    }

    let photo_bytes = photo_bytes.ok_or_else(|| {
        crate::FaceError::Config("Missing required multipart field 'photo'".to_string())
    })?;
    let person_id_str = person_id_raw.ok_or_else(|| {
        crate::FaceError::Config("Missing required multipart field 'person_id'".to_string())
    })?;

    let image = load_image_from_bytes(&photo_bytes)?;

    let scope = if scope_raw.is_empty() || scope_raw == "global" {
        RegistrationScope::Global
    } else {
        RegistrationScope::Event(scope_raw.as_str().into())
    };

    let opts = crate::pipeline::PipelineOptions {
        detect_face,
        check_liveness,
        apply_mask,
    };

    let (record_id, total_registrations, record_opt) = {
        let mut guard = state.inner.lock().await;
        let record_id = guard.pipeline.enroll_full_opts(
            &image,
            PersonId::from(person_id_str.as_str()),
            scope.clone(),
            label,
            opts,
        )?;
        let total = guard.pipeline.store().len();
        let record = guard.pipeline.store().record(record_id).cloned();
        (record_id, total, record)
    };

    if let Some(record) = record_opt {
        state.append_record_to_registry(&record).await?;
    }

    Ok(Json(EnrollResponse {
        record_id,
        person_id: person_id_str,
        scope: scope_raw,
        total_registrations,
    }))
}

#[derive(ToSchema)]
pub struct IdentifyRequest {
    /// Probe image file to search against registry
    #[schema(value_type = String, format = Binary)]
    pub photo: String,
    /// Event scope context to search alongside global registry (optional)
    pub event: Option<String>,
    /// Minimum similarity score threshold (optional, default: 0.45)
    pub threshold: Option<f32>,
    /// Enable face detection ("true"/"false", default: true)
    pub detect_face: Option<bool>,
    /// Enable anti-spoofing check ("true"/"false", default: true)
    pub antispoof: Option<bool>,
    /// Enable 36-point MediaPipe face oval contour background mask ("true"/"false", default: true)
    pub mask: Option<bool>,
}

#[derive(Serialize, ToSchema)]
pub struct MatchInfo {
    pub record_id: u32,
    pub person_id: String,
    pub similarity: f32,
    pub origin: String,
    pub label: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct IdentifyResponse {
    pub matched: bool,
    pub match_result: Option<MatchInfo>,
}

#[utoipa::path(
    post,
    path = "/api/v1/identify",
    tag = "Face Recognition",
    params(
        PipelineQueryParams
    ),
    request_body(content = IdentifyRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Identification result", body = IdentifyResponse)
    )
)]
pub async fn identify_face(
    State(state): State<AppState>,
    Query(query): Query<PipelineQueryParams>,
    mut multipart: Multipart,
) -> Result<Json<IdentifyResponse>, AppError> {
    let mut photo_bytes: Option<Vec<u8>> = None;
    let mut event_raw: Option<String> = None;
    let mut threshold_raw: Option<f32> = None;
    let mut detect_face: bool = query.get_detect().unwrap_or(true);
    let mut check_liveness: bool = query.get_antispoof().unwrap_or(true);
    let mut apply_mask: bool = query.get_mask().unwrap_or(true);

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        crate::FaceError::Config(format!("Failed to parse multipart field: {e}"))
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "photo" => {
                let bytes = field.bytes().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read photo field: {e}"))
                })?;
                photo_bytes = Some(bytes.to_vec());
            }
            "event" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read event field: {e}"))
                })?;
                if !text.is_empty() {
                    event_raw = Some(text);
                }
            }
            "threshold" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read threshold field: {e}"))
                })?;
                if let Ok(val) = text.parse::<f32>() {
                    threshold_raw = Some(val);
                }
            }
            "detect" | "detect_face" | "check_detect" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read detect_face field: {e}"))
                })?;
                detect_face = parse_bool_str(&text);
            }
            "antispoof" | "liveness" | "check_liveness" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read antispoof field: {e}"))
                })?;
                check_liveness = parse_bool_str(&text);
            }
            "mask" | "apply_mask" | "use_mask" | "mask_face" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read mask field: {e}"))
                })?;
                apply_mask = parse_bool_str(&text);
            }
            _ => {}
        }
    }

    let photo_bytes = photo_bytes.ok_or_else(|| {
        crate::FaceError::Config("Missing required multipart field 'photo'".to_string())
    })?;

    let image = load_image_from_bytes(&photo_bytes)?;

    let context = match event_raw {
        Some(evt) => RecognitionContext::Event(evt.as_str().into()),
        None => RecognitionContext::GlobalOnly,
    };

    let opts = crate::pipeline::PipelineOptions {
        detect_face,
        check_liveness,
        apply_mask,
    };

    let match_opt = {
        let mut guard = state.inner.lock().await;
        guard.pipeline.identify_full_opts(&image, &context, opts)?
    };

    match match_opt {
        Some(m) if threshold_raw.is_none_or(|t| m.similarity >= t) => Ok(Json(IdentifyResponse {
            matched: true,
            match_result: Some(MatchInfo {
                record_id: m.record_id,
                person_id: m.person_id.to_string(),
                similarity: m.similarity,
                origin: m.origin.to_string(),
                label: m.label,
            }),
        })),
        _ => Ok(Json(IdentifyResponse {
            matched: false,
            match_result: None,
        })),
    }
}

#[derive(ToSchema)]
pub struct VerifyRequest {
    /// First photo file containing a face
    #[schema(value_type = String, format = Binary)]
    pub photo_a: String,
    /// Second photo file containing a face
    #[schema(value_type = String, format = Binary)]
    pub photo_b: String,
    /// Enable face detection ("true"/"false", default: true)
    pub detect_face: Option<bool>,
    /// Enable anti-spoofing check ("true"/"false", default: true)
    pub antispoof: Option<bool>,
    /// Enable 36-point MediaPipe face oval contour background mask ("true"/"false", default: true)
    pub mask: Option<bool>,
}

#[derive(Serialize, ToSchema)]
pub struct VerifyResponse {
    pub similarity: f32,
    pub is_same_person: bool,
}

#[utoipa::path(
    post,
    path = "/api/v1/verify",
    tag = "Face Recognition",
    params(
        PipelineQueryParams
    ),
    request_body(content = VerifyRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "1:1 verification similarity result", body = VerifyResponse),
        (status = 400, description = "Invalid face count in one or both photos")
    )
)]
pub async fn verify_photos(
    State(state): State<AppState>,
    Query(query): Query<PipelineQueryParams>,
    mut multipart: Multipart,
) -> Result<Json<VerifyResponse>, AppError> {
    let mut photo_a_bytes: Option<Vec<u8>> = None;
    let mut photo_b_bytes: Option<Vec<u8>> = None;
    let mut detect_face: bool = query.get_detect().unwrap_or(true);
    let mut check_liveness: bool = query.get_antispoof().unwrap_or(true);
    let mut apply_mask: bool = query.get_mask().unwrap_or(true);

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        crate::FaceError::Config(format!("Failed to parse multipart field: {e}"))
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "photo_a" => {
                let bytes = field.bytes().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read photo_a field: {e}"))
                })?;
                photo_a_bytes = Some(bytes.to_vec());
            }
            "photo_b" => {
                let bytes = field.bytes().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read photo_b field: {e}"))
                })?;
                photo_b_bytes = Some(bytes.to_vec());
            }
            "detect" | "detect_face" | "check_detect" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read detect_face field: {e}"))
                })?;
                detect_face = parse_bool_str(&text);
            }
            "antispoof" | "liveness" | "check_liveness" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read antispoof field: {e}"))
                })?;
                check_liveness = parse_bool_str(&text);
            }
            "mask" | "apply_mask" | "use_mask" | "mask_face" => {
                let text = field.text().await.map_err(|e| {
                    crate::FaceError::Config(format!("Failed to read mask field: {e}"))
                })?;
                apply_mask = parse_bool_str(&text);
            }
            _ => {}
        }
    }

    let photo_a_bytes = photo_a_bytes.ok_or_else(|| {
        crate::FaceError::Config("Missing required multipart field 'photo_a'".to_string())
    })?;
    let photo_b_bytes = photo_b_bytes.ok_or_else(|| {
        crate::FaceError::Config("Missing required multipart field 'photo_b'".to_string())
    })?;

    let img_a = load_image_from_bytes(&photo_a_bytes)?;
    let img_b = load_image_from_bytes(&photo_b_bytes)?;

    let opts = crate::pipeline::PipelineOptions {
        detect_face,
        check_liveness,
        apply_mask,
    };

    let similarity = {
        let mut guard = state.inner.lock().await;
        guard.pipeline.verify_photos_full_opts(&img_a, &img_b, opts)?
    };

    let is_same_person = similarity >= 0.45;

    Ok(Json(VerifyResponse {
        similarity,
        is_same_person,
    }))
}

#[derive(Serialize, ToSchema)]
pub struct RegistryRecordItem {
    pub record_id: u32,
    pub person_id: String,
    pub scope: String,
    pub label: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct RegistryStatsResponse {
    pub total_records: usize,
    pub records: Vec<RegistryRecordItem>,
}

#[utoipa::path(
    get,
    path = "/api/v1/registry/stats",
    tag = "Registry Management",
    responses(
        (status = 200, description = "Registry status and all registered face records", body = RegistryStatsResponse)
    )
)]
pub async fn get_registry_stats(
    State(state): State<AppState>,
) -> Result<Json<RegistryStatsResponse>, AppError> {
    let guard = state.inner.lock().await;
    let store = guard.pipeline.store();
    let records: Vec<RegistryRecordItem> = store
        .all_records()
        .map(|r| RegistryRecordItem {
            record_id: r.id,
            person_id: r.person_id.to_string(),
            scope: match &r.scope {
                RegistrationScope::Global => "global".to_string(),
                RegistrationScope::Event(ev) => format!("event:{ev}"),
            },
            label: r.label.clone(),
        })
        .collect();

    Ok(Json(RegistryStatsResponse {
        total_records: records.len(),
        records,
    }))
}

#[utoipa::path(
    delete,
    path = "/api/v1/registry/records/{record_id}",
    tag = "Registry Management",
    params(
        ("record_id" = u32, Path, description = "Numeric record ID to delete")
    ),
    responses(
        (status = 200, description = "Record successfully deleted"),
        (status = 404, description = "Record ID not found")
    )
)]
pub async fn delete_record(
    State(state): State<AppState>,
    Path(record_id): Path<RecordId>,
) -> Result<Json<serde_json::Value>, AppError> {
    {
        let mut guard = state.inner.lock().await;
        guard.pipeline.store_mut().unregister(record_id)?;
    }

    state.save_registry().await?;

    Ok(Json(json!({
        "deleted": true,
        "record_id": record_id
    })))
}
