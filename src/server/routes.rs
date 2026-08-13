use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::server::handlers::*;
use crate::server::state::AppState;

#[derive(OpenApi)]
#[openapi(
    paths(
        health_check,
        enroll_face,
        identify_face,
        verify_photos,
        get_registry_stats,
        delete_record
    ),
    components(
        schemas(
            HealthResponse,
            PipelineQueryParams,
            EnrollRequest,
            EnrollResponse,
            IdentifyRequest,
            MatchInfo,
            IdentifyResponse,
            VerifyRequest,
            VerifyResponse,
            RegistryRecordItem,
            RegistryStatsResponse
        )
    ),
    tags(
        (name = "Diagnostics", description = "Server health & status metrics"),
        (name = "Face Recognition", description = "On-device face enrollment, identification & 1:1 verification"),
        (name = "Registry Management", description = "Vector store registry inspection & deletion")
    ),
    info(
        title = "FaceID REST API",
        version = "0.1.0",
        description = "Production-ready HTTP REST API service for on-device face recognition, anti-spoofing, and scoped vector search."
    )
)]
pub struct ApiDoc;

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/", get(root_index))
        .route("/health", get(health_check))
        .route("/api/v1/enroll", post(enroll_face))
        .route("/api/v1/identify", post(identify_face))
        .route("/api/v1/verify", post(verify_photos))
        .route("/api/v1/registry/stats", get(get_registry_stats))
        .route("/api/v1/registry/records/:record_id", delete(delete_record))
        .layer(cors)
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
