use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::error::FaceError;

pub struct AppError(pub FaceError);

impl From<FaceError> for AppError {
    fn from(err: FaceError) -> Self {
        AppError(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            FaceError::NoFaceDetected => (StatusCode::BAD_REQUEST, self.0.to_string()),
            FaceError::AmbiguousFaceCount { .. } => (StatusCode::BAD_REQUEST, self.0.to_string()),
            FaceError::SpoofDetected { .. } => (StatusCode::UNPROCESSABLE_ENTITY, self.0.to_string()),
            FaceError::NotRegistered { .. } => (StatusCode::NOT_FOUND, self.0.to_string()),
            FaceError::RecordNotFound(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            FaceError::ScopeMismatch { .. } => (StatusCode::BAD_REQUEST, self.0.to_string()),
            FaceError::Io(_) | FaceError::Image(_) | FaceError::Json(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                self.0.to_string(),
            ),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };

        let body = Json(json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}
