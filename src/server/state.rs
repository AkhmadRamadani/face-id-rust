use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::embedder::Embedder;
use crate::pipeline::FacePipeline;
use crate::recognition::persistence;
use crate::error::Result;

pub struct AppStateInner {
    pub pipeline: FacePipeline<Box<dyn Embedder>>,
    pub registry_path: PathBuf,
    pub auto_save: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Mutex<AppStateInner>>,
}

impl AppState {
    pub fn new(
        pipeline: FacePipeline<Box<dyn Embedder>>,
        registry_path: PathBuf,
        auto_save: bool,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AppStateInner {
                pipeline,
                registry_path,
                auto_save,
            })),
        }
    }

    /// Automatically flushes current vector store to registry JSON file if auto_save is enabled.
    pub async fn save_registry(&self) -> Result<()> {
        let guard = self.inner.lock().await;
        if guard.auto_save {
            persistence::save(guard.pipeline.store(), &guard.registry_path)?;
        }
        Ok(())
    }
}
