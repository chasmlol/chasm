//! UI system domain — the three OS-integration endpoints the guided
//! manual-model-placement flow (theme B) needs, since the Tauri webview can't do
//! them itself:
//!
//!   * `POST /system/open-url`    — open a URL (a Hugging Face model page) in the
//!     user's REAL default browser (`cmd /C start`).
//!   * `POST /system/open-folder` — open a model folder in Explorer. The dir is
//!     resolved SERVER-side from a fixed `kind` key (never a client path), so
//!     there is no path injection.
//!   * `POST /models/:domain/place` — accept a model file the user dropped (or
//!     picked) and MOVE it into the resolved target folder, after validating its
//!     extension. The body is the raw file bytes streamed to disk (so multi-GB
//!     GGUFs don't buffer in RAM); the filename rides `?name=`.
//!
//! Stays under `/api/ui/v1`.

use std::io::Write as _;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, State},
    Json,
};
use futures_util::StreamExt as _;
use serde::{Deserialize, Serialize};

use crate::AppState;

// ---------------------------------------------------------------------------
// open-url
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct OpenUrlBody {
    #[serde(default)]
    url: String,
}

#[derive(Serialize)]
pub(crate) struct OpResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl OpResult {
    fn ok() -> Self {
        Self { ok: true, error: None, path: None }
    }
    fn err(message: impl Into<String>) -> Self {
        Self { ok: false, error: Some(message.into()), path: None }
    }
}

/// `POST /api/ui/v1/system/open-url` — open `url` in the default browser. Only
/// http(s) URLs are accepted (so this can't be turned into a shell/file opener).
pub(crate) async fn open_url(Json(body): Json<OpenUrlBody>) -> Json<OpResult> {
    let url = body.url.trim().to_string();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Json(OpResult::err("Only http(s) URLs can be opened."));
    }
    // `start` is a cmd builtin; the empty "" is the window title so a URL with
    // spaces/quotes isn't mistaken for the title. `url` is validated http(s) above.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let spawned = std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
        match spawned {
            Ok(_) => Json(OpResult::ok()),
            Err(error) => Json(OpResult::err(format!("could not open browser: {error}"))),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = &url;
        Json(OpResult::err("open-url is only implemented on Windows."))
    }
}

// ---------------------------------------------------------------------------
// open-folder
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct OpenFolderBody {
    /// A fixed key resolved server-side to a config dir (never a client path).
    #[serde(default)]
    kind: String,
}

/// Resolves a fixed `kind` to a model folder, so open-folder / place never take a
/// client-supplied path. `llm` → the LLM models dir, `embed` → the embedder cache
/// dir, `engines` → the managed engines dir.
fn folder_for(state: &AppState, kind: &str) -> Option<std::path::PathBuf> {
    match kind {
        "llm" => Some(state.config.llm_models_dir.clone()),
        "embed" => Some(chasm_embed::embed_cache_dir()),
        "engines" => Some(state.config.engines_dir.clone()),
        _ => None,
    }
}

/// `POST /api/ui/v1/system/open-folder` — open the `kind` folder in Explorer
/// (creating it first so a fresh machine opens the right place, not an error).
pub(crate) async fn open_folder(
    State(state): State<Arc<AppState>>,
    Json(body): Json<OpenFolderBody>,
) -> Json<OpResult> {
    let Some(dir) = folder_for(&state, body.kind.trim()) else {
        return Json(OpResult::err("Unknown folder."));
    };
    let _ = std::fs::create_dir_all(&dir);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = std::process::Command::new("explorer")
            .arg(dir.display().to_string())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
    }
    Json(OpResult {
        ok: true,
        error: None,
        path: Some(dir.display().to_string()),
    })
}

// ---------------------------------------------------------------------------
// place-model (drag-drop / choose-file upload)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct PlaceQuery {
    #[serde(default)]
    name: String,
}

/// Validates a dropped/picked model filename for `domain` and returns the safe
/// basename to write. Rejects path separators (no traversal) and the wrong
/// extension (`.gguf` for the LLM, `.onnx` for the embedder). Pure + unit-tested.
pub(crate) fn validate_model_filename(domain: &str, name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("No filename provided.".to_string());
    }
    // Strip any directory components a browser might include — keep the basename.
    let base = trimmed
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(trimmed)
        .to_string();
    if base.is_empty() || base == "." || base == ".." {
        return Err("Invalid filename.".to_string());
    }
    let lower = base.to_ascii_lowercase();
    let ok = match domain {
        "llm" => lower.ends_with(".gguf"),
        "embed" | "retrieval" => lower.ends_with(".onnx"),
        _ => return Err(format!("Unknown model domain '{domain}'.")),
    };
    if !ok {
        let want = if domain == "llm" { ".gguf" } else { ".onnx" };
        return Err(format!(
            "'{base}' is not a {want} file — the {domain} model must be a {want}."
        ));
    }
    Ok(base)
}

fn place_dir(state: &AppState, domain: &str) -> Option<std::path::PathBuf> {
    match domain {
        "llm" => Some(state.config.llm_models_dir.clone()),
        "embed" | "retrieval" => Some(chasm_embed::embed_cache_dir()),
        _ => None,
    }
}

/// `POST /api/ui/v1/models/:domain/place?name=<file>` — stream the request body
/// (the raw model file bytes) to the resolved folder for `domain`, after
/// validating the extension. Streams to a `.part` then renames on success so a
/// partial upload never looks like a finished model. Returns the final path.
pub(crate) async fn place_model(
    State(state): State<Arc<AppState>>,
    Path(domain): Path<String>,
    Query(query): Query<PlaceQuery>,
    body: Body,
) -> Json<OpResult> {
    let filename = match validate_model_filename(&domain, &query.name) {
        Ok(name) => name,
        Err(error) => return Json(OpResult::err(error)),
    };
    let Some(dir) = place_dir(&state, &domain) else {
        return Json(OpResult::err(format!("Unknown model domain '{domain}'.")));
    };
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return Json(OpResult::err(format!("could not create {}: {error}", dir.display())));
    }
    let final_path = dir.join(&filename);
    let part_path = dir.join(format!("{filename}.part"));

    // Stream the body to the .part file. Per-chunk blocking writes are fine for a
    // one-off localhost import; nothing else contends for this thread.
    let mut file = match std::fs::File::create(&part_path) {
        Ok(f) => f,
        Err(error) => {
            return Json(OpResult::err(format!(
                "could not create {}: {error}",
                part_path.display()
            )))
        }
    };
    let mut stream = body.into_data_stream();
    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = std::fs::remove_file(&part_path);
                return Json(OpResult::err(format!("upload failed: {error}")));
            }
        };
        if let Err(error) = file.write_all(&chunk) {
            let _ = std::fs::remove_file(&part_path);
            return Json(OpResult::err(format!("write failed: {error}")));
        }
        total += chunk.len() as u64;
    }
    if let Err(error) = file.flush() {
        let _ = std::fs::remove_file(&part_path);
        return Json(OpResult::err(format!("flush failed: {error}")));
    }
    drop(file);
    if total == 0 {
        let _ = std::fs::remove_file(&part_path);
        return Json(OpResult::err("Uploaded file was empty."));
    }
    // Replace any existing model of the same name, then rename .part → final.
    let _ = std::fs::remove_file(&final_path);
    if let Err(error) = std::fs::rename(&part_path, &final_path) {
        let _ = std::fs::remove_file(&part_path);
        return Json(OpResult::err(format!("could not finalize: {error}")));
    }
    tracing::info!(
        "placed {domain} model '{filename}' ({total} bytes) -> {}",
        final_path.display()
    );
    Json(OpResult {
        ok: true,
        error: None,
        path: Some(final_path.display().to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::validate_model_filename;

    #[test]
    fn accepts_matching_extension() {
        assert_eq!(
            validate_model_filename("llm", "gemma-4-12b-it-UD-Q4_K_XL.gguf").unwrap(),
            "gemma-4-12b-it-UD-Q4_K_XL.gguf"
        );
        assert_eq!(
            validate_model_filename("embed", "model_optimized.onnx").unwrap(),
            "model_optimized.onnx"
        );
    }

    #[test]
    fn rejects_wrong_extension() {
        assert!(validate_model_filename("llm", "model.onnx").is_err());
        assert!(validate_model_filename("llm", "notes.txt").is_err());
        assert!(validate_model_filename("embed", "weights.gguf").is_err());
    }

    #[test]
    fn strips_directory_components_and_rejects_traversal() {
        // A browser "fullPath" is reduced to the basename.
        assert_eq!(
            validate_model_filename("llm", "C:\\Users\\me\\Downloads\\x.gguf").unwrap(),
            "x.gguf"
        );
        assert_eq!(
            validate_model_filename("llm", "sub/dir/y.gguf").unwrap(),
            "y.gguf"
        );
        assert!(validate_model_filename("llm", "").is_err());
        assert!(validate_model_filename("llm", "..").is_err());
    }

    #[test]
    fn rejects_unknown_domain() {
        assert!(validate_model_filename("tts", "x.gguf").is_err());
    }
}
