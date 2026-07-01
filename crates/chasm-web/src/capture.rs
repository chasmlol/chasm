//! Request-body capture for true 1:1 trace replay.
//!
//! When `CHASM_CAPTURE_DIR` is set, every incoming request (the plugin's
//! bridge calls + any UI POST) is recorded verbatim — method, path+query,
//! content-type, and the raw body (base64) — as one JSON file per request under
//! that dir, ordered by a sequence counter. A replay harness can POST each file
//! back to a fresh chasm to reproduce the exact in-game traffic byte-for-byte.
//!
//! No-op (just one env check) when the var is unset, so it's safe to leave wired
//! into the router permanently. Static-asset GETs are skipped as noise.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use axum::{body::Body, extract::Request, http::Method, middleware::Next, response::Response};
use base64::{engine::general_purpose::STANDARD, Engine as _};

static CAPTURE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

/// The capture dir from `CHASM_CAPTURE_DIR` (resolved once). `None` ⇒ off.
fn capture_dir() -> Option<&'static PathBuf> {
    CAPTURE_DIR
        .get_or_init(|| {
            std::env::var("CHASM_CAPTURE_DIR")
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
        })
        .as_ref()
}

/// Middleware: record each request verbatim when capture is enabled, then pass it
/// through unchanged. The body is buffered so it can be both recorded and replayed
/// to the handler; the file write is fire-and-forget so capture adds ~no latency.
pub(crate) async fn capture_request(req: Request, next: Next) -> Response {
    let Some(dir) = capture_dir() else {
        return next.run(req).await;
    };
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    // Skip static assets / favicon GETs — not part of the bridge traffic.
    if method == Method::GET && (path.starts_with("/static") || path == "/favicon.ico") {
        return next.run(req).await;
    }
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| path.clone());
    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let (parts, body) = req.into_parts();
    // Buffer the whole body so we can record it AND hand it to the handler.
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => return next.run(Request::from_parts(parts, Body::empty())).await,
    };

    let seq = SEQ.fetch_add(1, Ordering::SeqCst);
    let record = serde_json::json!({
        "seq": seq,
        "method": method.as_str(),
        "path": path_and_query,
        "content_type": content_type,
        "body_b64": STANDARD.encode(&bytes),
    });
    let dir = dir.clone();
    tokio::spawn(async move {
        let _ = tokio::fs::create_dir_all(&dir).await;
        let safe = path.trim_start_matches('/').replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let safe = if safe.is_empty() { "root".to_string() } else { safe };
        let name = format!("{seq:06}_{}_{safe}.json", method.as_str());
        if let Ok(text) = serde_json::to_string(&record) {
            let _ = tokio::fs::write(dir.join(name), text).await;
        }
    });

    next.run(Request::from_parts(parts, Body::from(bytes))).await
}
