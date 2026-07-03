//! In-process two-stage semantic retriever (embed + rerank) for the Chasm
//! game-AI backend, built on [`fastembed`] (ONNX via `ort`).
//!
//! Design goals (see the crate README / phase plan):
//!
//! * **Low-end first.** Defaults to small, CPU-capable, INT8-quantized ONNX
//!   models. On a weak box the LLM owns the GPU, so embeddings + reranking must
//!   run on CPU. Hardware is detected at load time to pick the execution
//!   provider and model tier; detection never panics and falls back to CPU.
//! * **Scale up.** When built with the `cuda` feature and a capable GPU is
//!   detected, full-precision models run on the CUDA execution provider.
//! * **Embed once, cache forever.** [`EmbeddingCache`] stores vectors keyed by
//!   `sha256(model_id + text)` so unchanged content is never re-embedded, even
//!   across restarts.
//!
//! The corpus is hundreds–low-thousands of items, so retrieval is brute-force
//! cosine (top-N) followed by a cross-encoder rerank (top-K) — no vector DB.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use fastembed::{
    EmbeddingModel, RerankInitOptions, RerankerModel, TextEmbedding, TextInitOptions, TextRerank,
};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Public configuration / device
// ---------------------------------------------------------------------------

/// The execution device a [`Retriever`] resolved to at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

/// Construction config for a [`Retriever`], mirroring the persisted
/// `RetrievalSettings` (see `chasm-core`).
///
/// All tier/execution fields accept the string values the settings UI uses;
/// unknown values are treated as `"auto"`.
#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// `"auto" | "small" | "base" | "quality"`.
    pub embedder_tier: String,
    pub reranker_enabled: bool,
    /// `"auto" | "small" | "large"`.
    pub reranker_tier: String,
    /// `"auto" | "cpu" | "gpu"`.
    pub execution: String,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            embedder_tier: "auto".to_string(),
            reranker_enabled: true,
            reranker_tier: "auto".to_string(),
            execution: "auto".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Retriever
// ---------------------------------------------------------------------------

/// Shared, cloneable handle to the loaded ONNX models.
///
/// `TextEmbedding::embed` / `TextRerank::rerank` take `&mut self`, so the models
/// live behind `Mutex`es; the whole thing is `Arc`-wrapped so cloning is cheap
/// and the handle is `Send + Sync`.
#[derive(Clone)]
pub struct Retriever {
    inner: Arc<RetrieverInner>,
}

struct RetrieverInner {
    embedder: Mutex<TextEmbedding>,
    /// Present only when reranking is enabled in the config.
    reranker: Option<Mutex<TextRerank>>,
    device: Device,
    model_id: String,
}

impl Retriever {
    /// Loads (downloading + caching on first use) the embedding model and, if
    /// enabled, the reranker, choosing the execution provider + model tier from
    /// the detected hardware. Expensive (model load / first-time download) —
    /// call once and clone the handle.
    pub fn load(cfg: &RetrieverConfig) -> Result<Self> {
        let device = resolve_device(&cfg.execution);
        let cache_dir = cache_dir();
        fs::create_dir_all(&cache_dir).ok();

        let embed_model = pick_embedder(&cfg.embedder_tier, device);
        let model_id = format!("{embed_model:?}");

        let mut embed_opts = TextInitOptions::new(embed_model)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(true);
        embed_opts = apply_execution_providers(embed_opts, device);
        let embedder = TextEmbedding::try_new(embed_opts)
            .context("failed to load embedding model (first use downloads it)")?;

        let reranker = if cfg.reranker_enabled {
            let rerank_model = pick_reranker(&cfg.reranker_tier, device);
            let mut rerank_opts = RerankInitOptions::new(rerank_model)
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(true);
            rerank_opts = apply_rerank_execution_providers(rerank_opts, device);
            let reranker = TextRerank::try_new(rerank_opts)
                .context("failed to load reranker model (first use downloads it)")?;
            Some(Mutex::new(reranker))
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(RetrieverInner {
                embedder: Mutex::new(embedder),
                reranker,
                device,
                model_id,
            }),
        })
    }

    /// The execution device the embedder resolved to.
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// A stable identifier for the loaded embedding model (used as part of the
    /// cache key so a model change invalidates cached vectors).
    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    /// Embeds a single string.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text])?;
        out.pop()
            .context("embedder returned no vector for the input")
    }

    /// Embeds a batch of strings, returning one vector per input in order.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut embedder = self
            .inner
            .embedder
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder mutex poisoned"))?;
        let docs: Vec<&str> = texts.to_vec();
        let vectors = embedder.embed(docs, None).context("embedding failed")?;
        Ok(vectors)
    }

    /// Whether a real cross-encoder reranker is loaded (vs. the cosine
    /// fallback). Lets [`search`] interpret [`Self::rerank`] scores correctly:
    /// unbounded logits (reranker) vs. already-bounded cosine (fallback).
    pub fn has_reranker(&self) -> bool {
        self.inner.reranker.is_some()
    }

    /// Cross-encoder relevance scores for `query` against each candidate,
    /// aligned to `candidates` order. When the reranker is disabled this falls
    /// back to the cosine similarity of the embeddings so callers are uniform.
    pub fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        match &self.inner.reranker {
            Some(reranker) => {
                let mut reranker = reranker
                    .lock()
                    .map_err(|_| anyhow::anyhow!("reranker mutex poisoned"))?;
                let docs: Vec<&str> = candidates.to_vec();
                let results = reranker
                    .rerank(query, docs, false, None)
                    .context("rerank failed")?;
                // `rerank` returns results sorted by score; realign to input order.
                let mut scores = vec![0.0f32; candidates.len()];
                for r in results {
                    if let Some(slot) = scores.get_mut(r.index) {
                        *slot = r.score;
                    }
                }
                Ok(scores)
            }
            None => {
                // Fallback: cosine of the query embedding vs each candidate.
                let query_vec = self.embed(query)?;
                let cand_vecs = self.embed_batch(candidates)?;
                Ok(cand_vecs
                    .iter()
                    .map(|v| cosine_similarity(&query_vec, v))
                    .collect())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hardware detection + model-tier selection
// ---------------------------------------------------------------------------

/// Resolves the execution device from the `execution` setting and detected
/// hardware. Never panics; defaults to CPU when uncertain.
fn resolve_device(execution: &str) -> Device {
    match execution {
        "cpu" => Device::Cpu,
        "gpu" => {
            if cuda_available() {
                Device::Cuda
            } else {
                Device::Cpu
            }
        }
        // "auto" / unknown
        _ => {
            if cuda_available() {
                Device::Cuda
            } else {
                Device::Cpu
            }
        }
    }
}

/// Whether a usable CUDA GPU is present. Returns false unless the crate was
/// built with the `cuda` feature *and* a CUDA device is detected. Detection is
/// best-effort (probes `nvidia-smi`) and never panics.
fn cuda_available() -> bool {
    if !cfg!(feature = "cuda") {
        return false;
    }
    detect_cuda_gpu().is_some()
}

/// Best-effort free VRAM (in MiB) of the first CUDA GPU, via `nvidia-smi`.
/// `None` when no GPU / `nvidia-smi` is unavailable. Used only to pick a tier.
fn detect_cuda_gpu() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .next()
        .and_then(|line| line.trim().parse::<u64>().ok())
}

/// VRAM (MiB) at or above which we consider the GPU "strong" enough for
/// quality-tier embeddings / large reranker. ~10 GB.
const STRONG_GPU_VRAM_MIB: u64 = 10_000;

/// Picks the embedding model for the requested tier + device.
///
/// Tiers: small=BGE-small (quantized on CPU), base=BGE-base, quality=BGE-large.
/// `"auto"` -> small on CPU, base on a modest GPU, large on a strong GPU.
fn pick_embedder(tier: &str, device: Device) -> EmbeddingModel {
    match tier {
        "small" => small_embedder(device),
        "base" => base_embedder(device),
        "quality" => EmbeddingModel::BGELargeENV15,
        // "auto" / unknown
        _ => match device {
            Device::Cpu => small_embedder(device),
            Device::Cuda => {
                if detect_cuda_gpu().is_some_and(|free| free >= STRONG_GPU_VRAM_MIB) {
                    EmbeddingModel::BGELargeENV15
                } else {
                    base_embedder(device)
                }
            }
        },
    }
}

/// Small tier: the INT8-quantized BGE-small on CPU (fast, low-RAM), the
/// full-precision variant on GPU.
fn small_embedder(device: Device) -> EmbeddingModel {
    match device {
        Device::Cpu => EmbeddingModel::BGESmallENV15Q,
        Device::Cuda => EmbeddingModel::BGESmallENV15,
    }
}

/// Base tier: quantized BGE-base on CPU, full-precision on GPU.
fn base_embedder(device: Device) -> EmbeddingModel {
    match device {
        Device::Cpu => EmbeddingModel::BGEBaseENV15Q,
        Device::Cuda => EmbeddingModel::BGEBaseENV15,
    }
}

/// Picks the reranker model for the requested tier + device.
///
/// Tiers: small=jina-reranker-v1-turbo-en (light), large=bge-reranker-v2-m3.
/// `"auto"` -> small on CPU / modest GPU, large on a strong GPU.
fn pick_reranker(tier: &str, device: Device) -> RerankerModel {
    match tier {
        "small" => RerankerModel::JINARerankerV1TurboEn,
        "large" => RerankerModel::BGERerankerV2M3,
        // "auto" / unknown
        _ => match device {
            Device::Cpu => RerankerModel::JINARerankerV1TurboEn,
            Device::Cuda => {
                if detect_cuda_gpu().is_some_and(|free| free >= STRONG_GPU_VRAM_MIB) {
                    RerankerModel::BGERerankerV2M3
                } else {
                    RerankerModel::JINARerankerV1TurboEn
                }
            }
        },
    }
}

// --- Execution-provider wiring --------------------------------------------
//
// The CUDA EP types come from `ort` (not re-exported by fastembed) and only
// exist when built with the `cuda` feature. Keeping these in cfg-gated helpers
// means the default CPU build never references CUDA types, so it compiles with
// no CUDA toolkit. On CPU we leave the default (CPU) provider in place.

#[cfg(feature = "cuda")]
fn apply_execution_providers(opts: TextInitOptions, device: Device) -> TextInitOptions {
    match device {
        Device::Cuda => opts.with_execution_providers(vec![cuda_execution_provider()]),
        Device::Cpu => opts,
    }
}

#[cfg(not(feature = "cuda"))]
fn apply_execution_providers(opts: TextInitOptions, _device: Device) -> TextInitOptions {
    opts
}

#[cfg(feature = "cuda")]
fn apply_rerank_execution_providers(opts: RerankInitOptions, device: Device) -> RerankInitOptions {
    match device {
        Device::Cuda => opts.with_execution_providers(vec![cuda_execution_provider()]),
        Device::Cpu => opts,
    }
}

#[cfg(not(feature = "cuda"))]
fn apply_rerank_execution_providers(opts: RerankInitOptions, _device: Device) -> RerankInitOptions {
    opts
}

/// Builds a CUDA execution-provider dispatch. ort registers it lazily and falls
/// back to CPU at session-build time if the GPU/driver is unusable, so this is
/// safe even when detection was optimistic.
#[cfg(feature = "cuda")]
fn cuda_execution_provider() -> ort::ep::ExecutionProviderDispatch {
    ort::ep::CUDAExecutionProvider::default().build()
}

// ---------------------------------------------------------------------------
// Cache dir
// ---------------------------------------------------------------------------

/// Resolves the model/cache directory: `CHASM_EMBED_DIR` if set, else
/// `<workspace>/models/embed` discovered by walking up from the current dir.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CHASM_EMBED_DIR") {
        return PathBuf::from(dir);
    }
    workspace_root().join("models").join("embed")
}

/// Public accessor for the embed model cache directory (where the
/// `models--<org>--<repo>` weight dirs live). The web layer uses this to detect
/// per-model download status and to host the download markers.
pub fn embed_cache_dir() -> PathBuf {
    cache_dir()
}

// ---------------------------------------------------------------------------
// On-demand model download (mirrors the settings registry ids)
// ---------------------------------------------------------------------------

/// Forces the weights for a registry model `id` to be downloaded + cached,
/// reusing the same fastembed loaders the retriever uses. Constructing the model
/// triggers the hf-hub download into [`embed_cache_dir`]; we drop it immediately.
///
/// The ids mirror `chasm_core::RETRIEVAL_MODELS`. Downloads always fetch
/// the CPU/quantized variant where the tier has one, matching the default
/// (CPU-only) build's runtime selection. Returns an error for unknown ids or on
/// download/load failure.
pub fn download_model(id: &str) -> Result<()> {
    let cache_dir = cache_dir();
    fs::create_dir_all(&cache_dir).ok();

    match id {
        // Embedders. Use the same per-device pick the retriever resolves to on a
        // CPU build so the cached weights match what gets loaded at runtime.
        "bge-small" => force_embedder(EmbeddingModel::BGESmallENV15Q, &cache_dir),
        "bge-base" => force_embedder(EmbeddingModel::BGEBaseENV15Q, &cache_dir),
        "bge-large" => force_embedder(EmbeddingModel::BGELargeENV15, &cache_dir),
        // Rerankers.
        "jina-turbo" => force_reranker(RerankerModel::JINARerankerV1TurboEn, &cache_dir),
        "bge-reranker-v2-m3" => force_reranker(RerankerModel::BGERerankerV2M3, &cache_dir),
        other => Err(anyhow::anyhow!("unknown retrieval model id: {other}")),
    }
}

/// Whether the weights [`Retriever::load`] would resolve for `cfg` are ALREADY
/// present on disk (no network). Resolves the exact embedder — and, when
/// `reranker_enabled`, the exact reranker — the loader would pick for the current
/// device/tier, maps each to the `models--<org>--<repo>` directory hf-hub creates
/// under [`embed_cache_dir`], and checks it exists.
///
/// The web layer calls this BEFORE `Retriever::load` so warm-up / lazy-load never
/// triggers a download: only load when the model is already downloaded, else fall
/// back to the keyword path (the user fetches it via Settings → Retrieval).
pub fn models_present(cfg: &RetrieverConfig) -> bool {
    let cache_dir = cache_dir();
    let device = resolve_device(&cfg.execution);

    let embed_model = pick_embedder(&cfg.embedder_tier, device);
    let Some(embed_code) = TextEmbedding::get_model_info(&embed_model)
        .ok()
        .map(|info| info.model_code.clone())
    else {
        return false;
    };
    if !cache_dir.join(model_code_to_cache_dir(&embed_code)).is_dir() {
        return false;
    }

    if cfg.reranker_enabled {
        let rerank_model = pick_reranker(&cfg.reranker_tier, device);
        let rerank_code = TextRerank::get_model_info(&rerank_model).model_code;
        if !cache_dir.join(model_code_to_cache_dir(&rerank_code)).is_dir() {
            return false;
        }
    }

    true
}

/// Maps a fastembed `model_code` (`<org>/<repo>`) to the `models--<org>--<repo>`
/// directory hf-hub creates under the cache dir. Mirrors `RetrievalModelDef.cache_dir`.
fn model_code_to_cache_dir(model_code: &str) -> String {
    format!("models--{}", model_code.replace('/', "--"))
}

/// Constructs (and drops) a `TextEmbedding` to force its weights to download.
fn force_embedder(model: EmbeddingModel, cache_dir: &Path) -> Result<()> {
    let opts = TextInitOptions::new(model)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_show_download_progress(true);
    let _embedder = TextEmbedding::try_new(opts).context("downloading embedding model")?;
    Ok(())
}

/// Constructs (and drops) a `TextRerank` to force its weights to download.
fn force_reranker(model: RerankerModel, cache_dir: &Path) -> Result<()> {
    let opts = RerankInitOptions::new(model)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_show_download_progress(true);
    let _reranker = TextRerank::try_new(opts).context("downloading reranker model")?;
    Ok(())
}

/// Walks up from the current dir to the workspace root (the dir holding the
/// top-level `Cargo.toml`). Falls back to the current dir.
fn workspace_root() -> PathBuf {
    let start = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for candidate in start.ancestors() {
        if candidate.join("Cargo.toml").is_file() && candidate.join("static").is_dir() {
            return candidate.to_path_buf();
        }
    }
    start
}

// ---------------------------------------------------------------------------
// Cosine + brute-force search
// ---------------------------------------------------------------------------

/// Cosine similarity of two equal-length vectors. Returns 0.0 for mismatched
/// lengths or zero-norm inputs.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Brute-force two-stage search:
///
/// 1. Cosine of the query embedding against every candidate vector; keep the
///    `top_n` best.
/// 2. Rerank those `top_n` by text with the cross-encoder (or cosine fallback).
/// 3. Return the `top_k` `(id, score)` sorted by score desc, filtered to scores
///    `>= min_score`.
pub fn search(
    retriever: &Retriever,
    query: &str,
    candidates: &[(String, Vec<f32>, String)],
    top_n: usize,
    top_k: usize,
    min_score: f32,
) -> Result<Vec<(String, f32)>> {
    let query_vec = retriever.embed(query)?;
    search_with_query_vec(retriever, &query_vec, query, candidates, top_n, top_k, min_score)
}

/// [`search`] with a PRECOMPUTED query vector. Every retrieval subsystem (lore,
/// chat memory, quests, actions) queries with the same turn text; embedding it
/// once per turn instead of once per subsystem removes 3-4 CPU ONNX inferences
/// from every prompt assembly.
pub fn search_with_query_vec(
    retriever: &Retriever,
    query_vec: &[f32],
    query: &str,
    candidates: &[(String, Vec<f32>, String)],
    top_n: usize,
    top_k: usize,
    min_score: f32,
) -> Result<Vec<(String, f32)>> {
    if candidates.is_empty() || top_n == 0 || top_k == 0 {
        return Ok(Vec::new());
    }

    // Stage 1: cosine recall.
    let query_vec = query_vec.to_vec();
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (_, vec, _))| (i, cosine_similarity(&query_vec, vec)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(top_n);

    // Stage 2: rerank the recalled candidates by text.
    let texts: Vec<&str> = scored
        .iter()
        .map(|(i, _)| candidates[*i].2.as_str())
        .collect();
    let rerank_scores = retriever.rerank(query, &texts)?;

    // Normalize to a 0..1 relevance so `min_score` is a meaningful knob — but the
    // mapping DEPENDS on which scorer ran:
    //   * Real cross-encoder reranker: unbounded logits (very negative for
    //     irrelevant, positive for relevant) -> sigmoid gives clean 0..1
    //     separation (irrelevant ~0.0, relevant ~0.8+).
    //   * Cosine fallback (reranker disabled): scores are ALREADY cosine
    //     similarities (~0.4..0.8 for BGE, which has a high baseline). Sigmoiding
    //     those compresses the whole useful range into ~0.60..0.69, so `min_score`
    //     can no longer separate relevant from irrelevant and every book dumps its
    //     full top-K. So pass cosine through (clamped to >= 0) instead.
    // Without this split, the no-reranker path silently over-injects everything.
    let has_reranker = retriever.has_reranker();
    let normalize = |x: f32| {
        if has_reranker {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.max(0.0)
        }
    };
    let mut reranked: Vec<(String, f32)> = scored
        .iter()
        .enumerate()
        .map(|(pos, (i, _))| {
            let score = normalize(rerank_scores.get(pos).copied().unwrap_or(0.0));
            (candidates[*i].0.clone(), score)
        })
        .collect();

    reranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    reranked.truncate(top_k);
    reranked.retain(|(_, score)| *score >= min_score);
    Ok(reranked)
}

// ---------------------------------------------------------------------------
// Persistent embedding cache
// ---------------------------------------------------------------------------

/// Directory-backed embedding cache with an in-process memory layer. Vectors
/// are stored as JSON files named by `sha256(model_id + text)`, so the same
/// content embeds exactly once and the cache survives restarts; hot vectors are
/// additionally memoized in RAM, because re-reading hundreds of small vector
/// files per TURN (lore candidates + chat-memory history) measured ~0.35s of
/// prompt-assembly time on Windows. Keying on the model id means switching
/// models won't return stale vectors.
#[derive(Clone)]
pub struct EmbeddingCache {
    dir: PathBuf,
    memory: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<f32>>>>>,
}

impl EmbeddingCache {
    /// Opens (creating if needed) a cache rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating embedding cache dir {}", dir.display()))?;
        Ok(Self {
            dir,
            memory: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        })
    }

    /// Memoizes `vec` under `key` (best-effort: a poisoned lock skips the memo).
    fn remember(&self, key: &str, vec: &std::sync::Arc<Vec<f32>>) {
        if let Ok(mut memory) = self.memory.lock() {
            memory.insert(key.to_string(), std::sync::Arc::clone(vec));
        }
    }

    /// Returns the cached vector for `text` (keyed by `sha256(model_id + text)`),
    /// checking RAM first, then disk, embedding and storing on a full miss.
    pub fn get_or_embed(&self, retriever: &Retriever, text: &str) -> Result<Vec<f32>> {
        let key = cache_key(retriever.model_id(), text);
        if let Ok(memory) = self.memory.lock() {
            if let Some(vec) = memory.get(&key) {
                return Ok(vec.as_ref().clone());
            }
        }
        let path = self.dir.join(format!("{key}.json"));
        if let Ok(bytes) = fs::read(&path) {
            if let Ok(vec) = serde_json::from_slice::<Vec<f32>>(&bytes) {
                let vec = std::sync::Arc::new(vec);
                self.remember(&key, &vec);
                return Ok(vec.as_ref().clone());
            }
            // Corrupt entry: fall through and re-embed/overwrite.
        }
        let vec = std::sync::Arc::new(retriever.embed(text)?);
        if let Ok(json) = serde_json::to_vec(vec.as_ref()) {
            // Best-effort write; a failed cache write must not fail the call.
            let _ = fs::write(&path, json);
        }
        self.remember(&key, &vec);
        Ok(vec.as_ref().clone())
    }

    /// Pre-warms the cache for `texts`, embedding only the misses in batches.
    /// Far faster than `get_or_embed` in a loop (one model call per batch vs per
    /// item) — used to vectorize large catalogs (~8k items) up front so the first
    /// catalog search doesn't stall. Best-effort: returns how many were embedded.
    pub fn warm_batch(&self, retriever: &Retriever, texts: &[String], batch_size: usize) -> usize {
        let model_id = retriever.model_id();
        let missing: Vec<&String> = texts
            .iter()
            .filter(|text| {
                if text.trim().is_empty() {
                    return false;
                }
                let key = cache_key(model_id, text);
                !self.dir.join(format!("{key}.json")).exists()
            })
            .collect();
        let mut embedded = 0usize;
        for chunk in missing.chunks(batch_size.max(1)) {
            let refs: Vec<&str> = chunk.iter().map(|text| text.as_str()).collect();
            let Ok(vectors) = retriever.embed_batch(&refs) else {
                continue;
            };
            for (text, vector) in chunk.iter().zip(vectors) {
                let key = cache_key(model_id, text);
                let path = self.dir.join(format!("{key}.json"));
                if let Ok(json) = serde_json::to_vec(&vector) {
                    let _ = fs::write(&path, json);
                    embedded += 1;
                }
            }
        }
        embedded
    }
}

/// `sha256(model_id + "\0" + text)` as lowercase hex.
fn cache_key(model_id: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(model_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
        // Mismatched / empty -> 0.0, never panics.
        assert_eq!(cosine_similarity(&a, &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cache_key_is_model_and_text_sensitive() {
        let k1 = cache_key("model-a", "hello");
        let k2 = cache_key("model-a", "hello");
        let k3 = cache_key("model-a", "world");
        let k4 = cache_key("model-b", "hello");
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k1, k4);
    }

    /// Smoke test for the real models. Ignored by default because it downloads
    /// model weights (a few hundred MB) on first run.
    ///
    /// Run with:
    ///   cargo test -p chasm-embed -- --ignored --nocapture
    ///
    /// Verifies (1) related sentences score higher than unrelated ones via
    /// embeddings, and (2) the reranker puts the on-topic passage first.
    #[test]
    #[ignore = "downloads model weights on first run"]
    fn smoke_related_beats_unrelated_and_reranker_orders() {
        let retriever = Retriever::load(&RetrieverConfig::default()).expect("load retriever");

        let query = "How do I repair my weapon in the wasteland?";
        let related = "Use a weapon repair kit or pay a vendor to fix your gun.";
        let unrelated = "The capital of France is Paris and it is very pretty.";

        let q = retriever.embed(query).expect("embed query");
        let r = retriever.embed(related).expect("embed related");
        let u = retriever.embed(unrelated).expect("embed unrelated");
        let sim_related = cosine_similarity(&q, &r);
        let sim_unrelated = cosine_similarity(&q, &u);
        println!("cosine related={sim_related} unrelated={sim_unrelated}");
        assert!(
            sim_related > sim_unrelated,
            "related ({sim_related}) should beat unrelated ({sim_unrelated})"
        );

        // Reranker should rank the related passage above the unrelated one.
        let scores = retriever
            .rerank(query, &[unrelated, related])
            .expect("rerank");
        println!("rerank scores [unrelated, related] = {scores:?}");
        assert!(
            scores[1] > scores[0],
            "reranker should score related ({}) above unrelated ({})",
            scores[1],
            scores[0]
        );

        // End-to-end brute-force search returns the related item first.
        let candidates = vec![
            ("unrelated".to_string(), u.clone(), unrelated.to_string()),
            ("related".to_string(), r.clone(), related.to_string()),
        ];
        let hits = search(&retriever, query, &candidates, 10, 5, f32::MIN).expect("search");
        println!("search hits = {hits:?}");
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some("related"));
    }

    /// Per-turn cost with a LARGE corpus (300 lore entries), to show the reranker
    /// is bounded by top-N (not corpus size) and what actually scales with 300.
    /// Run: cargo test -p chasm-embed --release -- --ignored --nocapture corpus_300
    #[test]
    #[ignore = "loads model weights; diagnostic"]
    fn corpus_300_per_turn_cost() {
        let cfg = RetrieverConfig {
            embedder_tier: "small".to_string(),
            reranker_enabled: true,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
        };
        let retriever = Retriever::load(&cfg).expect("load");
        let query = "tell me about joe cobb and the powder gangers near goodsprings";
        let docs: Vec<String> = (0..300)
            .map(|i| format!("Goodsprings lore entry {i}: a paragraph of descriptive worldbuilding text about people, places, factions, and history in the Mojave near Goodsprings."))
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();

        let dir = std::env::temp_dir().join("sb_embed_cache_corpus300");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = EmbeddingCache::open(&dir).expect("cache");

        // ONE-TIME: embed + cache all 300 (happens once ever, then cached on disk).
        let t = std::time::Instant::now();
        for d in &refs {
            cache.get_or_embed(&retriever, d).unwrap();
        }
        println!(
            "ONE-TIME  embed+cache 300 entries:        {:.0} ms (happens once, then cached)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // PER-TURN stage 1: embed query + read 300 cached vectors + cosine + sort.
        let t = std::time::Instant::now();
        let qv = retriever.embed(query).unwrap();
        let vecs: Vec<Vec<f32>> = refs
            .iter()
            .map(|d| cache.get_or_embed(&retriever, d).unwrap())
            .collect();
        let mut scored: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(&qv, v)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "STAGE 1   per-turn (embed q + recall 300): {:.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // PER-TURN stage 2: rerank ONLY the top-N (independent of the 300).
        for n in [20usize, 40] {
            let top: Vec<&str> = scored.iter().take(n).map(|(i, _)| refs[*i]).collect();
            let t = std::time::Instant::now();
            let _ = retriever.rerank(query, &top).unwrap();
            println!(
                "STAGE 2   rerank top-{n:<2} (of 300):          {:.1} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Breaks the two-stage search into its parts so the dominant cost is clear:
    /// stage 1 = embed query + cosine recall (candidate embeds are CACHED, so this
    /// is ~free after the first turn); stage 2 = cross-encoder rerank (per-turn,
    /// uncacheable). Run: cargo test -p chasm-embed --release -- --ignored
    /// --nocapture stage_breakdown
    #[test]
    #[ignore = "loads model weights; diagnostic"]
    fn stage_breakdown() {
        let cfg = RetrieverConfig {
            embedder_tier: "small".to_string(),
            reranker_enabled: true,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
        };
        let retriever = Retriever::load(&cfg).expect("load");
        let query = "tell me about joe cobb and the powder gangers near goodsprings";
        let docs: Vec<String> = (0..40)
            .map(|i| format!("Goodsprings lore/action candidate entry number {i} with some descriptive text."))
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();

        // Warm both models once (exclude one-time init from the numbers).
        let _ = retriever.embed(query).unwrap();
        let _ = retriever.rerank(query, &refs).unwrap();

        // Stage 1a: embed the query (one short text) — happens every turn.
        let t = std::time::Instant::now();
        let qv = retriever.embed(query).unwrap();
        println!(
            "stage1  embed query (1 text):      {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Stage 1b: embed 40 candidates — but this is CACHED in production, so it
        // only happens the first time each entry is seen, never again.
        let t = std::time::Instant::now();
        let cvs = retriever.embed_batch(&refs).unwrap();
        println!(
            "stage1  embed 40 candidates COLD:  {:.2} ms  (cached after 1st turn -> ~0)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Stage 1c: cosine recall over 40 cached vectors — every turn, but trivial.
        let t = std::time::Instant::now();
        let mut _sink = 0.0f32;
        for v in &cvs {
            _sink += cosine_similarity(&qv, v);
        }
        println!(
            "stage1  cosine over 40 vectors:    {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // Stage 2: rerank 40 candidates — every turn, uncacheable.
        let t = std::time::Instant::now();
        let _ = retriever.rerank(query, &refs).unwrap();
        println!(
            "stage2  rerank 40 candidates:      {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }

    /// Measures reranker latency at realistic per-turn candidate counts (CPU).
    /// Run: cargo test -p chasm-embed --release -- --ignored --nocapture rerank_latency
    #[test]
    #[ignore = "loads model weights; diagnostic"]
    fn rerank_latency_scaling() {
        let cfg = RetrieverConfig {
            embedder_tier: "small".to_string(),
            reranker_enabled: true,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
        };
        let retriever = Retriever::load(&cfg).expect("load");
        let query = "tell me about joe cobb and the powder gangers near goodsprings";
        // Short (action/lore-ish) and long (chat-message-ish) candidate texts.
        let short =
            "Easy Pete guards the town dynamite and won't hand it out lightly. ai.sandbox_here";
        let long = "Player: I was walking through Goodsprings earlier and ran into Trudy at the saloon, \
            she mentioned the Powder Gangers have been causing trouble and that Ringo is hiding out at \
            the gas station, then Sunny Smiles offered to teach me how to shoot geckos by the spring.";
        for &(label, text, n) in &[
            ("10 short", short, 10usize),
            ("40 short", short, 40),
            ("40 long", long, 40),
            ("80 long", long, 80),
        ] {
            let docs: Vec<String> = (0..n).map(|i| format!("{text} (#{i})")).collect();
            let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            let _ = retriever.rerank(query, &refs).expect("warm");
            let t0 = std::time::Instant::now();
            let _ = retriever.rerank(query, &refs).expect("rerank");
            println!(
                "rerank {label:<10} ({n:>3} items): {:.1} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    /// Reproduces the "tell me about joe cobb dumps the whole book" report and
    /// proves the reranker fixes it. Prints, for each lore/action candidate, the
    /// raw cosine (no-reranker path) vs. the reranker's sigmoid score, so the
    /// `min_score` threshold can be picked from real numbers. Also times rerank.
    ///
    /// Run: cargo test -p chasm-embed -- --ignored --nocapture joe_cobb
    #[test]
    #[ignore = "downloads/loads model weights; diagnostic"]
    fn joe_cobb_reranker_separates_relevant_from_book_dump() {
        let cfg = RetrieverConfig {
            embedder_tier: "small".to_string(),
            reranker_enabled: true,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
        };
        let retriever = Retriever::load(&cfg).expect("load retriever w/ reranker");
        assert!(retriever.has_reranker(), "reranker should be loaded");

        let query = "tell me about joe cobb";
        // (id, vector_text) approximating the FNV books from the screenshots.
        let items: &[(&str, &str)] = &[
            // The one genuinely on-topic lore entry.
            ("lore:joe-cobb", "Joe Cobb and the Powder Ganger threat. Joe Cobb is a Powder Ganger leader menacing Goodsprings, planning to attack the town."),
            // Loosely-related Goodsprings lore that was wrongly injected.
            ("lore:easy-pete-explosives", "Opinion: Easy Pete on explosives. Easy Pete guards the town's dynamite and won't hand it out lightly."),
            ("lore:ghost-town-gunfight", "Ghost Town Gunfight: the quest to defend Goodsprings from the Powder Gangers."),
            ("lore:chet-store", "Chet and the General Store. Chet runs the Goodsprings general store and stocks supplies."),
            ("lore:ringo-gas", "Ringo at the gas station. Ringo is a courier hiding in the abandoned gas station."),
            ("lore:prospector-saloon", "Prospector Saloon, the social hub of Goodsprings run by Trudy."),
            ("lore:easy-pete-ncr", "Opinion: Easy Pete on NCR and Legion politics in the Mojave."),
            // Action-book entries (none relevant to a pure chat question).
            ("action:spawn_item", "Spawn an item at a player or nearby NPC anchor. world.spawn_item"),
            ("action:spawn_entity", "Spawn an NPC or creature at a player or nearby anchor. world.spawn_entity"),
            ("action:combat_start", "Start combat with a target. combat.start attack"),
            ("action:combat_stop", "Stop combat with the player. combat.stop"),
            ("action:sandbox", "Sandbox around current position. ai.sandbox_here"),
            ("action:sit", "Sit down or use nearby furniture. ai.sit_down"),
            ("action:smoke", "Smoke idle gesture. npc.gesture_smoke"),
        ];

        let cosines: Vec<f32> = {
            let q = retriever.embed(query).expect("embed q");
            items
                .iter()
                .map(|(_, t)| cosine_similarity(&q, &retriever.embed(t).expect("embed t")))
                .collect()
        };

        let texts: Vec<&str> = items.iter().map(|(_, t)| *t).collect();
        let t0 = std::time::Instant::now();
        let logits = retriever.rerank(query, &texts).expect("rerank");
        let rerank_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

        let mut rows: Vec<(&str, f32, f32)> = items
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, cosines[i], sigmoid(logits[i])))
            .collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));

        println!(
            "\nquery: {query:?}   (rerank {} items in {rerank_ms:.1} ms)",
            items.len()
        );
        println!("{:<28} {:>10} {:>12}", "id", "cosine", "rerank(sig)");
        for (id, cos, rr) in &rows {
            println!("{id:<28} {cos:>10.3} {rr:>12.3}");
        }

        // The reranker must put the on-topic entry well above the irrelevant ones,
        // and the irrelevant action entries must score low (so a sane min_score
        // drops them). Cosine alone (no reranker) does NOT achieve this margin.
        let joe = rows.iter().find(|(id, ..)| *id == "lore:joe-cobb").unwrap();
        let worst_action = rows
            .iter()
            .filter(|(id, ..)| id.starts_with("action:"))
            .map(|(_, _, rr)| *rr)
            .fold(0.0f32, f32::max);
        println!(
            "\njoe-cobb rerank={:.3}  best-action rerank={:.3}",
            joe.2, worst_action
        );
        assert!(
            joe.2 > worst_action + 0.2,
            "reranker should rank joe-cobb (rr={:.3}) clearly above any action (rr={:.3})",
            joe.2,
            worst_action
        );
    }

    /// Prints the reranker relevance of the FNV action set for a GENERIC message
    /// ("What?") vs targeted requests, so `action_min_score` can be chosen to stop
    /// generic chat from offering unrelated gestures.
    /// Run: cargo test -p chasm-embed -- --ignored --nocapture action_min_score_probe
    #[test]
    #[ignore = "downloads/loads model weights; diagnostic"]
    fn action_min_score_probe() {
        let cfg = RetrieverConfig {
            embedder_tier: "small".to_string(),
            reranker_enabled: true,
            reranker_tier: "small".to_string(),
            execution: "cpu".to_string(),
        };
        let retriever = Retriever::load(&cfg).expect("load retriever");
        let actions: &[(&str, &str)] = &[
            ("combat.start", "Start combat with a target. attack fight hostile."),
            ("movement.follow_target", "Follow the player. follow me come with escort."),
            ("npc.gesture_wave", "Wave hello, say hello, greet gesture."),
            ("ai.sit_down", "Sit down or use nearby furniture."),
            ("ai.wait_here", "Wait here, stay here, hold position."),
            ("npc.gesture_sneeze", "Sneeze idle gesture. achoo, act sick."),
            ("npc.gesture_pushups", "Do pushups, exercise."),
            ("npc.gesture_look_around", "Look around, scan the area, keep watch."),
            ("npc.gesture_scratch", "Scratch self, scratch yourself."),
            ("npc.gesture_hands_up", "Hands up harmless, surrender."),
            ("npc.gesture_finger_up", "Finger up emphasis, one second."),
            ("npc.gesture_give_item", "Give an item to the player."),
        ];
        let texts: Vec<&str> = actions.iter().map(|(_, t)| *t).collect();
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        for query in ["What?", "wave at me", "follow me", "sit down please", "let's fight them"] {
            let logits = retriever.rerank(query, &texts).expect("rerank");
            let mut rows: Vec<(&str, f32)> =
                actions.iter().enumerate().map(|(i, (id, _))| (*id, sigmoid(logits[i]))).collect();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("\nquery {query:?}:");
            for (id, s) in &rows {
                println!("  {s:>6.3}  {id}");
            }
        }
    }
}
