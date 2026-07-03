//! UI providers domain — the per-capability PROVIDER picker (managed-local vs a
//! hosted API) + each provider's config, for the LLM / STT / TTS settings pages.
//!
//! Each capability page lets the user choose a provider: `"local"` (the managed
//! llama.cpp / Parakeet / faster-qwen3-tts) or one of the hosted APIs from the
//! catalogs in `chasm_core::providers`. Picking an API reveals its config (key,
//! model, and — where needed — base URL / voice). This module surfaces that
//! catalog + the saved config, persists a selection/config, and reports whether
//! the capability's LOCAL runtime is installed (for the "needs the … engine"
//! hint the pages show — theme C).
//!
//! API keys are secrets: they round-trip through the settings JSON (like
//! `nexus_api_key`) and are never logged. Stays under `/api/ui/v1`.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};

use chasm_core::{ApiProviderConfig, ApiProviderDef, AppSettings};

use crate::AppState;

/// One provider option for a capability page (the managed-local option OR a
/// hosted API), plus this user's saved config for it.
#[derive(Serialize)]
pub(crate) struct UiProvider {
    pub id: String,
    pub name: String,
    /// `"local"` or `"api"`.
    pub kind: &'static str,
    pub blurb: String,
    /// Suggested model ids (datalist; editable). Empty for the local option.
    pub models: Vec<String>,
    /// Suggested voices `[{id,label}]` — TTS only.
    pub voices: Vec<UiVoice>,
    pub needs_base_url: bool,
    pub needs_voice: bool,
    pub default_base_url: String,
    pub default_model: String,
    /// OpenRouter routing choices `[{id,label}]` (empty for other providers); when
    /// present the UI shows a Price/Balanced/Speed dropdown bound to `config.routing`.
    pub routing_options: Vec<UiVoice>,
    /// This user's saved config (blank/defaults for a provider never configured).
    pub config: UiProviderConfig,
}

#[derive(Serialize)]
pub(crate) struct UiVoice {
    pub id: String,
    pub label: String,
}

/// The saved config for one provider, sent to the UI so its fields round-trip.
/// `api_key` is the user's own key on localhost (like the Nexus key field); it is
/// never logged.
#[derive(Serialize)]
pub(crate) struct UiProviderConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub voice: String,
    /// OpenRouter routing preference (`speed`/`balanced`/`price`); empty elsewhere.
    pub routing: String,
}

/// The managed-LOCAL runtime a capability needs + whether it's installed (theme C).
#[derive(Serialize)]
pub(crate) struct UiLocalRuntime {
    /// Human name, e.g. `llama.cpp` / `Parakeet engine` / `qwen3-tts engine`.
    pub name: String,
    pub installed: bool,
    /// One-line hint pointing at the Runtimes page when not installed.
    pub hint: String,
}

/// The full providers payload for one capability page.
#[derive(Serialize)]
pub(crate) struct UiProviders {
    pub capability: String,
    pub selected: String,
    pub local_runtime: UiLocalRuntime,
    pub providers: Vec<UiProvider>,
}

/// `effective_key` is the shared cross-capability key when this capability's own
/// key is blank, so the field pre-fills with a key entered elsewhere.
fn config_view(saved: Option<&ApiProviderConfig>, effective_key: String) -> UiProviderConfig {
    match saved {
        Some(c) => UiProviderConfig {
            api_key: effective_key,
            model: c.model.clone(),
            base_url: c.base_url.clone(),
            voice: c.voice.clone(),
            routing: chasm_core::normalize_openrouter_routing(&c.routing),
        },
        None => UiProviderConfig {
            api_key: effective_key,
            model: String::new(),
            base_url: String::new(),
            voice: String::new(),
            routing: chasm_core::normalize_openrouter_routing(""),
        },
    }
}

fn api_provider(
    def: &ApiProviderDef,
    saved: Option<&ApiProviderConfig>,
    effective_key: String,
) -> UiProvider {
    UiProvider {
        id: def.id.to_string(),
        name: def.name.to_string(),
        kind: "api",
        blurb: def.blurb.to_string(),
        models: def.models.iter().map(|m| m.to_string()).collect(),
        voices: def
            .voices
            .iter()
            .map(|(id, label)| UiVoice { id: id.to_string(), label: label.to_string() })
            .collect(),
        needs_base_url: def.needs_base_url,
        needs_voice: def.needs_voice,
        default_base_url: def.default_base_url.to_string(),
        default_model: def.default_model.to_string(),
        routing_options: if def.id == "openrouter" {
            chasm_core::OPENROUTER_ROUTING_OPTIONS
                .iter()
                .map(|(id, label)| UiVoice { id: id.to_string(), label: label.to_string() })
                .collect()
        } else {
            Vec::new()
        },
        config: config_view(saved, effective_key),
    }
}

fn local_provider(name: &str, blurb: &str) -> UiProvider {
    UiProvider {
        id: chasm_core::PROVIDER_LOCAL.to_string(),
        name: name.to_string(),
        kind: "local",
        blurb: blurb.to_string(),
        models: Vec::new(),
        voices: Vec::new(),
        needs_base_url: false,
        needs_voice: false,
        default_base_url: String::new(),
        default_model: String::new(),
        routing_options: Vec::new(),
        config: config_view(None, String::new()),
    }
}

/// `GET /api/ui/v1/providers/:capability` — the provider list + saved config +
/// local-runtime status for one capability page.
pub(crate) async fn get_providers(
    State(state): State<Arc<AppState>>,
    Path(capability): Path<String>,
) -> Json<UiProviders> {
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_providers(&state, &settings, &capability))
}

fn build_providers(state: &Arc<AppState>, settings: &AppSettings, capability: &str) -> UiProviders {
    match capability {
        "llm" => {
            let selected = chasm_core::normalize_llm_provider(&settings.llm.provider);
            let mut providers = vec![local_provider(
                "Local (managed llama.cpp)",
                "Runs your downloaded GGUF locally — no API key, fully offline.",
            )];
            for def in chasm_core::LLM_API_PROVIDERS {
                let cfg = settings.llm.api.get(def.id);
                providers.push(api_provider(def, cfg, settings.provider_key(cfg, def.id)));
            }
            let installed =
                crate::launcher::llamacpp_status(&state.config) == crate::launcher::RuntimeStatus::Installed;
            UiProviders {
                capability: capability.to_string(),
                selected,
                local_runtime: UiLocalRuntime {
                    name: "llama.cpp runtime".to_string(),
                    installed,
                    hint: "Install it with one click on Settings → Runtimes.".to_string(),
                },
                providers,
            }
        }
        "stt" => {
            let selected = chasm_core::normalize_stt_provider(&settings.stt.provider);
            let mut providers = vec![local_provider(
                "Local (managed Parakeet)",
                "NVIDIA Parakeet on a local GPU server — no API key, fully offline.",
            )];
            for def in chasm_core::STT_API_PROVIDERS {
                let cfg = settings.stt.api.get(def.id);
                providers.push(api_provider(def, cfg, settings.provider_key(cfg, def.id)));
            }
            let installed = crate::parakeet_engine_status(state) == "installed";
            UiProviders {
                capability: capability.to_string(),
                selected,
                local_runtime: UiLocalRuntime {
                    name: "Parakeet engine".to_string(),
                    installed,
                    hint: "Install it with one click on Settings → Runtimes.".to_string(),
                },
                providers,
            }
        }
        "tts" => {
            let selected = chasm_core::normalize_tts_provider(&settings.tts.provider);
            let mut providers = vec![local_provider(
                "Local (managed qwen3-tts)",
                "Streaming faster-qwen3-tts engine locally — no API key, fully offline.",
            )];
            for def in chasm_core::TTS_API_PROVIDERS {
                let cfg = settings.tts.api.get(def.id);
                providers.push(api_provider(def, cfg, settings.provider_key(cfg, def.id)));
            }
            let installed =
                crate::launcher::faster_qwen3_tts_installed(settings, &state.config);
            UiProviders {
                capability: capability.to_string(),
                selected,
                local_runtime: UiLocalRuntime {
                    name: "qwen3-tts engine".to_string(),
                    installed,
                    hint: "Install it with one click on Settings → Runtimes.".to_string(),
                },
                providers,
            }
        }
        _ => UiProviders {
            capability: capability.to_string(),
            selected: chasm_core::PROVIDER_LOCAL.to_string(),
            local_runtime: UiLocalRuntime {
                name: String::new(),
                installed: false,
                hint: String::new(),
            },
            providers: Vec::new(),
        },
    }
}

/// `{ "provider": "openai" }` — the selection body.
#[derive(Deserialize)]
pub(crate) struct SelectBody {
    #[serde(default)]
    provider: String,
}

/// `POST /api/ui/v1/providers/:capability/select` — set the active provider and
/// return the fresh payload. Applies the switch live (spawns/stops the local
/// server as needed) off the async path.
pub(crate) async fn select_provider(
    State(state): State<Arc<AppState>>,
    Path(capability): Path<String>,
    Json(body): Json<SelectBody>,
) -> Json<UiProviders> {
    let mut settings = AppSettings::load(&state.config.settings_path);
    let provider = body.provider.trim();
    match capability.as_str() {
        "llm" => {
            settings.llm.provider = chasm_core::normalize_llm_provider(provider);
            let _ = settings.save(&state.config.settings_path);
            // No live action: the target is resolved per request.
        }
        "stt" => {
            settings.stt.provider = chasm_core::normalize_stt_provider(provider);
            if settings.save(&state.config.settings_path).is_ok() {
                let state = Arc::clone(&state);
                tokio::task::spawn_blocking(move || {
                    crate::launcher::apply_selected_stt_provider(&state);
                });
            }
        }
        "tts" => {
            let normalized = chasm_core::normalize_tts_provider(provider);
            let to_local = normalized == chasm_core::PROVIDER_LOCAL;
            settings.tts.provider = normalized;
            if settings.save(&state.config.settings_path).is_ok() {
                let state = Arc::clone(&state);
                tokio::task::spawn_blocking(move || {
                    if to_local {
                        crate::launcher::apply_selected_tts_engine(&state);
                    } else {
                        // Free the local engine's VRAM when moving to a hosted API.
                        crate::launcher::stop_tts_engines(&state);
                    }
                });
            }
        }
        _ => {}
    }
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_providers(&state, &settings, &capability))
}

/// `{ provider, apiKey?, model?, baseUrl?, voice? }` — a provider's config edit.
/// Absent fields are left unchanged; present ones overwrite.
#[derive(Deserialize)]
pub(crate) struct ConfigBody {
    #[serde(default)]
    provider: String,
    #[serde(default, rename = "apiKey")]
    api_key: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default, rename = "baseUrl")]
    base_url: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    /// OpenRouter routing preference (`speed`/`balanced`/`price`).
    #[serde(default)]
    routing: Option<String>,
}

// ---------------------------------------------------------------------------
// TTS voice cloning over API
// ---------------------------------------------------------------------------

/// `{ "character": "Easy Pete" }` — clone that character's recorded reference into
/// the active hosted-TTS provider.
#[derive(Deserialize)]
pub(crate) struct CloneBody {
    #[serde(default)]
    character: String,
}

/// Result of an API voice-clone: the new provider voice id, or a readable error.
#[derive(Serialize)]
pub(crate) struct CloneResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `POST /api/ui/v1/tts/clone` — clone `character`'s recorded reference clip
/// (`<voices>/<character>/reference.wav`) into the ACTIVE hosted-TTS provider and
/// store the returned voice id per character, so subsequent synthesis for that
/// character uses the cloned voice. Requires an API TTS provider to be selected
/// and a reference recorded (the same reference the local clone uses).
pub(crate) async fn clone_api_voice(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CloneBody>,
) -> Json<CloneResult> {
    let err = |m: String| CloneResult { ok: false, voice_id: None, error: Some(m) };
    let character = body.character.trim().to_string();
    if character.is_empty() {
        return Json(err("No character specified.".to_string()));
    }
    let settings = AppSettings::load(&state.config.settings_path);
    let provider = chasm_core::normalize_tts_provider(&settings.tts.provider);
    if provider == chasm_core::PROVIDER_LOCAL {
        return Json(err(
            "Select a hosted TTS provider first (cloning here uses its API).".to_string(),
        ));
    }
    let Some(def) = chasm_core::tts_api_provider(&provider) else {
        return Json(err(format!("Unknown TTS provider '{provider}'.")));
    };
    let cfg = settings.tts.api.get(&provider);
    let mut resolved = chasm_core::resolve_api(def, cfg);
    resolved.api_key = settings.provider_key(cfg, &provider);
    if resolved.api_key.is_empty() {
        return Json(err(format!("{}: no API key set (Settings → TTS).", def.name)));
    }
    // The reference clip is the same one the local clone records.
    let reference = crate::active_voices_dir(&state.config)
        .join(&character)
        .join("reference.wav");
    let sample = match std::fs::read(&reference) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            return Json(err(format!(
                "No reference clip for '{character}' — record one on the voice panel first."
            )))
        }
    };
    let client = reqwest::Client::new();
    match crate::tts_api::clone_voice(&client, &provider, &resolved, &character, sample).await {
        Ok(voice_id) => {
            if let Err(error) =
                crate::save_api_voice(&state.config, &provider, &character, &voice_id)
            {
                return Json(err(format!("cloned but could not save the voice id: {error}")));
            }
            Json(CloneResult { ok: true, voice_id: Some(voice_id), error: None })
        }
        Err(error) => Json(err(error)),
    }
}

/// `GET /api/ui/v1/tts/api-voices` — the map of characters that already have a
/// cloned voice for the ACTIVE hosted-TTS provider (so the UI can show status).
pub(crate) async fn list_api_voices(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let settings = AppSettings::load(&state.config.settings_path);
    let provider = chasm_core::normalize_tts_provider(&settings.tts.provider);
    // Reuse the per-character lookup by reading the store directly for the provider.
    let path = crate::active_voices_dir(&state.config).join("api-voices.json");
    let voices = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get(&provider).cloned())
        .unwrap_or_else(|| serde_json::json!({}));
    Json(serde_json::json!({ "provider": provider, "voices": voices }))
}

/// `POST /api/ui/v1/providers/:capability/config` — persist one provider's
/// key/model/base-url/voice, then return the fresh payload.
pub(crate) async fn save_provider_config(
    State(state): State<Arc<AppState>>,
    Path(capability): Path<String>,
    Json(body): Json<ConfigBody>,
) -> Json<UiProviders> {
    let provider = body.provider.trim().to_string();
    let is_capability = matches!(capability.as_str(), "llm" | "stt" | "tts");
    if !provider.is_empty() && provider != chasm_core::PROVIDER_LOCAL && is_capability {
        let mut settings = AppSettings::load(&state.config.settings_path);

        // Phase 1 — the API KEY goes to the SHARED cross-capability store (one key
        // serves LLM/STT/TTS). Also clear any stale per-capability copies so the
        // shared key is authoritative everywhere.
        if let Some(v) = body.api_key {
            let key = v.trim().to_string();
            if key.is_empty() {
                settings.api_keys.remove(&provider);
            } else {
                settings.api_keys.insert(provider.clone(), key);
            }
            for m in [
                &mut settings.llm.api,
                &mut settings.stt.api,
                &mut settings.tts.api,
            ] {
                if let Some(e) = m.get_mut(&provider) {
                    e.api_key = String::new();
                }
            }
        }

        // Phase 2 — the per-capability fields (model / base URL / voice / routing).
        let map = match capability.as_str() {
            "llm" => &mut settings.llm.api,
            "stt" => &mut settings.stt.api,
            _ => &mut settings.tts.api,
        };
        let entry = map.entry(provider.clone()).or_default();
        if let Some(v) = body.model {
            entry.model = v.trim().to_string();
        }
        if let Some(v) = body.base_url {
            entry.base_url = v.trim().to_string();
        }
        if let Some(v) = body.voice {
            entry.voice = v.trim().to_string();
        }
        if let Some(v) = body.routing {
            entry.routing = chasm_core::normalize_openrouter_routing(&v);
        }
        let _ = settings.save(&state.config.settings_path);
    }
    let settings = AppSettings::load(&state.config.settings_path);
    Json(build_providers(&state, &settings, &capability))
}
