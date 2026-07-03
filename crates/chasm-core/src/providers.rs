//! Hosted-API provider catalogs for the three AI capabilities (LLM / STT / TTS),
//! plus the persisted per-provider credential shape.
//!
//! Each capability page lets the user pick a PROVIDER: the managed-local option
//! (`"local"` — llama.cpp / Parakeet / faster-qwen3-tts, always the default) or
//! one of the hosted APIs below. Picking an API reveals its config (key, model,
//! and — where needed — base URL / voice); the runtime then routes generation /
//! transcription / synthesis to the selected provider transparently.
//!
//! API keys are secrets: they are stored in the settings JSON exactly like the
//! existing `LauncherSettings::nexus_api_key`, and MUST never be logged. The
//! provider catalogs only carry NON-secret metadata (ids, default models, base
//! URLs), so they are safe to serialize to the UI.
//!
//! Model / voice ids are given as EDITABLE suggestions (rendered as a datalist in
//! the UI), never as a hard allow-list — hosted providers rotate their ids often,
//! so a user can always type the exact current id even if a suggestion is stale.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The sentinel provider id for the managed-local option of any capability
/// (llama.cpp for the LLM, the Parakeet engine for STT, faster-qwen3-tts for
/// TTS). Selecting it keeps chasm's original no-API behaviour.
pub const PROVIDER_LOCAL: &str = "local";

/// One hosted-API provider's persisted config for a single capability. Stored per
/// provider id under each capability's `api` map, so switching providers keeps
/// each one's key/model without re-entering it.
///
/// `api_key` is a SECRET — persisted plaintext in the settings JSON (the same way
/// `nexus_api_key` already is) and never written to logs or traces.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiProviderConfig {
    /// The provider API key / token. Secret; never logged.
    pub api_key: String,
    /// The selected model id (free text; empty falls back to the catalog default).
    pub model: String,
    /// Base-URL override. Required for the generic OpenAI-compatible provider;
    /// empty falls back to the catalog default for the named providers.
    pub base_url: String,
    /// Selected voice id/name (TTS only; ignored for LLM/STT).
    pub voice: String,
    /// OpenRouter provider-routing preference: `"price"` (cheapest), `"balanced"`
    /// (OpenRouter's default load-balancing), or `"speed"` (fastest tok/s). Empty
    /// defaults to `"speed"` — chasm's real-time NPC dialogue favours speed unless
    /// the user picks otherwise. Ignored by non-OpenRouter providers.
    pub routing: String,
}

/// OpenRouter routing options `(value, label)` for the settings dropdown.
pub const OPENROUTER_ROUTING_OPTIONS: &[(&str, &str)] = &[
    ("speed", "Speed (fastest provider)"),
    ("balanced", "Balanced (OpenRouter default)"),
    ("price", "Price (cheapest provider)"),
];

/// Normalizes a stored routing preference to a known value, defaulting to
/// `"speed"` (fast by default for real-time dialogue).
pub fn normalize_openrouter_routing(value: &str) -> String {
    match value.trim() {
        "price" => "price".to_string(),
        "balanced" => "balanced".to_string(),
        _ => "speed".to_string(),
    }
}

/// Static, non-secret description of a hosted-API provider for one capability.
#[derive(Debug, Clone, Copy)]
pub struct ApiProviderDef {
    /// Stable id persisted as the capability's `provider` value and `api` map key.
    pub id: &'static str,
    /// Human label for the picker.
    pub name: &'static str,
    /// Default base URL. Empty ⇒ the user MUST supply one (generic compat).
    pub default_base_url: &'static str,
    /// Default model id used when the stored config's `model` is blank.
    pub default_model: &'static str,
    /// Suggested model ids (rendered as a datalist; not an allow-list).
    pub models: &'static [&'static str],
    /// Suggested voices `(id, label)` — TTS only.
    pub voices: &'static [(&'static str, &'static str)],
    /// The base URL is user-required (the generic OpenAI-compatible provider).
    pub needs_base_url: bool,
    /// A voice selection is required (TTS providers).
    pub needs_voice: bool,
    /// One-line help shown under the provider config in the UI.
    pub blurb: &'static str,
}

// ---------------------------------------------------------------------------
// LLM providers
// ---------------------------------------------------------------------------

/// Hosted LLM providers offered alongside the managed-local llama.cpp option.
///
/// OpenAI / OpenRouter / the generic compat option all speak the OpenAI
/// `/v1/chat/completions` shape the local path already uses, differing only by
/// base URL + auth + model id. Anthropic (Messages) and Gemini (generateContent)
/// have their own request/response shapes and get dedicated adapters.
pub const LLM_API_PROVIDERS: &[ApiProviderDef] = &[
    ApiProviderDef {
        id: "openai",
        name: "OpenAI",
        default_base_url: "https://api.openai.com/v1",
        default_model: "gpt-5.1-mini",
        models: &["gpt-5.1", "gpt-5.1-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "OpenAI Chat Completions. Structured replies use JSON mode.",
    },
    ApiProviderDef {
        id: "anthropic",
        name: "Anthropic (Claude)",
        default_base_url: "https://api.anthropic.com/v1",
        default_model: "claude-sonnet-5",
        models: &["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "Anthropic Messages API. Structured replies via a JSON prefill.",
    },
    ApiProviderDef {
        id: "gemini",
        name: "Google Gemini",
        default_base_url: "https://generativelanguage.googleapis.com/v1beta",
        default_model: "gemini-2.5-flash",
        models: &["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "Google Gemini generateContent. Structured replies via JSON mime.",
    },
    ApiProviderDef {
        id: "openrouter",
        name: "OpenRouter",
        default_base_url: "https://openrouter.ai/api/v1",
        default_model: "openai/gpt-5.1-mini",
        models: &[
            "openai/gpt-5.1-mini",
            "anthropic/claude-sonnet-5",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "OpenRouter — one key, many models (OpenAI-compatible).",
    },
    ApiProviderDef {
        id: "openai_compat",
        name: "OpenAI-compatible (custom)",
        default_base_url: "",
        default_model: "",
        models: &[],
        voices: &[],
        needs_base_url: true,
        needs_voice: false,
        blurb: "Any OpenAI-compatible server — Groq, Together, LM Studio, vLLM, a \
                local proxy. Set the base URL (ending in /v1) and model id.",
    },
];

// ---------------------------------------------------------------------------
// STT providers
// ---------------------------------------------------------------------------

/// Hosted STT providers offered alongside the managed-local Parakeet engine.
/// OpenAI + Groq share the OpenAI multipart `/audio/transcriptions` shape;
/// Deepgram (`/v1/listen`) and AssemblyAI (upload + poll) get their own adapters.
pub const STT_API_PROVIDERS: &[ApiProviderDef] = &[
    ApiProviderDef {
        id: "openai",
        name: "OpenAI",
        default_base_url: "https://api.openai.com/v1",
        default_model: "gpt-4o-mini-transcribe",
        models: &["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "OpenAI /v1/audio/transcriptions (multipart form).",
    },
    ApiProviderDef {
        id: "groq",
        name: "Groq",
        default_base_url: "https://api.groq.com/openai/v1",
        default_model: "whisper-large-v3-turbo",
        models: &["whisper-large-v3-turbo", "whisper-large-v3"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "Groq Whisper — very fast, OpenAI-compatible multipart.",
    },
    ApiProviderDef {
        id: "deepgram",
        name: "Deepgram",
        default_base_url: "https://api.deepgram.com/v1",
        default_model: "nova-3",
        models: &["nova-3", "nova-2"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "Deepgram /v1/listen — raw audio POST with model query.",
    },
    ApiProviderDef {
        id: "assemblyai",
        name: "AssemblyAI",
        default_base_url: "https://api.assemblyai.com/v2",
        default_model: "universal",
        models: &["universal", "best"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "AssemblyAI — uploads audio then polls the transcript.",
    },
    ApiProviderDef {
        id: "openrouter",
        name: "OpenRouter",
        default_base_url: "https://openrouter.ai/api/v1",
        default_model: "openai/whisper-1",
        models: &["openai/whisper-1", "openai/gpt-4o-transcribe"],
        voices: &[],
        needs_base_url: false,
        needs_voice: false,
        blurb: "OpenRouter transcription — one key, many STT models \
                (OpenAI-compatible /audio/transcriptions).",
    },
];

// ---------------------------------------------------------------------------
// TTS providers
// ---------------------------------------------------------------------------

/// Hosted TTS providers offered alongside the managed-local faster-qwen3-tts
/// engine. Each returns audio chasm wraps into the WAV chunks the bridge plays.
/// ALL of these support VOICE CLONING over their API (chasm can clone a
/// character's voice through the selected provider) — providers without API
/// cloning (e.g. OpenAI's fixed voices) are intentionally not listed. Note:
/// OpenRouter offers TTS but only fixed OpenAI/Google/Mistral voices (no
/// cloning), so it is a LLM+STT provider here, not a TTS one.
pub const TTS_API_PROVIDERS: &[ApiProviderDef] = &[
    ApiProviderDef {
        id: "elevenlabs",
        name: "ElevenLabs",
        default_base_url: "https://api.elevenlabs.io/v1",
        default_model: "eleven_turbo_v2_5",
        models: &["eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_flash_v2_5"],
        voices: &[
            ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
            ("EXAVITQu4vr4xnSDxMaL", "Sarah"),
            ("AZnzlk1XvdvUeBnXmlld", "Domi"),
            ("VR6AewLTigWG4xSOukaG", "Arnold"),
            ("pNInz6obpgDQGcFmaJgB", "Adam"),
        ],
        needs_base_url: false,
        needs_voice: true,
        blurb: "ElevenLabs — stock voices OR clone each character's voice via API \
                (instant voice cloning). Paste a voice id or use Clone below.",
    },
    ApiProviderDef {
        id: "cartesia",
        name: "Cartesia",
        default_base_url: "https://api.cartesia.ai",
        default_model: "sonic-2",
        models: &["sonic-2", "sonic-turbo"],
        voices: &[],
        needs_base_url: false,
        needs_voice: true,
        blurb: "Cartesia Sonic — ultra-low latency, instant voice cloning from a \
                few seconds of audio. Paste a voice id or use Clone below.",
    },
    ApiProviderDef {
        id: "inworld",
        name: "Inworld",
        default_base_url: "https://api.inworld.ai",
        default_model: "inworld-tts-1",
        models: &["inworld-tts-1", "inworld-tts-1-max"],
        voices: &[
            ("Ashley", "Ashley"),
            ("Alex", "Alex"),
            ("Hades", "Hades"),
            ("Mark", "Mark"),
        ],
        needs_base_url: false,
        needs_voice: true,
        blurb: "Inworld TTS — built for AI characters, zero-shot voice cloning via \
                API. Pick a stock voice or use Clone below.",
    },
];

// ---------------------------------------------------------------------------
// Lookup + normalization
// ---------------------------------------------------------------------------

/// The resolved, ready-to-use API target for one provider: base URL + key +
/// model (+ voice for TTS), each falling back to the catalog default when the
/// stored config left it blank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedApi {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub voice: String,
}

fn find(list: &'static [ApiProviderDef], id: &str) -> Option<&'static ApiProviderDef> {
    list.iter().find(|p| p.id == id)
}

/// The LLM provider def for `id`, or `None` for `"local"` / unknown.
pub fn llm_api_provider(id: &str) -> Option<&'static ApiProviderDef> {
    find(LLM_API_PROVIDERS, id)
}
/// The STT provider def for `id`, or `None` for `"local"` / unknown.
pub fn stt_api_provider(id: &str) -> Option<&'static ApiProviderDef> {
    find(STT_API_PROVIDERS, id)
}
/// The TTS provider def for `id`, or `None` for `"local"` / unknown.
pub fn tts_api_provider(id: &str) -> Option<&'static ApiProviderDef> {
    find(TTS_API_PROVIDERS, id)
}

/// Resolves a provider def + optional stored config into a ready [`ResolvedApi`],
/// applying catalog defaults for any blank field. `base_url` is always returned
/// without a trailing slash so callers can append paths cleanly.
pub fn resolve_api(def: &ApiProviderDef, cfg: Option<&ApiProviderConfig>) -> ResolvedApi {
    let pick = |stored: &str, default: &str| {
        let s = stored.trim();
        if s.is_empty() {
            default.to_string()
        } else {
            s.to_string()
        }
    };
    let (api_key, model, base_url, voice) = match cfg {
        Some(c) => (
            c.api_key.trim().to_string(),
            pick(&c.model, def.default_model),
            pick(&c.base_url, def.default_base_url),
            pick(&c.voice, def.voices.first().map(|(id, _)| *id).unwrap_or("")),
        ),
        None => (
            String::new(),
            def.default_model.to_string(),
            def.default_base_url.to_string(),
            def.voices.first().map(|(id, _)| id.to_string()).unwrap_or_default(),
        ),
    };
    ResolvedApi {
        base_url: base_url.trim_end_matches('/').to_string(),
        api_key,
        model,
        voice,
    }
}

/// Normalizes a stored/posted LLM provider id: a known hosted id or `"local"`;
/// anything else (empty, legacy, unknown) collapses to `"local"`.
pub fn normalize_llm_provider(value: &str) -> String {
    let v = value.trim();
    if !v.is_empty() && llm_api_provider(v).is_some() {
        v.to_string()
    } else {
        PROVIDER_LOCAL.to_string()
    }
}

/// Normalizes a stored/posted STT provider id. Legacy managed values
/// (`"whisper"`, `"parakeet"`, empty) map to `"local"` (Parakeet is now the only
/// managed STT); a known hosted id passes through; anything else → `"local"`.
pub fn normalize_stt_provider(value: &str) -> String {
    let v = value.trim();
    if !v.is_empty() && stt_api_provider(v).is_some() {
        v.to_string()
    } else {
        PROVIDER_LOCAL.to_string()
    }
}

/// Normalizes a stored/posted TTS provider id: a known hosted id or `"local"`;
/// anything else → `"local"`.
pub fn normalize_tts_provider(value: &str) -> String {
    let v = value.trim();
    if !v.is_empty() && tts_api_provider(v).is_some() {
        v.to_string()
    } else {
        PROVIDER_LOCAL.to_string()
    }
}

/// Reads a provider's stored config out of a capability `api` map (never inserts).
pub fn stored_config<'a>(
    map: &'a BTreeMap<String, ApiProviderConfig>,
    provider: &str,
) -> Option<&'a ApiProviderConfig> {
    map.get(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_falls_back_to_catalog_defaults() {
        let def = llm_api_provider("openai").unwrap();
        let r = resolve_api(def, None);
        assert_eq!(r.base_url, "https://api.openai.com/v1");
        assert_eq!(r.model, "gpt-5.1-mini");
        assert!(r.api_key.is_empty());
    }

    #[test]
    fn resolve_prefers_stored_over_default_and_trims_base_slash() {
        let def = llm_api_provider("openai").unwrap();
        let cfg = ApiProviderConfig {
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
            base_url: "https://proxy.local/v1/".into(),
            voice: String::new(),
            routing: String::new(),
        };
        let r = resolve_api(def, Some(&cfg));
        assert_eq!(r.api_key, "sk-test");
        assert_eq!(r.model, "gpt-4o");
        assert_eq!(r.base_url, "https://proxy.local/v1");
    }

    #[test]
    fn tts_resolve_defaults_voice_to_first_catalog_voice() {
        let def = tts_api_provider("elevenlabs").unwrap();
        let r = resolve_api(def, None);
        assert_eq!(r.voice, "21m00Tcm4TlvDq8ikWAM");
    }

    #[test]
    fn tts_providers_are_cloning_capable_set() {
        // OpenAI TTS (no API cloning) is intentionally absent; OpenRouter TTS
        // (fixed voices) too. Inworld was added.
        let ids: Vec<&str> = TTS_API_PROVIDERS.iter().map(|p| p.id).collect();
        assert_eq!(ids, vec!["elevenlabs", "cartesia", "inworld"]);
        assert!(tts_api_provider("openai").is_none());
        // OpenRouter IS an LLM + STT provider.
        assert!(llm_api_provider("openrouter").is_some());
        assert!(stt_api_provider("openrouter").is_some());
    }

    #[test]
    fn stt_normalizes_legacy_managed_values_to_local() {
        assert_eq!(normalize_stt_provider("whisper"), "local");
        assert_eq!(normalize_stt_provider("parakeet"), "local");
        assert_eq!(normalize_stt_provider(""), "local");
        assert_eq!(normalize_stt_provider("groq"), "groq");
        assert_eq!(normalize_stt_provider("nonsense"), "local");
    }

    #[test]
    fn llm_and_tts_normalize_unknown_to_local() {
        assert_eq!(normalize_llm_provider(""), "local");
        assert_eq!(normalize_llm_provider("koboldcpp"), "local");
        assert_eq!(normalize_llm_provider("anthropic"), "anthropic");
        assert_eq!(normalize_tts_provider("elevenlabs"), "elevenlabs");
        assert_eq!(normalize_tts_provider("xyz"), "local");
    }
}
