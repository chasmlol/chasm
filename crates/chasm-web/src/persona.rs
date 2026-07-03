//! Player persona — the SillyTavern-style "user persona" built from the FNV
//! mod's stealth capture.
//!
//! The mod POSTs `/api/game/v1/persona` (see `mod-source/docs/persona.md` for
//! the frozen contract): the player's stats snapshot (display strings reusing
//! the gamestate-macro extractors) plus an optional base64 JPEG/PNG of the
//! player photographed from the front. This module:
//!
//!   * stores the capture profile-aware under [`chasm_core::ProfilePaths::persona_dir`]
//!     (`capture.json` + `capture.jpg|png`),
//!   * generates a compact third-person description of the player with a
//!     vision-capable LLM when one is reachable — the optional separate
//!     `persona.vision_endpoint` first, then the main LLM endpoint with the
//!     image — and ALWAYS falls back to a stats-only text generation so the
//!     feature works with no vision model at all,
//!   * writes the result to `persona.json`, which prompt assembly injects at
//!     SillyTavern's story-string persona slot (see `chasm-prompt`) and the
//!     Persona UI page reads (see `ui/persona.rs`).
//!
//! Generation is spawned on a background task and guarded by a busy flag —
//! it can never block or break an NPC turn. A failed generation keeps the
//! previous good description and records the error for the UI.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

use axum::{extract::State, Json};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Map, Value};
use chasm_core::AppSettings;

use crate::{AppState, WebError, WebResult};

/// Max decoded screenshot size accepted from the mod (a 1080p JPEG is well
/// under 1 MB; this only guards against garbage).
const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;

/// Request-body limit for `POST /api/game/v1/persona` (applied as a
/// route-scoped [`axum::extract::DefaultBodyLimit`] in `lib.rs`; axum's 2 MB
/// default would otherwise 413 a large capture before this module ever saw
/// it). Sized for [`MAX_IMAGE_BYTES`] of image after base64's 4/3 inflation
/// (~10.7 MB) plus the stats snapshot and JSON framing.
pub(crate) const MAX_BODY_BYTES: usize = 12 * 1024 * 1024;

/// `max_tokens` for the persona generation. The prompt demands TWO compact
/// paragraphs (looks, then manner) of at most ~80 words each (~230 tokens
/// total); this clamp keeps the output bounded even when the model ignores
/// the instruction.
const PERSONA_MAX_TOKENS: i64 = 320;

/// One persona generation at a time, process-wide. A capture arriving while a
/// generation runs is stored; its generation is skipped (the next capture or a
/// manual Regenerate re-runs it) — persona is periodic + self-healing.
static GENERATING: AtomicBool = AtomicBool::new(false);

/// True while a persona generation task is in flight (read by the UI view).
pub(crate) fn generation_in_flight() -> bool {
    GENERATING.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Store layout
// ---------------------------------------------------------------------------

/// The persona store dir for the ACTIVE profile (created on demand).
pub(crate) fn persona_dir(state: &AppState) -> PathBuf {
    state.config.active_profile_paths().persona_dir()
}

/// Path of the stored stats snapshot (the last capture, minus image bytes).
pub(crate) fn capture_path(dir: &Path) -> PathBuf {
    dir.join("capture.json")
}

/// Path of the generated persona (description + provenance + stats used).
pub(crate) fn persona_path(dir: &Path) -> PathBuf {
    dir.join("persona.json")
}

/// Path of the stored screenshot for `format` (`jpeg` → capture.jpg).
fn image_path(dir: &Path, format: &str) -> PathBuf {
    dir.join(if format.eq_ignore_ascii_case("png") {
        "capture.png"
    } else {
        "capture.jpg"
    })
}

/// The stored screenshot, if any: `(path, mime)`. JPEG wins when both exist
/// (the writer removes the other format, so both only exist transiently).
pub(crate) fn stored_image(dir: &Path) -> Option<(PathBuf, &'static str)> {
    let jpg = dir.join("capture.jpg");
    if jpg.is_file() {
        return Some((jpg, "image/jpeg"));
    }
    let png = dir.join("capture.png");
    if png.is_file() {
        return Some((png, "image/png"));
    }
    None
}

/// Reads a JSON file as a `Value`; `None` when absent or unparseable.
pub(crate) fn read_json(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Writes `value` as pretty JSON via a temp file + rename, so concurrent
/// readers (prompt assembly reads persona.json on every turn) never observe a
/// half-written file. `fs::rename` replaces the destination on Windows.
fn write_json_atomic(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(value).unwrap_or_default())?;
    fs::rename(&tmp, path)
}

// ---------------------------------------------------------------------------
// Stats snapshot helpers
// ---------------------------------------------------------------------------

/// The capture fields that constitute the "stats snapshot" (the display
/// strings the mod extracts — see `mod-source/docs/persona.md`).
const STAT_KEYS: [&str; 13] = [
    "player_name",
    "level",
    "special",
    "skills",
    "perks",
    "equipped_weapon",
    "equipped_apparel",
    "location",
    "sex",
    "race",
    "hair_color",
    "hair_style",
    "eye_color",
];

/// Projects the stats snapshot out of a capture/persona body (string/number
/// fields only, in stable order).
fn stats_of(body: &Value) -> Value {
    let mut map = Map::new();
    for key in STAT_KEYS {
        if let Some(value) = body.get(key) {
            if value.is_string() || value.is_number() {
                map.insert(key.to_string(), value.clone());
            }
        }
    }
    Value::Object(map)
}

/// Fallout: New Vegas S.P.E.C.I.A.L. attributes: the mod's abbreviation, the
/// full name, and what the stat measures (per the in-game manual/wiki
/// definitions), so the model can interpret extremes without game knowledge.
const SPECIAL_ATTRIBUTES: [(&str, &str, &str); 7] = [
    ("STR", "Strength", "raw physical power and muscle"),
    ("PER", "Perception", "senses and environmental awareness"),
    ("END", "Endurance", "physical fitness and toughness"),
    ("CHA", "Charisma", "charm and social grace"),
    ("INT", "Intelligence", "reasoning and wits"),
    ("AGI", "Agility", "coordination and nimbleness"),
    ("LCK", "Luck", "plain good fortune"),
];

/// Qualitative band for a 1-10 S.P.E.C.I.A.L. value.
fn special_band(value: i64) -> &'static str {
    match value {
        i64::MIN..=1 => "abysmal",
        2..=3 => "very low",
        4 => "below average",
        5..=6 => "average",
        7 => "above average",
        8 => "high",
        9 => "exceptional",
        _ => "peak human",
    }
}

/// Qualitative band for a 0-100 skill value — only genuine extremes get one
/// (`None` otherwise). Low-but-ordinary values are deliberately NOT flagged:
/// fresh characters sit at 10-30 in most skills, so calling 25 "poor" would
/// drown the real outliers in noise.
fn skill_band(value: i64) -> Option<&'static str> {
    match value {
        i64::MIN..=10 => Some("dreadful"),
        70..=84 => Some("highly skilled"),
        85..=i64::MAX => Some("masterful"),
        _ => None,
    }
}

/// Parses a mod display string of `Label N` pairs (`"STR 9, PER 6, ..."` /
/// `"Barter 15, Energy Weapons 20, ..."`) into `(label, value)` entries.
/// Entries whose tail is not an integer are returned with `None` so callers
/// can pass them through verbatim rather than dropping data.
fn parse_stat_pairs(text: &str) -> Vec<(String, Option<i64>)> {
    text.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| match entry.rsplit_once(' ') {
            Some((label, number)) => match number.trim().parse::<i64>() {
                Ok(value) => (label.trim().to_string(), Some(value)),
                Err(_) => (entry.to_string(), None),
            },
            None => (entry.to_string(), None),
        })
        .collect()
}

/// Renders the mod's abbreviated SPECIAL string as one natural-language line
/// per attribute (full name, explicit 1-10 scale, qualitative band, meaning
/// hint). Unknown labels or unparseable entries pass through verbatim; a
/// wholly unparseable string falls back to the raw `SPECIAL: ...` line.
fn special_lines(raw: &str) -> Vec<String> {
    let pairs = parse_stat_pairs(raw);
    if pairs.is_empty() || pairs.iter().all(|(_, value)| value.is_none()) {
        return vec![format!("SPECIAL: {raw}")];
    }
    pairs
        .into_iter()
        .map(|(label, value)| match value {
            Some(value) => {
                let known = SPECIAL_ATTRIBUTES
                    .iter()
                    .find(|(abbrev, _, _)| abbrev.eq_ignore_ascii_case(&label));
                match known {
                    Some((_, name, meaning)) => format!(
                        "- {name} {value} of 10 — {band} ({meaning})",
                        band = special_band(value)
                    ),
                    None => format!("- {label} {value} of 10 — {}", special_band(value)),
                }
            }
            None => format!("- {label}"),
        })
        .collect()
}

/// Renders the mod's skills string naturally: the full list with the 0-100
/// scale made explicit, plus a callout line for the extremes (the only values
/// the prompt lets the model express).
fn skills_lines(raw: &str) -> Vec<String> {
    let pairs = parse_stat_pairs(raw);
    if pairs.is_empty() || pairs.iter().all(|(_, value)| value.is_none()) {
        return vec![format!("Skills: {raw}")];
    }
    let mut lines = vec![format!("Skills, each rated 0 to 100: {raw}")];
    let extremes: Vec<String> = pairs
        .iter()
        .filter_map(|(label, value)| {
            let value = (*value)?;
            let band = skill_band(value)?;
            Some(format!("{label} {value} of 100 ({band})"))
        })
        .collect();
    if !extremes.is_empty() {
        lines.push(format!("Notable skills: {}", extremes.join("; ")));
    }
    lines
}

/// The human-readable stat sheet embedded in the generation prompt. SPECIAL
/// and skills are rendered as natural language (full attribute names, explicit
/// scales, qualitative bands) so the model never sees bare `STR 9`-style
/// abbreviations it might misread.
fn stats_block(stats: &Value) -> String {
    let field = |key: &str| -> String {
        match stats.get(key) {
            Some(Value::String(text)) => text.trim().to_string(),
            Some(Value::Number(number)) => number.to_string(),
            _ => String::new(),
        }
    };
    let mut lines: Vec<String> = Vec::new();
    let name = field("player_name");
    if !name.is_empty() {
        lines.push(format!("Name: {name}"));
    }
    let level = field("level");
    if !level.is_empty() {
        lines.push(format!("Level: {level}"));
    }
    let special = field("special");
    if !special.is_empty() {
        lines.push("Attributes (each 1 to 10):".to_string());
        lines.extend(special_lines(&special));
    }
    let skills = field("skills");
    if !skills.is_empty() {
        lines.extend(skills_lines(&skills));
    }
    // Clothing IS in the sheet (the photo is a face shot, so the outfit must
    // come from game data); weapon and perks stay out (weapon is scene-specific
    // and perk names pulled the description in odd directions).
    let apparel = field("equipped_apparel");
    if !apparel.is_empty() {
        lines.push(format!("Wearing (game data): {apparel}"));
    }
    lines.join("\n")
}

/// Truncates to at most `max_chars`, at a word boundary, appending an ellipsis
/// when something was cut. Never splits a UTF-8 character.
fn cap_description(text: &str, max_chars: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    let trimmed = match cut.rfind(char::is_whitespace) {
        Some(index) if index > max_chars / 2 => cut[..index].trim_end().to_string(),
        _ => cut,
    };
    format!("{trimmed}…")
}

// ---------------------------------------------------------------------------
// The generation prompt
// ---------------------------------------------------------------------------

/// Builds the persona-generation prompt. `with_image` selects the vision
/// variant ("the person in this photo") vs. the stats-only variant.
fn persona_prompt(stats: &Value, with_image: bool) -> String {
    let name = stats
        .get("player_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("the person");
    let opening = if with_image {
        "You are looking at a close-up photo of a person's face. Write a sketch of their face and \
         head, as if telling someone how to recognize this person in a crowd."
            .to_string()
    } else {
        format!(
            "Write a sketch of the face and presence of {name}, a person someone is about to \
             meet, working only from the stat sheet below — as if telling someone how to \
             recognize them in a crowd."
        )
    };
    let rules = if with_image {
        "- FIRST PARAGRAPH — their looks, purely from the photo: apparent age and sex, face, \
         complexion, eyes, hair (color and style), facial hair, scars and other marks, and any \
         headwear (hat, helmet, goggles, mask). Open it with one plain sentence like \
         \"<name> is a middle-aged man with slicked-back blonde hair and pale blue eyes.\" \
         Trust ONLY the photo here — no stats, no personality. Do NOT describe clothing below \
         the neck, their body, build, or hands.\n\
         - Mention facial hair, scars, or headwear ONLY when actually present — never remark \
         on their absence (never write \"no visible scars\", \"clean-shaven\", or \
         \"wears no hat\"). Never describe their expression or what it lacks.\n\
         - END the first paragraph with what they are wearing, from the \"Wearing\" line of \
         the stat sheet (the photo does not show their outfit) — phrase it naturally as worn \
         clothing, not as a list. If the Wearing line names headwear, treat it as worn.\n\
         - SECOND PARAGRAPH — how they come across: manner, capability, presence, drawn from \
         the stat sheet. Never quote numbers or stat names. Only extremes deserve expression \
         (brilliant Intelligence → sharp, appraising; rock-bottom Speech → halting and \
         wordless; immense Strength → radiates physical power). Say nothing about middling \
         values.\n\
         - Never describe the background, surroundings, lighting, or framing. Never mention that \
         this is a photo, screenshot, render, or game.\n\
         - Write in third person, present tense. Use their name.\n\
         - Output ONLY the description: exactly TWO compact paragraphs separated by one blank \
         line, each at most about 80 words. No headings, no lists, no preamble.\n"
    } else {
        "- Describe the impression their face and presence give. Invent no specific facial \
         features; keep physical details to what the stats imply.\n\
         - Use the stat sheet, but never quote numbers or stat names. Only extremes deserve \
         expression; say nothing about middling values.\n\
         - Write in third person, present tense. Use their name.\n\
         - Output ONLY the description: exactly ONE compact paragraph, at most about 100 words. \
         No headings, no lists, no preamble.\n"
    };
    format!(
        "{opening}\n\n\
         Rules:\n\
         {rules}\n\
         Stat sheet:\n{stats}",
        stats = stats_block(stats)
    )
}

/// The OpenAI-compatible message list for one persona generation. With an
/// image, the user content is the multimodal parts array (text + `image_url`
/// data URI) that koboldcpp/llama.cpp/OpenAI-compat vision servers accept.
fn persona_messages(prompt: &str, image: Option<(&str, &str)>) -> Vec<Value> {
    let content = match image {
        Some((mime, base64)) => json!([
            { "type": "text", "text": prompt },
            { "type": "image_url",
              "image_url": { "url": format!("data:{mime};base64,{base64}") } },
        ]),
        None => json!(prompt),
    };
    vec![json!({ "role": "user", "content": content })]
}

// ---------------------------------------------------------------------------
// LLM transport
// ---------------------------------------------------------------------------

/// Minimal OpenAI-compatible chat completion against an EXPLICIT endpoint,
/// with optional bearer auth + model override — used for the separate
/// `persona.vision_endpoint`. (The main-endpoint attempts reuse
/// [`crate::llm::chat_completion_capturing_sampled`].) Kept here so the
/// persona feature never touches `llm.rs`.
async fn openai_chat_completion(
    endpoint: &str,
    api_key: &str,
    model: &str,
    messages: &[Value],
    max_tokens: i64,
) -> Result<String, String> {
    let endpoint = endpoint.trim_end_matches('/');
    let client = reqwest::Client::new();

    // Explicit model wins; else probe /v1/models like the main client does.
    let model = if model.trim().is_empty() {
        let mut probe = client.get(format!("{endpoint}/v1/models"));
        if !api_key.is_empty() {
            probe = probe.bearer_auth(api_key);
        }
        match probe.send().await {
            Ok(response) if response.status().is_success() => response
                .json::<Value>()
                .await
                .ok()
                .and_then(|body| {
                    body.get("data")?
                        .as_array()?
                        .first()?
                        .get("id")?
                        .as_str()
                        .map(str::to_string)
                }),
            _ => None,
        }
    } else {
        Some(model.trim().to_string())
    };

    let mut body = json!({
        "messages": messages,
        "stream": false,
        "max_tokens": max_tokens,
    });
    if let Some(model) = model {
        body["model"] = json!(model);
    }

    let mut request = client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&body);
    if !api_key.is_empty() {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("vision endpoint request failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("vision endpoint returned {status}: {text}"));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("vision endpoint decode failed: {error}"))?;
    let content = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        return Err("vision endpoint returned an empty completion".to_string());
    }
    Ok(content)
}

/// One attempt against the MAIN local LLM endpoint (the same client NPC turns
/// use), honoring the saved sampling with a persona max_tokens override.
async fn main_llm_completion(
    state: &AppState,
    messages: &[Value],
) -> Result<String, String> {
    let sampling = crate::llm::Sampling::from_settings(
        &AppSettings::load(&state.config.settings_path).llm.sampling,
    )
    .with_overrides(crate::llm::GenerationOptions {
        temperature: None,
        max_tokens: Some(PERSONA_MAX_TOKENS),
    });
    let (text, _metrics) = crate::llm::chat_completion_capturing_sampled(
        &state.config.llm_endpoint,
        messages,
        None,
        sampling,
    )
    .await?;
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("LLM returned an empty completion".to_string());
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Runs one persona generation from the STORED capture (capture.json +
/// capture image) and writes `persona.json`. Attempt order:
///
///   1. image + separate `persona.vision_endpoint` (when configured),
///   2. image + the main LLM endpoint (multimodal content parts — works when
///      the loaded model has a projector; llama.cpp without one rejects the
///      request, which cleanly falls through),
///   3. stats-only text prompt on the main LLM endpoint (always available).
///
/// On total failure the previous good `persona.json` description is KEPT and
/// only the error fields are refreshed, so a transient LLM outage never
/// destroys a working persona.
pub(crate) async fn generate_from_stored_capture(state: &AppState) -> WebResult<Value> {
    let dir = persona_dir(state);
    let capture = read_json(&capture_path(&dir)).ok_or_else(|| {
        WebError::from(anyhow::anyhow!(
            "no capture stored yet — the mod has not uploaded a persona capture"
        ))
    })?;

    let settings = AppSettings::load(&state.config.settings_path).persona;
    let stats = stats_of(&capture);

    // The stored screenshot, as (mime, base64), when present + decodeable.
    let image = stored_image(&dir).and_then(|(path, mime)| {
        fs::read(&path)
            .ok()
            .map(|bytes| (mime, STANDARD.encode(bytes)))
    });

    let mut note = String::new();
    let mut source = "stats_only";
    let mut description: Option<String> = None;
    // The exact prompt text of the attempt that PRODUCED the description
    // (persisted with the record so the Persona page can show precisely what
    // the LLM was asked; the screenshot rides along as an image part when
    // source == "vision").
    let mut used_prompt: Option<String> = None;

    if !settings.enabled {
        note = "persona generation is disabled in settings".to_string();
    } else {
        // 1. Separate vision endpoint (explicitly vision-capable).
        if let Some((mime, base64)) = image.as_ref() {
            let prompt = persona_prompt(&stats, true);
            let messages = persona_messages(&prompt, Some((mime, base64)));
            let vision_endpoint = settings.vision_endpoint.trim();
            if !vision_endpoint.is_empty() {
                match openai_chat_completion(
                    vision_endpoint,
                    settings.vision_api_key.trim(),
                    settings.vision_model.trim(),
                    &messages,
                    PERSONA_MAX_TOKENS,
                )
                .await
                {
                    Ok(text) => {
                        description = Some(text);
                        source = "vision";
                        used_prompt = Some(prompt.clone());
                        note = "generated from the screenshot via the separate vision endpoint"
                            .to_string();
                    }
                    Err(error) => {
                        note = format!("separate vision endpoint failed ({error}); ");
                        tracing::info!(target: "chasm::persona", %error, "vision endpoint failed");
                    }
                }
            }
            // 2. Main endpoint with the image.
            if description.is_none() {
                match main_llm_completion(state, &messages).await {
                    Ok(text) => {
                        description = Some(text);
                        source = "vision";
                        used_prompt = Some(prompt.clone());
                        note.push_str("generated from the screenshot via the main LLM endpoint");
                    }
                    Err(error) => {
                        note.push_str(&format!(
                            "main LLM did not accept the image ({error}); "
                        ));
                        tracing::info!(target: "chasm::persona", %error, "main-LLM vision failed");
                    }
                }
            }
        }
        // 3. Stats-only fallback (also the no-image path).
        if description.is_none() {
            let prompt = persona_prompt(&stats, false);
            let messages = persona_messages(&prompt, None);
            match main_llm_completion(state, &messages).await {
                Ok(text) => {
                    description = Some(text);
                    source = "stats_only";
                    used_prompt = Some(prompt);
                    note.push_str("generated from the stats snapshot (no vision model available)");
                }
                Err(error) => {
                    note.push_str(&format!("stats-only generation failed ({error})"));
                    tracing::warn!(target: "chasm::persona", %error, "persona generation failed");
                }
            }
        }
    }

    let now = chrono_now_iso();
    let previous = read_json(&persona_path(&dir)).unwrap_or_else(|| json!({}));
    let image_file = stored_image(&dir)
        .map(|(path, _)| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let captured_at = capture
        .get("captured_at")
        .and_then(Value::as_str)
        .or_else(|| capture.get("received_at").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();

    let persona = match description {
        Some(text) => {
            let capped = cap_description(&text, settings.effective_max_chars());
            json!({
                "description": capped,
                "generated_at": now,
                "captured_at": captured_at,
                "source": source,
                "model_note": note,
                "image_file": image_file,
                "stats": stats,
                // The exact prompt text sent to the LLM for this description
                // (the screenshot is attached as an image part when source is
                // "vision" — see persona_messages). Shown on the Persona page.
                "prompt": used_prompt.unwrap_or_default(),
            })
        }
        None => {
            // Keep the previous good description (if any); record the failure.
            let mut kept = previous.clone();
            if !kept.is_object() {
                kept = json!({});
            }
            kept["generation_error"] = json!(note);
            kept["generation_error_at"] = json!(now);
            kept
        }
    };

    write_json_atomic(&persona_path(&dir), &persona).map_err(WebError::from)?;
    Ok(persona)
}

/// Spawns [`generate_from_stored_capture`] on a background task unless one is
/// already running. Returns whether a task was started. NPC turn generation is
/// never awaited on this — the whole point.
pub(crate) fn spawn_generation(state: Arc<AppState>) -> bool {
    if GENERATING.swap(true, Ordering::SeqCst) {
        return false; // one at a time; the next capture re-triggers
    }
    tokio::spawn(async move {
        if let Err(error) = generate_from_stored_capture(&state).await {
            tracing::warn!(target: "chasm::persona", error = %format!("{error:?}"), "persona generation task failed");
        }
        GENERATING.store(false, Ordering::SeqCst);
    });
    true
}

/// RFC3339 UTC "now" without pulling in chrono: seconds precision is plenty.
/// Shared with the gamemaster pass for relationship entry timestamps.
pub(crate) fn chrono_now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Days-from-civil (Howard Hinnant's algorithm) to avoid a chrono dep.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let a = days + 719_468;
    let era = a.div_euclid(146_097);
    let doe = a.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// ---------------------------------------------------------------------------
// The game-transport receive endpoint
// ---------------------------------------------------------------------------

/// `POST /api/game/v1/persona` — the mod's capture upload (frozen contract in
/// `mod-source/docs/persona.md`). Stores the stats snapshot + screenshot under
/// the active profile's persona dir and queues an async generation. Returns
/// immediately; never blocks on the LLM.
///
/// Response: `{ "status": "stored", "generation": "queued" | "busy" |
/// "unchanged" | "disabled", "image": "stored" | "none" | "rejected: …" }`.
pub async fn receive_capture(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    if !body.is_object() {
        return Err(WebError::from(anyhow::anyhow!("body must be a JSON object")));
    }

    let dir = persona_dir(&state);
    fs::create_dir_all(&dir).map_err(WebError::from)?;

    // --- Screenshot (optional): decode, bound, store; degrade gracefully. ---
    let mut image_status = "none".to_string();
    let image_b64 = body
        .get("image_base64")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|data| !data.is_empty());
    if let Some(data) = image_b64 {
        match STANDARD.decode(data) {
            Ok(bytes) if bytes.len() <= MAX_IMAGE_BYTES && !bytes.is_empty() => {
                let format = body
                    .get("image_format")
                    .and_then(Value::as_str)
                    .unwrap_or("jpeg");
                let path = image_path(&dir, format);
                match fs::write(&path, &bytes) {
                    Ok(()) => {
                        // Drop the other-format leftover so stored_image() is unambiguous.
                        let other = if path.ends_with("capture.jpg") {
                            dir.join("capture.png")
                        } else {
                            dir.join("capture.jpg")
                        };
                        let _ = fs::remove_file(other);
                        image_status = "stored".to_string();
                    }
                    Err(error) => image_status = format!("rejected: write failed ({error})"),
                }
            }
            Ok(bytes) => {
                image_status = format!("rejected: {} bytes exceeds limit", bytes.len());
            }
            Err(error) => image_status = format!("rejected: base64 decode failed ({error})"),
        }
    }

    // --- Stats snapshot: everything except the image bytes. -----------------
    let mut capture = body.clone();
    if let Some(map) = capture.as_object_mut() {
        map.remove("image_base64");
        map.insert("received_at".to_string(), json!(chrono_now_iso()));
        map.insert("image".to_string(), json!(image_status.clone()));
    }
    write_json_atomic(&capture_path(&dir), &capture).map_err(WebError::from)?;

    // --- Queue the Gamemaster relationships pass on SAVE captures. ----------
    // Same trigger vocabulary as the persona regeneration below, but a fully
    // independent background task with its own busy flag: it neither delays
    // this response nor the persona generation, and a pass skipped while one
    // is already running loses nothing (its content stays past the watermark
    // for the next save).
    if is_save_trigger(&capture) {
        crate::gamemaster::spawn_pass(state.clone());
    }

    // --- Queue generation. ---------------------------------------------------
    let settings = AppSettings::load(&state.config.settings_path).persona;
    let generation = if !settings.enabled {
        "disabled"
    } else if !is_save_trigger(&capture) && is_unchanged(&dir, &capture) {
        // Same stats as the stored persona and a description already exists:
        // skip the LLM. Save-driven captures (the mod's whole trigger model,
        // see docs/persona.md) always regenerate — the player saved, the
        // screenshot is fresh, and they expect a fresh description even with
        // identical stats. The unchanged short-circuit only remains for
        // non-save uploads. The fresh image was still stored above either way.
        "unchanged"
    } else if spawn_generation(state.clone()) {
        "queued"
    } else {
        "busy"
    };

    Ok(Json(json!({
        "status": "stored",
        "generation": generation,
        "image": image_status,
    })))
}

/// True when the capture came from a game save (the mod's trigger vocabulary:
/// `save` / `quicksave` / `autosave`). Save-driven captures always regenerate,
/// bypassing [`is_unchanged`].
fn is_save_trigger(capture: &Value) -> bool {
    matches!(
        capture.get("trigger").and_then(Value::as_str),
        Some("save") | Some("quicksave") | Some("autosave")
    )
}

/// True when the stored persona was generated from an IDENTICAL stats snapshot
/// and has a non-empty description (→ nothing to regenerate).
fn is_unchanged(dir: &Path, capture: &Value) -> bool {
    let Some(persona) = read_json(&persona_path(dir)) else {
        return false;
    };
    let has_description = persona
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|description| !description.trim().is_empty());
    has_description && persona.get("stats") == Some(&stats_of(capture))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_body(name: &str) -> Value {
        json!({
            "captured_at": "2026-07-02T10:00:00Z",
            "player_name": name,
            "level": 12,
            "special": "STR 9, PER 6, END 5, CHA 1, INT 3, AGI 6, LCK 5",
            "skills": "Barter 15, Guns 45, Speech 4, Unarmed 80",
            "perks": "Toughness 2, Educated",
            "equipped_weapon": "9mm Pistol",
            "equipped_apparel": "Leather Armor, Goggles",
            "location": "Goodsprings",
            "sex": "male",
            "race": "Caucasian Old",
            "hair_color": "#D6B569",
            "hair_style": "Wavy",
            "eye_color": "Blue",
            "trigger": "quicksave",
        })
    }

    #[test]
    fn prompt_variants_carry_rules_and_stats() {
        let stats = stats_of(&capture_body("Courier"));
        let vision = persona_prompt(&stats, true);
        assert!(vision.contains("close-up photo of a person's face"));
        assert!(vision.contains("Never mention that this is a photo, screenshot, render, or game"));
        // Natural-language stats: full attribute names + explicit scale +
        // qualitative band + meaning hint, never bare abbreviations.
        assert!(vision.contains("Strength 9 of 10 — exceptional (raw physical power and muscle)"));
        assert!(vision.contains("Charisma 1 of 10 — abysmal (charm and social grace)"));
        assert!(vision.contains("Intelligence 3 of 10 — very low (reasoning and wits)"));
        assert!(!vision.contains("STR 9,"), "no raw abbreviations in the rendered attributes");
        assert!(vision.contains("Wearing (game data): Leather Armor, Goggles"), "outfit drives the clothing sentence");
        assert!(vision.contains("any headwear"));
        assert!(!vision.contains("Appearance facts"), "looks must come from the photo alone");
        assert!(!vision.contains("#D6B569"), "no character-data colors in the prompt");
        assert!(!vision.contains("Eyes: Blue"), "no character-data eye color in the prompt");
        assert!(!vision.contains("Perks:"), "perks must stay out of the prompt");
        assert!(vision.contains("FIRST PARAGRAPH"));
        assert!(vision.contains("SECOND PARAGRAPH"));
        assert!(vision.contains("exactly TWO compact paragraphs"));
        // Exactly one paragraph, ~100 words — the 180-word allowance is gone.
        assert!(!vision.contains("180 words"));

        let stats_only = persona_prompt(&stats, false);
        assert!(stats_only.contains("the face and presence of Courier"));
        assert!(!stats_only.contains("photo"));
        assert!(stats_only.contains("Skills, each rated 0 to 100: Barter 15, Guns 45, Speech 4, Unarmed 80"));
        // Only genuine extremes are called out (Barter 15 is ordinary early-game).
        assert!(stats_only.contains("Notable skills: Speech 4 of 100 (dreadful); Unarmed 80 of 100 (highly skilled)"));
    }

    #[test]
    fn special_rendering_is_natural_language_with_fallback() {
        let lines = special_lines("STR 10, PER 5, CHA 2, LCK 7");
        assert_eq!(
            lines,
            vec![
                "- Strength 10 of 10 — peak human (raw physical power and muscle)",
                "- Perception 5 of 10 — average (senses and environmental awareness)",
                "- Charisma 2 of 10 — very low (charm and social grace)",
                "- Luck 7 of 10 — above average (plain good fortune)",
            ]
        );
        // Unknown label still renders with the scale; unparseable entry passes through.
        let lines = special_lines("VIG 8, mystery");
        assert_eq!(lines, vec!["- VIG 8 of 10 — high", "- mystery"]);
        // A wholly unparseable string falls back to the raw line.
        assert_eq!(special_lines("???"), vec!["SPECIAL: ???"]);
    }

    #[test]
    fn skills_rendering_calls_out_extremes_only() {
        let lines = skills_lines("Guns 45, Speech 4, Sneak 30, Unarmed 90");
        assert_eq!(lines[0], "Skills, each rated 0 to 100: Guns 45, Speech 4, Sneak 30, Unarmed 90");
        assert_eq!(
            lines[1],
            "Notable skills: Speech 4 of 100 (dreadful); Unarmed 90 of 100 (masterful)"
        );
        // Middling and ordinary-low values → no extremes line at all.
        let lines = skills_lines("Guns 45, Sneak 30, Barter 15");
        assert_eq!(lines.len(), 1);
        // Garbage falls back to the raw line.
        assert_eq!(skills_lines("-"), vec!["Skills: -"]);
    }

    #[test]
    fn save_triggers_bypass_unchanged_short_circuit() {
        for trigger in ["save", "quicksave", "autosave"] {
            assert!(is_save_trigger(&json!({ "trigger": trigger })), "{trigger}");
        }
        assert!(!is_save_trigger(&json!({ "trigger": "initial" })));
        assert!(!is_save_trigger(&json!({ "trigger": "stats_changed" })));
        assert!(!is_save_trigger(&json!({})));
    }

    #[test]
    fn messages_embed_image_as_data_uri_parts() {
        let with_image = persona_messages("describe", Some(("image/jpeg", "QUJD")));
        let content = &with_image[0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/jpeg;base64,QUJD"
        );

        let text_only = persona_messages("describe", None);
        assert_eq!(text_only[0]["content"], "describe");
    }

    #[test]
    fn stats_projection_and_unchanged_detection() {
        let dir = std::env::temp_dir().join(format!(
            "chasm-persona-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();

        let capture = capture_body("Courier");
        // No persona yet → changed.
        assert!(!is_unchanged(&dir, &capture));

        // Persona with the SAME stats + a description → unchanged.
        let persona = json!({
            "description": "A hulking courier.",
            "stats": stats_of(&capture),
        });
        write_json_atomic(&persona_path(&dir), &persona).unwrap();
        assert!(is_unchanged(&dir, &capture));

        // A stats change (new apparel) → changed again.
        let mut changed = capture.clone();
        changed["equipped_apparel"] = json!("NCR Trooper Armor");
        assert!(!is_unchanged(&dir, &changed));

        // Empty description → always regenerate.
        let persona = json!({ "description": "", "stats": stats_of(&capture) });
        write_json_atomic(&persona_path(&dir), &persona).unwrap();
        assert!(!is_unchanged(&dir, &capture));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn description_cap_cuts_at_word_boundary() {
        let text = "one two three four five six seven";
        assert_eq!(cap_description(text, 100), text);
        let capped = cap_description(text, 12);
        assert!(capped.chars().count() <= 13); // 12 + ellipsis
        assert_eq!(capped, "one two…");
    }

    #[test]
    fn iso_timestamp_shape() {
        let now = chrono_now_iso();
        // 2026-07-02T12:34:56Z
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.ends_with('Z'));
        assert!(now.starts_with("20"));
        assert_eq!(&now[4..5], "-");
        assert_eq!(&now[10..11], "T");
    }
}
