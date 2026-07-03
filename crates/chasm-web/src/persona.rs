//! Player persona — the SillyTavern-style "user persona" built from the FNV
//! mod's game-data capture.
//!
//! The mod POSTs `/api/game/v1/persona` on EVERY save (see
//! `mod-source/docs/persona.md` for the frozen contract): the player's stats
//! snapshot plus appearance records (sex, race, hair style/color/length, eye
//! color, facial hair, worn apparel) — all display strings reusing the
//! gamestate-macro extractors. Pure data; no screenshot (the old offscreen
//! portrait + vision-LLM path is fully retired). This module:
//!
//!   * stores the capture profile-aware under [`chasm_core::ProfilePaths::persona_dir`]
//!     (`capture.json`),
//!   * generates a two-paragraph third-person description with the main text
//!     LLM: looks written from the appearance facts, manner written from the
//!     attribute/skill magnitudes,
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
use serde_json::{json, Map, Value};
use chasm_core::AppSettings;

use crate::{AppState, WebError, WebResult};

/// Request-body limit for `POST /api/game/v1/persona` (applied as a
/// route-scoped [`axum::extract::DefaultBodyLimit`] in `lib.rs`). Captures are
/// pure game-data JSON now (a few KB); this is a generous guard, not a budget.
pub(crate) const MAX_BODY_BYTES: usize = 256 * 1024;

/// `max_tokens` for the persona generation. The prompt demands TWO paragraphs
/// (looks at ~80 words, then manner at ~120 words — ~280 tokens total); this
/// clamp keeps the output bounded even when the model ignores the
/// instruction.
const PERSONA_MAX_TOKENS: i64 = 480;

/// One persona generation at a time, process-wide. A capture arriving while a
/// generation runs is stored and sets [`RERUN_REQUESTED`]; the running task
/// picks it up as soon as the current generation finishes, so EVERY save
/// regenerates (the mod uploads on every save, unthrottled).
static GENERATING: AtomicBool = AtomicBool::new(false);

/// Set when a capture arrives mid-generation; the in-flight task re-runs from
/// the (newer) stored capture before clearing [`GENERATING`]. Latest wins:
/// several captures during one generation collapse into a single re-run.
static RERUN_REQUESTED: AtomicBool = AtomicBool::new(false);

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

/// Path of the user's custom persona addition — a free-text paragraph the user
/// authors on the Persona page. Stored SEPARATELY from `persona.json` so it
/// survives the description regeneration that runs on every game save; prompt
/// assembly appends it as a final paragraph (see `read_player_persona` in
/// `chasm-st-compat`). Shape: `{ "note": "..." }`.
pub(crate) fn custom_note_path(dir: &Path) -> PathBuf {
    dir.join("custom-note.json")
}

/// Reads the user's custom addition (trimmed); empty string when absent, blank,
/// or unparseable — the injection path then behaves exactly as if there were no
/// addition at all.
pub(crate) fn read_custom_note(dir: &Path) -> String {
    read_json(&custom_note_path(dir))
        .as_ref()
        .and_then(|value| value.get("note"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Persists the user's custom addition (atomically). The stored text is
/// trimmed; an empty result writes `{ "note": "" }` so the file still exists and
/// the UI reads back exactly what injection will use.
pub(crate) fn write_custom_note(dir: &Path, note: &str) -> std::io::Result<()> {
    write_json_atomic(&custom_note_path(dir), &json!({ "note": note.trim() }))
}

/// Deletes screenshots left behind by the retired portrait feature so old
/// stores converge on the data-only layout. Best-effort.
fn remove_stale_images(dir: &Path) {
    let _ = fs::remove_file(dir.join("capture.jpg"));
    let _ = fs::remove_file(dir.join("capture.png"));
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
const STAT_KEYS: [&str; 16] = [
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
    "age_years",
    "hair_color",
    "hair_style",
    "hair_length",
    "eye_color",
    "facial_hair",
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
    let mut lines = vec![format!(
        "Skills, each rated 0 to 100 — untrained people sit around 10 to 30, 85 or more is \
         true mastery: {raw}"
    )];
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

/// Pulls a trimmed string/number field out of the stats snapshot.
fn stat_field(stats: &Value, key: &str) -> String {
    match stats.get(key) {
        Some(Value::String(text)) => text.trim().to_string(),
        Some(Value::Number(number)) => number.to_string(),
        _ => String::new(),
    }
}

/// Maps the mod's `#RRGGBB` hair color to a plain color word the LLM can use
/// (`None` on malformed input — the line is then omitted rather than guessed).
fn hair_color_name(hex: &str) -> Option<&'static str> {
    let hex = hex.trim().trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f32;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f32;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f32;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let lightness = (max + min) / 2.0 / 255.0;
    let saturation = if max <= 0.0 { 0.0 } else { (max - min) / max };
    Some(if lightness < 0.09 {
        "black"
    } else if saturation < 0.15 {
        // Desaturated: the gray/white ramp.
        if lightness > 0.8 {
            "white"
        } else if lightness > 0.5 {
            "gray"
        } else if lightness > 0.25 {
            "graying dark"
        } else {
            "black"
        }
    } else if r > g * 1.6 && r > b * 1.6 && lightness < 0.55 {
        "red"
    } else if lightness > 0.72 {
        "platinum blonde"
    } else if lightness > 0.55 {
        "blonde"
    } else if lightness > 0.42 {
        if r > g * 1.25 {
            "auburn"
        } else {
            "light brown"
        }
    } else if lightness > 0.28 {
        "brown"
    } else {
        "dark brown"
    })
}

/// Age phrase implied by the race record's name, when it carries a marker.
/// FNV encodes coarse age in race VARIANTS (the GECK AgeRace system):
/// `Caucasian Old`, `Hispanic Middle Aged`, child races... A plain adult race
/// says nothing (`None`) — the FaceGen-derived `age_years` speaks instead.
fn race_age_marker(race: &str) -> Option<&'static str> {
    let lower = race.to_ascii_lowercase();
    if lower.contains("child") || lower.contains("young") {
        Some("a child")
    } else if lower.contains("middle") {
        Some("middle-aged")
    } else if lower.contains("old") || lower.contains("elder") || lower.contains("aged") {
        Some("older, well past middle age")
    } else {
        None
    }
}

/// The ethnicity implied by the race name: age tokens stripped, the stock FNV
/// race names mapped to plain words; anything else (mod races, ghouls...)
/// passes through as-is so nothing is ever invented.
fn race_ethnicity(race: &str) -> String {
    let stripped: Vec<&str> = race
        .split_whitespace()
        .filter(|token| {
            !matches!(
                token.to_ascii_lowercase().as_str(),
                "old" | "older" | "aged" | "middle" | "child" | "young"
            )
        })
        .collect();
    let base = stripped.join(" ");
    match base.to_ascii_lowercase().as_str() {
        "caucasian" => "white".to_string(),
        "african american" | "africanamerican" => "Black".to_string(),
        "hispanic" => "Hispanic".to_string(),
        "asian" => "Asian".to_string(),
        _ => base,
    }
}

/// The appearance facts block: one `- Label: value` line per fact the capture
/// carries, natural-language values only (hex colors mapped to words, style
/// names lowercased into prose). Absent facts are simply not listed — the
/// prompt forbids the model from remarking on absence.
fn appearance_lines(stats: &Value) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let sex = stat_field(stats, "sex");
    if !sex.is_empty() {
        lines.push(format!("- Sex: {sex}"));
    }
    // Age: an explicit race marker (Old / Middle Aged / child races) wins —
    // aged races look aged regardless of the face. Otherwise the FaceGen-
    // derived years (the chargen Age slider) speak. With neither, the line is
    // omitted and the description claims nothing about age.
    let race = stat_field(stats, "race");
    let marker = if race.is_empty() {
        None
    } else {
        race_age_marker(&race)
    };
    if let Some(phrase) = marker {
        lines.push(format!("- Age: {phrase}"));
    } else if let Ok(years) = stat_field(stats, "age_years").parse::<u32>() {
        lines.push(format!("- Age: about {years} years old"));
    }
    if !race.is_empty() {
        let ethnicity = race_ethnicity(&race);
        if !ethnicity.is_empty() {
            lines.push(format!("- Ethnicity: {ethnicity}"));
        }
    }
    let mut hair_parts: Vec<String> = Vec::new();
    if let Some(color) = hair_color_name(&stat_field(stats, "hair_color")) {
        hair_parts.push(color.to_string());
    }
    let style = stat_field(stats, "hair_style");
    if !style.is_empty() {
        hair_parts.push(format!("styled in a {}", style.to_lowercase()));
    }
    // The GECK hair-length slider (0..1) scales the chosen style; only the
    // extremes say anything a stranger would notice.
    if let Ok(length) = stat_field(stats, "hair_length").parse::<f64>() {
        if length >= 0.75 {
            hair_parts.push("worn long".to_string());
        } else if length <= 0.25 {
            hair_parts.push("kept short".to_string());
        }
    }
    if !hair_parts.is_empty() {
        lines.push(format!("- Hair: {}", hair_parts.join(", ")));
    }
    let eyes = stat_field(stats, "eye_color");
    if !eyes.is_empty() {
        lines.push(format!("- Eyes: {}", eyes.to_lowercase()));
    }
    let facial_hair = stat_field(stats, "facial_hair");
    if !facial_hair.is_empty() {
        lines.push(format!("- Facial hair: {}", facial_hair.to_lowercase()));
    }
    let apparel = stat_field(stats, "equipped_apparel");
    if !apparel.is_empty() {
        lines.push(format!("- Wearing: {apparel}"));
    }
    lines
}

/// The human-readable character sheet embedded in the generation prompt:
/// appearance facts first (natural-language values), then SPECIAL and skills
/// rendered with full attribute names, explicit scales, and qualitative bands
/// so the model never sees bare `STR 9`-style abbreviations it might misread.
fn stats_block(stats: &Value) -> String {
    let mut lines: Vec<String> = Vec::new();
    let name = stat_field(stats, "player_name");
    if !name.is_empty() {
        lines.push(format!("Name: {name}"));
    }
    let level = stat_field(stats, "level");
    if !level.is_empty() {
        lines.push(format!("Level: {level}"));
    }
    let appearance = appearance_lines(stats);
    if !appearance.is_empty() {
        lines.push("Appearance (authoritative character data):".to_string());
        lines.extend(appearance);
    }
    let special = stat_field(stats, "special");
    if !special.is_empty() {
        lines.push(
            "Attributes, each rated 1 to 10 — 5 is an ordinary adult, 1 or less is a \
             crippling deficiency, 10 is the human limit:"
                .to_string(),
        );
        lines.extend(special_lines(&special));
    }
    let skills = stat_field(stats, "skills");
    if !skills.is_empty() {
        lines.extend(skills_lines(&skills));
    }
    // Weapon and perks stay out (weapon is scene-specific and perk names
    // pulled the description in odd directions).
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

/// Builds the persona-generation prompt: one text-only prompt writing TWO
/// paragraphs from the character sheet — looks from the appearance facts,
/// manner from the attribute/skill magnitudes.
fn persona_prompt(stats: &Value) -> String {
    let name = stats
        .get("player_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("the person");
    let opening = format!(
        "Write a sketch of {name}, a person someone is about to meet, working only from the \
         character sheet below — as if telling someone how to recognize this person in a \
         crowd."
    );
    // The honesty/calibration rule for the manner paragraph. Deliberately
    // generic — it teaches the model how to weigh ANY rating against its
    // scale, never naming specific attributes or skills (special-casing a
    // skill here would silently bias every other sheet).
    let calibration = "- Take the character sheet at face value and give every rating its full \
         weight — each line states its scale and a plain-language judgment; trust them \
         over politeness. Ratings near the middle are ordinary: pass over them in silence \
         rather than padding the paragraph — and when nearly everything is ordinary, say \
         that plainly in a sentence or two and stop, instead of restating each ordinary \
         thing. The further a rating sits from the middle, the \
         more space and force it deserves. A rating at the very bottom of its scale is a \
         glaring, crippling deficiency a stranger would notice within a minute of meeting \
         them — say what it actually looks like, bluntly; do not soften it, dress it up in \
         dignity, or shrink it to a quirk. A rating at the very top is genuine mastery or a \
         formidable gift — show it concretely in how they move, act, or react, not as a \
         one-word compliment. When the sheet mixes both, let the contrast stand at full \
         strength.\n\
         - Never quote numbers, ratings, stat names, or game terms — render them as \
         observed character. Describe only what the sheet supports; never mention what it \
         lacks or leaves unsaid.\n";
    let rules = format!(
        "- FIRST PARAGRAPH — their looks, from the \"Appearance\" facts ONLY: age and sex, \
         ethnicity, hair, eyes, facial hair when listed — ending with what they are \
         wearing, phrased naturally as worn clothing, not as a list. Open it with one \
         plain sentence like \"{name} is a middle-aged man with dark brown hair and blue \
         eyes.\" Use every Appearance fact given and invent nothing beyond connective \
         phrasing — never guess at features the sheet does not state (face shape, \
         complexion, scars, marks, build).\n\
         - Describe only what is listed — never remark on what is absent or unstated \
         (never write \"no visible scars\", \"clean-shaven\", or \"wears no hat\"). Never \
         describe their expression.\n\
         - SECOND PARAGRAPH — how they come across in person, drawn ONLY from the \
         attributes and skills: their mind, their social presence, and how they move and \
         carry themselves. Give each of those a sentence or two whenever the sheet speaks \
         to it.\n\
         {calibration}\
         - Never mention that this is a game, a character, or a sheet.\n\
         - Write in third person, present tense. Use their name.\n\
         - Output ONLY the description: exactly TWO paragraphs separated by one blank \
         line — the FIRST at most about 80 words, the SECOND at most about 120 words. \
         No headings, no lists, no preamble.\n"
    );
    format!(
        "{opening}\n\n\
         Rules:\n\
         {rules}\n\
         Character sheet:\n{stats}",
        stats = stats_block(stats)
    )
}

/// The OpenAI-compatible message list for one persona generation.
fn persona_messages(prompt: &str) -> Vec<Value> {
    vec![json!({ "role": "user", "content": prompt })]
}

// ---------------------------------------------------------------------------
// LLM transport
// ---------------------------------------------------------------------------

/// One attempt against the MAIN local LLM endpoint (the same client NPC turns
/// use), honoring the saved sampling with a persona max_tokens override.
async fn main_llm_completion(
    state: &AppState,
    messages: &[Value],
) -> Result<String, String> {
    let persona_settings = AppSettings::load(&state.config.settings_path);
    let sampling = crate::llm::Sampling::from_settings(&persona_settings.llm.sampling)
        .with_overrides(crate::llm::GenerationOptions {
            temperature: None,
            max_tokens: Some(PERSONA_MAX_TOKENS),
        });
    let target = crate::llm::LlmTarget::resolve(&persona_settings, &state.config);
    let (text, _metrics) = crate::llm::chat_completion_capturing_sampled(
        &target,
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

/// Runs one persona generation from the STORED capture (capture.json) and
/// writes `persona.json`: one text prompt against the main LLM endpoint, the
/// character sheet (appearance facts + stats) as the only source.
///
/// On failure the previous good `persona.json` description is KEPT and only
/// the error fields are refreshed, so a transient LLM outage never destroys a
/// working persona.
pub(crate) async fn generate_from_stored_capture(state: &AppState) -> WebResult<Value> {
    let dir = persona_dir(state);
    let capture = read_json(&capture_path(&dir)).ok_or_else(|| {
        WebError::from(anyhow::anyhow!(
            "no capture stored yet — the mod has not uploaded a persona capture"
        ))
    })?;

    let settings = AppSettings::load(&state.config.settings_path).persona;
    let stats = stats_of(&capture);

    let note;
    let mut description: Option<String> = None;
    // The exact prompt text that PRODUCED the description (persisted with the
    // record so the Persona page can show precisely what the LLM was asked).
    let mut used_prompt: Option<String> = None;

    if !settings.enabled {
        note = "persona generation is disabled in settings".to_string();
    } else {
        let prompt = persona_prompt(&stats);
        match main_llm_completion(state, &persona_messages(&prompt)).await {
            Ok(text) => {
                description = Some(text);
                used_prompt = Some(prompt);
                note = "generated from the character-data snapshot".to_string();
            }
            Err(error) => {
                note = format!("generation failed ({error})");
                tracing::warn!(target: "chasm::persona", %error, "persona generation failed");
            }
        }
    }

    let now = chrono_now_iso();
    let previous = read_json(&persona_path(&dir)).unwrap_or_else(|| json!({}));
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
                "source": "game_data",
                "model_note": note,
                "stats": stats,
                // The exact prompt text sent to the LLM for this description.
                // Shown on the Persona page.
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

/// Spawns [`generate_from_stored_capture`] on a background task. If one is
/// already running, requests a re-run instead (the running task regenerates
/// from the freshly stored capture right after the current pass) — so a save
/// landing mid-generation is never lost. Returns whether a NEW task was
/// started. NPC turn generation is never awaited on this — the whole point.
pub(crate) fn spawn_generation(state: Arc<AppState>) -> bool {
    if GENERATING.swap(true, Ordering::SeqCst) {
        RERUN_REQUESTED.store(true, Ordering::SeqCst);
        return false; // the in-flight task re-runs for us
    }
    tokio::spawn(async move {
        loop {
            if let Err(error) = generate_from_stored_capture(&state).await {
                tracing::warn!(target: "chasm::persona", error = %format!("{error:?}"), "persona generation task failed");
            }
            if !RERUN_REQUESTED.swap(false, Ordering::SeqCst) {
                break;
            }
            tracing::info!(target: "chasm::persona", "re-running persona generation for a capture that arrived mid-generation");
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
/// `mod-source/docs/persona.md`): a pure game-data snapshot (stats +
/// appearance). Stores it under the active profile's persona dir and queues an
/// async generation. Returns immediately; never blocks on the LLM.
///
/// Response: `{ "status": "stored", "generation": "queued" | "busy" |
/// "unchanged" | "disabled" }`. `busy` still regenerates: the in-flight
/// generation re-runs from this capture as soon as it finishes (see
/// [`spawn_generation`]).
pub async fn receive_capture(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> WebResult<Json<Value>> {
    if !body.is_object() {
        return Err(WebError::from(anyhow::anyhow!("body must be a JSON object")));
    }

    let dir = persona_dir(&state);
    fs::create_dir_all(&dir).map_err(WebError::from)?;
    remove_stale_images(&dir);

    let mut capture = body.clone();
    if let Some(map) = capture.as_object_mut() {
        // Tolerate uploads from an outdated plugin: the screenshot feature is
        // retired, so any image payload is simply dropped, never stored.
        map.remove("image_base64");
        map.remove("image_format");
        map.insert("received_at".to_string(), json!(chrono_now_iso()));
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
        // Same snapshot as the stored persona and a description already
        // exists: skip the LLM. Save-driven captures (the mod's whole trigger
        // model, see docs/persona.md) always regenerate. The unchanged
        // short-circuit only remains for non-save uploads.
        "unchanged"
    } else if spawn_generation(state.clone()) {
        "queued"
    } else {
        "busy"
    };

    Ok(Json(json!({
        "status": "stored",
        "generation": generation,
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
            "hair_length": "0.80",
            "eye_color": "Blue",
            "facial_hair": "Chin Curtain",
            "trigger": "quicksave",
        })
    }

    #[test]
    fn prompt_carries_appearance_facts_and_stats() {
        let stats = stats_of(&capture_body("Courier"));
        let prompt = persona_prompt(&stats);
        assert!(prompt.contains("Write a sketch of Courier"));
        assert!(!prompt.contains("photo"), "the vision path is retired");
        // Appearance facts render as natural language, never raw game data.
        assert!(prompt.contains("Appearance (authoritative character data):"));
        assert!(prompt.contains("- Sex: male"));
        assert!(prompt.contains("- Age: older, well past middle age"), "race variant drives age");
        assert!(prompt.contains("- Ethnicity: white"), "Caucasian maps to a plain word");
        assert!(prompt.contains("- Hair: blonde, styled in a wavy, worn long"));
        assert!(!prompt.contains("#D6B569"), "hex colors are mapped, never shown");
        assert!(prompt.contains("- Eyes: blue"));
        assert!(prompt.contains("- Facial hair: chin curtain"));
        assert!(prompt.contains("- Wearing: Leather Armor, Goggles"));
        // Natural-language stats: full attribute names + explicit scale +
        // qualitative band + meaning hint, never bare abbreviations.
        assert!(prompt.contains("Strength 9 of 10 — exceptional (raw physical power and muscle)"));
        assert!(prompt.contains("Charisma 1 of 10 — abysmal (charm and social grace)"));
        assert!(!prompt.contains("STR 9,"), "no raw abbreviations in the rendered attributes");
        assert!(prompt.contains(
            "Skills, each rated 0 to 100 — untrained people sit around 10 to 30, 85 or more \
             is true mastery: Barter 15, Guns 45, Speech 4, Unarmed 80"
        ));
        // Only genuine extremes are called out (Barter 15 is ordinary early-game).
        assert!(prompt.contains("Notable skills: Speech 4 of 100 (dreadful); Unarmed 80 of 100 (highly skilled)"));
        assert!(!prompt.contains("Perks:"), "perks must stay out of the prompt");
        assert!(prompt.contains("FIRST PARAGRAPH"));
        assert!(prompt.contains("SECOND PARAGRAPH"));
        assert!(prompt.contains("exactly TWO paragraphs"));
    }

    #[test]
    fn prompt_calibration_is_honest_and_stat_agnostic() {
        let stats = stats_of(&capture_body("Courier"));
        let prompt = persona_prompt(&stats);
        // The honesty rule is present with its full weight...
        assert!(prompt.contains("crippling deficiency"));
        assert!(prompt.contains("do not soften it"));
        assert!(prompt.contains("genuine mastery"));
        // ...the old euphemism-teaching examples are gone...
        assert!(!prompt.contains("halting and wordless"));
        assert!(!prompt.contains("rock-bottom Speech"));
        // ...and the RULES never special-case a skill (skill names may only
        // appear as sheet data, i.e. after the character sheet starts).
        let rules_end = prompt.find("Character sheet:").unwrap();
        let rules = &prompt[..rules_end];
        for skill in ["Unarmed", "Speech", "Guns", "Barter"] {
            assert!(!rules.contains(skill), "rules must not name {skill}");
        }
    }

    #[test]
    fn hair_color_hex_maps_to_plain_words() {
        assert_eq!(hair_color_name("#000000"), Some("black"));
        assert_eq!(hair_color_name("#54462E"), Some("dark brown"));
        assert_eq!(hair_color_name("#8A6E4B"), Some("brown"));
        assert_eq!(hair_color_name("#D6B569"), Some("blonde"));
        assert_eq!(hair_color_name("#F2E6C9"), Some("platinum blonde"));
        assert_eq!(hair_color_name("#8B2E16"), Some("red"));
        assert_eq!(hair_color_name("#AAAAAA"), Some("gray"));
        assert_eq!(hair_color_name("#F4F4F4"), Some("white"));
        assert_eq!(hair_color_name("not-a-color"), None);
        assert_eq!(hair_color_name(""), None);
    }

    #[test]
    fn race_maps_to_age_and_ethnicity() {
        // Plain adult races claim nothing — age_years speaks instead.
        assert_eq!(race_age_marker("Caucasian"), None);
        assert_eq!(race_age_marker("Caucasian Old"), Some("older, well past middle age"));
        assert_eq!(race_age_marker("Hispanic Old Aged"), Some("older, well past middle age"));
        assert_eq!(race_age_marker("Asian Middle Aged"), Some("middle-aged"));
        assert_eq!(race_age_marker("African American Child"), Some("a child"));
        assert_eq!(race_ethnicity("Caucasian Old"), "white");
        assert_eq!(race_ethnicity("African American"), "Black");
        assert_eq!(race_ethnicity("Hispanic Old Aged"), "Hispanic");
        assert_eq!(race_ethnicity("Asian"), "Asian");
        // Unknown/mod races pass through minus the age tokens.
        assert_eq!(race_ethnicity("Ghoul Old"), "Ghoul");
    }

    #[test]
    fn appearance_lines_skip_absent_facts_and_length_middles() {
        // Minimal capture: nothing appearance-ish → no lines at all.
        let empty = appearance_lines(&json!({}));
        assert!(empty.is_empty());
        // Facial hair and hair length only appear when meaningful.
        let stats = json!({
            "sex": "female",
            "race": "Hispanic",
            "age_years": "23",
            "hair_color": "#101010",
            "hair_style": "Bob",
            "hair_length": "0.50",
            "eye_color": "Green",
        });
        let lines = appearance_lines(&stats).join("\n");
        assert!(lines.contains("- Sex: female"));
        assert!(lines.contains("- Age: about 23 years old"), "FaceGen years drive the age");
        assert!(lines.contains("- Ethnicity: Hispanic"));
        assert!(lines.contains("- Hair: black, styled in a bob"));
        assert!(!lines.contains("worn long"), "middling hair length says nothing");
        assert!(!lines.contains("kept short"));
        assert!(lines.contains("- Eyes: green"));
        assert!(!lines.contains("Facial hair"), "absent facts are never mentioned");
    }

    #[test]
    fn age_line_priority_marker_then_years_then_silence() {
        // Race marker wins even when years are present (aged races look aged).
        let both = appearance_lines(&json!({ "race": "Caucasian Old", "age_years": "30" })).join("\n");
        assert!(both.contains("- Age: older, well past middle age"));
        assert!(!both.contains("about 30 years old"));
        // Plain race + years → years.
        let years = appearance_lines(&json!({ "race": "Caucasian", "age_years": "44" })).join("\n");
        assert!(years.contains("- Age: about 44 years old"));
        // Plain race, no years → NO age claim at all.
        let silent = appearance_lines(&json!({ "race": "Caucasian" })).join("\n");
        assert!(!silent.contains("- Age:"), "no data, no claim: {silent}");
        assert!(silent.contains("- Ethnicity: white"));
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
        assert_eq!(
            lines[0],
            "Skills, each rated 0 to 100 — untrained people sit around 10 to 30, 85 or more \
             is true mastery: Guns 45, Speech 4, Sneak 30, Unarmed 90"
        );
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
    fn messages_are_a_single_text_user_turn() {
        let messages = persona_messages("describe");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "describe");
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
    fn custom_note_survives_description_regeneration() {
        let dir = std::env::temp_dir().join(format!(
            "chasm-persona-note-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();

        // Absent → empty (injection sees nothing).
        assert_eq!(read_custom_note(&dir), "");

        // User saves a custom addition (leading/trailing space trimmed).
        write_custom_note(&dir, "  The Courier owes Benny a bullet.  ").unwrap();
        assert_eq!(read_custom_note(&dir), "The Courier owes Benny a bullet.");

        // Simulate a persona regeneration: persona.json is rewritten from
        // scratch (as generate_from_stored_capture does on every save). The
        // custom note lives in its OWN file and must be untouched.
        write_json_atomic(
            &persona_path(&dir),
            &json!({ "description": "A first generated description.", "stats": {} }),
        )
        .unwrap();
        assert_eq!(read_custom_note(&dir), "The Courier owes Benny a bullet.");
        write_json_atomic(
            &persona_path(&dir),
            &json!({ "description": "A totally rewritten description.", "stats": {} }),
        )
        .unwrap();
        assert_eq!(
            read_custom_note(&dir),
            "The Courier owes Benny a bullet.",
            "custom note must survive regeneration"
        );

        // Saving an empty/whitespace note clears it back to nothing.
        write_custom_note(&dir, "   ").unwrap();
        assert_eq!(read_custom_note(&dir), "");

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
