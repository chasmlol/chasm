//! Companions — author a brand-new character in chasm and get them as a
//! spawned, named, voiced follower in Fallout: New Vegas.
//!
//! Design: mod-source `docs/companions-architecture.md`. chasm's half:
//!   * `GET  /api/ui/v1/companions`            — pool status (plugin registry
//!     merged with card + voice-clone state) + recent command acks.
//!   * `POST /api/ui/v1/companions`            — create: real character card
//!     PNG in the active profile's characters dir (so chat/retrieval treat the
//!     companion like any character), optional voice clip through the existing
//!     clone pipeline, and a `create` command file for the NVSE plugin.
//!   * `POST /api/ui/v1/companions/:slot/op`   — summon / dismiss / despawn /
//!     release / face_design / rename relayed as command files.
//!
//! The file protocol mirrors the plugin side (`CHASM_COMPANION_V1` key=value
//! command files under `<bridge>/control/companions/`, acks under `acks/`, the
//! plugin-owned registry at `<bridge>/companions/registry.txt`).

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Path as AxPath, State},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use chasm_core::AppSettings;

use crate::{AppState, WebError, WebResult};

use super::books::write_png_character_json;

const COMMAND_VERSION: &str = "CHASM_COMPANION_V1";
const ACK_VERSION: &str = "CHASM_COMPANION_ACK_V1";
const REGISTRY_VERSION: &str = "CHASM_COMPANION_REGISTRY_V1";
/// Upper bound on registry slots parsed (sanity cap; the real pool size comes
/// from the game profile's `companions.bodies[].slots`).
const MAX_POOL_SLOTS: usize = 256;

// ===========================================================================
// Game-declared capabilities (profile.json `companions` block)
//
// Everything game-specific — pool layout, body variants and the command
// fields they map to, whether the game can do in-game face design, hint copy —
// is declared by the ACTIVE game profile, which the game's mod ships and
// injects. chasm renders capabilities; it knows nothing about any one game.
// ===========================================================================

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionBodyView {
    pub id: String,
    pub label: String,
    pub slots: usize,
    pub free: usize,
}

#[derive(Clone, Default)]
struct BodyVariant {
    id: String,
    label: String,
    slots: usize,
    command_fields: Vec<(String, String)>,
}

#[derive(Clone, Default)]
struct CompanionCapabilities {
    enabled: bool,
    bodies: Vec<BodyVariant>,
    in_game_face_design: bool,
    face_design_hint: String,
    voice_hint: String,
}

impl CompanionCapabilities {
    fn pool_size(&self) -> usize {
        self.bodies.iter().map(|b| b.slots).sum::<usize>().min(MAX_POOL_SLOTS)
    }
}

fn capabilities(state: &AppState) -> CompanionCapabilities {
    let settings = AppSettings::load(&state.config.settings_path);
    let profile_id = settings.active_profile_id(&state.config.profiles_dir);
    let Some(profile) = chasm_core::GameProfile::read(&state.config.profiles_dir, &profile_id)
    else {
        return CompanionCapabilities::default();
    };
    let Some(block) = profile.extra.get("companions") else {
        return CompanionCapabilities::default();
    };
    let enabled = block.get("enabled").and_then(Value::as_bool).unwrap_or(false);
    let mut bodies = Vec::new();
    for body in block.get("bodies").and_then(Value::as_array).into_iter().flatten() {
        let id = body.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
        if id.is_empty() {
            continue;
        }
        let label = body
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or(&id)
            .to_string();
        let slots = body.get("slots").and_then(Value::as_u64).unwrap_or(0) as usize;
        let command_fields = body
            .get("commandFields")
            .and_then(Value::as_object)
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        bodies.push(BodyVariant { id, label, slots, command_fields });
    }
    CompanionCapabilities {
        enabled: enabled && !bodies.is_empty(),
        bodies,
        in_game_face_design: block
            .get("inGameFaceDesign")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        face_design_hint: block
            .get("faceDesignHint")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        voice_hint: block
            .get("voiceHint")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    }
}

// ===========================================================================
// Bridge root + file protocol plumbing
// ===========================================================================

/// The rendezvous root the NVSE plugin uses — same resolution as the in-process
/// bridge fold: helper config's first `nativeBridgeRoots` entry, else the fixed
/// default (`%LOCALAPPDATA%\chasm\bridge`).
fn bridge_root(state: &AppState) -> PathBuf {
    let settings = AppSettings::load(&state.config.settings_path);
    let config_path = settings.launcher.helper_config.trim().to_string();
    if let Ok(config) = chasm_fnv_bridge::load_config(Path::new(&config_path)) {
        if let Some(root) = config.native_bridge_roots.first() {
            return root.clone();
        }
    }
    chasm_core::default_bridge_root()
}

fn command_dir(root: &Path) -> PathBuf {
    root.join("control").join("companions")
}

fn ack_dir(root: &Path) -> PathBuf {
    command_dir(root).join("acks")
}

fn registry_path(root: &Path) -> PathBuf {
    root.join("companions").join("registry.txt")
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Writes one `CHASM_COMPANION_V1` command file (atomic tmp+rename) and returns
/// the request id the plugin will ack under `acks/<request_id>.txt`.
fn write_command(state: &AppState, op: &str, fields: &[(String, String)]) -> WebResult<String> {
    let root = bridge_root(state);
    let dir = command_dir(&root);
    fs::create_dir_all(ack_dir(&root))?;
    let request_id = format!("comp_{}_{}", op, now_millis());
    let mut body = format!("{COMMAND_VERSION}\r\nrequest_id={request_id}\r\nop={op}\r\n");
    for (key, value) in fields {
        // one key=value per line; values are base64 wherever they can contain
        // newlines or non-ASCII (the *_base64 convention the plugin expects)
        body.push_str(&format!("{key}={value}\r\n"));
    }
    let final_path = dir.join(format!("{request_id}.txt"));
    let temp_path = dir.join(format!("{request_id}.tmp"));
    fs::write(&temp_path, body.as_bytes())?;
    fs::rename(&temp_path, &final_path)?;
    tracing::info!("companions: queued {op} command {request_id}");
    Ok(request_id)
}

/// Parses `key=value` lines after a required header line. Mirrors the plugin's
/// `ParseKeyValueLines` (first `=` splits; keys/values trimmed).
fn parse_key_value_lines(text: &str, header: &str) -> Option<std::collections::BTreeMap<String, String>> {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some(header) {
        return None;
    }
    let mut fields = std::collections::BTreeMap::new();
    for line in lines {
        if let Some((key, value)) = line.split_once('=') {
            fields.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    Some(fields)
}

fn decode_base64_text(value: Option<&String>) -> String {
    value
        .and_then(|v| STANDARD.decode(v).ok())
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

// ===========================================================================
// Views
// ===========================================================================

#[derive(Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionSlotView {
    pub slot: usize,
    pub claimed: bool,
    pub npc_key: String,
    pub name: String,
    pub character_name: String,
    pub voice: String,
    /// Body-variant id as reported by the game plugin's registry.
    pub body: String,
    pub face_designed: bool,
    pub waiting: bool,
    pub status: String,
    pub appearance_saved: bool,
    pub has_card: bool,
    /// none | reference | cloning | cloned | failed
    pub voice_status: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionAckView {
    pub request_id: String,
    pub ok: bool,
    pub error: String,
    pub op: String,
    pub slot: i32,
    pub npc_key: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionsView {
    pub enabled: bool,
    pub in_game_face_design: bool,
    pub face_design_hint: String,
    pub voice_hint: String,
    pub bodies: Vec<CompanionBodyView>,
    pub registry_found: bool,
    pub registry_rev: u32,
    pub slots: Vec<CompanionSlotView>,
    pub acks: Vec<CompanionAckView>,
}

/// `GET /api/ui/v1/companions`
pub(crate) async fn list_companions(
    State(state): State<Arc<AppState>>,
) -> WebResult<Json<CompanionsView>> {
    let caps = capabilities(&state);
    let pool_size = caps.pool_size();
    let root = bridge_root(&state);
    let mut view = CompanionsView {
        enabled: caps.enabled,
        in_game_face_design: caps.in_game_face_design,
        face_design_hint: caps.face_design_hint.clone(),
        voice_hint: caps.voice_hint.clone(),
        bodies: caps
            .bodies
            .iter()
            .map(|b| CompanionBodyView {
                id: b.id.clone(),
                label: b.label.clone(),
                slots: b.slots,
                free: b.slots,
            })
            .collect(),
        registry_found: false,
        registry_rev: 0,
        slots: (0..pool_size)
            .map(|slot| CompanionSlotView {
                slot,
                status: "unclaimed".into(),
                voice_status: "none".into(),
                ..Default::default()
            })
            .collect(),
        acks: Vec::new(),
    };
    if !caps.enabled {
        return Ok(Json(view));
    }

    if let Ok(text) = fs::read_to_string(registry_path(&root)) {
        if let Some(fields) = parse_key_value_lines(&text, REGISTRY_VERSION) {
            view.registry_found = true;
            view.registry_rev = fields
                .get("rev")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            for slot in 0..pool_size {
                let prefix = format!("slot{slot}.");
                let get = |key: &str| fields.get(&format!("{prefix}{key}")).cloned();
                let entry = &mut view.slots[slot];
                entry.claimed = get("claimed").as_deref() == Some("1");
                if !entry.claimed {
                    continue;
                }
                entry.npc_key = get("npc_key").unwrap_or_default();
                entry.name = decode_base64_text(fields.get(&format!("{prefix}name_base64")));
                entry.character_name =
                    decode_base64_text(fields.get(&format!("{prefix}character_base64")));
                entry.voice = get("voice").unwrap_or_default();
                entry.body = get("body").unwrap_or_default();
                entry.face_designed = get("face_designed").as_deref() == Some("1");
                entry.waiting = get("waiting").as_deref() == Some("1");
                entry.status = get("status").unwrap_or_else(|| "claimed".into());
                entry.appearance_saved = get("app.valid").as_deref() == Some("1");
                if let Some(body) = view.bodies.iter_mut().find(|b| b.id == entry.body) {
                    body.free = body.free.saturating_sub(1);
                }
            }
        }
    }

    // Card presence + voice-clone status come from chasm's own storage.
    let characters_dir = state.config.active_profile_paths().characters_dir();
    let settings = AppSettings::load(&state.config.settings_path);
    let profile_id = settings.active_profile_id(&state.config.profiles_dir);
    let voices_dir = state.config.profile_paths(&profile_id).voices_dir();
    let engine = settings.tts.local_engine.clone();
    for entry in &mut view.slots {
        if !entry.claimed {
            continue;
        }
        let card_name = if entry.character_name.is_empty() {
            &entry.name
        } else {
            &entry.character_name
        };
        entry.has_card = !card_name.is_empty()
            && characters_dir.join(format!("{card_name}.png")).is_file();
        entry.voice_status = voice_status(&voices_dir, card_name, &engine);
    }

    // Recent acks (newest first, capped) so the UI can surface command errors.
    let mut acks: Vec<(SystemTime, CompanionAckView)> = Vec::new();
    if let Ok(read) = fs::read_dir(ack_dir(&root)) {
        for file in read.flatten() {
            let path = file.path();
            if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Some(fields) = parse_key_value_lines(&text, ACK_VERSION) else {
                continue;
            };
            let modified = file
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            acks.push((
                modified,
                CompanionAckView {
                    request_id: fields.get("request_id").cloned().unwrap_or_default(),
                    ok: fields.get("ok").map(String::as_str) == Some("1"),
                    error: fields.get("error").cloned().unwrap_or_default(),
                    op: fields.get("op").cloned().unwrap_or_default(),
                    slot: fields
                        .get("slot")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(-1),
                    npc_key: fields.get("npc_key").cloned().unwrap_or_default(),
                },
            ));
        }
    }
    acks.sort_by(|a, b| b.0.cmp(&a.0));
    view.acks = acks.into_iter().take(20).map(|(_, ack)| ack).collect();

    Ok(Json(view))
}

fn voice_status(voices_dir: &Path, character_name: &str, engine: &str) -> String {
    if character_name.is_empty() {
        return "none".into();
    }
    let base = voices_dir.join(character_name);
    let engine_dir = base.join(engine);
    if engine_dir.join("sample.wav").is_file() {
        return "cloned".into();
    }
    if engine_dir.join(".cloning").is_file() {
        return "cloning".into();
    }
    if engine_dir.join(".failed").is_file() {
        return "failed".into();
    }
    if base.join("reference.wav").is_file() {
        return "reference".into();
    }
    "none".into()
}

// ===========================================================================
// Create
// ===========================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreateCompanionBody {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub personality: String,
    #[serde(default)]
    pub first_message: String,
    #[serde(default)]
    pub example_dialogue: String,
    #[serde(default)]
    pub system_prompt: String,
    /// Body-variant id from the profile's `companions.bodies` (defaults to the first).
    #[serde(default)]
    pub body: String,
    /// Design the face in game before spawning (only when the profile supports it).
    #[serde(default = "default_true")]
    pub face_design: bool,
    /// Voice clip for the clone pipeline (base64; WAV/FLAC/OGG recommended).
    #[serde(default)]
    pub voice_base64: String,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreateCompanionResponse {
    pub request_id: String,
    pub card_id: String,
    pub voice_saved: bool,
    pub clone_started: bool,
}

/// `POST /api/ui/v1/companions`
pub(crate) async fn create_companion(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateCompanionBody>,
) -> WebResult<Json<CreateCompanionResponse>> {
    let caps = capabilities(&state);
    if !caps.enabled {
        return Err(WebError::from(anyhow::anyhow!(
            "the active game profile does not support companions"
        )));
    }
    let variant = if body.body.trim().is_empty() {
        caps.bodies.first().cloned()
    } else {
        caps.bodies.iter().find(|b| b.id == body.body.trim()).cloned()
    }
    .ok_or_else(|| WebError::from(anyhow::anyhow!("unknown body variant '{}'", body.body)))?;
    let face_design = body.face_design && caps.in_game_face_design;

    let name = body.name.trim().to_string();
    if name.is_empty() {
        return Err(WebError::from(anyhow::anyhow!("companion name is required")));
    }
    if name.contains(['/', '\\', ':']) || name.contains("..") {
        return Err(WebError::from(anyhow::anyhow!(
            "companion name must not contain path separators"
        )));
    }

    // Refuse only when this name is already an ACTIVE companion; a plain card
    // collision is the normal retry-after-failed-spawn case and is upserted.
    let root = bridge_root(&state);
    if let Ok(text) = fs::read_to_string(registry_path(&root)) {
        if let Some(fields) = parse_key_value_lines(&text, REGISTRY_VERSION) {
            for slot in 0..MAX_POOL_SLOTS {
                let prefix = format!("slot{slot}.");
                if fields.get(&format!("{prefix}claimed")).map(String::as_str) != Some("1") {
                    continue;
                }
                let slot_character =
                    decode_base64_text(fields.get(&format!("{prefix}character_base64")));
                if slot_character.eq_ignore_ascii_case(&name) {
                    return Err(WebError::from(anyhow::anyhow!(
                        "'{name}' is already an active companion (slot {}); release them first",
                        slot + 1
                    )));
                }
            }
        }
    }

    // 1) The character card — a real card in the player's character book. If a
    //    card with this name already exists (e.g. an earlier create whose
    //    in-game half failed), overlay the typed fields onto it instead of
    //    erroring, so Create is safely retryable.
    let characters_dir = state.config.active_profile_paths().characters_dir();
    fs::create_dir_all(&characters_dir)?;
    let card_path = characters_dir.join(format!("{name}.png"));
    let body_index = caps.bodies.iter().position(|b| b.id == variant.id).unwrap_or(0);
    let card_png = match fs::read(&card_path) {
        Ok(existing) => overlay_card_fields(&existing, &name, &body)?,
        Err(_) => build_card_png(&name, &body, body_index)?,
    };
    fs::write(&card_path, &card_png)?;
    tracing::info!("companions: wrote character card {}", card_path.display());

    // 2) The voice: land the clip where the clone pipeline expects it, then
    //    kick the existing per-engine clone run (it clones every folder that
    //    has a reference.wav, companions included).
    let mut voice_saved = false;
    let mut clone_started = false;
    if !body.voice_base64.trim().is_empty() {
        let bytes = STANDARD
            .decode(body.voice_base64.trim())
            .map_err(|e| WebError::from(anyhow::anyhow!("voice clip is not valid base64: {e}")))?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(WebError::from(anyhow::anyhow!("voice clip too large (>64MB)")));
        }
        let settings = AppSettings::load(&state.config.settings_path);
        let profile_id = settings.active_profile_id(&state.config.profiles_dir);
        let voices_dir = state.config.profile_paths(&profile_id).voices_dir();
        let voice_dir = voices_dir.join(&name);
        fs::create_dir_all(&voice_dir)?;
        // Peak-normalize at upload: TTS engines reproduce reference loudness and
        // user recordings are routinely ~20 dB quieter than game-extracted
        // voices. This also covers the pre-clone window where the runtime falls
        // back to reference.wav. Non-PCM formats pass through untouched (the
        // clone step normalizes its prompt separately).
        let bytes = normalize_wav_peak(bytes);
        fs::write(voice_dir.join("reference.wav"), &bytes)?;
        voice_saved = true;
        // Pre-mark the engine dir so the UI shows "cloning" immediately.
        let engine_dir = voice_dir.join(&settings.tts.local_engine);
        let _ = fs::create_dir_all(&engine_dir);
        let _ = fs::remove_file(engine_dir.join(".failed"));
        let _ = fs::write(engine_dir.join(".cloning"), "");
        crate::start_voice_clone(&state);
        clone_started = true;
    }

    // 3) The plugin command: claim a slot, name it, spawn + follow (face
    //    design first when supported + requested). Executes when the user is
    //    in game. Body-variant specifics ride the PROFILE-declared command
    //    fields — chasm forwards them without interpreting.
    let mut fields: Vec<(String, String)> = vec![
        ("name_base64".into(), STANDARD.encode(name.as_bytes())),
        ("character_base64".into(), STANDARD.encode(name.as_bytes())),
        ("face_design".into(), if face_design { "1" } else { "0" }.into()),
        ("voice".into(), name.clone()),
    ];
    fields.extend(variant.command_fields.iter().cloned());
    let request_id = write_command(&state, "create", &fields)?;

    Ok(Json(CreateCompanionResponse {
        request_id,
        card_id: name,
        voice_saved,
        clone_started,
    }))
}

/// Peak-normalizes a 16-bit PCM WAV to ~-1 dBFS in place (data chunk scaled).
/// Anything that isn't a plain PCM16 WAV is returned unchanged — the clone
/// pipeline separately normalizes the prompt it derives.
fn normalize_wav_peak(bytes: Vec<u8>) -> Vec<u8> {
    // Minimal RIFF walk: find fmt (PCM16 mono/stereo any rate) + data chunk.
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return bytes;
    }
    let mut pos = 12usize;
    let mut is_pcm16 = false;
    let mut data_range: Option<(usize, usize)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            return bytes;
        }
        if id == b"fmt " && size >= 16 {
            let format = u16::from_le_bytes([bytes[body], bytes[body + 1]]);
            let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            is_pcm16 = format == 1 && bits == 16;
        } else if id == b"data" {
            data_range = Some((body, size & !1));
        }
        pos = body + size + (size & 1);
    }
    let (Some((start, len)), true) = (data_range, is_pcm16) else {
        return bytes;
    };
    let mut out = bytes;
    let mut peak: i32 = 0;
    for i in (start..start + len).step_by(2) {
        let v = i16::from_le_bytes([out[i], out[i + 1]]) as i32;
        peak = peak.max(v.abs());
    }
    if peak < 16 {
        return out; // effectively silence; don't amplify noise
    }
    let gain = (0.89f64 * 32767.0) / peak as f64;
    if gain <= 1.0 {
        return out; // already loud enough; never attenuate
    }
    for i in (start..start + len).step_by(2) {
        let v = i16::from_le_bytes([out[i], out[i + 1]]) as f64 * gain;
        let clamped = v.clamp(-32768.0, 32767.0) as i16;
        out[i..i + 2].copy_from_slice(&clamped.to_le_bytes());
    }
    out
}

/// Overlays the create-form fields onto an EXISTING card PNG (same dual-write
/// as the Characters book: `data.<key>` + legacy top-level). Only non-empty
/// form fields overwrite, so a retry with a sparse form can't wipe a card the
/// user has since edited. Falls back to a fresh card if the PNG has no data.
fn overlay_card_fields(existing: &[u8], name: &str, body: &CreateCompanionBody) -> WebResult<Vec<u8>> {
    let Some(json) = super::books::read_png_character_json(existing) else {
        return build_card_png(name, body, 0);
    };
    let mut card: Value = serde_json::from_str(&json).unwrap_or_else(|_| Value::Object(Map::new()));
    if !card.is_object() {
        card = Value::Object(Map::new());
    }
    {
        let obj = card.as_object_mut().expect("card is an object");
        if !obj.get("data").map(Value::is_object).unwrap_or(false) {
            obj.insert("data".to_string(), Value::Object(Map::new()));
        }
        let fields: [(&str, &str); 6] = [
            ("name", name),
            ("description", &body.description),
            ("personality", &body.personality),
            ("first_mes", &body.first_message),
            ("mes_example", &body.example_dialogue),
            ("system_prompt", &body.system_prompt),
        ];
        for (key, value) in fields {
            if value.trim().is_empty() && key != "name" {
                continue;
            }
            let value = Value::String(value.to_string());
            obj.insert(key.to_string(), value.clone());
            if let Some(data) = obj.get_mut("data").and_then(Value::as_object_mut) {
                data.insert(key.to_string(), value);
            }
        }
    }
    let updated = serde_json::to_string(&card)?;
    super::books::write_png_character_json(existing, &updated)
        .ok_or_else(|| WebError::from(anyhow::anyhow!("failed to re-embed card JSON")))
}

/// Builds a brand-new V2+V3 character card PNG: a generated placeholder
/// portrait with the card JSON in `chara` + `ccv3` tEXt chunks (same chunk
/// layout the Characters book round-trips).
fn build_card_png(name: &str, body: &CreateCompanionBody, body_index: usize) -> WebResult<Vec<u8>> {
    let mut data = Map::new();
    data.insert("name".into(), Value::String(name.into()));
    data.insert("description".into(), Value::String(body.description.clone()));
    data.insert("personality".into(), Value::String(body.personality.clone()));
    data.insert("first_mes".into(), Value::String(body.first_message.clone()));
    data.insert(
        "mes_example".into(),
        Value::String(body.example_dialogue.clone()),
    );
    data.insert(
        "system_prompt".into(),
        Value::String(body.system_prompt.clone()),
    );
    data.insert("scenario".into(), Value::String(String::new()));

    let mut card = Map::new();
    card.insert("spec".into(), Value::String("chara_card_v3".into()));
    card.insert("spec_version".into(), Value::String("3.0".into()));
    // Legacy top-level mirror (V2-only readers), matching the book's dual-write.
    for (key, value) in &data {
        card.insert(key.clone(), value.clone());
    }
    card.insert("data".into(), Value::Object(data));
    let card_json = serde_json::to_string(&Value::Object(card))?;

    let base_png = placeholder_portrait_png(body_index);
    write_png_character_json(&base_png, &card_json)
        .ok_or_else(|| WebError::from(anyhow::anyhow!("failed to embed card JSON")))
}

/// A minimal, valid 96×96 RGB PNG generated in-process (no image deps): zlib
/// "stored" deflate blocks + the shared CRC helper. Tint rotates per body
/// variant so cards are distinguishable in the book until the user drops in a
/// portrait.
fn placeholder_portrait_png(body_index: usize) -> Vec<u8> {
    const SIZE: usize = 96;
    const TINTS: [(u8, u8, u8); 4] = [(58, 74, 94), (94, 64, 84), (64, 88, 70), (92, 84, 58)];
    let (r, g, b) = TINTS[body_index % TINTS.len()];

    // Raw scanlines: filter byte 0 + RGB per pixel, with a simple border shade.
    let mut raw = Vec::with_capacity(SIZE * (1 + SIZE * 3));
    for y in 0..SIZE {
        raw.push(0u8);
        for x in 0..SIZE {
            let edge = x < 4 || y < 4 || x >= SIZE - 4 || y >= SIZE - 4;
            let shade = if edge { 24 } else { 0 };
            raw.push(r.saturating_add(shade));
            raw.push(g.saturating_add(shade));
            raw.push(b.saturating_add(shade));
        }
    }

    // zlib stream with stored (uncompressed) deflate blocks.
    let mut zlib = vec![0x78u8, 0x01];
    let mut offset = 0usize;
    while offset < raw.len() {
        let chunk = (raw.len() - offset).min(0xFFFF);
        let last = offset + chunk == raw.len();
        zlib.push(if last { 1 } else { 0 });
        zlib.extend_from_slice(&(chunk as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(chunk as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw[offset..offset + chunk]);
        offset += chunk;
    }
    let mut a = 1u32;
    let mut bsum = 0u32;
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        bsum = (bsum + a) % 65521;
    }
    zlib.extend_from_slice(&((bsum << 16) | a).to_be_bytes());

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, RGB

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    png.extend_from_slice(&raw_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&raw_chunk(b"IDAT", &zlib));
    png.extend_from_slice(&raw_chunk(b"IEND", &[]));
    png
}

/// `[len][type][data][crc]` — same wire layout `text_chunk` emits for tEXt.
fn raw_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    // Reuse the books module's tEXt builder shape via its public crc path:
    // text_chunk only builds tEXt, so compose generically here with its crc.
    let mut chunk = Vec::with_capacity(12 + data.len());
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    chunk.extend_from_slice(&super::books::crc32(&crc_input).to_be_bytes());
    chunk
}

// ===========================================================================
// Slot ops
// ===========================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionOpBody {
    pub op: String,
    #[serde(default)]
    pub name: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CompanionOpResponse {
    pub request_id: String,
}

/// `POST /api/ui/v1/companions/:slot/op`
pub(crate) async fn companion_op(
    State(state): State<Arc<AppState>>,
    AxPath(slot): AxPath<usize>,
    Json(body): Json<CompanionOpBody>,
) -> WebResult<Json<CompanionOpResponse>> {
    if slot >= capabilities(&state).pool_size().max(1) {
        return Err(WebError::from(anyhow::anyhow!("invalid slot {slot}")));
    }
    let op = body.op.trim().to_lowercase();
    let mut fields: Vec<(String, String)> = vec![("slot".into(), slot.to_string())];
    match op.as_str() {
        "summon" | "dismiss" | "despawn" | "release" => {}
        "face_design" => {
            if !capabilities(&state).in_game_face_design {
                return Err(WebError::from(anyhow::anyhow!(
                    "the active game profile does not support in-game face design"
                )));
            }
        }
        "rename" => {
            let name = body.name.trim();
            if name.is_empty() {
                return Err(WebError::from(anyhow::anyhow!("rename needs a name")));
            }
            fields.push(("name_base64".into(), STANDARD.encode(name.as_bytes())));
        }
        other => {
            return Err(WebError::from(anyhow::anyhow!("unknown op '{other}'")));
        }
    }
    let request_id = write_command(&state, &op, &fields)?;
    Ok(Json(CompanionOpResponse { request_id }))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn command_file_round_trips_key_values() {
        let mut body = format!("{COMMAND_VERSION}\r\nrequest_id=comp_x_1\r\nop=create\r\n");
        body.push_str("name_base64=QWRh\r\nfemale=1\r\n");
        let fields = parse_key_value_lines(&body, COMMAND_VERSION).expect("parses");
        assert_eq!(fields.get("op").map(String::as_str), Some("create"));
        assert_eq!(fields.get("female").map(String::as_str), Some("1"));
        assert_eq!(
            decode_base64_text(fields.get("name_base64")),
            "Ada".to_string()
        );
    }

    #[test]
    fn registry_parse_reads_claimed_slots() {
        let text = format!(
            "{REGISTRY_VERSION}\r\nrev=7\r\nslot0.claimed=1\r\nslot0.npc_key=ada\r\n\
             slot0.name_base64={}\r\nslot0.female=0\r\nslot0.status=spawned\r\n\
             slot0.app.valid=1\r\nslot1.claimed=0\r\n",
            STANDARD.encode("Ada")
        );
        let fields = parse_key_value_lines(&text, REGISTRY_VERSION).expect("parses");
        assert_eq!(fields.get("rev").map(String::as_str), Some("7"));
        assert_eq!(fields.get("slot0.status").map(String::as_str), Some("spawned"));
        assert_eq!(
            decode_base64_text(fields.get("slot0.name_base64")),
            "Ada".to_string()
        );
    }

    #[test]
    fn registry_with_wrong_header_is_rejected() {
        assert!(parse_key_value_lines("BOGUS\r\nrev=1\r\n", REGISTRY_VERSION).is_none());
    }

    #[test]
    fn placeholder_png_is_structurally_valid_and_card_embeds() {
        let png = placeholder_portrait_png(1);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR directly after signature, IEND at the tail.
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

        let body = CreateCompanionBody {
            name: "Ada Venture".into(),
            description: "A wandering tinkerer.".into(),
            personality: "curious".into(),
            first_message: "Hey there.".into(),
            example_dialogue: String::new(),
            system_prompt: String::new(),
            body: "female".into(),
            face_design: true,
            voice_base64: String::new(),
        };
        let card = build_card_png("Ada Venture", &body, 1).expect("card builds");
        let json = super::super::books::read_png_character_json(&card).expect("json embedded");
        let value: Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["data"]["name"], json!("Ada Venture"));
        assert_eq!(value["data"]["first_mes"], json!("Hey there."));
        assert_eq!(value["name"], json!("Ada Venture"));
    }

    #[test]
    fn overlay_updates_typed_fields_but_keeps_existing_when_blank() {
        let body_v1 = CreateCompanionBody {
            name: "Chamz".into(),
            description: "Original description.".into(),
            personality: "Original personality.".into(),
            first_message: "Yo.".into(),
            example_dialogue: String::new(),
            system_prompt: String::new(),
            body: "male".into(),
            face_design: true,
            voice_base64: String::new(),
        };
        let original = build_card_png("Chamz", &body_v1, 0).expect("card builds");

        let body_v2 = CreateCompanionBody {
            description: "New description.".into(),
            personality: String::new(), // blank -> must keep original
            ..body_v1
        };
        let updated = overlay_card_fields(&original, "Chamz", &body_v2).expect("overlay works");
        let json = super::super::books::read_png_character_json(&updated).expect("json embedded");
        let value: Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["data"]["description"], json!("New description."));
        assert_eq!(value["data"]["personality"], json!("Original personality."));
        assert_eq!(value["name"], json!("Chamz"));
    }

    #[test]
    fn normalize_wav_peak_boosts_quiet_pcm16() {
        // 8 samples of a quiet sine-ish signal at ~-24 dBFS peak (2048/32768)
        let samples: [i16; 8] = [0, 1024, 2048, 1024, 0, -1024, -2048, -1024];
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + 16).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&22050u32.to_le_bytes());
        wav.extend_from_slice(&44100u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&16u32.to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }

        let out = normalize_wav_peak(wav.clone());
        let data = &out[44..];
        let peak = data
            .chunks_exact(2)
            .map(|c| (i16::from_le_bytes([c[0], c[1]]) as i32).abs())
            .max()
            .unwrap();
        assert!(peak > 28000, "peak {peak} should be near full scale");

        // Non-WAV bytes pass through untouched.
        let junk = vec![1u8, 2, 3, 4];
        assert_eq!(normalize_wav_peak(junk.clone()), junk);
    }

    #[test]
    fn voice_status_prefers_cloned_over_reference() {
        let dir = std::env::temp_dir().join(format!("chasm_comp_voice_{}", now_millis()));
        let base = dir.join("Ada");
        fs::create_dir_all(base.join("engineA")).unwrap();
        fs::write(base.join("reference.wav"), b"riff").unwrap();
        assert_eq!(voice_status(&dir, "Ada", "engineA"), "reference");
        fs::write(base.join("engineA").join(".cloning"), b"").unwrap();
        assert_eq!(voice_status(&dir, "Ada", "engineA"), "cloning");
        fs::write(base.join("engineA").join("sample.wav"), b"riff").unwrap();
        assert_eq!(voice_status(&dir, "Ada", "engineA"), "cloned");
        let _ = fs::remove_dir_all(&dir);
    }
}
