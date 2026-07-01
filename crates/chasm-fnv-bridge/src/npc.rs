//! NPC resolution + gamestate — port of the Node helper's NPC-mapping path
//! (`slugLookupKey`, `getNpcMappingEntry`, `normalizeNpcCandidate`,
//! `getNearbyNpcCandidates`, `buildNativeGamestate`, attention/distance, and the
//! `getGeneratedLineItems`/`getSelectedSpeakerInfo` turn extraction).
//!
//! Candidates arrive as JSON `Value`s from the request's `targeting.nearby_npcs`
//! (parsed from the request file's line-10 metadata blob), so this module works
//! directly over `serde_json::Value` like the Node code worked over plain objects.

use std::collections::HashSet;

use regex::Regex;
use serde_json::{json, Map, Value};

use crate::config::BridgeConfig;
use crate::protocol::NativeRequest;

const DEFAULT_NATIVE_MAX_DISTANCE_METERS: f64 = 10.0;

/// A resolved nearby NPC ⇒ chat participant (mirrors `normalizeNpcCandidate`).
#[derive(Debug, Clone)]
pub struct NpcParticipant {
    pub participant_id: String,
    pub character_id: String,
    pub character_name: String,
    pub native_npc_key: String,
    pub native_npc_name: String,
    pub voice_type_key: String,
    pub voice_type_name: String,
    pub distance_meters: Option<f64>,
    pub distance_game_units: Option<f64>,
    pub under_crosshair: bool,
}

impl NpcParticipant {
    /// The participant object sent in `participants[]` to `/live-chats` + `/presence`.
    pub fn to_presence_value(&self) -> Value {
        json!({
            "participantId": self.participant_id,
            "type": "npc",
            "characterId": self.character_id,
            "name": self.character_name,
            "present": true,
            "audible": true,
            "distance": self.distance_meters.or(self.distance_game_units),
            "metadata": self.metadata_value(),
        })
    }

    pub fn metadata_value(&self) -> Value {
        json!({
            "nativeNpcKey": self.native_npc_key,
            "nativeNpcName": self.native_npc_name,
            "characterName": self.character_name,
            "voiceTypeKey": self.voice_type_key,
            "voiceTypeName": self.voice_type_name,
            "distanceMeters": self.distance_meters,
            "distanceGameUnits": self.distance_game_units,
            "underCrosshair": self.under_crosshair,
        })
    }
}

/// The speaker + spoken text for one generated line.
#[derive(Debug, Clone)]
pub struct GeneratedLine {
    pub participant_id: String,
    pub native_npc_key: String,
    pub native_npc_name: String,
    pub character_name: String,
    pub character_id: String,
    pub text: String,
    /// The sub-turn this line came from (for per-speaker action classification).
    /// `Null` for the ephemeral lines built from streaming deltas (TTS only).
    pub turn: Value,
}

// ---------------------------------------------------------------------------
// Slug / lookup keys
// ---------------------------------------------------------------------------

/// `slugLookupKey`: lowercase, collapse runs of non-`[a-z0-9]` to `_`, trim `_`.
pub fn slug_lookup_key(value: &str) -> String {
    let lower = value.trim().to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut pending_underscore = false;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_underscore && !out.is_empty() {
                out.push('_');
            }
            pending_underscore = false;
            out.push(ch);
        } else {
            pending_underscore = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// NPC mapping
// ---------------------------------------------------------------------------

fn normalize_mapping_entry(entry: &Value) -> Value {
    match entry {
        Value::String(s) => json!({ "characterId": s }),
        Value::Object(_) => entry.clone(),
        _ => json!({}),
    }
}

/// `getNpcMappingEntry`: direct (raw + slug) key match, then prefix/wildcard.
fn get_npc_mapping_entry<'a>(map: &'a Map<String, Value>, candidate: &Value) -> Option<&'a Value> {
    const ID_KEYS: [&str; 10] = [
        "npc_key", "npcKey", "nativeNpcKey", "characterId", "character_id", "npc_name", "npcName",
        "name", "voice_type_key", "voiceTypeKey",
    ];
    let mut keys: Vec<String> = Vec::new();
    for k in ID_KEYS {
        if let Some(raw) = candidate.get(k).and_then(value_to_string) {
            let trimmed = raw.trim().to_string();
            if !trimmed.is_empty() {
                keys.push(trimmed);
            }
            let slug = slug_lookup_key(&raw);
            if !slug.is_empty() {
                keys.push(slug);
            }
        }
    }

    for key in &keys {
        if let Some(entry) = map.get(key) {
            return Some(entry);
        }
    }

    let slug_keys: Vec<String> = keys.iter().map(|k| slug_lookup_key(k)).filter(|s| !s.is_empty()).collect();
    for (map_key, entry) in map {
        let prefixes = mapping_prefixes(map_key, entry);
        if prefixes
            .iter()
            .any(|prefix| slug_keys.iter().any(|key| key.starts_with(prefix.as_str())))
        {
            return Some(entry);
        }
    }
    None
}

/// `getNpcMappingPrefixes`: configured prefixes + trailing-`*` wildcard + the
/// auto `<slug>__ref_` composite-key prefix, all slug-normalized.
fn mapping_prefixes(map_key: &str, entry: &Value) -> Vec<String> {
    let mapping = normalize_mapping_entry(entry);
    let mut prefixes: Vec<String> = Vec::new();
    for key in ["matchPrefix", "nativeNpcKeyPrefix", "keyPrefix"] {
        let s = str_field(&mapping, &[key]);
        if !s.is_empty() {
            prefixes.push(s);
        }
    }
    for key in ["matchPrefixes", "nativeNpcKeyPrefixes", "keyPrefixes"] {
        if let Some(arr) = mapping.get(key).and_then(Value::as_array) {
            for v in arr {
                if let Some(s) = value_to_string(v) {
                    prefixes.push(s);
                }
            }
        }
    }
    let trimmed = map_key.trim();
    if let Some(stripped) = trimmed.strip_suffix('*') {
        prefixes.push(stripped.to_string());
    }
    let slug_key = slug_lookup_key(trimmed.trim_end_matches('*'));
    if !slug_key.is_empty() {
        prefixes.push(format!("{slug_key}__ref_"));
    }
    let mut out: Vec<String> = prefixes.iter().map(|p| slug_lookup_key(p)).filter(|s| !s.is_empty()).collect();
    out.sort();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------
// Candidate normalization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct CandidateFallback {
    native_npc_key: String,
    native_npc_name: String,
    distance_meters: Option<f64>,
    distance_game_units: Option<f64>,
    require_mapped_character: bool,
    allow_config_fallback: bool,
}

fn normalize_npc_candidate(
    config: &BridgeConfig,
    candidate: &Value,
    fb: &CandidateFallback,
) -> Option<NpcParticipant> {
    if is_player_candidate(candidate) {
        return None;
    }
    let mapping_entry = get_npc_mapping_entry(&config.npc_character_map, candidate);
    let mapping = mapping_entry.map(normalize_mapping_entry).unwrap_or_else(|| json!({}));
    let explicit_character_id = strip_png(&str_field(candidate, &["characterId", "character_id"]));
    if fb.require_mapped_character && mapping_entry.is_none() && explicit_character_id.is_empty() {
        return None;
    }

    let native_npc_key = first_non_empty([str_field(candidate, &["npc_key", "npcKey"]), fb.native_npc_key.clone()]);
    let native_npc_name = first_non_empty([
        str_field(candidate, &["npc_name", "npcName", "name"]),
        fb.native_npc_name.clone(),
    ]);
    let character_id = strip_png(&first_non_empty([
        str_field(&mapping, &["characterId", "character_id", "id"]),
        explicit_character_id.clone(),
        if !fb.require_mapped_character { native_npc_name.clone() } else { String::new() },
        if !fb.require_mapped_character { native_npc_key.clone() } else { String::new() },
        if fb.allow_config_fallback { config.character_id.clone() } else { String::new() },
    ]));
    let character_name = first_non_empty([
        str_field(&mapping, &["characterName", "character_name", "name"]),
        str_field(candidate, &["characterName", "character_name"]),
        native_npc_name.clone(),
        character_id.clone(),
        config.character_name.clone(),
    ]);
    let participant_id = first_non_empty([
        str_field(&mapping, &["participantId", "participant_id"]),
        str_field(candidate, &["participantId", "participant_id"]),
        if !native_npc_key.is_empty() {
            format!("npc:{native_npc_key}")
        } else {
            format!("npc:{character_id}")
        },
    ]);
    if character_id.is_empty() || participant_id.is_empty() {
        return None;
    }

    Some(NpcParticipant {
        participant_id,
        character_id,
        character_name: character_name.clone(),
        native_npc_key,
        native_npc_name,
        voice_type_key: first_non_empty([
            str_field(candidate, &["voice_type_key", "voiceTypeKey"]),
            str_field(&mapping, &["voiceTypeKey", "voice_type_key"]),
        ]),
        voice_type_name: first_non_empty([
            str_field(candidate, &["voice_type_name", "voiceTypeName"]),
            str_field(&mapping, &["voiceTypeName", "voice_type_name"]),
        ]),
        distance_meters: num_field(candidate, &["distance_m", "distanceMeters"]).or(fb.distance_meters),
        distance_game_units: num_field(candidate, &["distanceGameUnits", "distance_game_units"])
            .or(fb.distance_game_units),
        under_crosshair: bool_field(candidate, &["under_crosshair", "underCrosshair"]),
    })
}

fn is_player_candidate(candidate: &Value) -> bool {
    const KEYS: [&str; 8] = [
        "npc_key", "npcKey", "nativeNpcKey", "npc_name", "npcName", "name", "voice_type_key",
        "voiceTypeKey",
    ];
    KEYS.iter().any(|k| {
        candidate
            .get(*k)
            .and_then(value_to_string)
            .map(|s| {
                let slug = slug_lookup_key(&s);
                slug == "player" || slug == "playervoicemale" || slug == "playervoicefemale"
            })
            .unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// Mapped actors (admin actor resolution)
// ---------------------------------------------------------------------------

/// A configured NPC the admin (Todd) can command, with slug search terms.
#[derive(Debug, Clone)]
pub struct MappedActor {
    pub native_npc_key: String,
    pub native_npc_name: String,
    pub character_id: String,
    pub character_name: String,
    pub search_terms: Vec<String>,
}

/// `getMappedNativeActors`: every NPC in the character map, with searchable slugs.
pub fn get_mapped_native_actors(config: &BridgeConfig) -> Vec<MappedActor> {
    let mut out = Vec::new();
    for (key, value) in &config.npc_character_map {
        let mapping = normalize_mapping_entry(value);
        let native_npc_key = first_non_empty([str_field(&mapping, &["nativeNpcKey", "npc_key"]), key.clone()]);
        let native_npc_name = first_non_empty([
            str_field(&mapping, &["nativeNpcName", "npc_name", "characterName", "name", "characterId"]),
            key.clone(),
        ]);
        let character_id = strip_png(&first_non_empty([
            str_field(&mapping, &["characterId", "character_id", "id"]),
            native_npc_name.clone(),
            native_npc_key.clone(),
        ]));
        let character_name = first_non_empty([
            str_field(&mapping, &["characterName", "character_name", "name"]),
            character_id.clone(),
            native_npc_name.clone(),
            native_npc_key.clone(),
        ]);
        if native_npc_key.is_empty() || character_name.is_empty() {
            continue;
        }
        let split_source = first_non_empty([native_npc_name.clone(), character_name.clone()]);
        let mut terms = vec![
            native_npc_key.clone(),
            native_npc_name.clone(),
            character_id.clone(),
            character_name.clone(),
            key.clone(),
        ];
        terms.extend(split_source.split_whitespace().map(str::to_string));
        let mut seen = HashSet::new();
        let search_terms = terms
            .iter()
            .map(|t| slug_lookup_key(t))
            .filter(|s| !s.is_empty() && seen.insert(s.clone()))
            .collect();
        out.push(MappedActor {
            native_npc_key,
            native_npc_name,
            character_id,
            character_name,
            search_terms,
        });
    }
    out
}

/// Resolve a candidate (hint or nearby NPC) to a mapped participant, requiring a
/// real character mapping (admin actor strategies 4 + 6).
pub fn resolve_required_mapped_candidate(
    config: &BridgeConfig,
    candidate: &Value,
) -> Option<NpcParticipant> {
    normalize_npc_candidate(
        config,
        candidate,
        &CandidateFallback {
            require_mapped_character: true,
            ..Default::default()
        },
    )
}

// ---------------------------------------------------------------------------
// Nearby NPCs / participants / attention / distance
// ---------------------------------------------------------------------------

fn nearby_npcs(request: &NativeRequest) -> Vec<Value> {
    request
        .metadata
        .get("targeting")
        .and_then(|t| t.get("nearby_npcs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn target_name(config: &BridgeConfig, request: &NativeRequest) -> String {
    first_non_empty([request.npc_name.clone(), config.character_name.clone()])
}

/// `getNativeDistanceMeters`: crosshair candidate, else the npc_key match, else
/// the first nearby candidate. Returns 0.0 when there is no finite distance.
pub fn native_distance_meters(request: &NativeRequest) -> f64 {
    let nearby = nearby_npcs(request);
    let focused = nearby
        .iter()
        .find(|c| bool_field(c, &["under_crosshair", "underCrosshair"]))
        .or_else(|| nearby.iter().find(|c| str_field(c, &["npc_key"]) == request.npc_key))
        .or_else(|| nearby.first());
    focused
        .and_then(|c| num_field(c, &["distance_m", "distanceMeters", "distance"]))
        .unwrap_or(0.0)
}

/// `getNearbyNpcCandidates`: nearby list (distance-filtered) → participants, or a
/// single synthetic candidate from the request identity when there is no list.
pub fn nearby_npc_candidates(
    config: &BridgeConfig,
    request: &NativeRequest,
    distance_meters: f64,
    distance_game_units: f64,
) -> Vec<NpcParticipant> {
    let nearby = nearby_npcs(request);
    let has_native_list = !nearby.is_empty();
    let native_key = request.npc_key.clone();
    let native_name = target_name(config, request);
    let has_native_identity = !native_key.is_empty() || !request.npc_name.is_empty();

    let candidates: Vec<Value> = if has_native_list {
        nearby
    } else {
        vec![json!({
            "npc_key": native_key,
            "npc_name": native_name,
            "distance_m": distance_meters,
            "distanceGameUnits": distance_game_units,
            "under_crosshair": true,
        })]
    };

    let max_meters = if config.native_max_distance_meters.is_finite() {
        config.native_max_distance_meters
    } else {
        DEFAULT_NATIVE_MAX_DISTANCE_METERS
    };

    let mut participants: Vec<NpcParticipant> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for candidate in &candidates {
        if let Some(d) = num_field(candidate, &["distance_m", "distanceMeters"]) {
            if d > max_meters {
                continue;
            }
        }
        let fb = CandidateFallback {
            native_npc_key: native_key.clone(),
            native_npc_name: native_name.clone(),
            distance_meters: Some(distance_meters),
            distance_game_units: Some(distance_game_units),
            require_mapped_character: has_native_list || has_native_identity,
            allow_config_fallback: !has_native_list && !has_native_identity,
        };
        if let Some(p) = normalize_npc_candidate(config, candidate, &fb) {
            if seen.insert(p.participant_id.clone()) {
                participants.push(p);
            }
        }
    }

    if participants.is_empty() && !has_native_list {
        let fb = CandidateFallback {
            native_npc_key: native_key,
            native_npc_name: native_name,
            distance_meters: Some(distance_meters),
            distance_game_units: Some(distance_game_units),
            require_mapped_character: has_native_identity,
            allow_config_fallback: !has_native_identity,
        };
        if let Some(p) = normalize_npc_candidate(config, &json!({}), &fb) {
            participants.push(p);
        }
    }

    participants
}

/// `getAttentionTargetParticipantId`: focus by key, else crosshair, else the lone NPC.
pub fn attention_target(request: &NativeRequest, participants: &[NpcParticipant]) -> Option<String> {
    let focus_key = first_non_empty([
        request
            .metadata
            .get("targeting")
            .and_then(|t| t.get("focus_npc_key"))
            .and_then(value_to_string)
            .unwrap_or_default(),
        request.npc_key.clone(),
    ]);
    participants
        .iter()
        .find(|p| !p.native_npc_key.is_empty() && p.native_npc_key == focus_key)
        .or_else(|| participants.iter().find(|p| p.under_crosshair))
        .or_else(|| if participants.len() == 1 { participants.first() } else { None })
        .map(|p| p.participant_id.clone())
}

// ---------------------------------------------------------------------------
// Gamestate
// ---------------------------------------------------------------------------

/// `buildNativeGamestate`: nearby NPCs within the gamestate radius, deduped.
pub fn build_gamestate(config: &BridgeConfig, request: &NativeRequest, location: &str) -> Value {
    let radius = if config.gamestate_radius_meters.is_finite() {
        config.gamestate_radius_meters
    } else {
        30.0
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut npcs: Vec<Value> = Vec::new();
    for candidate in nearby_npcs(request) {
        if let Some(d) = num_field(&candidate, &["distance_m", "distanceMeters", "distance"]) {
            if d > radius {
                continue;
            }
        }
        let Some(entry) = gamestate_npc_entry(config, &candidate) else {
            continue;
        };
        let key = slug_lookup_key(
            &entry
                .get("internalName")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| str_value(&entry, "nativeNpcName")),
        );
        if seen.insert(key) {
            npcs.push(entry);
        }
    }
    json!({
        "location": location,
        "radiusMeters": radius,
        "npcs": npcs,
        "includeEmptyNpcList": true,
    })
}

fn gamestate_npc_entry(config: &BridgeConfig, candidate: &Value) -> Option<Value> {
    if is_player_candidate(candidate) {
        return None;
    }
    let mapping = get_npc_mapping_entry(&config.npc_character_map, candidate)
        .map(normalize_mapping_entry)
        .unwrap_or_else(|| json!({}));
    let native_key = str_field(candidate, &["npc_key", "npcKey", "nativeNpcKey"]);
    let native_name = first_non_empty([str_field(candidate, &["npc_name", "npcName", "name"]), native_key.clone()]);
    let explicit_id = strip_png(&str_field(candidate, &["characterId", "character_id"]));
    let character_name = first_non_empty([
        str_field(&mapping, &["characterName", "character_name", "name"]),
        str_field(candidate, &["characterName", "character_name"]),
        explicit_id,
    ]);
    if native_key.is_empty() && native_name.is_empty() {
        return None;
    }
    Some(json!({
        "internalName": if native_key.is_empty() { slug_lookup_key(&native_name) } else { native_key },
        "nativeNpcName": native_name,
        "stCharacterName": if character_name.is_empty() { "unmapped".to_string() } else { character_name },
        "distanceMeters": num_field(candidate, &["distance_m", "distanceMeters", "distance"]),
        "coordinates": candidate_coordinates(candidate),
        "underCrosshair": bool_field(candidate, &["under_crosshair", "underCrosshair"]),
    }))
}

fn candidate_coordinates(candidate: &Value) -> Value {
    const SOURCES: [&str; 6] = ["coordinates", "coords", "position", "pos", "worldPosition", "world_position"];
    let axis = |lower: &str, upper: &str| -> Option<f64> {
        for source in SOURCES {
            if let Some(obj) = candidate.get(source) {
                if let Some(v) = num_field(obj, &[lower, upper]) {
                    return Some(v);
                }
            }
        }
        num_field(candidate, &[lower, upper])
    };
    match (axis("x", "X"), axis("y", "Y"), axis("z", "Z")) {
        (Some(x), Some(y), Some(z)) => json!({ "x": x, "y": y, "z": z }),
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Turn extraction (getGeneratedLineItems / getSelectedSpeakerInfo)
// ---------------------------------------------------------------------------

/// Extract per-speaker lines from a turn, with the speaker-name prefix stripped.
pub fn extract_lines(
    config: &BridgeConfig,
    participants: &[NpcParticipant],
    request: &NativeRequest,
    turn: &Value,
) -> Vec<GeneratedLine> {
    let owned;
    let items: &[Value] = match turn.get("turns").and_then(Value::as_array) {
        Some(arr) if !arr.is_empty() => arr,
        _ => {
            owned = vec![turn.clone()];
            &owned
        }
    };
    items
        .iter()
        .filter_map(|item| {
            let speaker = selected_speaker_info(config, participants, request, item);
            let content = item.pointer("/message/content").and_then(Value::as_str).unwrap_or("");
            let text = strip_speaker_prefix(content, &speaker.character_name);
            if text.is_empty() {
                None
            } else {
                Some(GeneratedLine {
                    participant_id: speaker.participant_id,
                    native_npc_key: speaker.native_npc_key,
                    native_npc_name: speaker.native_npc_name,
                    character_name: speaker.character_name,
                    character_id: speaker.character_id,
                    text,
                    turn: (*item).clone(),
                })
            }
        })
        .collect()
}

/// A speaker resolved from a streaming `speaker.start` event, reused to label each
/// `speech.delta` segment's audio chunks before the final turn arrives.
#[derive(Debug, Clone)]
pub struct ResolvedSpeaker {
    pub participant_id: String,
    pub native_npc_key: String,
    pub native_npc_name: String,
    pub character_name: String,
    pub character_id: String,
}

/// Resolve a streaming `speaker.start` event's speaker against the participants.
pub fn resolve_stream_speaker(
    config: &BridgeConfig,
    participants: &[NpcParticipant],
    request: &NativeRequest,
    speaker_value: &Value,
) -> ResolvedSpeaker {
    let item = json!({ "speaker": speaker_value });
    let info = selected_speaker_info(config, participants, request, &item);
    ResolvedSpeaker {
        participant_id: info.participant_id,
        native_npc_key: info.native_npc_key,
        native_npc_name: info.native_npc_name,
        character_name: info.character_name,
        character_id: info.character_id,
    }
}

/// Fallback speaker when a `speech.delta` arrives before any `speaker.start`.
pub fn default_stream_speaker(config: &BridgeConfig, request: &NativeRequest) -> ResolvedSpeaker {
    let name = first_non_empty([request.npc_name.clone(), config.character_name.clone()]);
    ResolvedSpeaker {
        participant_id: String::new(),
        native_npc_key: request.npc_key.clone(),
        native_npc_name: name.clone(),
        character_name: name,
        character_id: config.character_id.clone(),
    }
}

struct SpeakerInfo {
    participant_id: String,
    native_npc_key: String,
    native_npc_name: String,
    character_name: String,
    character_id: String,
}

fn selected_speaker_info(
    config: &BridgeConfig,
    participants: &[NpcParticipant],
    request: &NativeRequest,
    item: &Value,
) -> SpeakerInfo {
    let participant_id = item
        .pointer("/speaker/participantId")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let participant = participants.iter().find(|p| p.participant_id == participant_id);
    let turn_name = item.pointer("/speaker/name").and_then(Value::as_str).unwrap_or("");
    let turn_character_id = item.pointer("/speaker/characterId").and_then(Value::as_str).unwrap_or("");
    SpeakerInfo {
        participant_id,
        native_npc_key: first_non_empty([
            participant.map(|p| p.native_npc_key.clone()).unwrap_or_default(),
            request.npc_key.clone(),
        ]),
        native_npc_name: first_non_empty([
            participant.map(|p| p.native_npc_name.clone()).unwrap_or_default(),
            turn_name.to_string(),
            request.npc_name.clone(),
            config.character_name.clone(),
        ]),
        character_name: first_non_empty([
            turn_name.to_string(),
            participant.map(|p| p.character_name.clone()).unwrap_or_default(),
            config.character_name.clone(),
        ]),
        character_id: first_non_empty([
            turn_character_id.to_string(),
            participant.map(|p| p.character_id.clone()).unwrap_or_default(),
            config.character_id.clone(),
        ]),
    }
}

/// `stripSpeakerPrefix`: drop a leading `"<Name>:"` (case-insensitive, repeated).
pub(crate) fn strip_speaker_prefix(text: &str, speaker_name: &str) -> String {
    let trimmed = text.trim().to_string();
    if speaker_name.is_empty() {
        return trimmed;
    }
    let pattern = format!(r"(?i)^(?:{}\b\s*:?\s*)+", regex::escape(speaker_name));
    let Ok(re) = Regex::new(&pattern) else {
        return trimmed;
    };
    let mut current = trimmed;
    loop {
        let next = re.replace(&current, "").trim().to_string();
        if next == current {
            return current;
        }
        current = next;
    }
}

/// The location string Node joins from the request's location parts (`' / '`).
pub fn location_string(request: &NativeRequest) -> String {
    [
        request.location.major.as_str(),
        request.location.minor.as_str(),
        request.location.cell.as_str(),
        request.location.worldspace.as_str(),
        request.location.region.as_str(),
    ]
    .iter()
    .filter(|p| !p.is_empty())
    .copied()
    .collect::<Vec<_>>()
    .join(" / ")
}

// ---------------------------------------------------------------------------
// Value helpers (JS `||`/`Number()`/`Boolean()` semantics)
// ---------------------------------------------------------------------------

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn str_field(v: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = v.get(*k).and_then(value_to_string) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }
    String::new()
}

fn str_value(v: &Value, key: &str) -> String {
    str_field(v, &[key])
}

fn num_field(v: &Value, keys: &[&str]) -> Option<f64> {
    for k in keys {
        match v.get(*k) {
            Some(Value::Number(n)) => {
                if let Some(f) = n.as_f64() {
                    if f.is_finite() {
                        return Some(f);
                    }
                }
            }
            Some(Value::String(s)) => {
                let t = s.trim();
                if !t.is_empty() {
                    if let Ok(f) = t.parse::<f64>() {
                        if f.is_finite() {
                            return Some(f);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn bool_field(v: &Value, keys: &[&str]) -> bool {
    keys.iter().any(|k| v.get(*k).map(js_truthy).unwrap_or(false))
}

fn js_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0 && !f.is_nan()).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn first_non_empty<const N: usize>(values: [String; N]) -> String {
    values.into_iter().find(|s| !s.is_empty()).unwrap_or_default()
}

fn strip_png(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 4 && t[t.len() - 4..].eq_ignore_ascii_case(".png") {
        t[..t.len() - 4].trim().to_string()
    } else {
        t.to_string()
    }
}
