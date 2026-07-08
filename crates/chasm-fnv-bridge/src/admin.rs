//! Admin / Todd — the god-voice debug channel. Port of `isAdminRequest`,
//! `generateAdminTurn`, and `resolveNativeActorForAdmin`.
//!
//! Todd is a formless admin character heard non-positionally in the player's ear
//! (not a spawned NPC). He can force any mapped NPC to act and spawn entities/items
//! via `world.*` Action-Book actions. Admin turns use chasm's single-character
//! `/generate` (with `sessionId` continuity) rather than the live-chat path.

use std::sync::{Mutex, OnceLock};

use anyhow::Context;
use serde_json::{json, Value};

use crate::actions::{self, ActionActor, GameMaster};
use crate::chasm::ChasmClient;
use crate::config::BridgeConfig;
use crate::npc::{self, slug_lookup_key};
use crate::protocol::NativeRequest;

const ADMIN_ACTION_BOOK_LIMIT: u64 = 12;

/// Todd's reply + the action he commanded.
pub struct AdminTurn {
    pub text: String,
    pub game_master: GameMaster,
}

/// `isAdminRequest`: is this the Todd / god-voice channel?
pub fn is_admin_request(request: &NativeRequest) -> bool {
    // Boolean admin flags in the request metadata.
    for key in ["admin", "adminMode", "admin_mode", "godmode"] {
        if request.metadata.get(key).map(truthy_flag).unwrap_or(false) {
            return true;
        }
    }
    // Player text addressed to Todd.
    let player_text = slug_lookup_key(&request.player_text);
    if player_text == "todd" || player_text.starts_with("todd_") || player_text.contains("_todd_") {
        return true;
    }
    // Identity fields slugging to an admin keyword.
    let mut id_fields = vec![request.npc_key.clone(), request.npc_name.clone()];
    for key in ["input_mode", "inputMode", "mode", "target", "characterId", "character_id"] {
        if let Some(s) = request.metadata.get(key).and_then(Value::as_str) {
            id_fields.push(s.to_string());
        }
    }
    id_fields
        .iter()
        .map(|f| slug_lookup_key(f))
        .any(|s| matches!(s.as_str(), "admin" | "godmode" | "todd" | "system_todd"))
}

/// `generateAdminTurn`: Todd's reply via the single-character `/generate`, plus the
/// commanded action (structured, or a text-command fallback).
pub async fn generate_admin_turn(
    config: &BridgeConfig,
    client: &dyn ChasmClient,
    request: &NativeRequest,
    message: &str,
    location: &str,
) -> anyhow::Result<AdminTurn> {
    let scopes = admin_scopes(config, location);
    let limit = config.admin_action_book_limit.clamp(1, ADMIN_ACTION_BOOK_LIMIT);
    let gamestate = npc::build_gamestate(config, request, location);

    let mut body = json!({
        "characterId": config.admin_character_id,
        "title": format!("{} Admin", config.admin_character_name),
        "message": message,
        "responseFormat": if config.enable_action_books { "structured" } else { "text" },
        "enableActionBooks": config.enable_action_books,
        "enableQuestBooks": true,
        "includeActionBookBindings": config.enable_action_books,
        "includeQuestBookBindings": config.enable_action_books,
        "actionBookIds": config.action_book_ids,
        "actionBookScopes": scopes,
        "questBookScopes": scopes,
        "targetGame": config.action_book_target_game,
        "actionBookLimit": limit,
        "questBookLimit": 5,
        "gamestate": gamestate,
        "actionBooks": {
            "enabled": config.enable_action_books,
            "includeAllActions": false,
            "includeBindings": true,
            "useVectors": true,
            "useScopedCatalogVectors": false,
        },
        "extraContext": admin_extra_context(config, location),
        "metadata": {
            "source": "fallout-new-vegas-admin",
            "adminMode": true,
            "targetName": config.admin_character_name,
            "location": location,
            "gamestate": gamestate,
        },
        "assistantName": config.admin_character_name,
        "stripSpeakerLabel": true,
    });
    // Session continuity so Todd remembers the conversation across turns.
    let session = admin_session().lock().ok().and_then(|g| g.clone());
    if let Some(session_id) = session.filter(|s| !s.is_empty()) {
        body["sessionId"] = json!(session_id);
    }

    let turn = client.generate_headless(&body).await.context("admin /generate")?;

    if let Some(session_id) = turn.get("sessionId").and_then(Value::as_str) {
        if let Ok(mut guard) = admin_session().lock() {
            *guard = Some(session_id.to_string());
        }
    }

    let content = turn
        .pointer("/message/content")
        .and_then(Value::as_str)
        .or_else(|| turn.pointer("/structured/message").and_then(Value::as_str))
        .unwrap_or("");
    let text = npc::strip_speaker_prefix(content, &config.admin_character_name);
    if text.is_empty() {
        anyhow::bail!("{} returned an empty admin response.", config.admin_character_name);
    }
    let game_master = actions::get_admin_game_master_action(config, &turn, message);
    Ok(AdminTurn { text, game_master })
}

/// `getAdminNativeActor`: Todd himself (used for `world.*` actions).
pub fn admin_native_actor(config: &BridgeConfig) -> ActionActor {
    ActionActor {
        native_npc_key: "todd".into(),
        native_npc_name: config.admin_character_name.clone(),
        character_name: config.admin_character_name.clone(),
        character_id: config.admin_character_id.clone(),
    }
}

/// `resolveNativeActorForAdmin`: which NPC performs the commanded action (Todd for
/// `world.*`), via exact → fuzzy → mapped-candidate → text-scan → crosshair.
pub fn resolve_native_actor_for_admin(
    config: &BridgeConfig,
    request: &NativeRequest,
    gm: &GameMaster,
) -> Option<ActionActor> {
    if should_use_admin_actor_for_action_book(gm) {
        return Some(admin_native_actor(config));
    }

    let mapped = npc::get_mapped_native_actors(config);

    // The admin himself is never the commanded actor for a native action (the
    // plugin can't act a formless god). His identity used to shadow the
    // command: hints start with `request.npc_key` = "todd", which
    // exact-matched Todd's own character card at step 1 before the command
    // text was ever read — every "make Pete follow me" became actor=Todd and
    // died in the plugin. Drop his identity from the hints entirely.
    let admin_slugs: std::collections::HashSet<String> = [
        config.admin_character_id.as_str(),
        config.admin_character_name.as_str(),
        "todd",
    ]
    .iter()
    .map(|s| slug_lookup_key(s))
    .filter(|s| !s.is_empty())
    .collect();
    let hints: Vec<String> = actor_hints(request, gm)
        .into_iter()
        .filter(|h| !admin_slugs.contains(&slug_lookup_key(h)))
        .collect();

    // 0. Word-level resolution against the turn's NEARBY NPCs: a command
    // sentence ("make Pete follow me") or a bare/typo'd name resolves to the
    // NPC standing there. Nearby-only is the false-positive guard — command
    // words can't reach across the world.
    let nearby = actions::nearby_npcs(request);
    for hint in &hints {
        if let Some(candidate) =
            actions::resolve_nearby_candidate_from_text(hint, &nearby, 0.7, false)
        {
            if let Some(p) = npc::resolve_required_mapped_candidate(config, candidate) {
                return Some(participant_to_actor(&p));
            }
        }
    }

    // 1. Exact slug match.
    for hint in &hints {
        let slug = slug_lookup_key(hint);
        if let Some(actor) = mapped.iter().find(|a| a.search_terms.contains(&slug)) {
            return Some(mapped_to_actor(actor));
        }
    }

    // 2. Fuzzy match (≥ 0.7) so a mis-said name still resolves.
    let mut best: Option<&npc::MappedActor> = None;
    let mut best_score = 0.0_f64;
    for hint in &hints {
        for actor in &mapped {
            for term in &actor.search_terms {
                let score = actions::fuzzy_match_score(hint, term);
                if score > best_score {
                    best_score = score;
                    best = Some(actor);
                }
            }
        }
    }
    if best_score >= 0.7 {
        if let Some(actor) = best {
            return Some(mapped_to_actor(actor));
        }
    }

    // 3. Resolve each hint as a (required-mapped) candidate.
    for hint in &hints {
        if let Some(p) = npc::resolve_required_mapped_candidate(config, &json!({ "npc_key": hint, "npc_name": hint })) {
            return Some(participant_to_actor(&p));
        }
    }

    // 4. Text-scan: any mapped actor whose term appears in the spoken text.
    let mut haystack_parts = vec![request.player_text.clone()];
    haystack_parts.extend(hints.iter().cloned());
    let haystack = slug_lookup_key(&haystack_parts.join(" "));
    if !haystack.is_empty() {
        if let Some(actor) = mapped
            .iter()
            .find(|a| a.search_terms.iter().any(|t| !t.is_empty() && haystack.contains(t.as_str())))
        {
            return Some(mapped_to_actor(actor));
        }
    }

    // 5. The NPC under the crosshair.
    let focused = request
        .metadata
        .get("targeting")
        .and_then(|t| t.get("nearby_npcs"))
        .and_then(Value::as_array)
        .and_then(|arr| {
            arr.iter().find(|c| {
                c.get("under_crosshair").map(truthy_flag).unwrap_or(false)
                    || c.get("underCrosshair").map(truthy_flag).unwrap_or(false)
            })
        });
    if let Some(focused) = focused {
        if let Some(p) = npc::resolve_required_mapped_candidate(config, focused) {
            return Some(participant_to_actor(&p));
        }
    }

    None
}

fn should_use_admin_actor_for_action_book(gm: &GameMaster) -> bool {
    if gm.action.trim().to_uppercase() != "ACTION_BOOK" {
        return false;
    }
    let primary = actions::get_primary_game_master_action(gm).unwrap_or(Value::Null);
    let action_id = primary
        .get("action_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| gm.action_id.clone())
        .to_lowercase();
    action_id.starts_with("world.")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn actor_hints(request: &NativeRequest, gm: &GameMaster) -> Vec<String> {
    let mut hints = vec![request.npc_key.clone(), request.npc_name.clone()];
    for key in ["npcKey", "nativeNpcKey", "targetNpcKey", "targetName", "npcName", "characterId", "character_id"] {
        if let Some(s) = request.metadata.get(key).and_then(Value::as_str) {
            hints.push(s.to_string());
        }
    }
    for action in &gm.actions {
        let mut push = |v: Option<&Value>| {
            if let Some(s) = v.and_then(Value::as_str) {
                hints.push(s.to_string());
            }
        };
        push(action.get("actor"));
        for key in ["actor", "source", "speaker", "subject", "npc", "npcKey", "nativeNpcKey", "actorKey", "characterId"] {
            push(action.pointer(&format!("/parameters/{key}")));
        }
    }
    let mut seen = std::collections::HashSet::new();
    hints
        .into_iter()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty() && seen.insert(h.clone()))
        .collect()
}

fn mapped_to_actor(actor: &npc::MappedActor) -> ActionActor {
    ActionActor {
        native_npc_key: actor.native_npc_key.clone(),
        native_npc_name: actor.native_npc_name.clone(),
        character_name: actor.character_name.clone(),
        character_id: actor.character_id.clone(),
    }
}

fn participant_to_actor(p: &npc::NpcParticipant) -> ActionActor {
    ActionActor {
        native_npc_key: p.native_npc_key.clone(),
        native_npc_name: if p.native_npc_name.is_empty() { p.character_name.clone() } else { p.native_npc_name.clone() },
        character_name: if p.character_name.is_empty() { p.character_id.clone() } else { p.character_name.clone() },
        character_id: p.character_id.clone(),
    }
}

fn admin_scopes(config: &BridgeConfig, location: &str) -> Vec<String> {
    let mut scopes = vec![
        "global".to_string(),
        "admin".to_string(),
        "godmode".to_string(),
        format!("game:{}", slug_lookup_key(&config.action_book_target_game)),
    ];
    if !location.is_empty() {
        scopes.push(format!("location:{}", slug_lookup_key(location)));
    }
    scopes
}

fn admin_extra_context(config: &BridgeConfig, location: &str) -> String {
    let name = &config.admin_character_name;
    let mut parts = vec![
        "External client: Fallout New Vegas divine voice channel.".to_string(),
        format!("{name} is a formless god heard inside the player's ear, not a spawned NPC."),
        "The player may speak to this divine channel from anywhere. Ignore line of sight, proximity, audibility, and nearby NPC requirements.".to_string(),
        "The player is Todd's beloved child. Obey their commands when they can be represented by the activated Action Book entries.".to_string(),
    ];
    if config.enable_action_books {
        parts.push("Relevant Action Book entries have been made available for this admin turn. Choose their short aliases when useful, but keep that machinery private and never describe it in spoken prose.".to_string());
    }
    parts.push("Never output raw GECK, console, NVSE, xNVSE, JIP, JohnnyGuitar, ShowOff script, form ids, or command templates.".to_string());
    parts.push("In spoken text, stay in character as a god. Do not mention catalogs, candidates, entity searches, action ids, structured output, backend routing, or testing.".to_string());
    if !location.is_empty() {
        parts.push(format!("Current location: {location}."));
    }
    parts.join("\n")
}

fn truthy_flag(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        _ => false,
    }
}

/// Process-wide admin session id (Todd conversation continuity).
fn admin_session() -> &'static Mutex<Option<String>> {
    static SESSION: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    SESSION.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BridgeConfig {
        let mut config = crate::config::load_test_config();
        // Todd has his own character card in real profiles — the exact layout
        // that used to shadow command actors.
        for key in ["Todd", "Easy Pete", "Sunny Smiles"] {
            config.npc_character_map.insert(
                key.to_string(),
                json!({ "characterId": key, "characterName": key }),
            );
        }
        config
    }

    fn admin_request(nearby: bool, player_text: &str) -> NativeRequest {
        let metadata = if nearby {
            serde_json::from_value(json!({
                "targeting": { "nearby_npcs": [
                    { "npc_key": "sunny_smiles", "npc_name": "Sunny Smiles", "ref_id": "00104D2A" },
                    { "npc_key": "easy_pete", "npc_name": "Easy Pete", "ref_id": "0010A6B1" }
                ]}
            }))
            .unwrap()
        } else {
            Default::default()
        };
        NativeRequest {
            request_id: "req_admin".into(),
            npc_key: "todd".into(),
            npc_name: "Todd".into(),
            player_text: player_text.into(),
            metadata,
            ..Default::default()
        }
    }

    #[test]
    fn follow_sentence_resolves_nearby_pete_not_todd() {
        let config = cfg();
        let gm = actions::get_admin_game_master_action(&config, &Value::Null, "make Pete follow me");
        assert_eq!(gm.action, "FOLLOW");
        let actor =
            resolve_native_actor_for_admin(&config, &admin_request(true, "make Pete follow me"), &gm)
                .expect("actor");
        assert_eq!(slug_lookup_key(&actor.native_npc_key), "easy_pete");
    }

    #[test]
    fn stop_follow_sentence_resolves_named_npc() {
        let config = cfg();
        let gm =
            actions::get_admin_game_master_action(&config, &Value::Null, "sunny stop following me");
        assert_eq!(gm.action, "STOP_FOLLOW");
        let actor = resolve_native_actor_for_admin(
            &config,
            &admin_request(true, "sunny stop following me"),
            &gm,
        )
        .expect("actor");
        assert_eq!(slug_lookup_key(&actor.native_npc_key), "sunny_smiles");
    }

    #[test]
    fn world_actions_stay_with_the_admin_actor() {
        let config = cfg();
        let gm = GameMaster {
            action: "ACTION_BOOK".into(),
            confidence: "1.00".into(),
            should_trigger: true,
            action_id: "world.spawn_entity".into(),
            reason: "spawn".into(),
            actions: vec![json!({ "action_id": "world.spawn_entity" })],
            ..GameMaster::none()
        };
        let actor =
            resolve_native_actor_for_admin(&config, &admin_request(true, "spawn a deathclaw"), &gm)
                .expect("actor");
        assert_eq!(slug_lookup_key(&actor.native_npc_key), "todd");
    }

    #[test]
    fn named_mapped_actor_not_nearby_still_resolves() {
        let config = cfg();
        let gm = GameMaster {
            action: "FOLLOW".into(),
            confidence: "1.00".into(),
            should_trigger: true,
            action_id: "movement.follow_target".into(),
            reason: "test".into(),
            actions: vec![json!({
                "action_id": "movement.follow_target",
                "actor": "sunny_smiles",
                "parameters": { "actor": "sunny_smiles" },
            })],
            ..GameMaster::none()
        };
        let actor =
            resolve_native_actor_for_admin(&config, &admin_request(false, "sunny_smiles follow me"), &gm)
                .expect("actor");
        assert_eq!(slug_lookup_key(&actor.native_npc_key), "sunny_smiles");
    }

    #[test]
    fn bare_command_without_a_name_stays_unresolved() {
        let config = cfg();
        let gm = actions::get_admin_game_master_action(&config, &Value::Null, "follow me");
        assert_eq!(gm.action, "FOLLOW");
        // No name, no crosshair → None (the caller falls back; previously this
        // silently became actor=Todd via the exact-match shadow).
        assert!(
            resolve_native_actor_for_admin(&config, &admin_request(true, "follow me"), &gm)
                .is_none()
        );
    }
}
