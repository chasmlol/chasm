//! Action-resolution engine — port of the Node helper's structured-action path:
//! classify a generated turn's structured actions into a native game-master action
//! (ATTACK / FOLLOW / STOP_FOLLOW / ACTION_BOOK), resolve trusted-execution
//! arguments (catalog→FormID, `ref:`/`refid:`/`form:`/`number:`/`string:` encoding,
//! fuzzy targets, repeat expansion), and write `NVBRIDGE_ACTION_V2` / legacy command
//! files to every `<root>/control/actions`.
//!
//! Works over `serde_json::Value` (the turn + action objects) like the Node code
//! worked over plain objects. Command-file byte format is locked by the tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use base64::Engine;
use regex::Regex;
use serde_json::{json, Map, Value};
use tracing::warn;

use crate::config::BridgeConfig;
use crate::npc::slug_lookup_key;
use crate::protocol::{now_epoch_millis, safe_file_id, sanitize_bridge_line, NativeRequest};

const TRUSTED_FNV_ACTION_ENGINE: &str = "fallout-new-vegas:xnvse";
const NATIVE_ACTION_COMMAND_VERSION: &str = "NVBRIDGE_ACTION_V2";

/// The resolved native action attached to a generated line / turn.
#[derive(Debug, Clone)]
pub struct GameMaster {
    pub action: String,        // ATTACK | FOLLOW | STOP_FOLLOW | ACTION_BOOK | NONE
    pub confidence: String,    // "0.90" or ""
    pub should_trigger: bool,
    pub action_id: String,
    pub reason: String,
    pub actions: Vec<Value>,   // normalized action metadata
}

impl GameMaster {
    /// `getNoGameMasterAction`: the canonical "no action" object.
    pub fn none() -> Self {
        Self {
            action: "NONE".into(),
            confidence: String::new(),
            should_trigger: false,
            action_id: String::new(),
            reason: String::new(),
            actions: Vec::new(),
        }
    }
}

/// The acting NPC for a command (the speaker, or an admin-resolved actor).
#[derive(Debug, Clone, Default)]
pub struct ActionActor {
    pub native_npc_key: String,
    pub native_npc_name: String,
    pub character_name: String,
    pub character_id: String,
}

// ---------------------------------------------------------------------------
// Generate-body action-book scopes
// ---------------------------------------------------------------------------

/// `actionBookScopes` for the generate body: global + game + location + npc.
pub fn action_book_scopes(config: &BridgeConfig, native_npc_key: &str, location: &str) -> Vec<String> {
    let mut scopes = vec![
        "global".to_string(),
        format!("game:{}", slug_lookup_key(&config.action_book_target_game)),
    ];
    if !location.is_empty() {
        scopes.push(format!("location:{}", slug_lookup_key(location)));
    }
    if !native_npc_key.is_empty() {
        scopes.push(format!("npc:{}", slug_lookup_key(native_npc_key)));
    }
    unique(scopes)
}

// ---------------------------------------------------------------------------
// Action collection
// ---------------------------------------------------------------------------

fn normalize_structured_action_id(action: &Value) -> String {
    first_str(action, &["id", "actionId", "name"])
}

fn normalize_activated_action_id(action: &Value) -> String {
    first_str(action, &["actionId", "action_id", "id"])
}

/// `collectStructuredActions`: every `structured.actions[]` / `actions[]` across
/// the turn and its sub-turns, filtered to those with an id.
fn collect_structured_actions(turn: &Value) -> Vec<Value> {
    let mut actions = Vec::new();
    let mut collect = |node: &Value| {
        if let Some(arr) = node.pointer("/structured/actions").and_then(Value::as_array) {
            actions.extend(arr.iter().cloned());
        }
        if let Some(arr) = node.get("actions").and_then(Value::as_array) {
            actions.extend(arr.iter().cloned());
        }
    };
    collect(turn);
    if let Some(turns) = turn.get("turns").and_then(Value::as_array) {
        for item in turns {
            collect(item);
        }
    }
    actions
        .into_iter()
        .filter(|a| !normalize_structured_action_id(a).is_empty())
        .collect()
}

/// `collectActivatedActions`: trusted/relay execution metadata across the turn,
/// including quest events flattened into actions.
fn collect_activated_actions(turn: &Value) -> Vec<Value> {
    let mut actions = Vec::new();
    let mut collect = |node: &Value| {
        for key in ["/metadata/activatedActions", "/activatedActions"] {
            if let Some(arr) = node.pointer(key).and_then(Value::as_array) {
                actions.extend(arr.iter().cloned());
            }
        }
        for key in ["/metadata/activatedQuests", "/activatedQuests"] {
            if let Some(quests) = node.pointer(key).and_then(Value::as_array) {
                for quest in quests {
                    let events = quest.get("questEvents").and_then(Value::as_array);
                    for event in events.into_iter().flatten() {
                        let mut e = event.clone();
                        if let Some(obj) = e.as_object_mut() {
                            let action_id = first_str(event, &["actionId", "action_id"]);
                            obj.insert("actionId".into(), json!(action_id));
                            obj.insert("bookId".into(), json!(first_str(quest, &["bookId"])));
                            obj.insert("questId".into(), json!(first_str(quest, &["questId"])));
                            obj.insert("questName".into(), json!(first_str(quest, &["questName"])));
                        }
                        actions.push(e);
                    }
                }
            }
        }
    };
    collect(turn);
    if let Some(turns) = turn.get("turns").and_then(Value::as_array) {
        for item in turns {
            collect(item);
        }
    }
    actions
        .into_iter()
        .filter(|a| !normalize_activated_action_id(a).is_empty())
        .collect()
}

fn get_activated_action_map(turn: &Value) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    for action in collect_activated_actions(turn) {
        let id = normalize_activated_action_id(&action);
        if !id.is_empty() {
            map.entry(id).or_insert(action);
        }
    }
    map
}

/// `getTrustedActivatedExecution`: the binding/execution for an action IF it is a
/// trusted FNV xNVSE action with a script. Returns empty objects otherwise.
fn get_trusted_activated_execution(
    action: &Value,
    activated: &HashMap<String, Value>,
) -> (Option<Value>, Value, Value) {
    let action_id = normalize_structured_action_id(action);
    let activated_action = activated.get(&action_id).cloned();
    let binding = activated_action
        .as_ref()
        .and_then(|a| a.get("binding"))
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let execution = activated_action
        .as_ref()
        .and_then(|a| a.get("execution"))
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let engine = binding.get("engine").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
    let script = execution.get("script").and_then(Value::as_str).unwrap_or("").trim();
    if engine != TRUSTED_FNV_ACTION_ENGINE || script.is_empty() {
        return (activated_action, json!({}), json!({}));
    }
    (activated_action, binding, execution)
}

// ---------------------------------------------------------------------------
// Parameter folding
// ---------------------------------------------------------------------------

const RESERVED_ACTION_KEYS: [&str; 17] = [
    "id", "actionId", "action_id", "name", "alias", "target", "actor", "parameters", "params",
    "reason", "confidence", "binding", "execution", "scopedCatalogs", "scoped_catalogs",
    "action_book_id", "bookId",
];

/// `mergeActionParameters`: fold top-level custom fields into parameters (explicit
/// `parameters` win) — small models emit fields like `entity`/`count` at the top.
fn merge_action_parameters(action: &Value) -> Map<String, Value> {
    let explicit = get_action_parameters(action);
    let mut merged = Map::new();
    if let Some(obj) = action.as_object() {
        for (key, value) in obj {
            if !RESERVED_ACTION_KEYS.contains(&key.as_str()) {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    for (key, value) in explicit {
        merged.insert(key, value);
    }
    merged
}

fn get_action_parameters(action: &Value) -> Map<String, Value> {
    action
        .get("parameters")
        .or_else(|| action.get("params"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn get_action_parameter_value(action: &Value, name: &str) -> Option<Value> {
    let key = name.trim();
    if key.is_empty() {
        return None;
    }
    let params = get_action_parameters(action);
    params.get(key).cloned().or_else(|| action.get(key).cloned())
}

fn get_action_parameter_value_from_config(action: &Value, config: &Value) -> Option<Value> {
    let mut names: Vec<String> = Vec::new();
    let mut push = |s: String| {
        let t = s.trim().to_string();
        if !t.is_empty() && !names.contains(&t) {
            names.push(t);
        }
    };
    push(first_str(config, &["parameter"]));
    push(first_str(config, &["name"]));
    for key in ["parameters", "fallbackParameters", "aliases"] {
        if let Some(arr) = config.get(key).and_then(Value::as_array) {
            for v in arr {
                if let Some(s) = value_to_string(v) {
                    push(s);
                }
            }
        }
    }
    for name in &names {
        if let Some(v) = get_action_parameter_value(action, name) {
            if !v.is_null() && value_to_string(&v).map(|s| !s.is_empty()).unwrap_or(true) {
                return Some(v);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Action classification helpers
// ---------------------------------------------------------------------------

fn get_action_target(action: &Value) -> String {
    let params = get_action_parameters(action);
    let raw = first_str(action, &["target"]);
    let target = if raw.is_empty() {
        params.get("target").and_then(value_to_string).unwrap_or_default()
    } else {
        raw
    };
    target.trim().to_lowercase()
}

fn is_player_action_target(action: &Value) -> bool {
    matches!(
        get_action_target(action).as_str(),
        "" | "player" | "courier" | "the player" | "target:player"
    )
}

fn get_action_confidence(config: &BridgeConfig, action: &Value) -> f64 {
    let params = get_action_parameters(action);
    let raw = match action.get("confidence") {
        Some(v) if !v.is_null() => Some(v.clone()),
        _ => params.get("confidence").filter(|v| !v.is_null()).cloned(),
    };
    let conf = match raw {
        Some(v) => value_to_f64(&v).unwrap_or(f64::NAN),
        None => config.native_action_confidence,
    };
    if !conf.is_finite() {
        return config.native_action_confidence;
    }
    conf.clamp(0.0, 1.0)
}

fn get_action_actor_hint(action: &Value) -> String {
    let params = get_action_parameters(action);
    let action_keys = ["actor", "source", "speaker", "subject", "npc", "npcKey", "nativeNpcKey"];
    let param_keys = [
        "actor", "source", "speaker", "subject", "npc", "npcKey", "nativeNpcKey", "actorKey",
        "characterId",
    ];
    for k in action_keys {
        let v = first_str(action, &[k]);
        if !v.is_empty() {
            return v;
        }
    }
    for k in param_keys {
        if let Some(s) = params.get(k).and_then(value_to_string) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    String::new()
}

/// `getStructuredActionMetadata`: normalize each structured action (folding params,
/// attaching trusted binding/execution + catalogs).
fn get_structured_action_metadata(config: &BridgeConfig, turn: &Value) -> Vec<Value> {
    if !config.enable_action_books {
        return Vec::new();
    }
    let activated = get_activated_action_map(turn);
    collect_structured_actions(turn)
        .iter()
        .map(|action| {
            let parameters = merge_action_parameters(action);
            let action_id = normalize_structured_action_id(action);
            let (activated_action, binding, execution) =
                get_trusted_activated_execution(action, &activated);
            let mut obj = Map::new();
            obj.insert("action_id".into(), json!(action_id));
            obj.insert(
                "action_book_id".into(),
                json!(activated_action.as_ref().map(|a| first_str(a, &["bookId"])).unwrap_or_default()),
            );
            obj.insert("target".into(), json!(get_action_target(action)));
            obj.insert("actor".into(), json!(get_action_actor_hint(action)));
            obj.insert("parameters".into(), Value::Object(parameters));
            obj.insert(
                "scopedCatalogs".into(),
                activated_action
                    .as_ref()
                    .and_then(|a| a.get("scopedCatalogs"))
                    .filter(|v| v.is_array())
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            );
            obj.insert(
                "confidence".into(),
                json!(format!("{:.2}", get_action_confidence(config, action))),
            );
            obj.insert(
                "reason".into(),
                json!(sanitize_bridge_line(&first_str(action, &["reason"]))),
            );
            if binding.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
                obj.insert("binding".into(), binding);
            }
            if execution.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
                obj.insert("execution".into(), execution);
            }
            Value::Object(obj)
        })
        .collect()
}

/// `getPrimaryGameMasterAction`: the action driving the command (matched by id,
/// else first with an execution script).
pub(crate) fn get_primary_game_master_action(gm: &GameMaster) -> Option<Value> {
    gm.actions
        .iter()
        .find(|a| first_str(a, &["action_id"]) == gm.action_id)
        .or_else(|| {
            gm.actions
                .iter()
                .find(|a| !a.pointer("/execution/script").and_then(Value::as_str).unwrap_or("").is_empty())
        })
        .cloned()
}

// ---------------------------------------------------------------------------
// getNativeGameMasterAction — structured action → native action string
// ---------------------------------------------------------------------------

/// `getNativeGameMasterAction`: ATTACK → STOP_FOLLOW → FOLLOW → ACTION_BOOK → NONE.
pub fn get_native_game_master_action(config: &BridgeConfig, turn: &Value) -> GameMaster {
    if !config.enable_action_books {
        return GameMaster::none();
    }
    let structured = collect_structured_actions(turn);
    let activated = get_activated_action_map(turn);
    let actions = get_structured_action_metadata(config, turn);

    let base = |action: &Value, native_action: &str| -> Option<GameMaster> {
        let action_id = normalize_structured_action_id(action);
        let confidence = get_action_confidence(config, action);
        if confidence < config.native_action_confidence {
            return None;
        }
        Some(GameMaster {
            action: native_action.into(),
            confidence: format!("{confidence:.2}"),
            should_trigger: true,
            action_id,
            reason: sanitize_bridge_line(&first_str(action, &["reason"])),
            actions: actions.clone(),
        })
    };

    for action in &structured {
        if normalize_structured_action_id(action) == "combat.start" && is_player_action_target(action) {
            if let Some(gm) = base(action, "ATTACK") {
                return gm;
            }
        }
    }
    const STOP_IDS: [&str; 3] = [
        "movement.stop_follow_target",
        "movement.stop_following",
        "movement.stop_follow",
    ];
    for action in &structured {
        let id = normalize_structured_action_id(action);
        if STOP_IDS.contains(&id.as_str()) && is_player_action_target(action) {
            if let Some(gm) = base(action, "STOP_FOLLOW") {
                return gm;
            }
        }
    }
    for action in &structured {
        if normalize_structured_action_id(action) == "movement.follow_target" && is_player_action_target(action) {
            if let Some(gm) = base(action, "FOLLOW") {
                return gm;
            }
        }
    }
    for action in &structured {
        let (_, _, execution) = get_trusted_activated_execution(action, &activated);
        if execution.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
            if let Some(gm) = base(action, "ACTION_BOOK") {
                return gm;
            }
        }
    }
    GameMaster::none()
}

// ---------------------------------------------------------------------------
// Admin game-master action (structured native action, else a text-command fallback)
// ---------------------------------------------------------------------------

/// `getAdminGameMasterAction`: a structured native action if the model emitted one,
/// otherwise classify the player's text into FOLLOW/STOP_FOLLOW/ATTACK.
pub fn get_admin_game_master_action(config: &BridgeConfig, turn: &Value, message: &str) -> GameMaster {
    let structured = get_native_game_master_action(config, turn);
    if structured.should_trigger {
        return structured;
    }
    get_admin_text_command_action(config, message)
}

struct AdminRes {
    follow: Regex,
    negation: Regex,
    stop: Regex,
    dismiss: Regex,
    attack: Regex,
    follow_verb: Regex,
}

fn admin_res() -> &'static AdminRes {
    static RE: OnceLock<AdminRes> = OnceLock::new();
    RE.get_or_init(|| AdminRes {
        follow: Regex::new(r"_(?:follow|following|follower|escort)_").unwrap(),
        negation: Regex::new(r"_(?:do_not|dont|don_t|never|no)_").unwrap(),
        stop: Regex::new(r"_(?:stop_following|stop_follow|stop)_").unwrap(),
        dismiss: Regex::new(r"_(?:dismiss|wait_here|stay_here|stand_down|go_away)_").unwrap(),
        attack: Regex::new(r"_(?:attack|fight|start_combat|hostile|aggro|kill|shoot|hit)_").unwrap(),
        follow_verb: Regex::new(r"_(?:follow|follow_me|come_with|come_along|come_here|escort)_").unwrap(),
    })
}

/// `getAdminTextCommandAction`: regex-classify the admin's words. The action's
/// `actor` is the raw text — `resolve_native_actor_for_admin` turns it into an NPC.
fn get_admin_text_command_action(config: &BridgeConfig, message: &str) -> GameMaster {
    if !config.enable_action_books {
        return GameMaster::none();
    }
    let text = message.trim();
    if text.is_empty() {
        return GameMaster::none();
    }
    let slug = format!("_{}_", slug_lookup_key(text));
    let res = admin_res();
    let mentions_follow = res.follow.is_match(&slug);
    let has_negation = res.negation.is_match(&slug);
    let wants_stop_follow = ((has_negation || res.stop.is_match(&slug)) && mentions_follow)
        || res.dismiss.is_match(&slug);
    let wants_attack = res.attack.is_match(&slug);
    let wants_follow = res.follow_verb.is_match(&slug);
    if has_negation && !wants_stop_follow {
        return GameMaster::none();
    }
    let action = if wants_stop_follow {
        "STOP_FOLLOW"
    } else if wants_attack {
        "ATTACK"
    } else if wants_follow {
        "FOLLOW"
    } else {
        return GameMaster::none();
    };
    let action_id = match action {
        "ATTACK" => "combat.start",
        "STOP_FOLLOW" => "movement.stop_follow_target",
        _ => "movement.follow_target",
    };
    GameMaster {
        action: action.into(),
        confidence: "1.00".into(),
        should_trigger: true,
        action_id: action_id.into(),
        reason: "admin text command fallback".into(),
        actions: vec![json!({
            "action_id": action_id,
            "target": "player",
            "actor": text,
            "parameters": { "target": "player", "actor": text },
            "confidence": "1.00",
            "reason": "admin text command fallback",
        })],
    }
}

// ---------------------------------------------------------------------------
// Scoped catalog + trusted-execution argument resolver
// ---------------------------------------------------------------------------

fn get_catalog_candidate_lookup_keys(candidate: &Value) -> Vec<String> {
    let metadata = candidate.get("metadata").cloned().unwrap_or_else(|| json!({}));
    let public_metadata = candidate.get("publicMetadata").cloned().unwrap_or_else(|| json!({}));
    let mut sources: Vec<Option<String>> = vec![
        value_to_string(candidate.get("id").unwrap_or(&Value::Null)),
        value_to_string(candidate.get("name").unwrap_or(&Value::Null)),
        value_to_string(candidate.get("title").unwrap_or(&Value::Null)),
        value_to_string(metadata.get("editorId").unwrap_or(&Value::Null)),
        value_to_string(metadata.get("fullName").unwrap_or(&Value::Null)),
        value_to_string(metadata.get("recordType").unwrap_or(&Value::Null)),
        value_to_string(public_metadata.get("editorId").unwrap_or(&Value::Null)),
        value_to_string(public_metadata.get("fullName").unwrap_or(&Value::Null)),
        value_to_string(public_metadata.get("recordType").unwrap_or(&Value::Null)),
    ];
    if let Some(arr) = candidate.get("aliases").and_then(Value::as_array) {
        for v in arr {
            sources.push(value_to_string(v));
        }
    }
    let mut keys = Vec::new();
    for s in sources.into_iter().flatten() {
        let trimmed = s.trim().to_string();
        keys.push(trimmed);
        keys.push(slug_lookup_key(&s));
    }
    unique(keys)
}

fn get_object_path_value<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for part in path.split('.').map(str::trim).filter(|p| !p.is_empty()) {
        current = current.get(part)?;
    }
    Some(current)
}

fn find_scoped_catalog_item(action: &Value, config: &Value) -> Option<Value> {
    let item_id = first_str(config, &["itemId", "id"]);
    let item_id = if item_id.is_empty() {
        get_action_parameter_value_from_config(action, config)
            .and_then(|v| value_to_string(&v))
            .unwrap_or_default()
            .trim()
            .to_string()
    } else {
        item_id
    };
    if item_id.is_empty() {
        return None;
    }
    let normalized = slug_lookup_key(&item_id);
    let catalog_id = first_str(config, &["catalogId", "catalog"]);
    let catalogs = action.get("scopedCatalogs").and_then(Value::as_array)?;
    for catalog in catalogs {
        if !catalog_id.is_empty()
            && catalog.get("catalogId").and_then(Value::as_str).unwrap_or("").trim() != catalog_id
        {
            continue;
        }
        let items = catalog.get("items").and_then(Value::as_array);
        let items = match items {
            Some(i) => i,
            None => continue,
        };
        let exact = items
            .iter()
            .find(|c| c.get("id").and_then(Value::as_str).unwrap_or("").trim() == item_id);
        if let Some(item) = exact {
            return Some(item.clone());
        }
        let fuzzy = items
            .iter()
            .find(|c| get_catalog_candidate_lookup_keys(c).contains(&normalized));
        if let Some(item) = fuzzy {
            return Some(item.clone());
        }
    }
    None
}

/// `resolveTrustedExecutionArguments`: resolve every arg; a single required-but-
/// unresolvable arg aborts (`None`). Empty optional args are dropped.
fn resolve_trusted_execution_arguments(
    execution: &Value,
    action: &Value,
    request: &NativeRequest,
) -> Option<Vec<String>> {
    let arguments = match execution.get("arguments").and_then(Value::as_array) {
        Some(a) => a,
        None => return Some(Vec::new()),
    };
    let mut args = Vec::new();
    for argument in arguments {
        let resolved = resolve_trusted_execution_argument(argument, action, request)?;
        if !resolved.is_empty() {
            args.push(resolved);
        }
    }
    Some(args)
}

/// Resolve one argument. `None` = required + unresolvable (abort); `Some("")` = drop.
fn resolve_trusted_execution_argument(
    argument: &Value,
    action: &Value,
    request: &NativeRequest,
) -> Option<String> {
    if let Some(s) = argument.as_str() {
        return Some(s.trim().to_string());
    }
    let config = if argument.is_object() {
        argument.clone()
    } else {
        json!({})
    };
    let arg_type = first_str(&config, &["type", "source"]).to_lowercase();
    let required = match config.get("required") {
        Some(v) => js_truthy(v),
        None => true,
    };

    let mut value: Option<Value> = None;
    if matches!(
        arg_type.as_str(),
        "catalogmetadata" | "catalog_metadata" | "scopedcatalogmetadata" | "scoped_catalog_metadata"
    ) {
        if let Some(item) = find_scoped_catalog_item(action, &config) {
            let key = {
                let k = first_str(&config, &["metadataKey", "key"]);
                if k.is_empty() {
                    "formId".to_string()
                } else {
                    k
                }
            };
            value = item
                .get("metadata")
                .and_then(|m| get_object_path_value(m, &key))
                .cloned();
        }
    } else if matches!(arg_type.as_str(), "parameter" | "param") {
        value = get_action_parameter_value_from_config(action, &config);
    } else if arg_type == "literal" || config.get("value").is_some() {
        value = config.get("value").cloned();
    }

    let is_empty = value
        .as_ref()
        .map(|v| v.is_null() || value_to_string(v).map(|s| s.is_empty()).unwrap_or(false))
        .unwrap_or(true);
    if is_empty {
        if let Some(default) = config.get("default") {
            value = Some(default.clone());
        }
    }

    let arg_native_type = first_str(&config, &["argType", "nativeType", "valueType"]);
    let encoded = normalize_trusted_native_arg_value(&arg_native_type, value.as_ref(), &config, request);
    match encoded {
        Some(e) => Some(e),
        None if required => None,
        None => Some(String::new()),
    }
}

/// `normalizeTrustedNativeArgValue`: encode a value as `ref:`/`refid:`/`form:`/
/// `number:`/`string:`. `None` = could not encode.
fn normalize_trusted_native_arg_value(
    type_hint: &str,
    value: Option<&Value>,
    config: &Value,
    request: &NativeRequest,
) -> Option<String> {
    let arg_type = {
        let t = type_hint.trim().to_lowercase();
        if t.is_empty() {
            first_str(config, &["argType", "nativeType"]).to_lowercase()
        } else {
            t
        }
    };
    let arg_type = if arg_type.is_empty() { "string".to_string() } else { arg_type };

    let value = value?;
    if value.is_null() {
        return None;
    }
    let text = value_to_string(value).unwrap_or_default();
    if text.is_empty() {
        return None;
    }

    if matches!(arg_type.as_str(), "ref" | "reference" | "refid" | "reference_id") {
        return normalize_trusted_native_ref_arg_value(&text, request);
    }
    if matches!(arg_type.as_str(), "form" | "formid" | "form_id") {
        let normalized = text.trim().trim_start_matches("0x").trim_start_matches("0X").to_uppercase();
        if is_hex_1_8(&normalized) {
            return Some(format!("form:{:0>8}", normalized));
        }
        return None;
    }
    if matches!(arg_type.as_str(), "number" | "float" | "int" | "integer") {
        let mut number = value_to_f64(value)?;
        if !number.is_finite() {
            return None;
        }
        if let Some(min) = config.get("min").and_then(value_to_f64) {
            number = number.max(min);
        }
        if let Some(max) = config.get("max").and_then(value_to_f64) {
            number = number.min(max);
        }
        if matches!(arg_type.as_str(), "int" | "integer") {
            number = number.trunc();
        }
        return Some(format!("number:{}", js_number_to_string(number)));
    }

    // Default: string. Commas → spaces so the comma-join stays unambiguous.
    let text = sanitize_bridge_line(&text).replace(',', " ");
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(format!("string:{text}"))
    }
}

fn normalize_trusted_native_ref_arg_value(value: &str, request: &NativeRequest) -> Option<String> {
    let text = value.trim();
    let slug = slug_lookup_key(text);
    if text.is_empty()
        || matches!(
            slug.as_str(),
            "player" | "courier" | "me" | "myself" | "the_player" | "target_player"
        )
    {
        return Some("ref:player".to_string());
    }
    if matches!(slug.as_str(), "actor" | "speaker" | "npc" | "subject" | "source") {
        return Some("ref:actor".to_string());
    }
    let ref_id = normalize_native_ref_id(text);
    if !ref_id.is_empty() {
        return Some(format!("refid:{ref_id}"));
    }
    let candidates = nearby_npcs(request);
    let terms = get_spawn_ref_lookup_terms(text);
    for candidate in &candidates {
        let candidate_ref_id = get_candidate_ref_id(candidate);
        if candidate_ref_id.is_empty() {
            continue;
        }
        let keys = get_candidate_spawn_anchor_keys(candidate);
        if terms.iter().any(|t| keys.contains(t)) {
            return Some(format!("refid:{candidate_ref_id}"));
        }
    }
    // Word-tokenized + fuzzy resolution, so a whole admin sentence ("make Pete
    // follow me") resolves the same as a bare name would.
    resolve_nearby_candidate_from_text(text, &candidates, 0.7, true)
        .map(|candidate| format!("refid:{}", get_candidate_ref_id(candidate)))
}

fn normalize_native_ref_id(value: &str) -> String {
    let text = value.trim();
    if text.is_empty() {
        return String::new();
    }
    if text.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(n) = text.parse::<u64>() {
            return format!("{n:X}"); // decimal → hex, NOT zero-padded
        }
    }
    let normalized = text
        .trim_start_matches("0x")
        .trim_start_matches("0X")
        .trim_start_matches("refid:")
        .trim_start_matches("REFID:")
        .trim()
        .to_uppercase();
    if is_hex_1_8(&normalized) {
        format!("{normalized:0>8}")
    } else {
        String::new()
    }
}

fn split_terms_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)[,;/|]+|\s+(?:or|and)\s+").unwrap())
}

fn get_spawn_ref_lookup_terms(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in split_terms_re().split(value) {
        let text = part.trim();
        if !text.is_empty() {
            out.push(text.to_string());
        }
        let slug = slug_lookup_key(text);
        if !slug.is_empty() {
            out.push(slug);
        }
    }
    unique(out)
}

fn get_candidate_ref_id(candidate: &Value) -> String {
    for key in ["ref_id", "refId", "referenceId", "reference_id"] {
        if let Some(v) = candidate.get(key) {
            let s = value_to_string(v).unwrap_or_default();
            let id = normalize_native_ref_id(&s);
            if !id.is_empty() {
                return id;
            }
        }
    }
    String::new()
}

fn get_candidate_spawn_anchor_keys(candidate: &Value) -> Vec<String> {
    let keys = [
        "npc_key", "npcKey", "nativeNpcKey", "npc_name", "npcName", "name", "characterName",
        "character_name",
    ];
    let mut out = Vec::new();
    for k in keys {
        if let Some(s) = candidate.get(k).and_then(value_to_string) {
            let text = s.trim().to_string();
            out.push(text.clone());
            out.push(slug_lookup_key(&text));
            for word in text.split_whitespace() {
                out.push(slug_lookup_key(word));
            }
        }
    }
    unique(out)
}

pub(crate) fn nearby_npcs(request: &NativeRequest) -> Vec<Value> {
    request
        .metadata
        .get("targeting")
        .and_then(|t| t.get("nearby_npcs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Fuzzy target resolution
// ---------------------------------------------------------------------------

fn levenshtein(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

pub(crate) fn fuzzy_match_score(a: &str, b: &str) -> f64 {
    let x = slug_lookup_key(a);
    let y = slug_lookup_key(b);
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    if x == y {
        return 1.0;
    }
    let max_len = x.chars().count().max(y.chars().count());
    if max_len == 0 {
        0.0
    } else {
        1.0 - levenshtein(&x, &y) as f64 / max_len as f64
    }
}

/// Command verbs / filler that must never FUZZY-match an NPC name word. Exact
/// matches stay allowed (an NPC literally named "Follow" would be a legitimate
/// hit) — this only guards the per-word fuzzy fallback, so "make Pete follow
/// me" can resolve via "pete" but never via "follow" edging past the
/// threshold against some name.
const NPC_WORD_FUZZY_STOPWORDS: &[&str] = &[
    "make", "have", "tell", "ask", "force", "want", "need", "please", "follow", "following",
    "follower", "followed", "escort", "come", "coming", "stop", "stops", "stay", "stand", "wait",
    "leave", "dismiss", "attack", "fight", "kill", "shoot", "hostile", "aggro", "player",
    "courier", "everyone", "everybody", "here", "there", "with", "away", "them", "they", "him",
    "her", "his", "hers", "this", "that", "again", "down", "back", "around", "behind",
];

/// Resolves free text — a bare name OR a whole command sentence ("make Pete
/// follow me") — to ONE candidate NPC:
///   1. adjacent word-pair slugs (`easy_pete` beats its pieces),
///   2. single word slugs (>= 3 chars) against the candidates' anchor keys
///      (which already include per-word name slugs),
///   3. whole-text fuzzy >= `threshold` (the legacy path — typos in bare
///      names like "Sunny Smiels"),
///   4. per-word fuzzy >= `threshold`, guarded by [`NPC_WORD_FUZZY_STOPWORDS`]
///      and a 4-char minimum so command vocabulary can't sneak over the line.
/// Matching ONLY the turn's nearby candidates is the false-positive guard:
/// command words can never reach an NPC who isn't standing there.
pub(crate) fn resolve_nearby_candidate_from_text<'a>(
    text: &str,
    candidates: &'a [Value],
    threshold: f64,
    require_ref_id: bool,
) -> Option<&'a Value> {
    let text = text.trim();
    if text.is_empty() || candidates.is_empty() {
        return None;
    }
    let eligible: Vec<&Value> = candidates
        .iter()
        .filter(|c| !require_ref_id || !get_candidate_ref_id(c).is_empty())
        .collect();
    if eligible.is_empty() {
        return None;
    }

    let words: Vec<String> = text
        .split_whitespace()
        .map(slug_lookup_key)
        .filter(|w| !w.is_empty())
        .collect();

    // Exact passes: word pairs before single words, so a multi-word name wins
    // over (and can't be shadowed by) its pieces.
    let mut exact_terms: Vec<String> = words
        .windows(2)
        .map(|pair| format!("{}_{}", pair[0], pair[1]))
        .collect();
    exact_terms.extend(words.iter().filter(|w| w.chars().count() >= 3).cloned());
    for term in &exact_terms {
        for candidate in &eligible {
            if get_candidate_spawn_anchor_keys(candidate).contains(term) {
                return Some(*candidate);
            }
        }
    }

    // Fuzzy passes: whole text first (bare-name typo), then guarded words.
    fn best_fuzzy<'a>(value: &str, eligible: &[&'a Value]) -> (Option<&'a Value>, f64) {
        let mut best = None;
        let mut best_score = 0.0_f64;
        for candidate in eligible {
            for key in get_candidate_spawn_anchor_keys(candidate) {
                let score = fuzzy_match_score(value, &key);
                if score > best_score {
                    best_score = score;
                    best = Some(*candidate);
                }
            }
        }
        (best, best_score)
    }

    let (mut best, mut best_score) = best_fuzzy(text, &eligible);
    for word in words
        .iter()
        .filter(|w| w.chars().count() >= 4 && !NPC_WORD_FUZZY_STOPWORDS.contains(&w.as_str()))
    {
        let (candidate, score) = best_fuzzy(word, &eligible);
        if score > best_score {
            best_score = score;
            best = candidate;
        }
    }
    if best_score >= threshold {
        best
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Repeat-count expansion
// ---------------------------------------------------------------------------

fn get_native_action_repeat_config(execution: &Value) -> Option<Value> {
    let repeat = execution
        .get("repeat")
        .or_else(|| execution.get("repetition"))
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let parameter = first_str(&repeat, &["parameter", "parameterName", "name"]);
    if parameter.is_empty() {
        return None;
    }
    let mut obj = repeat.as_object().cloned().unwrap_or_default();
    obj.insert("parameter".into(), json!(parameter));
    Some(Value::Object(obj))
}

fn get_trusted_execution_parameter_argument(execution: &Value, parameter_name: &str) -> Value {
    let expected = parameter_name.trim();
    if expected.is_empty() {
        return json!({});
    }
    if let Some(args) = execution.get("arguments").and_then(Value::as_array) {
        for argument in args {
            if first_str(argument, &["parameter", "name"]) == expected {
                return argument.clone();
            }
        }
    }
    json!({})
}

fn clamp_repeat_count(value: Option<&Value>, repeat: &Value, argument_config: &Value) -> u32 {
    let fallback = repeat
        .get("default")
        .or_else(|| argument_config.get("default"))
        .and_then(value_to_f64)
        .unwrap_or(1.0);
    let mut count = value.and_then(value_to_f64).unwrap_or(fallback);
    if !count.is_finite() {
        count = if fallback.is_finite() { fallback } else { 1.0 };
    }
    if let Some(min) = repeat.get("min").or_else(|| argument_config.get("min")).and_then(value_to_f64) {
        count = count.max(min);
    }
    if let Some(max) = repeat.get("max").or_else(|| argument_config.get("max")).and_then(value_to_f64) {
        count = count.min(max);
    }
    (count.trunc().max(1.0)) as u32
}

fn get_native_action_repeat_count(execution: &Value, action: &Value) -> u32 {
    let repeat = match get_native_action_repeat_config(execution) {
        Some(r) => r,
        None => return 1,
    };
    let parameter = first_str(&repeat, &["parameter"]);
    let argument_config = get_trusted_execution_parameter_argument(execution, &parameter);
    clamp_repeat_count(
        get_action_parameter_value(action, &parameter).as_ref(),
        &repeat,
        &argument_config,
    )
}

fn get_native_action_repeat_argument_value(execution: &Value) -> Value {
    let repeat = match get_native_action_repeat_config(execution) {
        Some(r) => r,
        None => return Value::Null,
    };
    for key in ["argumentValue", "perCommandValue", "value"] {
        if let Some(v) = repeat.get(key) {
            return v.clone();
        }
    }
    json!(1)
}

fn clone_action_with_parameter(action: &Value, parameter_name: &str, value: &Value) -> Value {
    let mut params = get_action_parameters(action);
    params.insert(parameter_name.to_string(), value.clone());
    let mut cloned = action.as_object().cloned().unwrap_or_default();
    cloned.insert("parameters".into(), Value::Object(params.clone()));
    if action.get("params").is_some() && action.get("parameters").is_none() {
        cloned.insert("params".into(), Value::Object(params));
    }
    Value::Object(cloned)
}

fn clone_game_master_with_primary_action(
    gm: &GameMaster,
    primary: &Value,
    replacement: &Value,
) -> GameMaster {
    let mut replaced = false;
    let mut actions: Vec<Value> = gm
        .actions
        .iter()
        .map(|a| {
            if a == primary {
                replaced = true;
                replacement.clone()
            } else {
                a.clone()
            }
        })
        .collect();
    if !replaced {
        actions.push(replacement.clone());
    }
    GameMaster {
        actions,
        ..gm.clone()
    }
}

// ---------------------------------------------------------------------------
// Command-file builders
// ---------------------------------------------------------------------------

fn encode_command_value(value: &str) -> String {
    sanitize_bridge_line(value).replace('=', ":")
}

fn encode_base64_utf8(value: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(value.as_bytes())
}

struct BuiltCommand {
    request_id: String,
    format: &'static str,
    lines: Vec<String>,
}

/// `buildNativeActionCommandLines`: the V2 (trusted) or legacy line list.
fn build_native_action_command_lines(
    request: &NativeRequest,
    actor: &ActionActor,
    gm: &GameMaster,
    source: &str,
) -> Option<BuiltCommand> {
    let action = gm.action.trim().to_uppercase();
    let native_npc_key = first_non_empty([actor.native_npc_key.clone(), String::new()]);
    let native_npc_name = first_non_empty([
        actor.native_npc_name.clone(),
        actor.character_name.clone(),
        native_npc_key.clone(),
    ]);
    let primary = get_primary_game_master_action(gm).unwrap_or(Value::Null);
    let binding = primary.get("binding").cloned().unwrap_or_else(|| json!({}));
    let execution = primary.get("execution").cloned().unwrap_or_else(|| json!({}));
    let trusted_script = execution.get("script").and_then(Value::as_str).unwrap_or("").trim().to_string();
    let trusted_engine = binding.get("engine").and_then(Value::as_str).unwrap_or("").trim().to_lowercase();
    let request_id = {
        let raw = first_non_empty([request.request_id.clone()]);
        if raw.is_empty() {
            sanitize_bridge_line(&format!("{source}-{}", now_epoch_millis()))
        } else {
            sanitize_bridge_line(&raw)
        }
    };
    let player_text = request.player_text.clone();

    if trusted_engine == TRUSTED_FNV_ACTION_ENGINE && !trusted_script.is_empty() {
        let args = match resolve_trusted_execution_arguments(&execution, &primary, request) {
            Some(a) => a,
            None => {
                warn!(
                    "could not resolve trusted Action Book arguments for {}",
                    if gm.action_id.is_empty() { &action } else { &gm.action_id }
                );
                return None;
            }
        };
        let action_id = {
            let pid = primary.get("action_id").and_then(Value::as_str).unwrap_or("");
            if pid.is_empty() { gm.action_id.clone() } else { pid.to_string() }
        };
        let lines = vec![
            NATIVE_ACTION_COMMAND_VERSION.to_string(),
            format!("request_id={}", encode_command_value(&request_id)),
            format!("npc_key={}", encode_command_value(&native_npc_key)),
            format!("npc_name={}", encode_command_value(&native_npc_name)),
            format!("action={}", encode_command_value(&action)),
            format!("action_id={}", encode_command_value(&action_id)),
            format!(
                "action_book_id={}",
                encode_command_value(primary.get("action_book_id").and_then(Value::as_str).unwrap_or(""))
            ),
            format!("engine={}", encode_command_value(&trusted_engine)),
            format!(
                "template_id={}",
                encode_command_value(
                    &first_str(&execution, &["templateId", "template_id"])
                )
            ),
            format!(
                "language={}",
                encode_command_value(execution.get("language").and_then(Value::as_str).unwrap_or(""))
            ),
            format!("arguments={}", encode_command_value(&args.join(","))),
            format!("confidence={}", encode_command_value(&gm.confidence)),
            format!(
                "reason={}",
                encode_command_value(if gm.reason.is_empty() { source } else { &gm.reason })
            ),
            format!("player_text={}", encode_base64_utf8(&player_text)),
            format!("script_base64={}", encode_base64_utf8(&trusted_script)),
        ];
        return Some(BuiltCommand {
            request_id,
            format: NATIVE_ACTION_COMMAND_VERSION,
            lines,
        });
    }

    // Legacy: 7 bare positional lines.
    let lines = [
        request_id.clone(),
        native_npc_key,
        native_npc_name,
        action,
        gm.confidence.clone(),
        if gm.reason.is_empty() { source.to_string() } else { gm.reason.clone() },
        player_text,
    ]
    .iter()
    .map(|l| sanitize_bridge_line(l))
    .collect();
    Some(BuiltCommand {
        request_id,
        format: "legacy",
        lines,
    })
}

/// `buildNativeActionCommands`: the command set, with repeat expansion.
fn build_native_action_commands(
    request: &NativeRequest,
    actor: &ActionActor,
    gm: &GameMaster,
    source: &str,
) -> Option<Vec<BuiltCommand>> {
    let primary = get_primary_game_master_action(gm).unwrap_or(Value::Null);
    let execution = primary.get("execution").cloned().unwrap_or_else(|| json!({}));
    let repeat = get_native_action_repeat_config(&execution);
    let repeat_count = get_native_action_repeat_count(&execution, &primary);
    if repeat.is_none() || repeat_count <= 1 {
        return build_native_action_command_lines(request, actor, gm, source).map(|c| vec![c]);
    }
    let repeat = repeat.unwrap();
    let parameter = first_str(&repeat, &["parameter"]);
    let repeated_value = get_native_action_repeat_argument_value(&execution);
    let mut commands = Vec::new();
    for _ in 0..repeat_count {
        let repeated_action = clone_action_with_parameter(&primary, &parameter, &repeated_value);
        let repeated_gm = clone_game_master_with_primary_action(gm, &primary, &repeated_action);
        match build_native_action_command_lines(request, actor, &repeated_gm, source) {
            Some(c) => commands.push(c),
            None => return None,
        }
    }
    Some(commands)
}

/// Build the native action command FILE BODY (the exact `control/actions/*.txt`
/// content) WITHOUT writing it — for the scheduler to capture at schedule time
/// (where the turn context is available) and replay at fire time by writing it
/// verbatim. The command is self-contained: the actor is named by `npc_key` and
/// the script args are resolved live by the plugin (`actor`/`player`), so the
/// captured body works whenever it is written, regardless of conversation state.
/// Returns `None` if the action isn't a queueable native/Action-Book action.
/// Repeat expansion is ignored (a scheduled action fires once).
pub fn build_native_command_body(
    request: &NativeRequest,
    actor: &ActionActor,
    gm: &GameMaster,
    source: &str,
) -> Option<String> {
    if !gm.should_trigger {
        return None;
    }
    let action = gm.action.trim().to_uppercase();
    if !["ATTACK", "FOLLOW", "STOP_FOLLOW", "ACTION_BOOK"].contains(&action.as_str()) {
        return None;
    }
    if actor.native_npc_key.trim().is_empty() && actor.native_npc_name.trim().is_empty() {
        return None;
    }
    let command = build_native_action_command_lines(request, actor, gm, source)?;
    Some(format!("{}\r\n", command.lines.join("\r\n")))
}

/// `writeNativeGameMasterCommand`: gate + write a command file to every root's
/// `control/actions`. Returns true if anything was queued.
pub fn write_native_game_master_command(
    config: &BridgeConfig,
    request: &NativeRequest,
    actor: &ActionActor,
    gm: &GameMaster,
    source: &str,
) -> bool {
    if !gm.should_trigger {
        return false;
    }
    let action = gm.action.trim().to_uppercase();
    if !["ATTACK", "FOLLOW", "STOP_FOLLOW", "ACTION_BOOK"].contains(&action.as_str()) {
        return false;
    }
    if actor.native_npc_key.trim().is_empty() && actor.native_npc_name.trim().is_empty() {
        warn!("could not resolve actor for {action}; command not queued");
        return false;
    }
    let commands = match build_native_action_commands(request, actor, gm, source) {
        Some(c) if !c.is_empty() => c,
        _ => return false,
    };

    let mut queued = 0;
    for root in &config.native_bridge_roots {
        if root.parent().map(|p| !p.exists()).unwrap_or(true) {
            continue;
        }
        let directory = native_action_command_dir(root);
        if std::fs::create_dir_all(&directory).is_err() {
            continue;
        }
        for (index, command) in commands.iter().enumerate() {
            let file_id = safe_file_id(&format!(
                "{}-{}-{}-{:03}",
                command.request_id,
                action.to_lowercase(),
                now_epoch_millis(),
                index + 1
            ));
            let body = format!("{}\r\n", command.lines.join("\r\n"));
            if std::fs::write(directory.join(format!("{file_id}.txt")), body).is_ok() {
                queued += 1;
            }
        }
    }
    if queued > 0 {
        let format = commands.first().map(|c| c.format).unwrap_or("legacy");
        tracing::info!(
            "queued {} {action} ({format}) command(s) for {}",
            commands.len(),
            if actor.native_npc_name.is_empty() { &actor.native_npc_key } else { &actor.native_npc_name }
        );
    }
    queued > 0
}

fn native_action_command_dir(root: &Path) -> PathBuf {
    root.join("control").join("actions")
}

// ---------------------------------------------------------------------------
// Value helpers (JS `||`/`Number()`/`Boolean()` semantics)
// ---------------------------------------------------------------------------

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(js_number_to_string(n.as_f64()?)),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0.0)
            } else {
                t.parse::<f64>().ok()
            }
        }
        _ => None,
    }
}

fn js_number_to_string(f: f64) -> String {
    if f.fract() == 0.0 && f.is_finite() && f.abs() < 1e15 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
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

/// First non-empty trimmed string across the given keys (string/number/bool values).
fn first_str(v: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = v.get(*k).and_then(value_to_string) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    String::new()
}

fn first_non_empty<const N: usize>(values: [String; N]) -> String {
    values.into_iter().find(|s| !s.trim().is_empty()).unwrap_or_default()
}

fn is_hex_1_8(s: &str) -> bool {
    !s.is_empty() && s.len() <= 8 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn unique(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values.into_iter().filter(|v| seen.insert(v.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BridgeConfig {
        crate::config::load_test_config()
    }

    fn request() -> NativeRequest {
        NativeRequest {
            request_id: "req_x".into(),
            npc_key: "easy_pete".into(),
            npc_name: "Easy Pete".into(),
            player_text: "follow me".into(),
            ..Default::default()
        }
    }

    fn follow_turn() -> Value {
        json!({
            "structured": { "actions": [
                { "id": "movement.follow_target", "target": "player", "confidence": 0.9, "reason": "asked to follow" }
            ]}
        })
    }

    #[test]
    fn classifies_follow() {
        let gm = get_native_game_master_action(&cfg(), &follow_turn());
        assert_eq!(gm.action, "FOLLOW");
        assert!(gm.should_trigger);
        assert_eq!(gm.confidence, "0.90");
    }

    #[test]
    fn below_confidence_is_none() {
        let turn = json!({ "structured": { "actions": [
            { "id": "combat.start", "target": "player", "confidence": 0.3 }
        ]}});
        assert_eq!(get_native_game_master_action(&cfg(), &turn).action, "NONE");
    }

    #[test]
    fn legacy_command_format() {
        let gm = get_native_game_master_action(&cfg(), &follow_turn());
        let actor = ActionActor {
            native_npc_key: "easy_pete".into(),
            native_npc_name: "Easy Pete".into(),
            character_name: "Easy Pete".into(),
            character_id: "Easy Pete".into(),
        };
        let cmd = build_native_action_command_lines(&request(), &actor, &gm, "src").unwrap();
        assert_eq!(cmd.format, "legacy");
        assert_eq!(
            cmd.lines,
            vec!["req_x", "easy_pete", "Easy Pete", "FOLLOW", "0.90", "asked to follow", "follow me"]
        );
    }

    #[test]
    fn v2_command_format_with_args() {
        let turn = json!({
            "structured": { "actions": [
                { "id": "world.spawn_entity", "confidence": 0.95, "reason": "spawn", "entity": "deathclaw", "count": 1 }
            ]},
            "metadata": { "activatedActions": [
                { "actionId": "world.spawn_entity", "bookId": "FNV", "binding": { "engine": "fallout-new-vegas:xnvse" },
                  "execution": { "script": "SpawnCreature", "templateId": "tmpl", "language": "geckscript",
                    "arguments": [ { "type": "catalogMetadata", "catalogId": "fnv.entities", "argType": "form", "parameter": "entity" } ] },
                  "scopedCatalogs": [ { "catalogId": "fnv.entities", "items": [
                      { "id": "deathclaw", "name": "Deathclaw", "metadata": { "formId": "0014F42C" } } ] } ] }
            ]}
        });
        let gm = get_native_game_master_action(&cfg(), &turn);
        assert_eq!(gm.action, "ACTION_BOOK");
        let actor = ActionActor { native_npc_key: "todd".into(), native_npc_name: "Todd".into(), ..Default::default() };
        let cmd = build_native_action_command_lines(&request(), &actor, &gm, "src").unwrap();
        assert_eq!(cmd.lines[0], "NVBRIDGE_ACTION_V2");
        assert_eq!(cmd.lines[5], "action_id=world.spawn_entity");
        assert_eq!(cmd.lines[7], "engine=fallout-new-vegas:xnvse");
        // catalog deathclaw → formId 0014F42C → form:0014F42C
        assert_eq!(cmd.lines[10], "arguments=form:0014F42C");
        // script_base64 of "SpawnCreature"
        assert_eq!(cmd.lines[14], format!("script_base64={}", encode_base64_utf8("SpawnCreature")));
    }

    #[test]
    fn ref_and_number_encoders() {
        let req = request();
        assert_eq!(
            normalize_trusted_native_arg_value("ref", Some(&json!("player")), &json!({}), &req),
            Some("ref:player".into())
        );
        assert_eq!(
            normalize_trusted_native_arg_value("number", Some(&json!(5)), &json!({"max": 3}), &req),
            Some("number:3".into())
        );
        assert_eq!(
            normalize_trusted_native_arg_value("form", Some(&json!("14f42c")), &json!({}), &req),
            Some("form:0014F42C".into())
        );
    }

    #[test]
    fn fuzzy_resolves_close_name() {
        let candidates = vec![json!({ "npc_name": "Sunny Smiles", "ref_id": "00104D2A" })];
        // "sunny smiels" (typo) should still resolve to Sunny's refid.
        let resolved = resolve_nearby_candidate_from_text("Sunny Smiels", &candidates, 0.7, true);
        assert_eq!(
            resolved.map(get_candidate_ref_id),
            Some("00104D2A".to_string())
        );
    }

    /// A Todd-style request: the admin is the addressee, real NPCs are nearby.
    fn admin_request_with_nearby() -> NativeRequest {
        let metadata = serde_json::from_value(json!({
            "targeting": { "nearby_npcs": [
                { "npc_key": "sunny_smiles", "npc_name": "Sunny Smiles", "ref_id": "00104D2A" },
                { "npc_key": "easy_pete", "npc_name": "Easy Pete", "ref_id": "0010A6B1" }
            ]}
        }))
        .unwrap();
        NativeRequest {
            request_id: "req_admin".into(),
            npc_key: "todd".into(),
            npc_name: "Todd".into(),
            player_text: "make Pete follow me".into(),
            metadata,
            ..Default::default()
        }
    }

    #[test]
    fn ref_arg_resolves_npc_named_inside_a_sentence() {
        // The admin text-command actor is the WHOLE sentence; the word "Pete"
        // must resolve to the nearby Easy Pete's refid.
        let req = admin_request_with_nearby();
        assert_eq!(
            normalize_trusted_native_ref_arg_value("make Pete follow me", &req),
            Some("refid:0010A6B1".into())
        );
        // Multi-word name spoken in full, mid-sentence (word-pair pass).
        assert_eq!(
            normalize_trusted_native_ref_arg_value("tell easy pete to wave", &req),
            Some("refid:0010A6B1".into())
        );
    }

    #[test]
    fn command_words_do_not_false_match_nearby_npcs() {
        let req = admin_request_with_nearby();
        // No NPC named in the sentence → no resolution (the admin resolver's
        // later steps / fallback handle it), never a fuzzy hit off "follow".
        assert_eq!(
            normalize_trusted_native_ref_arg_value("follow me again", &req),
            None
        );
    }

    #[test]
    fn in_sentence_typo_resolves_via_guarded_word_fuzzy() {
        let candidates = vec![json!({ "npc_name": "Sunny Smiles", "ref_id": "00104D2A" })];
        // "sunnny" (typo, not a stopword, >= 4 chars) fuzzy-matches "sunny".
        let resolved =
            resolve_nearby_candidate_from_text("make sunnny follow me", &candidates, 0.7, true);
        assert_eq!(
            resolved.map(get_candidate_ref_id),
            Some("00104D2A".to_string())
        );
    }
}
