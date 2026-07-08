//! Manual (ignored) live-Gemma harness for the GROUP-DISCOVERY loot flow:
//!
//!   "loot the oven"
//!     -> actions(loot)            [group expansion, chasm-side]
//!     -> search_containers        [game lists exact names, no contents]
//!     -> loot_container("Oven")   [world action: walk over + open]
//!     (then, async in production: contents arrive, the model chooses items)
//!
//! The always-injected prompt carries only the ACTION GROUPS list — the loot
//! actions themselves are discovered mid-turn. Game queries are FAKED with the
//! mod's real text shapes; the group expansion uses the REAL production
//! formatter (npc_group_actions_result). Run:
//!   CHASM_AB_LLAMA=http://127.0.0.1:5001 \
//!   CHASM_EMBED_DIR=<hf models dir> \
//!   CHASM_SNAP_TEST_BOOK=<path to "Fallout New Vegas Action Book.json"> \
//!   cargo test -p chasm-prompt --test loot_battery --release -- --ignored --nocapture

use chasm_embed::{models_present, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_prompt::{
    action_alias_pairs, action_embed_candidates, action_enum_values, action_passes_scopes,
    action_verb_pairs, npc_actions_instruction, resolve_guess_to_action, slug_action_alias, RetrievalCtx,
    NPC_STRUCTURED_OUTPUT_INSTRUCTION,
};
use chasm_st_compat::ActionEntry;
use serde_json::{json, Value};
use std::collections::HashMap;

const FLOOR: f32 = 0.5;
const SAMPLES: usize = 3;
const MAX_ROUNDS: usize = 4;

fn entries_from_book_json(book: &Value) -> Vec<ActionEntry> {
    let str_vec = |e: &Value, k: &str| -> Vec<String> {
        e.get(k)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default()
    };
    let str_of =
        |e: &Value, k: &str| -> String { e.get(k).and_then(Value::as_str).unwrap_or("").to_string() };
    let Some(entries) = book.get("entries").and_then(Value::as_object) else {
        return Vec::new();
    };
    entries
        .values()
        .map(|e| ActionEntry {
            keys: str_vec(e, "key"),
            title: str_of(e, "comment"),
            description: str_of(e, "content"),
            constant: false,
            disable: e.get("disable").and_then(Value::as_bool).unwrap_or(false),
            vectorized: true,
            order: 0.0,
            case_sensitive: None,
            action_id: str_of(e, "actionId"),
            alias: e.get("alias").and_then(Value::as_str).map(str::to_string),
            short_name: None,
            verbs: str_vec(e, "verbs"),
            group: str_of(e, "group"),
            risk_tier: String::new(),
            parameters_schema: Value::Null,
            preconditions: Vec::new(),
            effects: Vec::new(),
            examples_when_to_use: Vec::new(),
            examples_when_not_to_use: Vec::new(),
            vectorizable_text: str_of(e, "vectorizableText"),
            execution: Value::Null,
            binding: e.get("binding").cloned().unwrap_or(Value::Null),
            requires_target: false,
            scoped_catalogs: Vec::new(),
            scopes: str_vec(e, "scopes"),
        })
        .filter(|entry| !entry.action_id.is_empty())
        .collect()
}

fn groups_from_book_json(book: &Value) -> Vec<(String, String)> {
    book.get("groups")
        .and_then(Value::as_object)
        .map(|groups| {
            groups
                .iter()
                .filter_map(|(name, blurb)| blurb.as_str().map(|b| (name.clone(), b.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Loot verbs (alias + verb lexicon) of the book's loot_container entry —
/// the set the production grammar excludes pre-search and pins post-search.
fn loot_verbs(entries: &[ActionEntry]) -> Vec<String> {
    entries
        .iter()
        .filter(|e| !e.disable && e.action_id == "world.loot_container")
        .flat_map(|e| e.alias.clone().into_iter().chain(e.verbs.iter().cloned()))
        .collect()
}

/// The production step schema per round (mirrors llm.rs LootGrammar): before
/// a search discovers containers the loot verbs are EXCLUDED from the enum;
/// after, a dedicated branch pins loot_container's target to the real names.
fn npc_schema_round(
    action_enum: &[String],
    loot: &[String],
    discovered_containers: &[String],
) -> Value {
    let step = |verbs: &[String], targets: Option<&[String]>| -> Value {
        let mut action = json!({ "type": "string" });
        if !verbs.is_empty() {
            action["enum"] = json!(verbs);
        }
        let mut target = json!({ "type": "string" });
        if let Some(t) = targets {
            target["enum"] = json!(t);
        }
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "action": action,
                "target": target,
                "items": { "type": "string" },
                "time": { "type": "string" },
                "condition": { "type": "string" },
                "delay": { "type": "string" }
            },
            "required": ["action", "target", "items"]
        })
    };
    let other: Vec<String> = action_enum
        .iter()
        .filter(|v| !loot.iter().any(|lv| lv.eq_ignore_ascii_case(v)))
        .cloned()
        .collect();
    let steps = if discovered_containers.is_empty() {
        step(&other, None)
    } else {
        json!({ "anyOf": [step(loot, Some(discovered_containers)), step(&other, None)] })
    };
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_npc_reply",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "speech": { "type": "string" },
                    "actions": { "type": "array", "items": steps }
                },
                "required": ["speech", "actions"]
            }
        }
    })
}

/// The production step schema (mirrors llm.rs: items grammar-required).
fn npc_schema(action_enum: &[String]) -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "chasm_npc_reply",
            "strict": true,
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "speech": { "type": "string" },
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "action": { "type": "string", "enum": action_enum },
                                "target": { "type": "string" },
                                "items": { "type": "string" },
                                "time": { "type": "string" },
                                "condition": { "type": "string" },
                                "delay": { "type": "string" }
                            },
                            "required": ["action", "target", "items"]
                        }
                    }
                },
                "required": ["speech", "actions"]
            }
        }
    })
}

/// Faked GAME query results in the mod's exact text shapes. The world:
/// an Oven, a locked Footlocker, Barton Thorn's body, bottles and a hat on
/// the floor.
fn fake_game_query(action_id: &str) -> Option<&'static str> {
    match action_id {
        "world.search_area" => Some(
            "Around you: Oven [container] (2m), Footlocker [container, locked] (3m), \
             body of Barton Thorn [body] (4m), 4x Milk Bottle (3m) and Straw Hat (5m). \
             Containers do not show their contents until opened. Use the exact names.",
        ),
        "movement.list_places" => Some(
            "Places you can travel to from here: Prospector Saloon (28m),              Goodsprings General Store (41m), Goodsprings Cemetery (220m) - and              \"player\" to return to the player. Use the exact names.",
        ),
        "chasm.who_is_here" => Some("Nearby: Easy Pete (7m), Chet (12m)."),
        "chasm.recall" => Some("(your memory) Nothing much lately."),
        _ => None,
    }
}

struct Case {
    text: &'static str,
    /// Required world action by the final round: "" | "world.loot_container" |
    /// "world.take_items".
    want_action: &'static str,
    /// The action step's target+items text must contain one of these.
    want_words_any: &'static [&'static str],
    /// Queries that must have run at some point (any order).
    want_queries: &'static [&'static str],
    /// Words the FINAL round's speech must contain one of (empty = don't care).
    want_answer_any: &'static [&'static str],
    /// No queries and no world actions allowed at all.
    forbid_all: bool,
}

fn battery() -> Vec<Case> {
    vec![
        // The oven, end to end: discover -> search -> open the exact name.
        Case {
            text: "loot the oven",
            want_action: "world.loot_container",
            want_words_any: &["oven"],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        Case {
            text: "go through that footlocker",
            want_action: "world.loot_container",
            want_words_any: &["footlocker"],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        Case {
            text: "search the dead body and take what's useful",
            want_action: "world.loot_container",
            want_words_any: &["barton", "body"],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        // Ground sweep: must find the EXACT name via search_items.
        Case {
            text: "pick up all the bottles in this room",
            want_action: "world.take_items",
            want_words_any: &["milk bottle"],
            want_queries: &["world.search_area"],
            want_answer_any: &[],
            forbid_all: false,
        },
        // Question: search + spoken answer, no committed loot required.
        Case {
            text: "anything worth taking around here?",
            want_action: "",
            want_words_any: &[],
            want_queries: &[],
            want_answer_any: &["oven", "footlocker", "bottle", "hat", "body", "barton"],
            forbid_all: false,
        },
        // Travel group: direct order resolves to the travel action with the
        // destination in its fields; questions ride list_places.
        Case {
            text: "go to the prospector saloon",
            want_action: "movement.travel_to_location",
            want_words_any: &["saloon"],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        Case {
            text: "come back to me",
            want_action: "movement.travel_to_location",
            want_words_any: &["player", "you", "courier"],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        Case {
            text: "where can we go around here?",
            want_action: "",
            want_words_any: &[],
            want_queries: &["movement.list_places"],
            want_answer_any: &["saloon", "store", "cemetery"],
            forbid_all: false,
        },
        Case {
            text: "actually stop, forget the trip",
            want_action: "movement.stop_travel|ai.wait_here",
            want_words_any: &[],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: false,
        },
        // Negatives: loot-adjacent words, nothing to do.
        Case {
            text: "nice hat you're wearing",
            want_action: "",
            want_words_any: &[],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: true,
        },
        Case {
            text: "this room is cozy, don't you think?",
            want_action: "",
            want_words_any: &[],
            want_queries: &[],
            want_answer_any: &[],
            forbid_all: true,
        },
    ]
}

#[test]
#[ignore = "drives a live llama-server; run manually"]
fn loot_battery_group_flow() {
    let llama = std::env::var("CHASM_AB_LLAMA").unwrap_or_else(|_| "http://127.0.0.1:5001".into());
    let Ok(book_path) = std::env::var("CHASM_SNAP_TEST_BOOK") else {
        eprintln!("SKIP: CHASM_SNAP_TEST_BOOK not set");
        return;
    };
    let raw = std::fs::read_to_string(&book_path).expect("read action book");
    let book: Value =
        serde_json::from_str(raw.trim_start_matches('\u{feff}')).expect("parse action book");
    let entries = entries_from_book_json(&book);
    let groups = groups_from_book_json(&book);
    assert!(!groups.is_empty(), "book has no groups map");
    let requested: Vec<String> = Vec::new();
    let eligible: Vec<ActionEntry> = entries
        .iter()
        .filter(|e| !e.disable && action_passes_scopes(e, &requested))
        .cloned()
        .collect();
    let enum_values = action_enum_values(&entries, &requested);
    let query_ids: std::collections::HashSet<String> = eligible
        .iter()
        .filter(|e| e.binding.get("engine").and_then(Value::as_str) == Some("chasm:query"))
        .map(|e| e.action_id.clone())
        .collect();
    for id in ["chasm.list_actions", "world.search_area"] {
        assert!(query_ids.contains(id), "{id} missing from the book queries");
    }
    assert!(eligible.iter().any(|e| e.action_id == "world.loot_container"));
    assert!(eligible.iter().any(|e| e.action_id == "world.take_items"));

    let mut by_alias: HashMap<String, String> = HashMap::new();
    for (id, alias) in &action_alias_pairs(&eligible) {
        by_alias.entry(slug_action_alias(alias)).or_insert_with(|| id.clone());
    }
    for (id, verb) in &action_verb_pairs(&eligible) {
        by_alias.entry(slug_action_alias(verb)).or_insert_with(|| id.clone());
    }
    let candidates = action_embed_candidates(&eligible);
    let cfg = RetrieverConfig::default();
    let retriever;
    let cache;
    let ctx = if models_present(&cfg) {
        retriever = Retriever::load(&cfg).expect("load retriever");
        cache = EmbeddingCache::open(std::env::temp_dir().join("chasm-loot-battery-cache"))
            .expect("open cache");
        Some(RetrievalCtx {
            retriever: &retriever,
            cache: &cache,
            chat_memory_enabled: false,
            lore_semantic_enabled: false,
            action_semantic_enabled: true,
            quest_semantic_enabled: false,
            candidates: 48,
            top_k: 8,
            min_score: 0.0,
            action_min_score: FLOOR,
            chat_memory_limit: 0,
            lore_limit: 0,
            quest_limit: 0,
        })
    } else {
        eprintln!("WARN: no embed models - cosine fallback disabled");
        None
    };
    let resolve = |verb: &str| -> Option<String> {
        if let Some(id) = by_alias.get(&slug_action_alias(verb)) {
            return Some(id.clone());
        }
        for word in verb.split_whitespace() {
            if let Some(id) = by_alias.get(&slug_action_alias(word)) {
                return Some(id.clone());
            }
        }
        ctx.as_ref()
            .and_then(|c| resolve_guess_to_action(c, verb, &candidates, FLOOR))
    };

    // Flat action space: every action is directly available; the model just
    // searches for what it needs.
    let query_section = npc_actions_instruction(&eligible, &requested);
    let groups_section = String::new();
    assert!(
        query_section.contains("search_area"),
        "search_area should be directly available"
    );
    assert!(groups_section.contains("\"loot\""), "groups section missing loot");
    eprintln!("--- always-on sections ---\n{query_section}{groups_section}--------------------------");
    let system = format!(
        "You are Sunny Smiles, a friendly gunslinger and guide in Goodsprings, talking to \
         the player (the Courier). The current in-game time is 12:30PM.\n\n{NPC_STRUCTURED_OUTPUT_INSTRUCTION}{query_section}{groups_section}"
    );
    let loot_verb_list = loot_verbs(&eligible);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();

    let mut failures: Vec<String> = Vec::new();
    let out_path = std::env::var("CHASM_AB_OUT").ok();
    let mut out_lines: Vec<String> = Vec::new();

    for case in battery() {
        eprintln!("\n== {:?}", case.text);
        for sample in 0..SAMPLES {
            let mut messages = vec![
                json!({ "role": "system", "content": system }),
                json!({ "role": "user", "content": case.text }),
            ];
            // Per round: (speech, resolved ids). World-action fields by id.
            let mut rounds: Vec<(String, Vec<String>)> = Vec::new();
            let mut action_text: HashMap<String, String> = HashMap::new();
            let mut ran_queries: Vec<String> = Vec::new();
            let mut executed: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut discovered: Vec<String> = Vec::new();
            let mut fail: Option<String> = None;
            for round in 0..MAX_ROUNDS {
                // Per-round grammar, exactly like production enum mode.
                let schema = npc_schema_round(&enum_values, &loot_verb_list, &discovered);
                let body = json!({
                    "model": "gemma-4-12b", "temperature": 0.7, "max_tokens": 300,
                    "messages": messages, "response_format": schema,
                });
                let resp = client
                    .post(format!("{llama}/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .body(body.to_string())
                    .send()
                    .expect("llama request");
                let parsed: Value = serde_json::from_str(&resp.text().unwrap()).unwrap();
                let content = parsed["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
                let turn: Value = serde_json::from_str(content.trim()).unwrap_or(Value::Null);
                let speech = turn.get("speech").and_then(Value::as_str).unwrap_or("").to_string();
                let steps: Vec<Value> = turn
                    .get("actions")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let mut ids: Vec<String> = Vec::new();
                let mut queries: Vec<(String, String)> = Vec::new();
                let mut world_this_round: Vec<(String, String)> = Vec::new();
                for step in &steps {
                    let Some(verb) = step.get("action").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(id) = resolve(verb) else { continue };
                    let target = step.get("target").and_then(Value::as_str).unwrap_or("");
                    let items = step.get("items").and_then(Value::as_str).unwrap_or("");
                    if query_ids.contains(&id) {
                        // Same dedup key as production: id + topic, so a
                        // corrected retry (actions("footlocker") -> actions("loot"))
                        // still runs while true repeats end the turn.
                        if executed.insert(format!("{id}|{}", target.trim().to_lowercase())) {
                            queries.push((id.clone(), target.to_string()));
                        }
                    } else {
                        let dest = step
                            .get("to")
                            .or_else(|| step.get("destination"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        world_this_round.push((id.clone(), format!("{target} {items} {dest}").to_lowercase()));
                    }
                    ids.push(id);
                }
                // Mirror production: a query round's world steps are PLANNING
                // against unseen listings - only query-free rounds act.
                if queries.is_empty() {
                    for (id, text) in world_this_round.drain(..) {
                        action_text.insert(id, text);
                    }
                }
                eprintln!("  [{sample}#r{round}] speech={:?} steps={:?}", truncate(&speech, 55), ids);
                out_lines.push(json!({
                    "case": case.text, "sample": sample, "round": round, "content": content,
                }).to_string());
                rounds.push((speech, ids));
                if queries.is_empty() {
                    break;
                }
                let mut results = String::new();
                for (id, topic) in &queries {
                    ran_queries.push(id.clone());
                    if id == "world.search_area" {
                        discovered = vec![
                            "Oven".to_string(),
                            "Footlocker".to_string(),
                            "body of Barton Thorn".to_string(),
                        ];
                    }
                    let text = fake_game_query(id).unwrap_or("(nothing)").to_string();
                    results.push_str(&text);
                    results.push('\n');
                }
                messages.push(json!({ "role": "assistant", "content": content }));
                messages.push(json!({ "role": "user", "content": format!(
                    "[QUERY RESULT]\n{results}[This result is complete and current. Continue the same reply: if the order needs an action from the result, emit it now with the EXACT names shown; answer questions naturally. Speaking is optional - stay silent when simply proceeding. NEVER repeat anything you already said this turn; only say what is new. Do not repeat a query.]"
                )}));
            }

            // Scoring.
            let final_speech = rounds.last().map(|r| r.0.to_lowercase()).unwrap_or_default();
            let all_ids: Vec<String> = rounds.iter().flat_map(|r| r.1.clone()).collect();
            if case.forbid_all {
                let offending: Vec<&String> = all_ids
                    .iter()
                    .filter(|id| id.starts_with("world.") || query_ids.contains(*id))
                    .collect();
                if !offending.is_empty() {
                    fail = Some(format!("expected a plain reply, got {offending:?}"));
                }
            }
            if !case.want_action.is_empty() {
                // "a|b" = any of these ids satisfies the case.
                let accepted: Vec<&str> = case.want_action.split('|').collect();
                match accepted.iter().find_map(|id| action_text.get(*id)) {
                    None => {
                        fail = Some(format!("never emitted {}: {:?}", case.want_action, all_ids));
                    }
                    Some(text) => {
                        if !case.want_words_any.is_empty()
                            && !case.want_words_any.iter().any(|w| text.contains(w))
                        {
                            fail = Some(format!(
                                "{} missing {:?} in fields: {:?}",
                                case.want_action, case.want_words_any, text
                            ));
                        }
                    }
                }
            }
            for wanted in case.want_queries {
                if !ran_queries.iter().any(|q| q == wanted) {
                    fail = Some(format!("required query {wanted} never ran ({ran_queries:?})"));
                }
            }
            if !case.want_answer_any.is_empty()
                && !case.want_answer_any.iter().any(|w| final_speech.contains(w))
            {
                fail = Some(format!("final speech missing answer words: {:?}", truncate(&final_speech, 80)));
            }
            match fail {
                Some(reason) => {
                    eprintln!("  [{sample}] FAIL: {reason}");
                    failures.push(format!("{:?} #{sample}: {reason}", case.text));
                }
                None => eprintln!("  [{sample}] PASS"),
            }
        }
    }

    if let Some(path) = out_path {
        std::fs::write(&path, out_lines.join("\n")).ok();
    }
    eprintln!("\n================ {} failures ================", failures.len());
    for f in &failures {
        eprintln!("  {f}");
    }
    assert!(
        failures.len() <= 2,
        "loot battery too flaky ({} failures):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}...", &s[..s.floor_char_boundary(n)]) }
}
