//! Manual (ignored) A/B harness: FREEFORM verb + resolver snap vs ENUM-GRAMMAR
//! verb constraint, against a LIVE llama-server and the REAL action book. Both
//! modes use the identical fully-enforced step schema (mirrors chasm-web's
//! `npc_structured_response_format`) — the ONLY variable is whether the step's
//! "action" field is a free string (resolved after generation: alias → book
//! verbs → cosine snap) or a grammar enum of aliases+verbs (the sampler steers,
//! the model never sees the list in either mode).
//!
//! Run (server must be up; uses one slot, sequential):
//!   CHASM_AB_LLAMA=http://127.0.0.1:5001 \
//!   CHASM_EMBED_DIR=<hf models dir> \
//!   CHASM_SNAP_TEST_BOOK=<path to "Fallout New Vegas Action Book.json"> \
//!   cargo test -p chasm-prompt --test enum_ab --release -- --ignored --nocapture

use chasm_embed::{models_present, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_prompt::{
    action_alias_pairs, action_embed_candidates, action_enum_values, action_passes_scopes,
    action_verb_pairs, resolve_guess_to_action, slug_action_alias, RetrievalCtx,
    NPC_STRUCTURED_OUTPUT_INSTRUCTION,
};
use chasm_st_compat::ActionEntry;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

const FLOOR: f32 = 0.5;
const SAMPLES: usize = 3;

// ---------------------------------------------------------------------------
// Book loading (same raw-JSON mirror as guess_snap.rs)
// ---------------------------------------------------------------------------

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
            alias: None,
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
            binding: Value::Null,
            requires_target: false,
            scoped_catalogs: Vec::new(),
            scopes: str_vec(e, "scopes"),
        })
        .filter(|entry| !entry.action_id.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Schema (mirror of chasm-web llm.rs npc_structured_response_format)
// ---------------------------------------------------------------------------

fn npc_schema(action_enum: Option<&[String]>) -> Value {
    let mut action = serde_json::Map::new();
    action.insert("type".into(), json!("string"));
    action.insert(
        "description".into(),
        json!("ONE short verb for what the NPC physically does."),
    );
    if let Some(values) = action_enum {
        action.insert("enum".into(), json!(values));
    }
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
                                "action": Value::Object(action),
                                "target": { "type": "string" },
                                "time": { "type": "string" },
                                "condition": { "type": "string" },
                                "delay": { "type": "string" }
                            },
                            "required": ["action"]
                        }
                    }
                },
                "required": ["speech", "actions"]
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Test battery
// ---------------------------------------------------------------------------

/// What the resolved plan must look like for a case to PASS.
struct Case {
    text: &'static str,
    /// Expected resolved action id; "" = no action; "SAFE" = anything except a
    /// combat/travel action (creative request with no matching book entry —
    /// no-op or a gesture both count as safe).
    expect_action: &'static str,
    /// Required substring of the step target ("" = don't care).
    expect_target: &'static str,
    /// true = the step must NOT be deferred (no time/condition/delay).
    expect_immediate: bool,
    /// Some(field) = the step MUST carry this deferral field non-empty.
    expect_deferred: Option<&'static str>,
}

const fn case(
    text: &'static str,
    expect_action: &'static str,
    expect_target: &'static str,
    expect_immediate: bool,
    expect_deferred: Option<&'static str>,
) -> Case {
    Case { text, expect_action, expect_target, expect_immediate, expect_deferred }
}

fn battery() -> Vec<Case> {
    vec![
        // Kill phrasings — the flagship flow.
        case("kill easy pete", "combat.start", "pete", true, None),
        case("kill easy pete for me", "combat.start", "pete", true, None),
        case("murder easy pete", "combat.start", "pete", true, None),
        case("take out easy pete", "combat.start", "pete", true, None),
        case("waste easy pete right now", "combat.start", "pete", true, None),
        case("shoot easy pete", "combat.start", "pete", true, None),
        case("put easy pete down", "combat.start", "pete", true, None),
        case("avenge me. easy pete just insulted my honor", "combat.start", "pete", true, None),
        // Ordinary actions.
        case("wave at me", "npc.gesture_wave", "", true, None),
        case("give me a salute", "npc.gesture_salute", "", true, None),
        case("follow me", "movement.follow_target", "", true, None),
        case("stop following me", "movement.stop_follow_target", "", true, None),
        case("wait here", "ai.wait_here", "", true, None),
        case("sit down", "ai.sit_down", "", true, None),
        // Genuine deferrals.
        case("wave at me in five minutes", "npc.gesture_wave", "", false, Some("time")),
        case("attack easy pete when i say go", "combat.start", "pete", false, Some("condition")),
        // Speech-only turns.
        case("how are you today?", "", "", true, None),
        case("tell me about goodsprings", "", "", true, None),
        // No matching action exists — the failure-mode probe.
        case("do a little dance for me", "SAFE", "", true, None),
        case("light that campfire over there", "SAFE", "", true, None),
    ]
}

const UNSAFE_IDS: [&str; 3] = ["combat.start", "combat.stop", "movement.travel_to_location"];

// ---------------------------------------------------------------------------
// Resolution (production order: alias/verb slug map -> per-word -> cosine snap)
// ---------------------------------------------------------------------------

struct Resolver<'a> {
    by_alias: HashMap<String, String>,
    ctx: Option<RetrievalCtx<'a>>,
    candidates: Vec<(String, String)>,
}

impl<'a> Resolver<'a> {
    fn resolve(&self, verb: &str) -> Option<String> {
        if let Some(id) = self.by_alias.get(&slug_action_alias(verb)) {
            return Some(id.clone());
        }
        for word in verb.split_whitespace() {
            if let Some(id) = self.by_alias.get(&slug_action_alias(word)) {
                return Some(id.clone());
            }
        }
        let ctx = self.ctx.as_ref()?;
        resolve_guess_to_action(ctx, verb, &self.candidates, FLOOR)
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Tally {
    total: usize,
    action_ok: usize,
    target_ok: usize,
    schedule_ok: usize,
    wrong_action: usize,
    dropped_action: usize,
    latency_ms: u128,
}

fn score(case: &Case, steps: &[(Option<String>, Value)], tally: &mut Tally) -> String {
    tally.total += 1;
    let first = steps.first();
    let resolved = first.and_then(|(id, _)| id.clone()).unwrap_or_default();
    let raw_verb = first
        .map(|(_, s)| s.get("action").and_then(Value::as_str).unwrap_or("").to_string())
        .unwrap_or_default();

    // Action correctness.
    let action_ok = match case.expect_action {
        "" => steps.is_empty(),
        "SAFE" => resolved.is_empty() || !UNSAFE_IDS.contains(&resolved.as_str()),
        want => resolved == want,
    };
    if action_ok {
        tally.action_ok += 1;
    } else if resolved.is_empty() && !case.expect_action.is_empty() {
        tally.dropped_action += 1;
    } else {
        tally.wrong_action += 1;
    }

    // Target correctness (only when an action was expected with a named target).
    let target = first
        .map(|(_, s)| s.get("target").and_then(Value::as_str).unwrap_or("").to_lowercase())
        .unwrap_or_default();
    let target_ok = case.expect_target.is_empty() || target.contains(case.expect_target);
    if target_ok {
        tally.target_ok += 1;
    }

    // Scheduling correctness.
    let deferral = |s: &Value, k: &str| -> bool {
        s.get(k).and_then(Value::as_str).map(|v| !v.trim().is_empty()).unwrap_or(false)
    };
    let schedule_ok = match (case.expect_immediate, case.expect_deferred) {
        (true, _) => first
            .map(|(_, s)| {
                !deferral(s, "time") && !deferral(s, "condition") && !deferral(s, "delay")
            })
            .unwrap_or(true),
        (false, Some(field)) => first.map(|(_, s)| deferral(s, field)).unwrap_or(false),
        (false, None) => true,
    };
    if schedule_ok {
        tally.schedule_ok += 1;
    }

    let ok = action_ok && target_ok && schedule_ok;
    format!(
        "{} action={}({}) target={:?} sched={}",
        if ok { "PASS" } else { "FAIL" },
        if resolved.is_empty() { "-" } else { &resolved },
        if raw_verb.is_empty() { "-" } else { &raw_verb },
        target,
        if schedule_ok { "ok" } else { "BAD" },
    )
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "drives a live llama-server; run manually"]
fn enum_vs_freeform_ab() {
    let llama = std::env::var("CHASM_AB_LLAMA").unwrap_or_else(|_| "http://127.0.0.1:5001".into());
    let Ok(book_path) = std::env::var("CHASM_SNAP_TEST_BOOK") else {
        eprintln!("SKIP: CHASM_SNAP_TEST_BOOK not set");
        return;
    };
    let raw = std::fs::read_to_string(&book_path).expect("read action book");
    let book: Value =
        serde_json::from_str(raw.trim_start_matches('\u{feff}')).expect("parse action book");
    let entries = entries_from_book_json(&book);
    // Regular-NPC eligibility: the real book scopes EVERY entry (global +
    // admin + game:*) — "global" passes for any request, admin-only stays out.
    let requested: Vec<String> = Vec::new();
    let eligible: Vec<ActionEntry> = entries
        .iter()
        .filter(|e| !e.disable && action_passes_scopes(e, &requested))
        .cloned()
        .collect();
    assert!(
        !eligible.is_empty(),
        "no eligible entries — book/scope filter broken (an empty enum makes an          UNSATISFIABLE grammar and llama-server silently returns empty content)"
    );

    // Enum values (shared production builder) + resolution map.
    let enum_values = action_enum_values(&entries, &requested);
    assert!(!enum_values.is_empty(), "enum values empty");
    let mut by_alias: HashMap<String, String> = HashMap::new();
    for (id, alias) in &action_alias_pairs(&eligible) {
        by_alias.entry(slug_action_alias(alias)).or_insert_with(|| id.clone());
    }
    for (id, verb) in &action_verb_pairs(&eligible) {
        by_alias.entry(slug_action_alias(verb)).or_insert_with(|| id.clone());
    }
    eprintln!("enum has {} values over {} eligible entries", enum_values.len(), eligible.len());

    // Embedder (freeform mode's semantic fallback). Optional: without it the
    // freeform side only gets alias+verb resolution.
    let cfg = RetrieverConfig::default();
    let retriever;
    let cache;
    let ctx = if models_present(&cfg) {
        retriever = Retriever::load(&cfg).expect("load retriever");
        cache = EmbeddingCache::open(std::env::temp_dir().join("chasm-enum-ab-cache"))
            .expect("open scratch cache");
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
        eprintln!("WARN: embed models absent — freeform mode runs without the cosine snap");
        None
    };
    let resolver = Resolver {
        by_alias,
        ctx,
        candidates: action_embed_candidates(&eligible),
    };

    let system = format!(
        "You are Sunny Smiles, a friendly gunslinger and guide in Goodsprings. You are \
         talking to the player (the Courier). The current in-game time is 12:30PM. People \
         nearby: Easy Pete, Chet, Trudy.\n\n{NPC_STRUCTURED_OUTPUT_INSTRUCTION}"
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("http client");

    let out_path = std::env::var("CHASM_AB_OUT").ok();
    let mut out_lines: Vec<String> = Vec::new();

    let modes: [(&str, Value); 2] = [
        ("freeform", npc_schema(None)),
        ("enum", npc_schema(Some(&enum_values))),
    ];
    let mut tallies: HashMap<&str, Tally> = HashMap::new();

    for case in battery() {
        eprintln!("\n== {:?}", case.text);
        for (mode, format) in &modes {
            let tally = tallies.entry(mode).or_default();
            for sample in 0..SAMPLES {
                let body = json!({
                    "model": "gemma-4-12b",
                    "temperature": 0.7,
                    "max_tokens": 300,
                    "messages": [
                        { "role": "system", "content": system },
                        { "role": "user", "content": case.text }
                    ],
                    "response_format": format,
                });
                let started = Instant::now();
                let resp = client
                    .post(format!("{llama}/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .body(body.to_string())
                    .send()
                    .expect("llama-server request");
                let elapsed = started.elapsed().as_millis();
                let text = resp.text().expect("response body");
                let parsed: Value = serde_json::from_str(&text).expect("response JSON");
                let content = parsed["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                tally.latency_ms += elapsed;

                let turn: Value = serde_json::from_str(content.trim()).unwrap_or(Value::Null);
                let steps: Vec<(Option<String>, Value)> = turn
                    .get("actions")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .map(|s| {
                                let verb = s.get("action").and_then(Value::as_str).unwrap_or("");
                                (resolver.resolve(verb), s.clone())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let verdict = score(&case, &steps, tally);
                eprintln!("  [{mode}#{sample}] {verdict} | {}ms", elapsed);
                out_lines.push(
                    json!({
                        "case": case.text, "mode": mode, "sample": sample,
                        "latency_ms": elapsed, "content": content, "verdict": verdict,
                    })
                    .to_string(),
                );
            }
        }
    }

    eprintln!("\n================ SUMMARY ================");
    for (mode, t) in [("freeform", &tallies["freeform"]), ("enum", &tallies["enum"])] {
        eprintln!(
            "{mode:9}: action {}/{} | target {}/{} | schedule {}/{} | wrong {} | dropped {} | avg {}ms",
            t.action_ok, t.total, t.target_ok, t.total, t.schedule_ok, t.total,
            t.wrong_action, t.dropped_action,
            t.latency_ms / (t.total.max(1) as u128),
        );
    }
    if let Some(path) = out_path {
        std::fs::write(&path, out_lines.join("\n")).expect("write raw outputs");
        eprintln!("raw outputs -> {path}");
    }
}
