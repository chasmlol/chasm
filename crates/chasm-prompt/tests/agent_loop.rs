//! Manual (ignored) live-Gemma harness for the NPC AGENT LOOP protocol:
//! query actions return results and the model continues speaking (rounds),
//! speech is optional (silent action rounds), turn ends when a round emits no
//! query. The harness replicates chasm's loop client-side with FAKED query
//! results, so the instruction + protocol can be tuned before the chasm wiring
//! ever runs — same philosophy as enum_ab.rs.
//!
//! Run:
//!   CHASM_AB_LLAMA=http://127.0.0.1:5001 \
//!   CHASM_EMBED_DIR=<hf models dir> \
//!   CHASM_SNAP_TEST_BOOK=<path to "Fallout New Vegas Action Book.json"> \
//!   cargo test -p chasm-prompt --test agent_loop --release -- --ignored --nocapture

use chasm_embed::{models_present, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_prompt::{
    action_alias_pairs, action_embed_candidates, action_enum_values, action_passes_scopes,
    action_verb_pairs, npc_actions_instruction, resolve_guess_to_action, slug_action_alias,
    RetrievalCtx, NPC_STRUCTURED_OUTPUT_INSTRUCTION,
};
use chasm_st_compat::ActionEntry;
use serde_json::{json, Value};
use std::collections::HashMap;

const FLOOR: f32 = 0.5;
const SAMPLES: usize = 3;
const MAX_ROUNDS: usize = 3;

// ---------------------------------------------------------------------------
// Book loading (same raw-JSON mirror as enum_ab.rs)
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

// ---------------------------------------------------------------------------
// Faked query results (chasm/mod would produce these in production)
// ---------------------------------------------------------------------------

fn fake_query_result(action_id: &str) -> Option<&'static str> {
    match action_id {
        "chasm.who_is_here" => Some(
            "Nearby: Easy Pete (7m), Sunny Smiles (4m), Chet (12m), Cheyenne the dog (5m).",
        ),
        "chasm.recall" => Some(
            "(your memory) Three days ago a pack of geckos attacked the water tower; Sunny killed two of them and one bit Chet's leg before Pete clubbed it.",
        ),
        "chasm.check_inventory" => Some(
            "Your pack contains: 9mm pistol, 23 rounds of 9mm, 2 stimpaks, a bottle of dirty water, 47 caps.",
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

struct Case {
    text: &'static str,
    /// Max rounds this case should take (1 = plain reply).
    want_rounds: usize,
    /// Query id that MUST fire in round 1 ("" = no query allowed).
    want_query: &'static str,
    /// Words the FINAL round's speech must contain one of ("" = don't care).
    want_answer_any: &'static [&'static str],
    /// Words round 1 speech must NOT contain (guessing guard).
    forbid_round1: &'static [&'static str],
}

fn battery() -> Vec<Case> {
    vec![
        Case { text: "hi", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        Case { text: "how are you today?", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        Case { text: "wait here for a moment", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        Case {
            text: "who else is around us right now?",
            want_rounds: 2,
            want_query: "chasm.who_is_here",
            want_answer_any: &["sunny", "pete", "chet", "cheyenne"],
            forbid_round1: &["sunny", "pete", "chet", "cheyenne"],
        },
        Case {
            text: "what do you remember about the gecko attack?",
            want_rounds: 2,
            want_query: "chasm.recall",
            want_answer_any: &["gecko", "water tower", "chet", "sunny"],
            forbid_round1: &["water tower"],
        },
        Case {
            text: "what's in your pack right now?",
            want_rounds: 2,
            want_query: "chasm.check_inventory",
            want_answer_any: &["stimpak", "pistol", "caps", "9mm"],
            forbid_round1: &["stimpak", "pistol", "caps", "9mm"],
        },
        Case {
            text: "list everything you're carrying",
            want_rounds: 2,
            want_query: "chasm.check_inventory",
            want_answer_any: &["stimpak", "pistol", "caps", "9mm"],
            forbid_round1: &["stimpak", "pistol", "caps", "9mm"],
        },
        // Opinion/small talk that MENTIONS query-adjacent topics must not query.
        Case { text: "do you like geckos?", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        // Clear opinion/praise - nothing to look up.
        Case { text: "you're a damn good shot, sunny", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        // Silent action turn: action must be present; speech empty is welcome.
        Case { text: "wait here and don't say a word", want_rounds: 1, want_query: "", want_answer_any: &[], forbid_round1: &[] },
        // A question needing a query even though phrased indirectly.
        Case {
            text: "got any stimpaks on you?",
            want_rounds: 2,
            want_query: "chasm.check_inventory",
            // "I've got two of 'em right here" is a correct answer that never
            // repeats the word - accept the count as evidence.
            want_answer_any: &["stimpak", "two", "couple", "2 "],
            forbid_round1: &[],
        },
    ]
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "drives a live llama-server; run manually"]
fn agent_loop_protocol() {
    let llama = std::env::var("CHASM_AB_LLAMA").unwrap_or_else(|_| "http://127.0.0.1:5001".into());
    let Ok(book_path) = std::env::var("CHASM_SNAP_TEST_BOOK") else {
        eprintln!("SKIP: CHASM_SNAP_TEST_BOOK not set");
        return;
    };
    let raw = std::fs::read_to_string(&book_path).expect("read action book");
    let book: Value =
        serde_json::from_str(raw.trim_start_matches('\u{feff}')).expect("parse action book");
    let entries = entries_from_book_json(&book);
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
    eprintln!("queries in book: {query_ids:?}; enum {} values", enum_values.len());

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
        cache = EmbeddingCache::open(std::env::temp_dir().join("chasm-agent-loop-cache"))
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

    let query_section = npc_actions_instruction(&eligible, &requested);
    eprintln!("--- query section ---\n{query_section}---------------------");
    let system = format!(
        "You are Sunny Smiles, a friendly gunslinger and guide in Goodsprings, talking to \
         the player (the Courier). The current in-game time is 12:30PM.\n\n{NPC_STRUCTURED_OUTPUT_INSTRUCTION}{query_section}"
    );
    let schema = npc_schema(&enum_values);
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
            let mut rounds: Vec<(String, Vec<String>)> = Vec::new(); // (speech, resolved action ids)
            let mut fail: Option<String> = None;
            for round in 0..MAX_ROUNDS {
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
                let ids: Vec<String> = turn
                    .get("actions")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|s| s.get("action").and_then(Value::as_str))
                            .filter_map(|v| resolve(v))
                            .collect()
                    })
                    .unwrap_or_default();
                let queries: Vec<String> =
                    ids.iter().filter(|id| query_ids.contains(*id)).cloned().collect();
                eprintln!("  [{sample}#r{round}] speech={:?} actions={:?}", truncate(&speech, 70), ids);
                out_lines.push(json!({
                    "case": case.text, "sample": sample, "round": round,
                    "content": content,
                }).to_string());
                rounds.push((speech, ids.clone()));
                if queries.is_empty() {
                    break;
                }
                let mut results = String::new();
                for q in &queries {
                    if let Some(r) = fake_query_result(q) {
                        results.push_str(r);
                        results.push('\n');
                    }
                }
                messages.push(json!({ "role": "assistant", "content": content }));
                messages.push(json!({ "role": "user", "content": format!(
                    "[QUERY RESULT]\n{results}[This result is complete and current. Continue the same reply: if the order needs an action from the result, emit it now with the EXACT names shown; answer questions naturally. Speaking is optional - stay silent when simply proceeding. NEVER repeat anything you already said this turn; only say what is new. Do not repeat a query.]"
                )}));
            }

            // Scoring.
            let r1 = &rounds[0];
            let final_speech = rounds.last().map(|r| r.0.to_lowercase()).unwrap_or_default();
            let r1_speech = r1.0.to_lowercase();
            let r1_query = r1.1.iter().find(|id| query_ids.contains(*id)).cloned().unwrap_or_default();
            if case.want_query.is_empty() {
                if !r1_query.is_empty() {
                    fail = Some(format!("unwanted query {r1_query}"));
                }
                if rounds.len() != 1 {
                    fail = Some(format!("expected 1 round, got {}", rounds.len()));
                }
            } else {
                if r1_query != case.want_query {
                    fail = Some(format!("expected query {} in round 1, got {:?}", case.want_query, r1.1));
                }
                if rounds.len() < 2 {
                    fail = fail.or(Some("never continued to round 2".into()));
                }
                for word in case.forbid_round1 {
                    if r1_speech.contains(word) {
                        fail = Some(format!("round 1 GUESSED forbidden word {word:?}"));
                    }
                }
                if !case.want_answer_any.is_empty()
                    && !case.want_answer_any.iter().any(|w| final_speech.contains(w))
                {
                    fail = Some(format!("final speech missing answer words: {:?}", truncate(&final_speech, 80)));
                }
                if rounds.len() > case.want_rounds {
                    fail = fail.or(Some(format!("took {} rounds (max {})", rounds.len(), case.want_rounds)));
                }
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
        "agent loop battery too flaky ({} failures):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}...", &s[..s.floor_char_boundary(n)]) }
}
