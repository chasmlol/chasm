//! RAG retrieval eval harness over the REAL Fallout: New Vegas books.
//!
//! Runs the actual embed + rerank pipeline (same models as production) against a
//! suite of scenarios and scores whether the right actions/lore surface. Tunable
//! without recompiling via env vars:
//!   MIN_SCORE   (default 0.2)   rerank score floor
//!   TOP_N       (default 10)    stage-1 recall count (the `candidates` setting)
//!   RICH        (default 0)     1 = fold vectorizableText + vectorSearchTexts into
//!                               the action embed text (the currently-unused fields)
//!
//! Run: cargo run -p chasm-embed --example rag_eval --release

use serde_json::Value;
use chasm_embed::{search, Retriever, RetrieverConfig};

const ACTION_BOOK: &str =
    "profiles/fallout-new-vegas/headless/action-books/Fallout New Vegas Action Book.json";
const LORE_BOOK: &str = "profiles/fallout-new-vegas/worlds/Fallout New Vegas.json";

struct Cand {
    id: String,
    text: String,
    keys: Vec<String>,
    constant: bool,
    vectorized: bool,
}

/// Production keyword activation: constant always, else any key as a
/// case-insensitive substring of the scan text (mirrors `key_matches` fallback).
fn keyword_active(c: &Cand, scan: &str) -> bool {
    if c.constant {
        return true;
    }
    let lower = scan.to_lowercase();
    c.keys.iter().any(|k| {
        let k = k.trim().to_lowercase();
        !k.is_empty() && lower.contains(&k)
    })
}

fn str_at(v: &Value, k: &str) -> String {
    v.get(k)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}
fn arr_at(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Action embed text by richness level:
///   0 = comment + content + keys (production today)
///   1 = + vectorizableText + vectorSearchTexts (full hand-written phrasings)
///   2 = + vectorizableText only (generic intent, NO name-laden phrasings that
///       cross-contaminate, e.g. "attack joe cobb" matching "tell me about joe cobb")
fn load_actions(rich: u32) -> Vec<Cand> {
    let txt = std::fs::read_to_string(ACTION_BOOK).expect("read action book");
    let root: Value = serde_json::from_str(&txt).expect("parse action book");
    let mut out = Vec::new();
    if let Some(entries) = root.get("entries").and_then(Value::as_object) {
        for entry in entries.values() {
            if entry
                .get("disable")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                continue;
            }
            let id = str_at(entry, "actionId");
            let mut parts: Vec<String> = Vec::new();
            let comment = str_at(entry, "comment");
            let content = str_at(entry, "content");
            let keys = arr_at(entry, "key");
            if !comment.is_empty() {
                parts.push(comment);
            }
            if !content.is_empty() {
                parts.push(content);
            }
            if !keys.is_empty() {
                parts.push(keys.join(", "));
            }
            if rich >= 1 {
                let vt = str_at(entry, "vectorizableText");
                if !vt.is_empty() {
                    parts.push(vt);
                }
            }
            if rich == 1 {
                let vst = arr_at(entry, "vectorSearchTexts");
                if !vst.is_empty() {
                    parts.push(vst.join("; "));
                }
            }
            out.push(Cand {
                id,
                text: parts.join("\n"),
                keys,
                constant: entry
                    .get("constant")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                vectorized: entry
                    .get("vectorized")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            });
        }
    }
    out
}

/// Lore embed text = comment + content (production `lore_vector_text`).
fn load_lore() -> Vec<Cand> {
    let txt = std::fs::read_to_string(LORE_BOOK).expect("read lore book");
    let root: Value = serde_json::from_str(&txt).expect("parse lore book");
    let mut out = Vec::new();
    if let Some(entries) = root.get("entries").and_then(Value::as_object) {
        for entry in entries.values() {
            if entry
                .get("disable")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                continue;
            }
            // Constants always inject in production (not via vector); skip from the
            // vector candidate pool so we see the semantic ranking of the rest.
            if entry
                .get("constant")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                continue;
            }
            let comment = str_at(entry, "comment");
            let content = str_at(entry, "content");
            let text = if comment.is_empty() {
                content
            } else {
                format!("{comment}\n{content}")
            };
            out.push(Cand {
                id: comment_or(entry),
                text,
                keys: arr_at(entry, "key"),
                constant: false,
                vectorized: true,
            });
        }
    }
    out
}
fn comment_or(entry: &Value) -> String {
    let c = str_at(entry, "comment");
    if c.is_empty() {
        arr_at(entry, "key").first().cloned().unwrap_or_default()
    } else {
        c
    }
}

fn embed_candidates(
    retriever: &Retriever,
    cands: &[Cand],
    vectorized_only: bool,
) -> Vec<(String, Vec<f32>, String)> {
    let filtered: Vec<&Cand> = cands
        .iter()
        .filter(|c| !vectorized_only || c.vectorized)
        .collect();
    let texts: Vec<&str> = filtered.iter().map(|c| c.text.as_str()).collect();
    let vecs = retriever.embed_batch(&texts).expect("embed candidates");
    filtered
        .iter()
        .zip(vecs)
        .map(|(c, v)| (c.id.clone(), v, c.text.clone()))
        .collect()
}

struct Scenario {
    name: &'static str,
    text: &'static str,
    gamestate: &'static str,
    /// action_ids that SHOULD surface (above min_score).
    want: &'static [&'static str],
    /// action_ids that should NOT surface (above min_score).
    avoid: &'static [&'static str],
    /// lore titles (comment substrings) that should surface.
    lore: &'static [&'static str],
}

fn sc(
    name: &'static str,
    text: &'static str,
    gamestate: &'static str,
    want: &'static [&'static str],
    avoid: &'static [&'static str],
    lore: &'static [&'static str],
) -> Scenario {
    Scenario {
        name,
        text,
        gamestate,
        want,
        avoid,
        lore,
    }
}

fn scenarios() -> Vec<Scenario> {
    let day = "Location: Goodsprings, near the Prospector Saloon. Time: 14:00 daytime.";
    let night = "Location: Goodsprings, near the Prospector Saloon. Time: 23:30 night.";
    let saloon = "Location: inside the Prospector Saloon, Goodsprings. Time: 18:00.";
    let gas = "Location: Goodsprings gas station. Time: 12:00.";
    let wild = "Location: Mojave wasteland, open desert outside Goodsprings. Time: 10:00.";
    vec![
        // --- Combat / threats -> should surface combat.start AND fear gestures ---
        sc("threat: shoot you", "I'm gonna shoot you", day, &["combat.start"], &["ai.sit_down", "npc.gesture_wave"], &[]),
        sc("threat: kill you", "I'll kill you where you stand", day, &["combat.start"], &[], &[]),
        sc("threat: youre dead", "you're dead, pal", day, &["combat.start"], &[], &[]),
        sc("threat: draw weapon", "draw your weapon, this ends now", day, &["combat.start"], &[], &[]),
        sc("threat: scared npc fear", "I'm gonna shoot you", day, &["npc.gesture_cower", "npc.gesture_hands_up", "npc.gesture_panic", "npc.gesture_ask_help"], &[], &[]),
        sc("surrender", "okay okay, I give up, please don't shoot", day, &["combat.stop", "npc.gesture_hands_up"], &[], &[]),
        sc("stop combat", "stand down, cease fire", day, &["combat.stop"], &[], &[]),
        sc("calm down", "hey, calm down, let's not fight", day, &["combat.stop"], &["combat.start"], &[]),
        // --- Movement: follow / stop follow / wait ---
        sc("follow direct", "follow me", day, &["movement.follow_target"], &[], &[]),
        sc("follow indirect", "come on, stick close to me", day, &["movement.follow_target"], &[], &[]),
        sc("follow indirect2", "I need you with me, let's go", day, &["movement.follow_target"], &[], &[]),
        sc("stop follow", "stop following me", day, &["movement.stop_follow_target"], &[], &[]),
        sc("dismiss", "you can go now, I don't need you", day, &["movement.stop_follow_target"], &[], &[]),
        sc("wait here", "wait here for me", day, &["ai.wait_here"], &[], &[]),
        sc("stay put", "stay put, don't move", day, &["ai.wait_here"], &[], &[]),
        // --- Idle / sit / sandbox ---
        sc("sit down", "take a seat", saloon, &["ai.sit_down"], &[], &[]),
        sc("sit indirect", "rest your legs for a bit", saloon, &["ai.sit_down"], &[], &[]),
        sc("relax", "just relax and do your own thing", day, &["ai.sandbox_here"], &[], &[]),
        sc("resume", "go back to normal, forget that", day, &["ai.resume_default"], &[], &[]),
        // --- Gestures ---
        sc("wave", "give them a wave", day, &["npc.gesture_wave"], &[], &[]),
        sc("greet nod", "give a little nod hello", day, &["npc.gesture_greet_head_nod", "npc.gesture_wave"], &[], &[]),
        sc("point", "point over there", day, &["npc.gesture_point"], &[], &[]),
        sc("shrug", "I dunno, shrug it off", day, &["npc.gesture_shrug"], &[], &[]),
        sc("laugh", "haha that's hilarious", day, &["npc.gesture_laugh"], &[], &[]),
        sc("cry", "this is so sad, I could cry", day, &["npc.gesture_cry"], &[], &[]),
        sc("cheer", "yeah! we did it!", day, &["npc.gesture_cheer", "npc.gesture_clap"], &[], &[]),
        sc("rude", "give them the finger", day, &["npc.gesture_middle_finger"], &[], &[]),
        sc("shush", "quiet, be silent", day, &["npc.gesture_shush"], &[], &[]),
        sc("smoke", "light up a cigarette and chill", day, &["npc.gesture_smoke"], &[], &[]),
        // --- Spawn (admin) ---
        sc("spawn item gun", "give me a 9mm pistol", day, &["world.spawn_item"], &["world.spawn_entity"], &[]),
        sc("spawn item caps", "spawn me some caps", day, &["world.spawn_item"], &[], &[]),
        sc("spawn item stim", "I need a stimpak", day, &["world.spawn_item"], &[], &[]),
        sc("spawn entity gecko", "spawn a gecko right here", day, &["world.spawn_entity"], &["world.spawn_item"], &[]),
        sc("spawn entity deathclaw", "summon a deathclaw", day, &["world.spawn_entity"], &[], &[]),
        // --- Neutral chat -> should surface FEW/NO actions ---
        sc("greeting", "hello there, how are you?", day, &[], &["combat.start", "world.spawn_item", "world.spawn_entity"], &[]),
        sc("smalltalk weather", "sure is a hot one today", day, &[], &["combat.start", "world.spawn_entity"], &[]),
        sc("smalltalk name", "what's your name, stranger?", day, &[], &["combat.start", "world.spawn_item"], &[]),
        sc("compliment", "I really like your hat", day, &[], &["combat.start", "world.spawn_entity"], &[]),
        // --- Lore queries (informational; should surface lore, NOT actions) ---
        sc("lore joe cobb", "tell me about Joe Cobb", day, &[], &["combat.start", "world.spawn_entity"], &["Joe Cobb and the Powder Ganger threat"]),
        sc("lore ringo", "who is Ringo?", gas, &[], &["combat.start"], &["Ringo at the gas station"]),
        sc("lore powder gangers", "what's the deal with the powder gangers?", day, &[], &["combat.start"], &["Joe Cobb and the Powder Ganger threat"]),
        sc("lore chet", "tell me about the general store", day, &[], &[], &["Chet and the General Store"]),
        sc("lore saloon", "who runs this saloon?", saloon, &[], &[], &["Prospector Saloon"]),
        sc("lore gunfight", "what happened in the gunfight?", day, &[], &[], &["Ghost Town Gunfight"]),
        sc("lore ncr", "what is the NCR?", day, &[], &[], &["NCR basics"]),
        sc("lore easy pete", "tell me about Easy Pete", day, &[], &[], &["Easy Pete"]),
        // --- Misspellings ---
        sc("misspell shoot", "im gona shoot u", day, &["combat.start"], &[], &[]),
        sc("misspell follow", "folow me pls", day, &["movement.follow_target"], &[], &[]),
        sc("misspell spawn", "spwan a gun for me", day, &["world.spawn_item"], &[], &[]),
        sc("misspell joecobb", "tell me about joe cob", day, &[], &["combat.start"], &["Joe Cobb and the Powder Ganger threat"]),
        sc("misspell ringo", "wheres ringoo", gas, &[], &[], &["Ringo at the gas station"]),
        sc("misspell sit", "sit dwn over ther", saloon, &["ai.sit_down"], &[], &[]),
        sc("misspell wait", "weight here", day, &["ai.wait_here"], &[], &[]),
        // --- Gamestate variations (same/empty query, different place/time) ---
        sc("ambient saloon", "what's going on around here?", saloon, &[], &[], &["Prospector Saloon"]),
        sc("ambient gas", "what's going on around here?", gas, &[], &[], &["Ringo at the gas station"]),
        sc("ambient wild night", "it's dark out here", &wild, &[], &[], &[]),
        sc("ambient night", "anything happening tonight?", night, &[], &[], &[]),
        // --- Edge cases ---
        sc("gibberish", "asdfgh qwerty zxcvb", day, &[], &["combat.start", "world.spawn_item", "world.spawn_entity"], &[]),
        sc("empty-ish", "...", day, &[], &["combat.start", "world.spawn_entity"], &[]),
        sc("rambling", "so anyway I was walking and thinking about the weather and my brahmin and how the road's been dusty lately and whether trudy restocked", day, &[], &["combat.start", "world.spawn_entity"], &[]),
    ]
}

fn run_search(
    retriever: &Retriever,
    query: &str,
    cands: &[(String, Vec<f32>, String)],
    top_n: usize,
    min_score: f32,
) -> Vec<(String, f32)> {
    search(retriever, query, cands, top_n, 10, min_score).unwrap_or_default()
}

fn main() {
    let min_score: f32 = std::env::var("MIN_SCORE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.2);
    let top_n: usize = std::env::var("TOP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let rich: u32 = std::env::var("RICH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let no_gamestate = std::env::var("NO_GAMESTATE").ok().as_deref() == Some("1");
    // Optional separate floor for actions (they score lower than lore passages).
    let action_min: f32 = std::env::var("ACTION_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(min_score);

    eprintln!("loading retriever (reranker on, cpu)...");
    let cfg = RetrieverConfig {
        embedder_tier: "small".into(),
        reranker_enabled: true,
        reranker_tier: "small".into(),
        execution: "cpu".into(),
    };
    let retriever = Retriever::load(&cfg).expect("load retriever");

    let actions = load_actions(rich);
    let lore = load_lore();
    eprintln!(
        "actions={} lore={}  MIN_SCORE={min_score} TOP_N={top_n} RICH={}",
        actions.len(),
        lore.len(),
        rich
    );
    let action_cands = embed_candidates(&retriever, &actions, true);
    let lore_cands = embed_candidates(&retriever, &lore, false);

    let scenarios = scenarios();
    let mut pass = 0usize;
    let mut fails: Vec<String> = Vec::new();

    println!("\n================ RAG EVAL (min_score={min_score}, top_n={top_n}, rich={rich}) ================");
    for s in &scenarios {
        // Keyword scans the full turn (message + gamestate); vector uses message
        // only (gamestate dilutes recall) when NO_GAMESTATE is set.
        let kw_scan = if s.gamestate.is_empty() {
            s.text.to_string()
        } else {
            format!("{}\n{}", s.text, s.gamestate)
        };
        let vquery = if no_gamestate {
            s.text.to_string()
        } else {
            kw_scan.clone()
        };
        let a_vhits = run_search(&retriever, &vquery, &action_cands, top_n, action_min);
        let l_vhits = run_search(&retriever, &vquery, &lore_cands, top_n, min_score);
        let a_kw: Vec<String> = actions
            .iter()
            .filter(|c| keyword_active(c, &kw_scan))
            .map(|c| c.id.clone())
            .collect();
        let l_kw: Vec<String> = lore
            .iter()
            .filter(|c| keyword_active(c, &kw_scan))
            .map(|c| c.id.clone())
            .collect();

        // Production injects keyword OR vector hits.
        let a_has = |id: &str| a_kw.iter().any(|k| k == id) || a_vhits.iter().any(|(h, _)| h == id);
        let l_has = |needle: &str| {
            l_kw.iter().any(|k| k.contains(needle))
                || l_vhits.iter().any(|(h, _)| h.contains(needle))
        };

        let mut problems: Vec<String> = Vec::new();
        for w in s.want {
            if !a_has(w) {
                problems.push(format!("MISSING {w}"));
            }
        }
        for a in s.avoid {
            if a_has(a) {
                let via = if a_kw.iter().any(|k| k == a) {
                    "kw".to_string()
                } else {
                    format!(
                        "v{:.2}",
                        a_vhits
                            .iter()
                            .find(|(h, _)| h == a)
                            .map(|(_, s)| *s)
                            .unwrap_or(0.0)
                    )
                };
                problems.push(format!("UNWANTED {a}@{via}"));
            }
        }
        for l in s.lore {
            if !l_has(l) {
                problems.push(format!("MISSING lore '{l}'"));
            }
        }

        let status = if problems.is_empty() { "PASS" } else { "FAIL" };
        if problems.is_empty() {
            pass += 1;
        } else {
            fails.push(format!("{}: {}", s.name, problems.join(", ")));
        }
        let fmt = |hits: &[(String, f32)]| {
            hits.iter()
                .take(6)
                .map(|(id, sc)| format!("{id}:{sc:.2}"))
                .collect::<Vec<_>>()
                .join("  ")
        };
        println!("\n[{status}] {}  | \"{}\"", s.name, s.text);
        println!("   act kw:  {}", a_kw.join(" "));
        println!("   act vec: {}", fmt(&a_vhits));
        println!("   lore kw: {}", l_kw.join(" "));
        println!("   lore vec:{}", fmt(&l_vhits));
        if !problems.is_empty() {
            println!("   -> {}", problems.join(", "));
        }
    }

    println!(
        "\n================ SUMMARY: {pass}/{} passed ================",
        scenarios.len()
    );
    if !fails.is_empty() {
        println!("FAILURES:");
        for f in &fails {
            println!("  - {f}");
        }
    }
}
