//! Manual (ignored) integration check for the verb->action resolution chain on
//! the REAL action book + REAL embedding model: exact alias, then the book's
//! deterministic `verbs` lexicon, then the cosine guess snap (the production
//! order in `normalize_structured_action_aliases`). "kill <name> for me" was
//! the flagship phrase with no offline coverage — this pins every layer.
//!
//! Findings that shaped this (2026-07-05): the original reranker-scored snap
//! put EVERY kill phrasing below the 0.5 floor ("attack" scored 0.25 against
//! the combat entry containing the word "attack") — cross-encoder relevance is
//! the wrong scale for terse verbs. Raw cosine separates bare verbs cleanly
//! ("kill" 0.63 vs junk ~0.45-0.52) but cross-matches idioms ("take him out"
//! -> take-item gesture 0.64), hence the deterministic `verbs` layer.
//!
//! Skips (passes with a note) when the book path or models are absent. Run:
//!   CHASM_EMBED_DIR=<hf models dir> \
//!   CHASM_SNAP_TEST_BOOK=<path to "Fallout New Vegas Action Book.json"> \
//!   cargo test -p chasm-prompt --test guess_snap --release -- --ignored --nocapture

use chasm_embed::{models_present, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_prompt::{
    action_alias_pairs, action_embed_candidates, action_verb_pairs, resolve_guess_to_action,
    slug_action_alias, RetrievalCtx,
};
use chasm_st_compat::ActionEntry;
use serde_json::Value;
use std::collections::HashMap;

/// Production floor from `generate.rs` (`GUESS_ACTION_FLOOR`).
const FLOOR: f32 = 0.5;

/// Build [`ActionEntry`]s from the raw book JSON, mirroring `action_from_raw`
/// for the fields resolution uses (keys/title/description/vectorizable_text
/// feed the embed text; verbs feed the deterministic layer). Everything else
/// is inert here.
fn entries_from_book_json(book: &Value) -> Vec<ActionEntry> {
    let str_vec = |e: &Value, k: &str| -> Vec<String> {
        e.get(k)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    let str_of = |e: &Value, k: &str| -> String {
        e.get(k).and_then(Value::as_str).unwrap_or("").to_string()
    };
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
            scopes: Vec::new(),
        })
        .filter(|entry| !entry.action_id.is_empty())
        .collect()
}

#[test]
#[ignore = "needs real embed models + CHASM_SNAP_TEST_BOOK; run manually"]
fn kill_phrasings_resolve_to_combat_start() {
    let Ok(book_path) = std::env::var("CHASM_SNAP_TEST_BOOK") else {
        eprintln!("SKIP: CHASM_SNAP_TEST_BOOK not set");
        return;
    };
    let cfg = RetrieverConfig::default();
    if !models_present(&cfg) {
        eprintln!("SKIP: embed models not present (set CHASM_EMBED_DIR)");
        return;
    }
    let raw = std::fs::read_to_string(&book_path).expect("read action book");
    let book: Value =
        serde_json::from_str(raw.trim_start_matches('\u{feff}')).expect("parse action book");
    let entries = entries_from_book_json(&book);
    assert!(
        entries.iter().any(|e| e.action_id == "combat.start"),
        "book at {book_path} has no combat.start entry"
    );
    let candidates = action_embed_candidates(&entries);

    let retriever = Retriever::load(&cfg).expect("load retriever");
    let cache_dir = std::env::temp_dir().join("chasm-guess-snap-test-cache");
    let cache = EmbeddingCache::open(&cache_dir).expect("open scratch cache");
    let ctx = RetrievalCtx {
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
    };

    // The production resolution order (normalize_structured_action_aliases):
    // whole-phrase slug against aliases+verbs, per-word slug, then cosine snap.
    let mut by_alias: HashMap<String, String> = HashMap::new();
    for (id, alias) in &action_alias_pairs(&entries) {
        by_alias.insert(slug_action_alias(alias), id.clone());
    }
    for (id, verb) in &action_verb_pairs(&entries) {
        by_alias
            .entry(slug_action_alias(verb))
            .or_insert_with(|| id.clone());
    }
    let resolve = |verb: &str| -> Option<String> {
        if let Some(id) = by_alias.get(&slug_action_alias(verb)) {
            return Some(id.clone());
        }
        for word in verb.split_whitespace() {
            if let Some(id) = by_alias.get(&slug_action_alias(word)) {
                return Some(id.clone());
            }
        }
        resolve_guess_to_action(&ctx, verb, &candidates, FLOOR)
    };

    let mut failures = Vec::new();

    // Every phrasing a model plausibly emits for "kill <name> for me".
    let must_resolve = [
        "kill", "kill easy pete", "attack", "murder", "shoot him", "take him out",
        "waste him", "fight", "finish him off", "execute",
    ];
    for guess in must_resolve {
        let resolved = resolve(guess);
        eprintln!("composite {guess:?} -> {resolved:?}");
        if resolved.as_deref() != Some("combat.start") {
            failures.push(format!("{guess:?} -> {resolved:?} (want combat.start)"));
        }
    }

    // The cosine snap alone must carry the bare-verb cases (verbs data absent).
    for (guess, want) in [
        ("kill", "combat.start"),
        ("shoot him", "combat.start"),
        ("sit down", "ai.sit_down"),
        ("follow me", "movement.follow_target"),
        ("wave", "npc.gesture_wave"),
        ("stop fighting", "combat.stop"),
    ] {
        let resolved = resolve_guess_to_action(&ctx, guess, &candidates, FLOOR);
        eprintln!("embedder {guess:?} -> {resolved:?}");
        if resolved.as_deref() != Some(want) {
            failures.push(format!("embedder {guess:?} -> {resolved:?} (want {want})"));
        }
    }

    // Ordinary chatter must NOT resolve to combat.
    for guess in ["wave", "sit down", "follow me", "stop fighting", "laugh"] {
        let resolved = resolve(guess);
        if resolved.as_deref() == Some("combat.start") {
            failures.push(format!("control {guess:?} wrongly resolved to combat.start"));
        }
    }

    assert!(
        failures.is_empty(),
        "resolution failures:\n{}",
        failures.join("\n")
    );
}
