//! Manual (ignored) probe: what does `find_action` actually rank for a given
//! query, on the REAL book + REAL embedder? Prints the full cosine/relevance
//! ranking plus what `search_actions_semantic` returns (floor+gap+top_k), so the
//! "pick up the pistol didn't expose search_area" report can be tuned from real
//! numbers instead of guesses. CPU execution on purpose — must not contend with
//! the live game's GPU.
//!
//! Run:
//!   CHASM_EMBED_DIR=<hf models dir> \
//!   CHASM_SNAP_TEST_BOOK=<path to "Fallout New Vegas Action Book.json"> \
//!   cargo test -p chasm-prompt --test find_action_rank --release -- --ignored --nocapture

use chasm_embed::{cosine_similarity, models_present, EmbeddingCache, Retriever, RetrieverConfig};
use chasm_prompt::{
    action_alias_pairs, action_embed_candidates, search_actions_semantic, RetrievalCtx,
    FIND_ACTION_ID,
};
use chasm_st_compat::ActionEntry;
use serde_json::Value;
use std::collections::HashMap;

// Mirror generate.rs.
const FLOOR: f32 = 0.15;
const GAP: f32 = 0.22;
const TOP_K: usize = 6;

fn entries_from_book_json(book: &Value) -> Vec<ActionEntry> {
    let str_vec = |e: &Value, k: &str| {
        e.get(k)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default()
    };
    let str_of = |e: &Value, k: &str| e.get(k).and_then(Value::as_str).unwrap_or("").to_string();
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
        .filter(|entry| !entry.action_id.is_empty() && !entry.disable)
        .collect()
}

#[test]
#[ignore = "needs real embed models + CHASM_SNAP_TEST_BOOK; run manually"]
fn find_action_ranking_probe() {
    let Ok(book_path) = std::env::var("CHASM_SNAP_TEST_BOOK") else {
        eprintln!("SKIP: CHASM_SNAP_TEST_BOOK not set");
        return;
    };
    // CPU on purpose: the live game owns the GPU.
    let cfg = RetrieverConfig {
        embedder_tier: "small".to_string(),
        reranker_enabled: false,
        reranker_tier: "small".to_string(),
        execution: "cpu".to_string(),
    };
    if !models_present(&cfg) {
        eprintln!("SKIP: embed models not present (set CHASM_EMBED_DIR)");
        return;
    }
    let raw = std::fs::read_to_string(&book_path).expect("read book");
    let book: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}')).expect("parse book");
    let entries = entries_from_book_json(&book);
    let alias_by_id: HashMap<String, String> =
        action_alias_pairs(&entries).into_iter().map(|(id, a)| (id, a)).collect();
    // The pool find_action searches: everything except itself (mirrors agent_find_action).
    let pool: Vec<(String, String)> = action_embed_candidates(&entries)
        .into_iter()
        .filter(|(id, _)| id != FIND_ACTION_ID)
        .collect();

    let retriever = Retriever::load(&cfg).expect("load retriever (cpu)");
    let cache = EmbeddingCache::open(std::env::temp_dir().join("chasm-find-rank-cache"))
        .expect("open cache");
    let ctx = RetrievalCtx {
        retriever: &retriever,
        cache: &cache,
        chat_memory_enabled: false,
        lore_semantic_enabled: false,
        action_semantic_enabled: true,
        quest_semantic_enabled: false,
        candidates: 64,
        top_k: TOP_K,
        min_score: 0.0,
        action_min_score: FLOOR,
        chat_memory_limit: 0,
        lore_limit: 0,
        quest_limit: 0,
    };

    let alias = |id: &str| alias_by_id.get(id).cloned().unwrap_or_else(|| id.to_string());
    let remap = |cos: f32| ((cos - 0.45) / 0.35).clamp(0.0, 1.0);

    let queries = [
        "pick up the pistol",
        "the pistol on the ground",
        "do you see that pistol on the ground",
        "pick up the hammer",
        "loot the fridge",
        "search the room for loot",
        "do some pushups",
        "sit down",
        "attack the raider",
        "shoot that guy",
        "kill easy pete",
    ];

    for q in queries {
        // Full manual ranking (raw cosine + relevance) so we SEE where each lands.
        let qv = cache.get_or_embed(&retriever, &retriever.query_text(q)).expect("embed q");
        let mut scored: Vec<(String, f32, f32)> = pool
            .iter()
            .map(|(id, text)| {
                let cv = cache.get_or_embed(&retriever, text).unwrap_or_default();
                let cos = cosine_similarity(&qv, &cv);
                (alias(id), cos, remap(cos))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = scored.first().map(|r| r.2).unwrap_or(0.0);

        // What production would actually expose.
        let returned = search_actions_semantic(&ctx, q, &pool, TOP_K, FLOOR, GAP);
        let returned_aliases: Vec<String> = returned.iter().map(|id| alias(id)).collect();

        println!("\n=== {q:?}");
        println!("   RETURNS: {returned_aliases:?}");
        println!("   {:<20} {:>7} {:>7}  {}", "alias", "cos", "relev", "kept?");
        for (al, cos, rel) in scored.iter().take(12) {
            let kept = if *rel >= FLOOR && *rel + GAP >= best { "KEEP" } else { "cut" };
            println!("   {al:<20} {cos:>7.3} {rel:>7.3}  {kept}");
        }
    }
}
