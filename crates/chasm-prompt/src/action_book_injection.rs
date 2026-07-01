//! Action Book selection, limiting, and formatting over the full-fidelity
//! [`ResolvedAction`] reader, ported from `formatActionBookPrompt` and the
//! limit logic in `src/headless/generation.js`.
//!
//! This is additive: `lib.rs` keeps its existing slim-`ActionEntry` formatter
//! and `assemble_prompt` signature unchanged. This module is what the read-only
//! viewer (and any future generation path) uses when it has the richer
//! [`ResolvedAction`] entries with risk tiers, scoped catalogs, scopes, etc.

use serde_json::Value;
use chasm_st_compat::{ActionBookDetail, ResolvedAction, ScopedCatalogConfig};

use crate::{key_matches, slug_action_alias};

/// Default `actionBookLimit` when the request supplies none (generation.js line
/// 885: `... > 0 ? Math.min(actionBookLimit, 40) : 10`).
pub const DEFAULT_ACTION_BOOK_LIMIT: usize = 10;
/// Hard cap on the number of injected entries (`Math.min(limit, 40)`).
pub const MAX_ACTION_BOOK_LIMIT: usize = 40;
/// The admin/helper default (`adminActionBookLimit` in nvbridge-helper.mjs).
pub const ADMIN_ACTION_BOOK_LIMIT: usize = 12;

/// Header prepended to the formatted body before injection (generation.js ~1011).
pub const ACTION_BOOK_HEADER: &str = "Activated Action Book entries:";

/// Clamps a requested limit the same way `prepareGenerationRun` does:
/// a positive integer is capped at [`MAX_ACTION_BOOK_LIMIT`], otherwise the
/// default of [`DEFAULT_ACTION_BOOK_LIMIT`] is used.
pub fn resolve_action_book_limit(requested: Option<usize>) -> usize {
    match requested {
        Some(limit) if limit > 0 => limit.min(MAX_ACTION_BOOK_LIMIT),
        _ => DEFAULT_ACTION_BOOK_LIMIT,
    }
}

/// Constant-or-keyword activation for a single resolved action against the scan
/// text. Disabled entries never activate; constant entries always do; otherwise
/// any primary key matching counts (case-insensitive unless `caseSensitive`).
pub fn action_is_active(action: &ResolvedAction, scan_text: &str) -> bool {
    if action.disable {
        return false;
    }
    if action.constant {
        return true;
    }
    let case_sensitive = action.case_sensitive.unwrap_or(false);
    action
        .keys
        .iter()
        .any(|key| key_matches(key, scan_text, case_sensitive))
}

/// Derives the structured action alias for a resolved entry — the short string
/// the model is told to emit. Mirrors `getStructuredActionAlias` /
/// `slugActionAlias`: known ids map to friendly aliases, `npc.gesture_*` keeps
/// its suffix, everything else slugifies the last dotted segment.
pub fn resolved_action_alias(action: &ResolvedAction) -> String {
    let action_id = action.action_id.trim();
    match action_id {
        "npc.gesture_wave" => return "wave".to_string(),
        "movement.follow_target" => return "follow".to_string(),
        "movement.stop_follow_target" | "movement.stop_following" | "movement.stop_follow" => {
            return "stop_follow".to_string()
        }
        "combat.start" => return "attack".to_string(),
        "combat.stop" => return "stop_combat".to_string(),
        "ai.wait_here" => return "wait".to_string(),
        "ai.sandbox_here" => return "sandbox".to_string(),
        "ai.resume_default" => return "resume".to_string(),
        "ai.sit_down" => return "sit".to_string(),
        "world.spawn_item" => return "spawn_item".to_string(),
        "world.spawn_entity" => return "spawn_entity".to_string(),
        _ => {}
    }

    if let Some(rest) = action_id.strip_prefix("npc.gesture_") {
        return slug_action_alias(rest);
    }

    let last = action_id
        .split('.')
        .filter(|part| !part.is_empty())
        .next_back();
    let fallback = match last {
        Some(part) => part.to_string(),
        None if !action.title.is_empty() => action.title.clone(),
        None => action.action_id.clone(),
    };
    slug_action_alias(&fallback)
}

/// Selects + sorts + limits the resolved actions from `books` for `scan_text`,
/// returning the entries that would be injected (already capped at `limit`).
pub fn select_action_entries(
    books: &[ActionBookDetail],
    scan_text: &str,
    limit: usize,
) -> Vec<ResolvedAction> {
    let mut items: Vec<ResolvedAction> = books
        .iter()
        .flat_map(|book| book.entries.iter())
        .filter(|action| action_is_active(action, scan_text))
        .cloned()
        .collect();
    // Sort by priority descending; entries arrive pre-sorted per-book, but
    // flattening across books needs a re-sort. Stable to preserve read order.
    items.sort_by(|left, right| {
        right
            .priority
            .partial_cmp(&left.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(limit);
    items
}

fn schema_non_empty(value: &Value) -> bool {
    match value {
        Value::Object(map) => !map.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Null => false,
        _ => true,
    }
}

fn format_scoped_catalog(catalog: &ScopedCatalogConfig) -> String {
    let mut bits: Vec<String> = Vec::new();
    let label = if catalog.title.is_empty() {
        catalog.catalog_id.clone()
    } else {
        catalog.title.clone()
    };
    bits.push(format!("catalog {} -> {}", catalog.parameter_name, label));
    if !catalog.description.is_empty() {
        bits.push(catalog.description.clone());
    }
    bits.join(" | ")
}

/// Formats one resolved action into its bullet block — alias line plus the
/// risk/description/parameters/preconditions/effects/use-when detail lines.
/// Mirrors `formatActionBookPrompt`'s per-entry output (runtime-only scoped
/// catalog *candidates* and nearby-NPC targets are described, not resolved).
pub fn format_resolved_action(action: &ResolvedAction) -> String {
    let alias = resolved_action_alias(action);
    let title = if action.title.is_empty() {
        action.action_id.clone()
    } else {
        action.title.clone()
    };
    let mut parts: Vec<String> = vec![
        format!("- {alias} => {}: {title}", action.action_id),
        format!("Action alias: {alias}. Prefer outputting this exact string in actions."),
    ];
    if !action.risk_tier.is_empty() {
        parts.push(format!("Risk: {}", action.risk_tier));
    }
    if !action.description.is_empty() {
        parts.push(format!("Description: {}", action.description));
    }
    if schema_non_empty(&action.parameters_schema) {
        parts.push(format!(
            "Parameters: {}",
            serde_json::to_string(&action.parameters_schema).unwrap_or_default()
        ));
    }
    if !action.preconditions.is_empty() {
        parts.push(format!(
            "Preconditions: {}",
            action.preconditions.join("; ")
        ));
    }
    if !action.effects.is_empty() {
        parts.push(format!("Effects: {}", action.effects.join("; ")));
    }
    for catalog in &action.scoped_catalogs {
        parts.push(format!(
            "Scoped catalog: {}",
            format_scoped_catalog(catalog)
        ));
    }
    if !action.examples_when_to_use.is_empty() {
        parts.push(format!(
            "Use when: {}",
            action.examples_when_to_use.join("; ")
        ));
    }
    if !action.examples_when_not_to_use.is_empty() {
        parts.push(format!(
            "Do not use when: {}",
            action.examples_when_not_to_use.join("; ")
        ));
    }
    parts.join("\n  ")
}

/// Formats the full Action Book body (the joined per-entry blocks, no header).
pub fn format_action_entries(actions: &[ResolvedAction]) -> String {
    actions
        .iter()
        .map(format_resolved_action)
        .collect::<Vec<_>>()
        .join("\n")
}

/// End-to-end: select, sort, limit, and render the injected block including the
/// `Activated Action Book entries:` header. Returns `None` when nothing is
/// active (so callers can skip the section entirely, like generation.js does).
pub fn build_action_book_block(
    books: &[ActionBookDetail],
    scan_text: &str,
    limit: Option<usize>,
) -> Option<String> {
    let limit = resolve_action_book_limit(limit);
    let selected = select_action_entries(books, scan_text, limit);
    if selected.is_empty() {
        return None;
    }
    Some(format!(
        "{ACTION_BOOK_HEADER}\n{}",
        format_action_entries(&selected)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn action(action_id: &str, title: &str, order: f64) -> ResolvedAction {
        ResolvedAction {
            id: "1".to_string(),
            uid: 1,
            action_id: action_id.to_string(),
            title: title.to_string(),
            description: String::new(),
            keys: Vec::new(),
            secondary_keys: Vec::new(),
            constant: false,
            vectorized: false,
            disable: false,
            case_sensitive: None,
            priority: order,
            risk_tier: "low".to_string(),
            target_game: String::new(),
            plugin_source: String::new(),
            parameters_schema: Value::Null,
            preconditions: Vec::new(),
            effects: Vec::new(),
            command_template: String::new(),
            binding: Value::Null,
            execution: Value::Null,
            examples_when_to_use: Vec::new(),
            examples_when_not_to_use: Vec::new(),
            scoped_catalogs: Vec::new(),
            scopes: Vec::new(),
            tags: Vec::new(),
            source_links: Vec::new(),
        }
    }

    fn book(entries: Vec<ResolvedAction>) -> ActionBookDetail {
        ActionBookDetail {
            id: "Book".to_string(),
            name: "Book".to_string(),
            description: String::new(),
            settings: Value::Null,
            binding: Value::Null,
            target_game: String::new(),
            catalogs: Vec::new(),
            entries,
        }
    }

    #[test]
    fn limit_clamps_to_forty_and_defaults_to_ten() {
        assert_eq!(resolve_action_book_limit(None), 10);
        assert_eq!(resolve_action_book_limit(Some(0)), 10);
        assert_eq!(resolve_action_book_limit(Some(5)), 5);
        assert_eq!(resolve_action_book_limit(Some(100)), 40);
    }

    #[test]
    fn selects_active_entries_sorted_by_priority() {
        let mut follow = action("movement.follow_target", "Follow", 235.0);
        follow.keys = vec!["follow".to_string()];
        let mut wave = action("npc.gesture_wave", "Wave", 252.0);
        wave.keys = vec!["wave".to_string()];
        let books = vec![book(vec![follow, wave])];

        let selected = select_action_entries(&books, "please wave then follow", 10);
        assert_eq!(selected.len(), 2);
        // Wave (252) outranks Follow (235).
        assert_eq!(selected[0].action_id, "npc.gesture_wave");

        // Only follow matches here.
        let only = select_action_entries(&books, "follow me", 10);
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].action_id, "movement.follow_target");
    }

    #[test]
    fn block_has_header_alias_and_fields() {
        let mut follow = action("movement.follow_target", "Follow target", 235.0);
        follow.constant = true;
        follow.risk_tier = "medium".to_string();
        follow.description = "Follow the player.".to_string();
        follow.parameters_schema = json!({ "target": "player" });
        let block = build_action_book_block(&[book(vec![follow])], "", None).unwrap();
        assert!(block.starts_with("Activated Action Book entries:\n"));
        assert!(block.contains("- follow => movement.follow_target: Follow target"));
        assert!(block.contains("Risk: medium"));
        assert!(block.contains("Description: Follow the player."));
        assert!(block.contains("Parameters: {\"target\":\"player\"}"));
    }

    #[test]
    fn empty_when_nothing_active() {
        let mut follow = action("movement.follow_target", "Follow", 235.0);
        follow.keys = vec!["follow".to_string()];
        assert!(build_action_book_block(&[book(vec![follow])], "hello there", None).is_none());
    }
}
