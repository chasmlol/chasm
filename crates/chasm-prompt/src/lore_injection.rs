//! Additive lorebook (World Info) activation + rendering.
//!
//! This module ports the keyword/constant activation path from
//! `src/headless/lorebooks.js` (`entryMatches`, `entryPassesCharacterFilter`)
//! and the injection format from `src/headless/generation.js`:
//!
//! ```text
//! Activated lore:
//! <entry contents joined by blank lines>
//! ```
//!
//! It operates on the full [`LoreEntryFull`] model so it sees every field
//! (secondary keys, position, vectorized, depth, role). It is wired *additively*:
//! the existing [`crate::assemble_prompt`] signature is unchanged; callers that
//! want full-fidelity lore activation call [`activate_lore`] / [`render_lore_block`]
//! (or the convenience [`build_activated_lore_block`]) directly.
//!
//! Parity gaps versus the live ST path are intentional and documented inline:
//! vector retrieval (needs the embeddings runtime), selective secondary-key
//! AND/NOT logic, probability rolls, and at-depth/role placement are not applied
//! here — see the crate-level docs and the final migration report.

use regex::RegexBuilder;
use chasm_st_compat::LoreEntryFull;

/// Default activation limit, matching the headless resolver's `limit` default.
pub const DEFAULT_LORE_LIMIT: usize = 10;

/// Tests one activation key against `text`. Tries a regex (case-insensitive
/// unless `case_sensitive`), falling back to a substring test — mirroring the
/// `entryMatches` try/catch in `lorebooks.js`.
fn key_matches(key: &str, text: &str, case_sensitive: bool) -> bool {
    let raw = key.trim();
    if raw.is_empty() {
        return false;
    }
    match RegexBuilder::new(raw)
        .case_insensitive(!case_sensitive)
        .size_limit(1 << 20)
        .build()
    {
        Ok(regex) => regex.is_match(text),
        Err(_) => {
            if case_sensitive {
                text.contains(raw)
            } else {
                text.to_lowercase().contains(&raw.to_lowercase())
            }
        }
    }
}

/// Mirrors `entryMatches`: constant entries (not disabled) always fire; keyword
/// entries fire when any primary key matches the scan text.
fn entry_matches(entry: &LoreEntryFull, text: &str) -> bool {
    if entry.constant && !entry.disable {
        return true;
    }
    entry
        .keys
        .iter()
        .any(|key| key_matches(key, text, entry.case_sensitive))
}

/// Mirrors `entryPassesCharacterFilter`: an entry with a character filter only
/// applies to the named characters (or everyone-but-named when `filter_exclude`).
/// Names are compared case-insensitively with a trailing `.png` stripped, like
/// `normalizeCharacterFilterName`.
fn passes_character_filter(entry: &LoreEntryFull, character_name: Option<&str>) -> bool {
    if entry.filter_names.is_empty() {
        return true;
    }
    let normalize = |value: &str| value.trim().trim_end_matches(".png").to_lowercase();
    let target = character_name.map(normalize);
    let matches = target
        .as_deref()
        .map(|name| {
            entry
                .filter_names
                .iter()
                .any(|candidate| normalize(candidate) == name)
        })
        .unwrap_or(false);
    if entry.filter_exclude {
        !matches
    } else {
        matches
    }
}

/// A single activated lore entry plus the book it came from and why it fired.
#[derive(Debug, Clone)]
pub struct ActivatedLore<'a> {
    pub book_id: &'a str,
    pub entry: &'a LoreEntryFull,
    /// `"constant"` or `"keyword"`.
    pub match_source: &'static str,
}

/// Activates lore across `(book_id, entries)` groups against `scan_text`.
///
/// Steps (ported from `resolveLorebooks`):
/// 1. skip disabled entries and entries failing the character filter,
/// 2. keep constant entries and keyword matches,
/// 3. dedup by `(book_id, uid)`,
/// 4. sort by `order` descending,
/// 5. truncate to `limit`.
///
/// Vector-marked entries are reported via [`LoreEntryFull::vectorized`] but are
/// NOT auto-activated here (no embeddings runtime).
pub fn activate_lore<'a>(
    books: impl IntoIterator<Item = (&'a str, &'a [LoreEntryFull])>,
    scan_text: &str,
    character_name: Option<&str>,
    limit: usize,
) -> Vec<ActivatedLore<'a>> {
    let mut seen: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    let mut activated: Vec<ActivatedLore<'a>> = Vec::new();

    for (book_id, entries) in books {
        for entry in entries {
            if entry.disable || !passes_character_filter(entry, character_name) {
                continue;
            }
            if !entry_matches(entry, scan_text) {
                continue;
            }
            if !seen.insert((book_id, entry.uid.as_str())) {
                continue;
            }
            let match_source = if entry.constant {
                "constant"
            } else {
                "keyword"
            };
            activated.push(ActivatedLore {
                book_id,
                entry,
                match_source,
            });
        }
    }

    activated.sort_by(|a, b| {
        b.entry
            .order
            .partial_cmp(&a.entry.order)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    activated.truncate(limit);
    activated
}

/// Renders the injected block exactly as `generation.js` does:
/// `"Activated lore:\n" + contents.join("\n\n")`. Returns `None` when nothing
/// activated (so the caller can skip the system part entirely).
pub fn render_lore_block(activated: &[ActivatedLore<'_>]) -> Option<String> {
    let body = activated
        .iter()
        .map(|item| item.entry.content.clone())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if body.is_empty() {
        None
    } else {
        Some(format!("Activated lore:\n{body}"))
    }
}

/// Convenience: activate + render in one call. Injected after character context
/// and before world-state, matching the headless `systemParts` order.
pub fn build_activated_lore_block<'a>(
    books: impl IntoIterator<Item = (&'a str, &'a [LoreEntryFull])>,
    scan_text: &str,
    character_name: Option<&str>,
    limit: usize,
) -> Option<String> {
    render_lore_block(&activate_lore(books, scan_text, character_name, limit))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(uid: &str, keys: &[&str], constant: bool, order: f64, content: &str) -> LoreEntryFull {
        LoreEntryFull {
            uid: uid.to_string(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            keys_secondary: Vec::new(),
            comment: String::new(),
            content: content.to_string(),
            constant,
            selective: false,
            disable: false,
            order,
            position: chasm_st_compat::WorldInfoPosition::Before,
            depth: None,
            probability: None,
            case_sensitive: false,
            match_whole_words: None,
            vectorized: false,
            role: None,
            filter_names: Vec::new(),
            filter_exclude: false,
        }
    }

    #[test]
    fn constant_and_keyword_activate() {
        let entries = vec![
            entry("0", &[], true, 10.0, "Always here."),
            entry("1", &["Goodsprings"], false, 50.0, "A dusty town."),
            entry("2", &["Vegas"], false, 99.0, "Bright lights."),
        ];
        let block = build_activated_lore_block(
            std::iter::once(("Fallout New Vegas", entries.as_slice())),
            "We rode into goodsprings at dawn.",
            None,
            DEFAULT_LORE_LIMIT,
        )
        .unwrap();
        // Sorted by order desc: dusty town (50) before always-here (10); Vegas not matched.
        assert_eq!(block, "Activated lore:\nA dusty town.\n\nAlways here.");
    }

    #[test]
    fn disabled_and_filtered_entries_are_skipped() {
        let mut disabled = entry("0", &[], true, 10.0, "Disabled.");
        disabled.disable = true;
        let mut filtered = entry("1", &[], true, 10.0, "Only for Sunny.");
        filtered.filter_names = vec!["Sunny Smiles".to_string()];
        let entries = vec![disabled, filtered];
        let activated = activate_lore(
            std::iter::once(("book", entries.as_slice())),
            "anything",
            Some("Easy Pete"),
            DEFAULT_LORE_LIMIT,
        );
        assert!(activated.is_empty());
    }

    #[test]
    fn dedup_by_book_and_uid_and_truncate() {
        let entries = vec![
            entry("0", &[], true, 1.0, "a"),
            entry("1", &[], true, 2.0, "b"),
            entry("2", &[], true, 3.0, "c"),
        ];
        let activated = activate_lore(std::iter::once(("book", entries.as_slice())), "", None, 2);
        assert_eq!(activated.len(), 2);
        // Highest order first.
        assert_eq!(activated[0].entry.content, "c");
        assert_eq!(activated[1].entry.content, "b");
    }
}
