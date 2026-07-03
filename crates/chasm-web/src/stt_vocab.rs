//! Gathers the STT "boost vocabulary" for the active profile: every character
//! name plus every lorebook entry title and trigger key. These are exactly the
//! proper nouns a generic ASR mangles ("sunny smells" for "Sunny Smiles"), so
//! the Parakeet server biases toward them (see `scripts/stt_vocab_boost.py`).
//!
//! The list is REFRESHABLE, not baked at startup: it is rebuilt whenever the
//! character/lore files change, detected cheaply via a directory signature
//! (file count + newest mtime + total size) so we never re-read PNG cards on
//! every push-to-talk unless something actually changed.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use crate::AppState;

/// Shortest single word kept as its own boost token. Short words are both
/// collision-prone and rarely the proper noun a player struggles to be heard
/// saying, so we drop them (the full phrase is still kept).
const MIN_TOKEN_LEN: usize = 3;

/// Upper bound on delivered vocabulary. A pathologically large lorebook should
/// not bloat every transcription request; if we truncate we log it (never a
/// silent cap). Generous headroom: even a few thousand terms cost only a couple
/// of ms in the server-side corrector.
const MAX_VOCAB: usize = 5_000;

/// How many delivered words to include as a UI preview sample.
const SAMPLE_SIZE: usize = 24;

/// Cached per-source word lists for the active profile. Split by source so the
/// Characters/Lorebooks toggles can be applied (and counted) without re-reading
/// files — the lists are rebuilt only when the underlying files change.
struct Sources {
    fingerprint: u64,
    characters: Arc<Vec<String>>,
    lore: Arc<Vec<String>>,
}

static CACHE: Mutex<Option<Sources>> = Mutex::new(None);

/// A boost vocabulary plus the numbers the settings UI shows.
pub(crate) struct BoostSummary {
    /// The words actually delivered to the STT server (respecting the toggles).
    pub words: Vec<String>,
    /// Distinct character-derived terms available (independent of the toggle).
    pub available_characters: usize,
    /// Distinct lore-derived terms available (independent of the toggle).
    pub available_lore: usize,
    /// A small preview of the delivered words.
    pub sample: Vec<String>,
}

/// The per-source lists for the active profile, cached and rebuilt only when
/// the character/lore files change (add/edit/remove) — so the vocabulary always
/// tracks the current Characters and Lore books with no manual refresh.
fn sources(state: &AppState) -> (Arc<Vec<String>>, Arc<Vec<String>>) {
    let fingerprint = fingerprint(state);
    if let Ok(guard) = CACHE.lock() {
        if let Some(entry) = guard.as_ref() {
            if entry.fingerprint == fingerprint {
                return (entry.characters.clone(), entry.lore.clone());
            }
        }
    }
    let characters = Arc::new(gather_characters(state));
    let lore = Arc::new(gather_lore(state));
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some(Sources {
            fingerprint,
            characters: characters.clone(),
            lore: lore.clone(),
        });
    }
    (characters, lore)
}

/// Build the boost vocabulary + UI numbers for the given toggle states. Master
/// off (or both sources off) yields an empty word list.
pub(crate) fn summarize(
    state: &AppState,
    master: bool,
    want_characters: bool,
    want_lore: bool,
) -> BoostSummary {
    let (characters, lore) = sources(state);
    let mut words: Vec<String> = Vec::new();
    if master {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for source in [
            (want_characters, characters.as_ref()),
            (want_lore, lore.as_ref()),
        ] {
            if !source.0 {
                continue;
            }
            for word in source.1 {
                if seen.insert(word.to_lowercase()) {
                    words.push(word.clone());
                }
            }
        }
        if words.len() > MAX_VOCAB {
            tracing::debug!(
                "stt boost vocab truncated to {} of {} combined entries",
                MAX_VOCAB,
                words.len()
            );
            words.truncate(MAX_VOCAB);
        }
    }
    let sample = words.iter().take(SAMPLE_SIZE).cloned().collect();
    BoostSummary {
        words,
        available_characters: characters.len(),
        available_lore: lore.len(),
        sample,
    }
}

/// Character display names (embedded card name, falling back to file stem) —
/// the same source the Characters book UI shows.
fn gather_characters(state: &AppState) -> Vec<String> {
    clean_and_dedupe(crate::ui::books::character_names(state))
}

/// Lorebook entry titles + trigger keys for the active profile. Keys are
/// frequently the exact proper noun ("Novac", "Caesar's Legion") so they are
/// valuable boost terms too.
fn gather_lore(state: &AppState) -> Vec<String> {
    let mut raw: Vec<String> = Vec::new();
    if let Ok(books) = state.repository.list_lorebooks() {
        for book in books {
            for entry in book.entries {
                let title = entry.comment.trim();
                if !title.is_empty() {
                    raw.push(title.to_string());
                }
                raw.extend(entry.keys);
            }
        }
    }
    clean_and_dedupe(raw)
}

/// Cleans, splits multi-word names into useful sub-tokens, and dedupes
/// (case-insensitively, first casing wins). Pure function — unit-tested below.
fn clean_and_dedupe(raw: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();

    for entry in raw {
        // Normalize internal whitespace.
        let phrase = entry.split_whitespace().collect::<Vec<_>>().join(" ");
        if phrase.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = phrase.split(' ').collect();

        // Keep the whole multi-word phrase ("Sunny Smiles", "Caesar's Legion").
        if tokens.len() >= 2 {
            if let Some(cleaned) = clean_phrase(&phrase) {
                push_unique(&mut out, &mut seen, cleaned);
            }
        }
        // Plus each useful individual word ("Sunny", "Smiles").
        for token in tokens {
            if let Some(word) = clean_token(token) {
                push_unique(&mut out, &mut seen, word);
            }
        }
    }
    // Per-source list is uncapped; the combined MAX_VOCAB cap is applied in
    // `summarize` after the two sources are merged.
    out
}

/// Trims surrounding punctuation from a whole phrase, keeping it only if it
/// still carries enough letters to be a meaningful proper noun.
fn clean_phrase(phrase: &str) -> Option<String> {
    let trimmed = phrase
        .trim_matches(|c: char| !c.is_alphanumeric())
        .trim()
        .to_string();
    let alnum = trimmed.chars().filter(|c| c.is_alphanumeric()).count();
    if alnum >= 4 && trimmed.contains(' ') {
        Some(trimmed)
    } else {
        None
    }
}

/// Cleans a single word token: strips edge punctuation, and drops it if too
/// short, purely numeric, or an ordinary English word (which would false-boost).
fn clean_token(token: &str) -> Option<String> {
    let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric());
    let alnum = cleaned.chars().filter(|c| c.is_alphanumeric()).count();
    if alnum < MIN_TOKEN_LEN {
        return None;
    }
    if cleaned.chars().all(|c| !c.is_alphabetic()) {
        return None; // purely numeric / symbolic
    }
    if common_words().contains(cleaned.to_lowercase().as_str()) {
        return None;
    }
    Some(cleaned.to_string())
}

fn push_unique(out: &mut Vec<String>, seen: &mut HashSet<String>, word: String) {
    let key = word.to_lowercase();
    if seen.insert(key) {
        out.push(word);
    }
}

/// Ordinary English words that must never be boosted as single tokens (they
/// would corrupt normal speech). Multi-word phrases bypass this — they are
/// inherently specific. Kept in sync with the server-side guard in spirit.
fn common_words() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "the", "and", "are", "you", "your", "for", "was", "with", "his", "her", "him", "she",
            "they", "them", "that", "this", "then", "than", "there", "here", "have", "has", "had",
            "not", "but", "all", "any", "one", "two", "who", "why", "how", "what", "when", "where",
            "which", "will", "would", "can", "could", "should", "did", "does", "done", "get", "got",
            "out", "off", "over", "into", "just", "like", "know", "now", "new", "old", "good",
            "bad", "yes", "man", "men", "way", "day", "say", "see", "let", "come", "want", "need",
            "take", "make", "give", "tell", "talk", "look", "well", "back", "down", "from", "some",
            "more", "most", "much", "many", "very", "also", "about", "still", "even", "only",
            "such", "these", "those", "been", "being", "were", "our", "its", "hey", "yeah", "okay",
            "sir", "please", "hello", "thanks", "sorry", "sure", "fine", "name", "people", "friend",
            "help", "thing", "things", "time", "place", "world", "wait", "stop", "town", "city",
            "area", "quest", "note", "info", "lore", "entry", "book", "npc", "npcs", "misc",
        ]
        .into_iter()
        .collect()
    })
}

// --- Cheap staleness fingerprint -------------------------------------------

fn fingerprint(state: &AppState) -> u64 {
    let paths = state.config.active_profile_paths();
    let mut hasher = DefaultHasher::new();
    for dir in [paths.characters_dir(), paths.worlds_dir()] {
        dir.to_string_lossy().hash(&mut hasher);
        dir_signature(&dir).hash(&mut hasher);
    }
    hasher.finish()
}

/// A cheap directory signature: (file count, newest mtime nanos, total bytes).
/// Changes on any add / remove / in-place edit, without reading file contents.
fn dir_signature(dir: &Path) -> (u64, u64, u64) {
    let mut count = 0u64;
    let mut newest = 0u64;
    let mut total_len = 0u64;
    if let Ok(read) = fs::read_dir(dir) {
        for entry in read.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            count += 1;
            total_len = total_len.wrapping_add(meta.len());
            if let Ok(modified) = meta.modified() {
                if let Ok(delta) = modified.duration_since(SystemTime::UNIX_EPOCH) {
                    newest = newest.max(delta.as_nanos() as u64);
                }
            }
        }
    }
    (count, newest, total_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_ci(list: &[String], needle: &str) -> bool {
        list.iter().any(|w| w.eq_ignore_ascii_case(needle))
    }

    #[test]
    fn splits_phrases_and_keeps_full_name() {
        let out = clean_and_dedupe(vec!["Sunny Smiles".to_string()]);
        assert!(contains_ci(&out, "Sunny Smiles"), "keeps full phrase: {out:?}");
        assert!(contains_ci(&out, "Sunny"), "keeps sub-token Sunny: {out:?}");
        assert!(contains_ci(&out, "Smiles"), "keeps sub-token Smiles: {out:?}");
    }

    #[test]
    fn dedupes_case_insensitively() {
        let out = clean_and_dedupe(vec![
            "Novac".to_string(),
            "novac".to_string(),
            "NOVAC".to_string(),
        ]);
        let hits = out.iter().filter(|w| w.eq_ignore_ascii_case("novac")).count();
        assert_eq!(hits, 1, "novac deduped to one entry: {out:?}");
    }

    #[test]
    fn drops_short_and_common_and_numeric_tokens() {
        let out = clean_and_dedupe(vec![
            "the".to_string(),   // common
            "of".to_string(),    // too short
            "a".to_string(),     // too short
            "123".to_string(),   // numeric
            "town".to_string(),  // common (lore noise)
            "Rex".to_string(),   // 3-letter proper noun -> kept
        ]);
        assert!(!contains_ci(&out, "the"), "drops 'the': {out:?}");
        assert!(!contains_ci(&out, "of"), "drops short 'of': {out:?}");
        assert!(!contains_ci(&out, "123"), "drops numeric: {out:?}");
        assert!(!contains_ci(&out, "town"), "drops common 'town': {out:?}");
        assert!(contains_ci(&out, "Rex"), "keeps 'Rex': {out:?}");
    }

    #[test]
    fn keeps_multiword_phrase_even_with_common_word() {
        // A phrase like "Doc Mitchell" stays whole (phrases bypass the common
        // guard); "Doc" alone is < MIN_TOKEN_LEN so only "Mitchell" splits out.
        let out = clean_and_dedupe(vec!["Doc Mitchell".to_string()]);
        assert!(contains_ci(&out, "Doc Mitchell"), "keeps phrase: {out:?}");
        assert!(contains_ci(&out, "Mitchell"), "keeps Mitchell: {out:?}");
    }

    #[test]
    fn strips_edge_punctuation_and_normalizes_space() {
        let out = clean_and_dedupe(vec!["  Caesar's   Legion!  ".to_string()]);
        assert!(contains_ci(&out, "Caesar's Legion"), "cleans phrase: {out:?}");
        assert!(contains_ci(&out, "Legion"), "keeps Legion token: {out:?}");
    }

    #[test]
    fn empty_and_whitespace_are_ignored() {
        let out = clean_and_dedupe(vec![
            String::new(),
            "   ".to_string(),
            "\t\n".to_string(),
        ]);
        assert!(out.is_empty(), "no entries from blanks: {out:?}");
    }
}
