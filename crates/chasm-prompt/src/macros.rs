//! Gamestate macro substitution (SillyTavern-style `{{key}}` placeholders).
//!
//! The mod sends a flat string→string macro table under `metadata.macros`
//! every turn (see `mod-source/docs/gamestate-macros.md` for the vocabulary);
//! whatever keys arrive that turn are exactly the macros available that turn.
//! [`apply_macros`] is the whole substitution engine: plain string replacement,
//! no defaults, conditionals, randoms, or nested expansion (future iterations).
//!
//! Production injection is scoped to ONE component: the GLOBAL scenario
//! template (see [`crate::scenario`]), resolved per turn by the generation
//! path with the turn's macros plus backend-computed ones (`{{participants}}`).
//! No other prompt component (cards, lore, system prompt) runs macros; the
//! Gamestate test endpoint (`POST /api/ui/v1/gamestate/test`) and the Globals
//! preview remain the free-form proof surfaces.

use std::collections::BTreeMap;

use serde_json::Value;

/// Replaces every `{{key}}` in `text` with its value from `macros`. Keys are
/// matched case-insensitively with surrounding whitespace trimmed, so
/// `{{ Player_Name }}` resolves like `{{player_name}}`. Unknown or missing
/// keys render as the empty string. No recursion: substituted values are
/// copied verbatim, never re-scanned, and a stray `{{` with no closing `}}`
/// is left untouched.
pub fn apply_macros(text: &str, macros: &BTreeMap<String, String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(open) = rest.find("{{") {
        let after_open = &rest[open + 2..];
        let Some(close) = after_open.find("}}") else {
            // No closing `}}` anywhere ahead: nothing further can be a macro.
            break;
        };

        out.push_str(&rest[..open]);
        let key = after_open[..close].trim().to_ascii_lowercase();
        if let Some(value) = macros.get(&key) {
            out.push_str(value);
        }
        // Unknown key -> substitute nothing (empty string).
        rest = &after_open[close + 2..];
    }

    out.push_str(rest);
    out
}

/// Extracts the macro table from a turn's `metadata` value: `metadata.macros`
/// as a `BTreeMap<String, String>`. Keys are lowercased (the case-insensitive
/// half of [`apply_macros`]' lookup); values are coerced to strings (strings
/// verbatim, numbers/bools formatted) and anything else is skipped. Returns an
/// empty map when `metadata` or `metadata.macros` is absent or not an object.
pub fn macros_from_metadata(metadata: &Value) -> BTreeMap<String, String> {
    macros_from_value(metadata.get("macros").unwrap_or(&Value::Null))
}

/// Coerces a bare `{ key: value }` object (e.g. `extra.chasm.macros`, or the
/// test endpoint's `macros` override) into the macro table shape.
pub fn macros_from_value(value: &Value) -> BTreeMap<String, String> {
    let mut macros = BTreeMap::new();
    let Value::Object(map) = value else {
        return macros;
    };
    for (key, value) in map {
        let key = key.trim().to_ascii_lowercase();
        if key.is_empty() {
            continue;
        }
        let text = match value {
            Value::String(text) => text.clone(),
            Value::Number(number) => number.to_string(),
            Value::Bool(flag) => flag.to_string(),
            _ => continue,
        };
        macros.insert(key, text);
    }
    macros
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn table(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn substitutes_a_basic_macro() {
        let macros = table(&[("player_name", "Courier")]);
        assert_eq!(
            apply_macros("Hello {{player_name}}!", &macros),
            "Hello Courier!"
        );
    }

    #[test]
    fn unknown_macro_renders_empty() {
        let macros = table(&[("player_name", "Courier")]);
        assert_eq!(apply_macros("Hi {{nope}}.", &macros), "Hi .");
        assert_eq!(apply_macros("{{also_nope}}", &BTreeMap::new()), "");
    }

    #[test]
    fn substitutes_multiple_macros_in_one_string() {
        let macros = table(&[("major_location", "Goodsprings"), ("time_of_day", "2:32PM")]);
        assert_eq!(
            apply_macros("You are in {{major_location}}. It is {{time_of_day}}.", &macros),
            "You are in Goodsprings. It is 2:32PM."
        );
    }

    #[test]
    fn substitutes_adjacent_macros() {
        let macros = table(&[("a", "1"), ("b", "2")]);
        assert_eq!(apply_macros("{{a}}{{b}}{{a}}", &macros), "121");
    }

    #[test]
    fn text_without_macros_passes_through_verbatim() {
        let macros = table(&[("player_name", "Courier")]);
        let text = "No placeholders here, just braces } and { alone.";
        assert_eq!(apply_macros(text, &macros), text);
    }

    #[test]
    fn malformed_open_without_close_is_left_alone() {
        let macros = table(&[("player_name", "Courier")]);
        assert_eq!(apply_macros("Broken {{player_name", &macros), "Broken {{player_name");
        // A macro BEFORE the stray opener still resolves.
        assert_eq!(
            apply_macros("{{player_name}} says {{oops", &macros),
            "Courier says {{oops"
        );
    }

    #[test]
    fn keys_match_case_insensitively_and_trimmed() {
        let macros = table(&[("player_name", "Courier")]);
        assert_eq!(apply_macros("{{PLAYER_NAME}}", &macros), "Courier");
        assert_eq!(apply_macros("{{ Player_Name }}", &macros), "Courier");
    }

    #[test]
    fn values_are_not_rescanned_for_macros() {
        // No recursion: a value containing `{{...}}` is copied verbatim.
        let macros = table(&[("a", "{{b}}"), ("b", "nested")]);
        assert_eq!(apply_macros("{{a}}", &macros), "{{b}}");
    }

    #[test]
    fn extracts_macros_from_turn_metadata() {
        let metadata = json!({
            "targeting": { "nearby_npcs": [] },
            "macros": {
                "Player_Name": "Courier",
                "level": 12,
                "hardcore": true,
                "ignored_object": { "nested": 1 },
                "ignored_null": null,
                "  ": "ignored blank key",
            }
        });
        let macros = macros_from_metadata(&metadata);
        assert_eq!(
            macros,
            table(&[("player_name", "Courier"), ("level", "12"), ("hardcore", "true")])
        );
    }

    #[test]
    fn missing_or_non_object_macros_yield_empty_table() {
        assert!(macros_from_metadata(&Value::Null).is_empty());
        assert!(macros_from_metadata(&json!({})).is_empty());
        assert!(macros_from_metadata(&json!({ "macros": "not-an-object" })).is_empty());
        assert!(macros_from_value(&json!([1, 2, 3])).is_empty());
    }
}
