//! The GLOBAL scenario: one app-wide `{{macro}}` template that replaces the
//! per-character card `scenario` field in prompt assembly.
//!
//! Every turn the generation path resolves the template (the saved Globals
//! store value, else [`DEFAULT_SCENARIO_TEMPLATE`]) through [`crate::apply_macros`]
//! with the turn's gamestate macro table PLUS backend-computed macros (today
//! just `{{participants}}`, built by [`participants_macro`]), and injects the
//! result where the card scenario used to sit in the ST-style assembly order.
//!
//! This module holds the pieces that are pure prompt-text policy — the default
//! template and the computed-macro formatters — so they are unit-testable
//! without a repository or web state. Reading/writing the stored template is
//! `chasm_st_compat::GlobalsStore`; per-turn wiring is `chasm-web`'s
//! `generate.rs`.

/// The built-in global scenario template, used until the user saves their own
/// on the Globals page (and restorable from there).
///
/// Wording notes:
/// * `apply_macros` renders a missing macro as the EMPTY string, so the
///   template is split into short, self-contained sentences — one absent value
///   degrades only its own sentence instead of garbling one long clause.
/// * `{{participants}}` is backend-computed every turn (see
///   [`participants_macro`]) and always names at least the player, so the
///   conversation sentence never renders empty.
/// * The closing instruction carries no macros at all — it survives even a
///   turn with no recorded gamestate.
pub const DEFAULT_SCENARIO_TEMPLATE: &str = "It is {{time_of_day}}. You are in \
{{minor_location}}. The surrounding area is {{major_location}}. You are in a \
conversation with {{participants}}. Speak and act consistently with this \
place, this time, and the people present.";

/// Formats the backend-computed `{{participants}}` macro: who the prompted NPC
/// is talking WITH — the player plus every OTHER NPC in the group conversation
/// (the speaking character itself is excluded by the caller).
///
/// * `player_name` is the turn's `player_name` macro; when it is empty the
///   player is still named as "the player" so the sentence never collapses.
/// * `other_npc_names` are the co-present NPC names, in presence order.
///
/// Examples: `Courier` · `Courier and Sunny Smiles`
/// · `Courier, Sunny Smiles, and Trudy`.
pub fn participants_macro(player_name: &str, other_npc_names: &[String]) -> String {
    let player = {
        let trimmed = player_name.trim();
        if trimmed.is_empty() {
            "the player".to_string()
        } else {
            trimmed.to_string()
        }
    };
    let mut names: Vec<String> = vec![player];
    names.extend(
        other_npc_names
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_string),
    );
    readable_list(&names)
}

/// Joins names as a readable English list: `a` · `a and b` · `a, b, and c`
/// (Oxford comma, so long NPC groups stay unambiguous).
fn readable_list(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [head @ .., last] => format!("{}, and {last}", head.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apply_macros;
    use std::collections::BTreeMap;

    fn table(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn participants_player_only() {
        assert_eq!(participants_macro("Courier", &[]), "Courier");
    }

    #[test]
    fn participants_one_npc() {
        assert_eq!(
            participants_macro("Courier", &["Sunny Smiles".to_string()]),
            "Courier and Sunny Smiles"
        );
    }

    #[test]
    fn participants_several_npcs_use_oxford_comma() {
        assert_eq!(
            participants_macro(
                "Courier",
                &["Sunny Smiles".to_string(), "Trudy".to_string()]
            ),
            "Courier, Sunny Smiles, and Trudy"
        );
        assert_eq!(
            participants_macro(
                "Courier",
                &[
                    "Sunny Smiles".to_string(),
                    "Trudy".to_string(),
                    "Easy Pete".to_string()
                ]
            ),
            "Courier, Sunny Smiles, Trudy, and Easy Pete"
        );
    }

    #[test]
    fn participants_missing_player_name_degrades_to_the_player() {
        assert_eq!(participants_macro("", &[]), "the player");
        assert_eq!(
            participants_macro("  ", &["Trudy".to_string()]),
            "the player and Trudy"
        );
    }

    #[test]
    fn participants_skips_blank_npc_names() {
        assert_eq!(
            participants_macro("Courier", &[String::new(), "Trudy".to_string()]),
            "Courier and Trudy"
        );
    }

    #[test]
    fn default_template_resolves_fully_populated() {
        let macros = table(&[
            ("time_of_day", "2:32PM"),
            ("minor_location", "Prospector Saloon"),
            ("major_location", "Goodsprings"),
            (
                "participants",
                "Courier, Sunny Smiles, and Trudy",
            ),
        ]);
        assert_eq!(
            apply_macros(DEFAULT_SCENARIO_TEMPLATE, &macros),
            "It is 2:32PM. You are in Prospector Saloon. The surrounding area \
             is Goodsprings. You are in a conversation with Courier, Sunny \
             Smiles, and Trudy. Speak and act consistently with this place, this time, \
             and the people present."
        );
    }

    #[test]
    fn default_template_degrades_per_sentence_when_macros_missing() {
        // Only participants present (backend-computed, so always there): the
        // location/time sentences degrade individually; the conversation and
        // closing sentences stay fully intact — no `{{` leaks, no long clause
        // is garbled across values.
        let macros = table(&[("participants", "the player")]);
        let resolved = apply_macros(DEFAULT_SCENARIO_TEMPLATE, &macros);
        assert!(!resolved.contains("{{"), "no unresolved macros: {resolved}");
        assert!(resolved.contains("You are in a conversation with the player."));
        assert!(resolved.contains("Speak and act consistently"));
        // Each location/time sentence is still its own short sentence.
        assert!(resolved.contains("It is ."));
        assert!(resolved.contains("You are in ."));
    }
}
