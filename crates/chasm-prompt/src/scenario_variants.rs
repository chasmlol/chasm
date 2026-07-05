//! DYNAMIC scenario variants: gamestate-selected wording for the global
//! scenario.
//!
//! The global scenario used to be ONE template for every situation. This
//! module adds a fixed catalog of situation VARIANTS — companion, following,
//! sneaking-together, traveling, waiting, and friends — each with its own
//! user-editable template. Per turn the generation path evaluates the
//! responding NPC's [`NpcStateFlags`] (mod-reported `metadata.npc_state` plus
//! chasm's own movement store for `traveling`) and picks the highest-priority
//! enabled variant whose condition holds; the winner's template then resolves
//! through the exact same `{{macro}}` pipeline as before. Exactly one scenario
//! is ever injected.
//!
//! Selection is driven by INTERNAL GAME STATE ONLY — engine-reported flags and
//! chasm's own stores — never by analyzing chat text or which actions fired.
//!
//! Conditions are a FIXED enum (the catalog below), not free-form expressions:
//! the UI shows each variant's condition read-only and the user edits wording,
//! priority, and enablement. Combined states (companion + sneaking) are their
//! own catalog entries that outrank their parts by default priority.
//!
//! This module is pure prompt-text policy (no repository, no web state):
//! flag parsing, the catalog, and deterministic selection — all unit-testable.
//! Storage is `chasm_st_compat::GlobalsStore::scenario_variants`; per-turn
//! wiring is `chasm-web`'s `generate.rs`.

use serde_json::Value;

/// The per-turn gamestate flags a variant condition can bind to.
///
/// All the mod-reported flags default to FALSE when `metadata.npc_state` is
/// absent (old mod build, admin path, another game's bridge) — a missing block
/// simply selects the `default` variant and never errors. `traveling` is
/// chasm-side state (an active journey in the movement store), filled in by
/// the caller, not parsed from metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NpcStateFlags {
    /// The NPC is the player's teammate (vanilla teammate state).
    pub teammate: bool,
    /// The NPC is actively following the player RIGHT NOW (a companion told
    /// to wait is a teammate but not following).
    pub following: bool,
    /// The NPC was told to wait/stay put and is parked expecting the player.
    pub waiting: bool,
    /// The NPC is sneaking.
    pub sneaking: bool,
    /// The player is sneaking.
    pub player_sneaking: bool,
    /// The NPC has their weapon drawn.
    pub weapon_drawn: bool,
    /// The player has their weapon drawn.
    pub player_weapon_drawn: bool,
    /// The NPC is sitting (or settling into / rising from a seat).
    pub sitting: bool,
    /// The player is swimming.
    pub player_swimming: bool,
    /// CHASM-SIDE: the NPC has an active (en-route) journey in the movement
    /// store. Set by the caller from the store, never from metadata.
    pub traveling: bool,
}

impl NpcStateFlags {
    /// Parses the mod-reported flags from a turn's request `metadata`.
    ///
    /// The bridge forwards the mod's `npc_state` block as `metadata.npcState`
    /// (inner keys stay snake_case, exactly as the mod wrote them); the raw
    /// `npc_state` spelling is accepted too for direct/test callers. Absent
    /// block, absent key, or non-bool value → false. `traveling` is NOT read
    /// here (chasm-side state).
    pub fn from_metadata(metadata: &Value) -> Self {
        let state = metadata
            .get("npcState")
            .or_else(|| metadata.get("npc_state"))
            .unwrap_or(&Value::Null);
        let flag = |key: &str| state.get(key).and_then(Value::as_bool).unwrap_or(false);
        NpcStateFlags {
            teammate: flag("teammate"),
            following: flag("following"),
            waiting: flag("waiting"),
            sneaking: flag("sneaking"),
            player_sneaking: flag("player_sneaking"),
            weapon_drawn: flag("weapon_drawn"),
            player_weapon_drawn: flag("player_weapon_drawn"),
            sitting: flag("sitting"),
            player_swimming: flag("player_swimming"),
            traveling: false,
        }
    }
}

/// One entry of the FIXED variant catalog: the condition it binds to (by id),
/// its display label + human condition description, and its shipped defaults.
#[derive(Debug, Clone, Copy)]
pub struct VariantDef {
    /// Stable id — the storage/UI key AND the condition selector.
    pub id: &'static str,
    /// Short display label for the UI.
    pub label: &'static str,
    /// Read-only, human description of when the variant applies (the UI shows
    /// this next to the template; the condition itself is not editable).
    pub condition_hint: &'static str,
    /// Default priority (higher wins). The `default` variant is priority 0 by
    /// construction and is not in this catalog.
    pub default_priority: i32,
    /// The shipped template wording (user-editable, restorable).
    pub default_template: &'static str,
}

/// The variant catalog, highest default priority first. The `default`
/// variant (the pre-existing global template) is NOT listed here: it lives in
/// `GlobalsStore::scenario_template` and is the final fallback.
pub const VARIANT_CATALOG: &[VariantDef] = &[
    VariantDef {
        id: "traveling",
        label: "Traveling",
        condition_hint: "The NPC has an active journey in the movement store (en route to a destination).",
        default_priority: 100,
        default_template: "It is {{time_of_day}}. You are on the road, traveling to \
            {{travel_destination}}. You expect to arrive around {{travel_arrival_time}}. \
            The surrounding area is {{major_location}}. You are in a conversation with \
            {{participants}}. Your mind is on the journey — where you are headed and why — \
            so speak like someone mid-trip, not someone idling around.",
    },
    VariantDef {
        id: "companion_sneaking",
        label: "Companion, sneaking",
        condition_hint: "The NPC is the player's companion and the pair is sneaking (the player \
            is crouched — companions sneak with them).",
        default_priority: 90,
        default_template: "It is {{time_of_day}}. You are crouched beside {{player_name}}, \
            moving quietly together {{inside_or_outside}} {{minor_location}}. The surrounding \
            area is {{major_location}}. You are in a hushed conversation with {{participants}}. \
            WHISPER: keep every reply terse and low, stay alert, and never raise your voice — \
            you do not want to be heard.",
    },
    VariantDef {
        id: "sneaking_stranger",
        label: "Player sneaking (not a companion)",
        condition_hint: "The player is sneaking while talking to an NPC who is NOT their \
            companion — someone is skulking up to you.",
        default_priority: 85,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}}. The surrounding area is {{major_location}}. {{player_name}} is \
            crouched low, skulking about while speaking to you. You are in a conversation with \
            {{participants}}. React to that suspicious behavior the way your character would — \
            puzzled, wary, or amused.",
    },
    VariantDef {
        id: "waiting",
        label: "Waiting",
        condition_hint: "The NPC was told to wait/stay put and is parked at that spot expecting \
            the player's return.",
        default_priority: 80,
        default_template: "It is {{time_of_day}}. You are waiting {{inside_or_outside}} \
            {{minor_location}}, staying put where {{player_name}} asked you to wait. The \
            surrounding area is {{major_location}}. You are in a conversation with \
            {{participants}}. Speak like someone holding position and expecting to be \
            collected — you are not going anywhere until told.",
    },
    VariantDef {
        id: "following",
        label: "Following",
        condition_hint: "The NPC is actively following the player right now (not merely a \
            teammate — a companion told to wait is not following).",
        default_priority: 70,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}}, walking at {{player_name}}'s side — where they go, you go. The \
            surrounding area is {{major_location}}. You are in a conversation with \
            {{participants}}. Speak like a traveling partner sharing the road, mindful of the \
            surroundings you pass through.",
    },
    VariantDef {
        id: "companion",
        label: "Companion",
        condition_hint: "The NPC is the player's companion/teammate (the vanilla teammate \
            state), regardless of follow/wait orders.",
        default_priority: 60,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}}, traveling with {{player_name}} as their companion — you share \
            the road, the fights, and the spoils. The surrounding area is {{major_location}}. \
            You are in a conversation with {{participants}}. Speak with the familiarity of a \
            trusted partner, not a stranger.",
    },
    VariantDef {
        id: "weapon_drawn",
        label: "Weapon drawn (NPC)",
        condition_hint: "The NPC has their own weapon out (not yet in combat — combat has its \
            own directive).",
        default_priority: 50,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}} with your weapon drawn and ready. The surrounding area is \
            {{major_location}}. You are in a conversation with {{participants}}. You are on \
            edge — keep your words clipped and watchful; this is not a relaxed chat.",
    },
    VariantDef {
        id: "player_weapon_drawn",
        label: "Weapon drawn (player)",
        condition_hint: "The player is holding a drawn weapon while speaking to the NPC.",
        default_priority: 45,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}}. The surrounding area is {{major_location}}. {{player_name}} is \
            speaking to you with a drawn weapon in hand. You are in a conversation with \
            {{participants}}. React to the bared weapon the way your character would — wary, \
            on guard, or defiant.",
    },
    VariantDef {
        id: "sitting",
        label: "Sitting",
        condition_hint: "The NPC is sitting down (bench, chair, stool).",
        default_priority: 40,
        default_template: "It is {{time_of_day}}. You are sitting down {{inside_or_outside}} \
            {{minor_location}}, settled and at ease. The surrounding area is \
            {{major_location}}. You are in a conversation with {{participants}}. Speak like \
            someone comfortable where they sit, in no hurry to be anywhere else.",
    },
    VariantDef {
        id: "player_swimming",
        label: "Player swimming",
        condition_hint: "The player is in the water, swimming.",
        default_priority: 30,
        default_template: "It is {{time_of_day}}. You are {{inside_or_outside}} \
            {{minor_location}}. The surrounding area is {{major_location}}. {{player_name}} is \
            in the water, swimming, while speaking with you. You are in a conversation with \
            {{participants}}. Speak and act consistently with this odd arrangement.",
    },
];

/// Looks up a catalog entry by id.
pub fn variant_def(id: &str) -> Option<&'static VariantDef> {
    VARIANT_CATALOG.iter().find(|def| def.id == id)
}

/// Evaluates the FIXED condition bound to a variant id against the turn's
/// flags. Unknown ids never match (forward-compat: a stored id this build
/// doesn't know is skipped, not an error). The `default` id always matches.
pub fn condition_matches(id: &str, flags: &NpcStateFlags) -> bool {
    match id {
        "traveling" => flags.traveling,
        // FNV auto-sneaks teammates with the player, so the pair is sneaking
        // when either flag reads true for a teammate.
        "companion_sneaking" => flags.teammate && (flags.sneaking || flags.player_sneaking),
        "sneaking_stranger" => flags.player_sneaking && !flags.teammate,
        "waiting" => flags.waiting,
        "following" => flags.following,
        "companion" => flags.teammate,
        "weapon_drawn" => flags.weapon_drawn,
        "player_weapon_drawn" => flags.player_weapon_drawn,
        "sitting" => flags.sitting,
        "player_swimming" => flags.player_swimming,
        "default" => true,
        _ => false,
    }
}

/// One RESOLVED variant (stored config merged over the catalog defaults) as
/// the selection input.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioVariant {
    pub id: String,
    pub enabled: bool,
    pub priority: i32,
    pub template: String,
}

/// The selection outcome: which variant won and its template.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedScenario<'a> {
    /// The winning variant id (`"default"` for the fallback).
    pub variant_id: &'a str,
    /// The winning template — may be blank ONLY for the default variant
    /// (blank default = the user disabled the scenario component entirely).
    pub template: &'a str,
}

/// Deterministic variant selection: walk the variants highest priority first
/// (ties break by catalog order, then id, so selection is stable) and return
/// the first that is enabled, whose condition holds, and whose template is
/// non-blank — a BLANK template for an enabled, matching variant falls
/// through to the next match (same "blank = omit" semantics the scenario
/// already has, applied per variant). Nothing matches → the `default`
/// variant/template (which may itself be blank = scenario omitted).
pub fn select_scenario<'a>(
    variants: &'a [ScenarioVariant],
    default_template: &'a str,
    flags: &NpcStateFlags,
) -> SelectedScenario<'a> {
    let catalog_rank = |id: &str| {
        VARIANT_CATALOG
            .iter()
            .position(|def| def.id == id)
            .unwrap_or(usize::MAX)
    };
    let mut ordered: Vec<&ScenarioVariant> = variants.iter().collect();
    ordered.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| catalog_rank(&a.id).cmp(&catalog_rank(&b.id)))
            .then_with(|| a.id.cmp(&b.id))
    });
    for variant in ordered {
        if !variant.enabled || variant.id == "default" {
            continue;
        }
        if !condition_matches(&variant.id, flags) {
            continue;
        }
        if variant.template.trim().is_empty() {
            continue; // blank = fall through to the next match
        }
        return SelectedScenario {
            variant_id: &variant.id,
            template: &variant.template,
        };
    }
    SelectedScenario {
        variant_id: "default",
        template: default_template,
    }
}

/// The shipped variant set as selection inputs: every catalog entry, enabled,
/// at its default priority and template. Used when nothing is stored yet and
/// as the merge base for partial configs.
pub fn default_variants() -> Vec<ScenarioVariant> {
    VARIANT_CATALOG
        .iter()
        .map(|def| ScenarioVariant {
            id: def.id.to_string(),
            enabled: true,
            priority: def.default_priority,
            template: def.default_template.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flags() -> NpcStateFlags {
        NpcStateFlags::default()
    }

    #[test]
    fn missing_npc_state_parses_to_all_false() {
        assert_eq!(NpcStateFlags::from_metadata(&Value::Null), flags());
        assert_eq!(NpcStateFlags::from_metadata(&json!({})), flags());
        assert_eq!(
            NpcStateFlags::from_metadata(&json!({ "npcState": "bogus" })),
            flags()
        );
    }

    #[test]
    fn parses_bridge_and_raw_spellings() {
        let bridge = json!({ "npcState": { "teammate": true, "sneaking": true } });
        let parsed = NpcStateFlags::from_metadata(&bridge);
        assert!(parsed.teammate && parsed.sneaking);
        assert!(!parsed.following && !parsed.traveling);

        let raw = json!({ "npc_state": { "player_sneaking": true, "waiting": true } });
        let parsed = NpcStateFlags::from_metadata(&raw);
        assert!(parsed.player_sneaking && parsed.waiting);
    }

    #[test]
    fn non_bool_flag_values_read_false() {
        let metadata = json!({ "npcState": { "teammate": "yes", "sitting": 1 } });
        assert_eq!(NpcStateFlags::from_metadata(&metadata), flags());
    }

    #[test]
    fn default_selection_when_no_flags() {
        let variants = default_variants();
        let selected = select_scenario(&variants, "DEFAULT", &flags());
        assert_eq!(selected.variant_id, "default");
        assert_eq!(selected.template, "DEFAULT");
    }

    #[test]
    fn companion_selects_companion_variant() {
        let variants = default_variants();
        let state = NpcStateFlags { teammate: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "companion");
    }

    #[test]
    fn combined_state_outranks_its_parts() {
        let variants = default_variants();
        // Companion + sneaking outranks both companion and any sneaking variant.
        let state = NpcStateFlags {
            teammate: true,
            sneaking: true,
            player_sneaking: true,
            ..flags()
        };
        assert_eq!(
            select_scenario(&variants, "D", &state).variant_id,
            "companion_sneaking"
        );
        // Player-only sneak still counts for a teammate (FNV auto-sneak).
        let state = NpcStateFlags { teammate: true, player_sneaking: true, ..flags() };
        assert_eq!(
            select_scenario(&variants, "D", &state).variant_id,
            "companion_sneaking"
        );
        // A sneaking player near a NON-companion hits the stranger variant.
        let state = NpcStateFlags { player_sneaking: true, ..flags() };
        assert_eq!(
            select_scenario(&variants, "D", &state).variant_id,
            "sneaking_stranger"
        );
    }

    #[test]
    fn traveling_outranks_companion_states() {
        let variants = default_variants();
        let state = NpcStateFlags { teammate: true, following: true, traveling: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "traveling");
    }

    #[test]
    fn waiting_outranks_following_and_companion() {
        let variants = default_variants();
        let state = NpcStateFlags { teammate: true, waiting: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "waiting");
        let state = NpcStateFlags { teammate: true, following: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "following");
    }

    #[test]
    fn disabled_variant_is_skipped() {
        let mut variants = default_variants();
        variants
            .iter_mut()
            .find(|variant| variant.id == "companion")
            .unwrap()
            .enabled = false;
        let state = NpcStateFlags { teammate: true, ..flags() };
        // Companion disabled → nothing else matches → default.
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "default");
    }

    #[test]
    fn blank_template_falls_through_to_next_match() {
        let mut variants = default_variants();
        variants
            .iter_mut()
            .find(|variant| variant.id == "companion_sneaking")
            .unwrap()
            .template = "   ".to_string();
        let state = NpcStateFlags { teammate: true, sneaking: true, ..flags() };
        // companion_sneaking matches first but is blank → falls to sitting?
        // No: next matching is companion (teammate).
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "companion");
    }

    #[test]
    fn priority_reorder_changes_selection() {
        let mut variants = default_variants();
        variants
            .iter_mut()
            .find(|variant| variant.id == "sitting")
            .unwrap()
            .priority = 999;
        let state = NpcStateFlags { teammate: true, sitting: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "sitting");
    }

    #[test]
    fn unknown_stored_id_is_skipped_not_an_error() {
        let mut variants = default_variants();
        variants.push(ScenarioVariant {
            id: "from_the_future".to_string(),
            enabled: true,
            priority: 10_000,
            template: "??".to_string(),
        });
        let state = NpcStateFlags { teammate: true, ..flags() };
        assert_eq!(select_scenario(&variants, "D", &state).variant_id, "companion");
    }

    #[test]
    fn default_templates_resolve_with_no_macro_leaks() {
        use std::collections::BTreeMap;
        let macros: BTreeMap<String, String> =
            [("participants".to_string(), "the player".to_string())].into();
        for def in VARIANT_CATALOG {
            let resolved = crate::apply_macros(def.default_template, &macros);
            assert!(
                !resolved.contains("{{"),
                "variant {} leaks a macro: {resolved}",
                def.id
            );
        }
    }

    #[test]
    fn catalog_ids_are_unique_and_condition_backed() {
        let mut seen = std::collections::BTreeSet::new();
        for def in VARIANT_CATALOG {
            assert!(seen.insert(def.id), "duplicate catalog id {}", def.id);
            // Every catalog condition must be reachable: some flag set makes it true.
            let all_on = NpcStateFlags {
                teammate: true,
                following: true,
                waiting: true,
                sneaking: true,
                player_sneaking: true,
                weapon_drawn: true,
                player_weapon_drawn: true,
                sitting: true,
                player_swimming: true,
                traveling: true,
            };
            let stranger_sneak = NpcStateFlags { player_sneaking: true, ..NpcStateFlags::default() };
            assert!(
                condition_matches(def.id, &all_on) || condition_matches(def.id, &stranger_sneak),
                "catalog condition {} can never match",
                def.id
            );
        }
    }
}
