//! Game profiles. Chasm core is generic; a profile (e.g. Fallout: New
//! Vegas) owns its characters and a game-specific voice extractor. Without an
//! active profile the app has no characters to show.
//!
//! A profile is also a self-contained content folder: characters, lorebooks,
//! quest/action books, voices, chats, the live-chats store, save-sync snapshots,
//! and the embed cache all live under `profiles/<id>/`. [`ProfilePaths`] resolves
//! each of those to the active profile's folder, falling back to the legacy
//! (pre-profile) location when the profile subdir does not exist — so an
//! un-migrated install keeps working unchanged.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One character owned by a game profile (matches a character card by name).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProfileCharacter {
    pub name: String,
    /// Game editor id for exact NPC resolution (optional).
    #[serde(default)]
    pub edid: String,
    /// Explicit voice-type override for generic/ambiguous NPCs (optional).
    #[serde(default)]
    pub voicetype: String,
}

/// A game profile loaded from `profiles/<id>/profile.json`.
///
/// Unknown top-level keys (e.g. the `voice` extractor block and `comment`) are
/// captured in `extra` so re-serializing a profile preserves them losslessly.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GameProfile {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub characters: Vec<ProfileCharacter>,
    /// Any other keys present in `profile.json` (e.g. `voice`, `comment`),
    /// preserved verbatim so a round-trip write does not drop them.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl GameProfile {
    /// Reads `profiles/<id>/profile.json`. Returns `None` when absent/invalid or
    /// when `id` is empty (no active profile -> empty app).
    pub fn read(profiles_dir: &Path, id: &str) -> Option<GameProfile> {
        if id.is_empty() {
            return None;
        }
        let text = fs::read_to_string(profiles_dir.join(id).join("profile.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Lists all valid profiles under `profiles_dir`, sorted by name.
    pub fn list(profiles_dir: &Path) -> Vec<GameProfile> {
        let mut out = Vec::new();
        if let Ok(entries) = fs::read_dir(profiles_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(id) = entry.file_name().to_str() {
                        if let Some(profile) = GameProfile::read(profiles_dir, id) {
                            out.push(profile);
                        }
                    }
                }
            }
        }
        out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        out
    }
}

/// Resolves every per-profile content directory/file to the active profile's
/// folder, with a fallback to the legacy (pre-profile) location.
///
/// Each accessor returns the `profiles/<id>/…` path **iff that path exists**;
/// otherwise it returns the legacy path. This keeps a half-migrated or
/// un-migrated install working: drop a `profiles/<id>/characters/` folder in and
/// it wins; leave it out and the old `{data_root}/characters/` is used.
///
/// When `profile_id` is empty (no active profile), every accessor returns the
/// legacy path — the app behaves exactly as it did before profiles existed.
///
/// Legacy bases:
/// * `data_root`  — `{data_root}/characters`, `/worlds`, `/chats`,
///   `/headless/...`, `/embed-cache`.
/// * `voices_dir` — `{workspace}/voices`.
/// * `embed_cache` — `CHASM_EMBED_DIR` when set, else `{data_root}/embed-cache`.
#[derive(Debug, Clone)]
pub struct ProfilePaths {
    profile_dir: Option<PathBuf>,
    data_root: PathBuf,
    legacy_voices_dir: PathBuf,
    legacy_embed_cache_dir: PathBuf,
}

impl ProfilePaths {
    /// Builds a resolver for the active `profile_id`. An empty id means
    /// "no profile" and yields legacy paths everywhere.
    pub fn new(
        profiles_dir: &Path,
        profile_id: &str,
        data_root: &Path,
        voices_dir: &Path,
        embed_cache_dir: &Path,
    ) -> Self {
        let profile_dir = if profile_id.is_empty() {
            None
        } else {
            Some(profiles_dir.join(profile_id))
        };
        Self {
            profile_dir,
            data_root: data_root.to_path_buf(),
            legacy_voices_dir: voices_dir.to_path_buf(),
            legacy_embed_cache_dir: embed_cache_dir.to_path_buf(),
        }
    }

    /// Resolves a content path: `profiles/<id>/<rel>` if it exists, else the
    /// supplied `legacy` path. `rel` is joined onto the profile dir component by
    /// component (e.g. `["headless", "quest-books"]`).
    fn resolve(&self, rel: &[&str], legacy: PathBuf) -> PathBuf {
        let Some(profile_dir) = self.profile_dir.as_ref() else {
            return legacy;
        };
        let mut candidate = profile_dir.clone();
        for part in rel {
            candidate.push(part);
        }
        if candidate.exists() {
            candidate
        } else {
            legacy
        }
    }

    /// Character cards dir: `profiles/<id>/characters` or `{data_root}/characters`.
    pub fn characters_dir(&self) -> PathBuf {
        self.resolve(&["characters"], self.data_root.join("characters"))
    }

    /// Lorebooks (World Info) dir: `profiles/<id>/worlds` or `{data_root}/worlds`.
    pub fn worlds_dir(&self) -> PathBuf {
        self.resolve(&["worlds"], self.data_root.join("worlds"))
    }

    /// Quest books dir: `profiles/<id>/headless/quest-books` or
    /// `{data_root}/headless/quest-books`.
    pub fn quest_books_dir(&self) -> PathBuf {
        self.resolve(
            &["headless", "quest-books"],
            self.data_root.join("headless").join("quest-books"),
        )
    }

    /// Action books dir: `profiles/<id>/headless/action-books` or
    /// `{data_root}/headless/action-books`.
    pub fn action_books_dir(&self) -> PathBuf {
        self.resolve(
            &["headless", "action-books"],
            self.data_root.join("headless").join("action-books"),
        )
    }

    /// Action catalogs dir (the full spawnable-record lists referenced by scoped
    /// catalogs): `profiles/<id>/headless/action-catalogs` or
    /// `{data_root}/headless/action-catalogs`.
    pub fn action_catalogs_dir(&self) -> PathBuf {
        self.resolve(
            &["headless", "action-catalogs"],
            self.data_root.join("headless").join("action-catalogs"),
        )
    }

    /// Voices dir (refs + clones): `profiles/<id>/voices` or `{workspace}/voices`.
    pub fn voices_dir(&self) -> PathBuf {
        self.resolve(&["voices"], self.legacy_voices_dir.clone())
    }

    /// Chats root (`<root>/chats/<name>/*.jsonl`): `profiles/<id>/chats` or
    /// `{data_root}/chats`. Backs `single`-mode (per-character) session files.
    pub fn chats_dir(&self) -> PathBuf {
        self.resolve(&["chats"], self.data_root.join("chats"))
    }

    /// Group-chats root (`<root>/group chats/<chat>.jsonl`):
    /// `profiles/<id>/group chats` or `{data_root}/group chats`. Backs
    /// `group`-mode session files (the live-chat conversation segments). Resolved
    /// on the `group chats` subdir itself so a profile that ships `chats/` but not
    /// `group chats/` still falls back to the legacy group-chats location.
    pub fn group_chats_dir(&self) -> PathBuf {
        self.resolve(&["group chats"], self.data_root.join("group chats"))
    }

    /// Live-chats store file: `profiles/<id>/headless/live-chats.json` or
    /// `{data_root}/headless/live-chats.json`.
    pub fn live_chats_store(&self) -> PathBuf {
        self.resolve(
            &["headless", "live-chats.json"],
            self.data_root.join("headless").join("live-chats.json"),
        )
    }

    /// Save-sync snapshots dir: `profiles/<id>/headless/save-sync` or
    /// `{data_root}/headless/save-sync`.
    pub fn save_sync_dir(&self) -> PathBuf {
        self.resolve(
            &["headless", "save-sync"],
            self.data_root.join("headless").join("save-sync"),
        )
    }

    /// Globals store file (app-wide prompt building blocks, e.g. the global
    /// scenario template): `profiles/<id>/headless/globals.json` or
    /// `{data_root}/headless/globals.json`. Resolved on the file itself (same
    /// rule as [`Self::live_chats_store`]): a fresh install writes the legacy
    /// location until the profile ships/migrates its own copy.
    pub fn globals_store(&self) -> PathBuf {
        self.resolve(
            &["headless", "globals.json"],
            self.data_root.join("headless").join("globals.json"),
        )
    }

    /// Embed cache dir: `profiles/<id>/embed-cache` or the legacy embed cache dir
    /// (`CHASM_EMBED_DIR` / `{data_root}/embed-cache`).
    pub fn embed_cache_dir(&self) -> PathBuf {
        self.resolve(&["embed-cache"], self.legacy_embed_cache_dir.clone())
    }

    /// Player-persona store dir (the capture image + generated description +
    /// stats snapshot): `profiles/<id>/headless/persona` when a profile is
    /// active and its folder exists, else `{data_root}/headless/persona`.
    ///
    /// Deliberately NOT the per-subpath [`Self::resolve`] rule the read-only
    /// content kinds use. Persona is a runtime store the backend WRITES, and
    /// the subdir does not exist until the first capture — `resolve` would
    /// route that first write to the legacy root even with a profile active,
    /// and the store would stick there. It is also a brand-new store (no
    /// pre-profile data exists in the wild), so there is no legacy content
    /// worth falling back to. Anchoring on [`Self::content_root`] — the same
    /// base the live-chats store and save-sync snapshots derive from — keeps
    /// reads and writes agreeing before and after the first write.
    pub fn persona_dir(&self) -> PathBuf {
        self.content_root().join("headless").join("persona")
    }

    /// The "content root" used by code paths that derive several sibling paths
    /// from one base (the live-chat repository, save-sync). When a profile is
    /// active *and* its folder exists, this is `profiles/<id>`; otherwise it is
    /// the legacy `data_root`. Note: callers that need the legacy `data_root`
    /// behavior for a specific subdir (e.g. world-state, which stays global)
    /// must not route through here.
    pub fn content_root(&self) -> PathBuf {
        match self.profile_dir.as_ref() {
            Some(dir) if dir.exists() => dir.clone(),
            _ => self.data_root.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown top-level keys in `profile.json` survive a deserialize→serialize
    /// round-trip (the `voice` extractor block + `comment`).
    #[test]
    fn profile_preserves_unknown_keys() {
        let json = serde_json::json!({
            "id": "fallout-new-vegas",
            "name": "Fallout: New Vegas",
            "description": "desc",
            "voice": { "extractor": "extract_voices.py", "plugin": "FalloutNV.esm" },
            "comment": "keep me",
            "characters": [ { "name": "Easy Pete", "edid": "GSEasyPete" } ]
        });
        let profile: GameProfile = serde_json::from_value(json).unwrap();
        assert_eq!(profile.id, "fallout-new-vegas");
        assert_eq!(profile.characters.len(), 1);
        assert!(profile.extra.contains_key("voice"));
        assert!(profile.extra.contains_key("comment"));

        let back = serde_json::to_value(&profile).unwrap();
        assert_eq!(back["voice"]["plugin"], "FalloutNV.esm");
        assert_eq!(back["comment"], "keep me");
        assert_eq!(back["characters"][0]["name"], "Easy Pete");
    }

    #[test]
    fn resolves_profile_path_when_subdir_exists_else_legacy() {
        let tmp = std::env::temp_dir().join(format!("sb-profilepaths-{}", std::process::id()));
        let profiles_dir = tmp.join("profiles");
        let data_root = tmp.join("data");
        let voices_dir = tmp.join("ws-voices");
        let embed_dir = data_root.join("embed-cache");
        let id = "fallout-new-vegas";

        // Only `characters` exists under the profile; nothing else does.
        fs::create_dir_all(profiles_dir.join(id).join("characters")).unwrap();

        let paths = ProfilePaths::new(&profiles_dir, id, &data_root, &voices_dir, &embed_dir);

        // Existing profile subdir wins.
        assert_eq!(
            paths.characters_dir(),
            profiles_dir.join(id).join("characters")
        );
        // Missing profile subdirs fall back to legacy bases.
        assert_eq!(paths.worlds_dir(), data_root.join("worlds"));
        assert_eq!(
            paths.action_books_dir(),
            data_root.join("headless").join("action-books")
        );
        assert_eq!(paths.voices_dir(), voices_dir);
        assert_eq!(paths.embed_cache_dir(), embed_dir);

        // content_root is the profile dir because profiles/<id> exists.
        assert_eq!(paths.content_root(), profiles_dir.join(id));
        // persona_dir anchors on content_root (write-safe before the subdir
        // exists), so it is the profile's headless/persona here.
        assert_eq!(
            paths.persona_dir(),
            profiles_dir.join(id).join("headless").join("persona")
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn empty_profile_id_always_uses_legacy() {
        let tmp =
            std::env::temp_dir().join(format!("sb-profilepaths-empty-{}", std::process::id()));
        let profiles_dir = tmp.join("profiles");
        let data_root = tmp.join("data");
        let voices_dir = tmp.join("ws-voices");
        let embed_dir = data_root.join("embed-cache");
        // Even if a folder exists, an empty id never routes into it.
        fs::create_dir_all(profiles_dir.join("x").join("characters")).unwrap();

        let paths = ProfilePaths::new(&profiles_dir, "", &data_root, &voices_dir, &embed_dir);
        assert_eq!(paths.characters_dir(), data_root.join("characters"));
        assert_eq!(paths.content_root(), data_root);
        assert_eq!(
            paths.live_chats_store(),
            data_root.join("headless").join("live-chats.json")
        );
        assert_eq!(
            paths.persona_dir(),
            data_root.join("headless").join("persona")
        );

        let _ = fs::remove_dir_all(&tmp);
    }
}
