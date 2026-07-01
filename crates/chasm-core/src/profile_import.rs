//! Profile bundle import.
//!
//! Chasm is game-agnostic and ships an empty `profiles/`. Game content arrives as
//! a **profile bundle** — one self-contained folder holding everything authored
//! about one game (characters, lorebooks, action/quest books, catalogs, groups),
//! keyed by `profile.json`. A bundle travels *with the game's mod*; when the mod
//! boots and stages its bundle into the shared bridge folder, chasm copies the
//! validated, authored-content-only subset into its own `profiles/<id>/`.
//!
//! See `SillyBridge-FNV/docs/PROFILES.md` for the full design. This module
//! implements the copy: [`import_bundle`] for one `<source>/` bundle folder and
//! [`import_from_source_root`] for a directory of them. It is deliberately
//! game-agnostic — nothing here hard-codes "Fallout"; it just copies bundle
//! folders whose id is a safe slug, and only the allowlisted content entries.
//!
//! Safety / correctness guarantees:
//! * **Slug-validated id** — the id (folder name + active-profile key) is rejected
//!   if it contains `/`, `\`, `..`, or is empty, so a malicious bundle can't write
//!   outside `profiles/`.
//! * **Version-gated** — an already-present profile is only replaced when the
//!   source `bundleVersion` is strictly newer, so a user's local edits to an
//!   equal/older bundle are never clobbered.
//! * **Allowlist-only copy** — only authored content is copied; runtime/per-user
//!   dirs (chats, save-sync, embed cache, vectors, voices, world-state, …) are
//!   never copied even if a malformed source contains them.
//! * **Atomic install** — the bundle is assembled in `profiles/<id>.tmp/` and
//!   renamed into place, replacing any old `profiles/<id>/`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::GameProfile;

/// Authored-content entries copied from a bundle into `profiles/<id>/`, relative
/// to the bundle root. Each maps 1:1 to a [`crate::ProfilePaths`] accessor. An
/// entry may be a file (`profile.json`) or a directory (copied recursively).
/// Anything not on this list is ignored, so a source can never smuggle extra
/// content in.
pub const ALLOWLIST: &[&str] = &[
    "profile.json",
    "characters",
    "worlds",
    "headless/action-books",
    "headless/quest-books",
    "headless/action-catalogs",
    "groups",
    // The game-specific voice extractor the mod authors + ships in its profile
    // bundle (named in profile.json's `voice.extractor`). Cloning runs it to pull
    // per-NPC reference audio from THAT game's data, so the clone system travels
    // with the profile instead of being baked into chasm per game. A single fixed
    // filename (not arbitrary `*.py`) so a bundle can't smuggle in other scripts.
    "extract_voices.py",
];

/// Per-user runtime / transient entries that must NEVER be copied even if a
/// (malformed) source contains them — chasm regenerates all of these locally per
/// user / playthrough. Purely documentary/defensive: the copy is allowlist-driven
/// so these are already excluded, but this makes the intent explicit and backs the
/// "denylisted dirs are not copied" test.
pub const DENYLIST: &[&str] = &[
    "chats",
    "group chats",
    "embed-cache",
    "vectors",
    "voices",
    "headless/live-chats.json",
    "headless/save-sync",
    "headless/world-state.json",
];

/// What [`import_bundle`] did with one bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportAction {
    /// The profile was absent and the bundle was installed fresh.
    Installed,
    /// A profile existed but the source bundle was newer; it was replaced.
    Updated,
    /// A profile existed at an equal-or-newer bundle version; left untouched.
    SkippedUpToDate,
    /// The bundle was invalid (missing/unreadable `profile.json`, unsafe id, or an
    /// IO failure during copy). Carries a human reason for the log.
    Rejected(String),
}

/// The outcome of importing one bundle: the (best-effort) profile id and what
/// happened. `id` is empty when the bundle was rejected before an id was known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportOutcome {
    pub id: String,
    pub action: ImportAction,
}

impl ImportOutcome {
    fn rejected(id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            action: ImportAction::Rejected(reason.into()),
        }
    }
}

/// True when `id` is a safe single-segment slug: non-empty, and free of path
/// separators or `..` traversal (so it can only ever name a direct child of
/// `profiles/`). Also rejects a leading/trailing dot form and absolute-ish inputs.
fn is_safe_slug(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && !id.contains('/')
        && !id.contains('\\')
        && !id.contains("..")
        && !id.contains('\0')
}

/// Reads the `bundleVersion` from a parsed profile — a first-class top-level key
/// captured in `extra` (it is not a typed field on [`GameProfile`]). Accepts an
/// integer or an integer-valued float; anything else (absent/garbage) is `None`,
/// treated as version 0 by the comparison so a versioned source always wins over
/// an unversioned installed profile.
fn bundle_version(profile: &GameProfile) -> Option<u64> {
    profile
        .extra
        .get("bundleVersion")
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
}

/// Imports one profile bundle folder into `profiles_dir`.
///
/// `source_bundle_dir` is the bundle root (the folder that directly contains
/// `profile.json`, e.g. `<staged>/fallout-new-vegas/`). See the module docs for
/// the full algorithm. Best-effort and non-panicking: any failure is returned as
/// [`ImportAction::Rejected`], never an `Err`/panic, so a bad bundle can't take
/// down the connect path.
pub fn import_bundle(source_bundle_dir: &Path, profiles_dir: &Path) -> ImportOutcome {
    // 1. Read + parse the manifest.
    let manifest_path = source_bundle_dir.join("profile.json");
    let text = match fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(err) => {
            return ImportOutcome::rejected("", format!("missing/unreadable profile.json: {err}"))
        }
    };
    let profile: GameProfile = match serde_json::from_str(&text) {
        Ok(profile) => profile,
        Err(err) => return ImportOutcome::rejected("", format!("invalid profile.json: {err}")),
    };
    let id = profile.id.trim().to_string();

    // 2. Validate the id is a safe slug.
    if !is_safe_slug(&id) {
        return ImportOutcome::rejected(
            id.clone(),
            format!("unsafe profile id {id:?} (must be a simple slug)"),
        );
    }

    // 3. Version-compare against any already-installed profile.
    let dest_dir = profiles_dir.join(&id);
    let source_version = bundle_version(&profile).unwrap_or(0);
    let existing = GameProfile::read(profiles_dir, &id);
    let action = match &existing {
        Some(installed) => {
            let installed_version = bundle_version(installed).unwrap_or(0);
            if source_version > installed_version {
                ImportAction::Updated
            } else {
                return ImportOutcome {
                    id,
                    action: ImportAction::SkippedUpToDate,
                };
            }
        }
        None => ImportAction::Installed,
    };

    // 4 + 5. Copy the allowlisted content into a temp dir, then atomically swap in.
    if let Err(reason) = install_atomically(source_bundle_dir, profiles_dir, &dest_dir) {
        return ImportOutcome::rejected(id, reason);
    }

    ImportOutcome { id, action }
}

/// Assembles the allowlisted content of `source_bundle_dir` into a sibling
/// `<dest>.tmp/` staging dir, then renames it over `dest`. Returns `Err(reason)`
/// on any IO failure (leaving the previous `dest` intact where possible).
fn install_atomically(
    source_bundle_dir: &Path,
    profiles_dir: &Path,
    dest_dir: &Path,
) -> Result<(), String> {
    // Sibling `<id>.tmp` under profiles_dir so the final rename is same-volume.
    let tmp_dir = {
        let name = dest_dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "destination has no file name".to_string())?;
        profiles_dir.join(format!("{name}.tmp"))
    };

    // Start from a clean tmp (a leftover from a crashed prior run must not leak in).
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir failed: {e}"))?;

    // Copy each allowlisted entry that exists in the source.
    for rel in ALLOWLIST {
        let src = join_rel(source_bundle_dir, rel);
        if !src.exists() {
            continue;
        }
        let dst = join_rel(&tmp_dir, rel);
        let copy_result = if src.is_dir() {
            copy_dir_recursive(&src, &dst)
        } else {
            copy_file(&src, &dst)
        };
        if let Err(e) = copy_result {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(format!("copy {rel:?} failed: {e}"));
        }
    }

    // Atomic swap: remove any existing dest, then rename tmp into place. (Windows
    // has no atomic replace-directory rename, so we remove-then-rename; the window
    // is tiny and the tmp dir is fully assembled before we touch dest.)
    if dest_dir.exists() {
        if let Err(e) = fs::remove_dir_all(dest_dir) {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(format!("remove existing profile failed: {e}"));
        }
    }
    if let Err(e) = fs::rename(&tmp_dir, dest_dir) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err(format!("rename tmp into place failed: {e}"));
    }
    Ok(())
}

/// Joins a `/`-separated relative allowlist entry onto `base` component by
/// component, so it maps to the right nested path on any platform.
fn join_rel(base: &Path, rel: &str) -> PathBuf {
    let mut path = base.to_path_buf();
    for part in rel.split('/') {
        path.push(part);
    }
    path
}

/// Copies a single file, creating parent dirs as needed.
fn copy_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(src, dst)?;
    Ok(())
}

/// Recursively copies a directory tree. Sub-directories and files are copied as-is
/// — the allowlist already scoped us to an authored-content root, so everything
/// under it belongs to that content (e.g. every card under `characters/`).
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Imports every bundle found directly under `source_root`.
///
/// A child of `source_root` is treated as a bundle iff it is a directory holding a
/// `profile.json`. Non-bundle children are silently ignored. Returns one
/// [`ImportOutcome`] per discovered bundle (empty when `source_root` is
/// missing/empty). Best-effort: never panics.
pub fn import_from_source_root(source_root: &Path, profiles_dir: &Path) -> Vec<ImportOutcome> {
    let mut outcomes = Vec::new();
    let entries = match fs::read_dir(source_root) {
        Ok(entries) => entries,
        Err(_) => return outcomes, // No staged bundles (the common case).
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("profile.json").is_file() {
            outcomes.push(import_bundle(&path, profiles_dir));
        }
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    /// A minimal valid bundle at `<root>/<id>/` with the given bundleVersion and a
    /// character card, plus a denylisted runtime dir that must NOT be copied.
    fn make_bundle(root: &Path, id: &str, version: u64) {
        let bundle = root.join(id);
        write(
            &bundle.join("profile.json"),
            &format!(r#"{{"id":"{id}","name":"Test Game","bundleVersion":{version}}}"#),
        );
        write(&bundle.join("characters").join("Todd.png"), "PNGDATA");
        write(&bundle.join("worlds").join("Lore.json"), "{}");
        // Denylisted runtime data present in the source — must be skipped.
        write(&bundle.join("chats").join("session.jsonl"), "runtime");
        write(
            &bundle.join("headless").join("save-sync").join("snap.json"),
            "runtime",
        );
        write(&bundle.join("headless").join("world-state.json"), "runtime");
        write(&bundle.join("embed-cache").join("v.bin"), "runtime");
    }

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "chasm-import-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn installs_into_empty_profiles() {
        let base = tmp("install");
        let source = base.join("source");
        let profiles = base.join("profiles");
        make_bundle(&source, "testgame", 1);

        let outcome = import_bundle(&source.join("testgame"), &profiles);
        assert_eq!(outcome.id, "testgame");
        assert_eq!(outcome.action, ImportAction::Installed);

        // Allowlisted content landed.
        assert!(profiles.join("testgame").join("profile.json").is_file());
        assert!(profiles
            .join("testgame")
            .join("characters")
            .join("Todd.png")
            .is_file());
        assert!(profiles
            .join("testgame")
            .join("worlds")
            .join("Lore.json")
            .is_file());

        // Denylisted runtime data did NOT.
        assert!(!profiles.join("testgame").join("chats").exists());
        assert!(!profiles.join("testgame").join("embed-cache").exists());
        assert!(!profiles
            .join("testgame")
            .join("headless")
            .join("save-sync")
            .exists());
        assert!(!profiles
            .join("testgame")
            .join("headless")
            .join("world-state.json")
            .exists());
        // The tmp staging dir was consumed by the rename.
        assert!(!profiles.join("testgame.tmp").exists());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn skips_when_same_version() {
        let base = tmp("skip");
        let source = base.join("source");
        let profiles = base.join("profiles");
        make_bundle(&source, "testgame", 2);

        assert_eq!(
            import_bundle(&source.join("testgame"), &profiles).action,
            ImportAction::Installed
        );
        // A user edit that an equal-version reconnect must not clobber.
        write(
            &profiles.join("testgame").join("characters").join("edit.txt"),
            "user edit",
        );
        assert_eq!(
            import_bundle(&source.join("testgame"), &profiles).action,
            ImportAction::SkippedUpToDate
        );
        assert!(profiles
            .join("testgame")
            .join("characters")
            .join("edit.txt")
            .is_file());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn updates_when_newer() {
        let base = tmp("update");
        let source_v1 = base.join("source-v1");
        let source_v2 = base.join("source-v2");
        let profiles = base.join("profiles");
        make_bundle(&source_v1, "testgame", 1);
        make_bundle(&source_v2, "testgame", 3);

        assert_eq!(
            import_bundle(&source_v1.join("testgame"), &profiles).action,
            ImportAction::Installed
        );
        let outcome = import_bundle(&source_v2.join("testgame"), &profiles);
        assert_eq!(outcome.action, ImportAction::Updated);

        // The manifest now reflects the newer version.
        let updated = GameProfile::read(&profiles, "testgame").unwrap();
        assert_eq!(bundle_version(&updated), Some(3));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_unsafe_id() {
        let base = tmp("reject");
        let source = base.join("source");
        let profiles = base.join("profiles");

        for bad in ["..", "../escape", "a/b", "a\\b", ""] {
            let bundle = source.join("holder");
            write(
                &bundle.join("profile.json"),
                &format!(r#"{{"id":"{}"}}"#, bad.replace('\\', "\\\\")),
            );
            let outcome = import_bundle(&bundle, &profiles);
            assert!(
                matches!(outcome.action, ImportAction::Rejected(_)),
                "id {bad:?} should be rejected, got {:?}",
                outcome.action
            );
            let _ = fs::remove_dir_all(&bundle);
        }
        // Nothing escaped into a parent.
        assert!(!base.join("escape").exists());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn source_root_imports_each_bundle_and_skips_non_bundles() {
        let base = tmp("root");
        let source = base.join("chasm-profile");
        let profiles = base.join("profiles");
        make_bundle(&source, "gamea", 1);
        make_bundle(&source, "gameb", 1);
        // A stray non-bundle dir (no profile.json) is ignored.
        fs::create_dir_all(source.join("not-a-bundle")).unwrap();

        let outcomes = import_from_source_root(&source, &profiles);
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|o| o.action == ImportAction::Installed));
        assert!(profiles.join("gamea").join("profile.json").is_file());
        assert!(profiles.join("gameb").join("profile.json").is_file());

        let _ = fs::remove_dir_all(&base);
    }
}
