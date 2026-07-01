//! Game launcher (Phase 1): detect Mod Organizer 2 + the required Fallout: New
//! Vegas mods, and resolve the headless launch command.
//!
//! This is the "users who already have MO2" path: we DETECT what's installed and
//! LINK to where missing pieces come from. Auto-download + full provisioning are
//! later phases — the [`RequiredMod`] registry already encodes each mod's source
//! so those phases can slot in without touching the detection logic here.
//!
//! Everything is overridable. Hard-coded values are *defaults* only: the MO2 exe
//! path, the instance name, the profile, and the game dir can all come from
//! [`LauncherSettings`] (persisted) or be auto-detected from the environment
//! (`%LOCALAPPDATA%`, MO2's `ModOrganizer.ini`, the Steam default).

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Serialize;

use crate::LauncherSettings;

// ---------------------------------------------------------------------------
// Defaults (this machine's ground truth; all overridable)
// ---------------------------------------------------------------------------

/// Default MO2 executable location when not set in settings / on PATH.
pub const DEFAULT_MO2_EXE: &str = r"C:\Modding\MO2\ModOrganizer.exe";

/// Default MO2 profile to launch under.
pub const DEFAULT_PROFILE: &str = "Default";

/// Default launch-executable title (the MO2 "executables" entry to run).
pub const DEFAULT_EXECUTABLE: &str = "NVSE";

/// Steam's default install dir for Fallout: New Vegas (fallback when MO2's
/// `ModOrganizer.ini` doesn't yield a readable `gamePath`).
pub const DEFAULT_GAME_DIR: &str =
    r"C:\Program Files (x86)\Steam\steamapps\common\Fallout New Vegas";

/// The script-extender loader that must exist in the game dir for a modded NVSE
/// launch. Its presence is how we detect xNVSE.
pub const NVSE_LOADER: &str = "nvse_loader.exe";

/// The base-game executable. Its presence in the game dir is how we confirm the
/// configured Fallout: New Vegas installation path actually points at the game.
pub const FALLOUT_NV_EXE: &str = "FalloutNV.exe";

/// MO2's `gameName` for a Fallout: New Vegas instance. Used to pick the FNV
/// instance when `%LOCALAPPDATA%\ModOrganizer` holds several (e.g. the user also
/// has a Skyrim instance) — we read each instance's `ModOrganizer.ini`.
pub const FNV_GAME_NAME: &str = "New Vegas";

// ---------------------------------------------------------------------------
// Required-mods registry
// ---------------------------------------------------------------------------

/// How a [`RequiredMod`] is detected on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectKind {
    /// A file relative to the game dir (e.g. `nvse_loader.exe`).
    GameFile,
    /// A mod folder under `<instance>/mods/<name>/`.
    ModFolder,
}

/// Where a required mod is obtained. Consumed by the auto-setup downloader
/// (`setup-mo2.ps1`) and the Phase 1 "Get" link in settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModSource {
    /// A GitHub `owner/repo` whose Releases page hosts the binary as a downloadable
    /// asset. The auto-setup hits the GitHub releases API, picks the asset whose
    /// name contains `asset_hint` (case-insensitive; empty = first archive asset),
    /// downloads + extracts it into the mod folder. Fully auto-installable.
    GitHubRelease {
        repo: &'static str,
        /// Substring used to pick the right asset among a release's assets (e.g.
        /// `".7z"` for xNVSE, `".zip"` otherwise). Empty = first archive asset.
        asset_hint: &'static str,
    },
    /// A subfolder of a GitHub repo that is ITSELF a ready-to-drop MO2 mod (has its
    /// own `meta.ini` + `Data`/`nvse` layout). The auto-setup downloads the repo
    /// tarball at `git_ref` and copies `subdir` into the mod folder. Used for the
    /// project's own NVBridge mod (no release; lives at `mo2-mod/NVBridge`). Fully
    /// auto-installable.
    GitHubRepoDir {
        repo: &'static str,
        git_ref: &'static str,
        subdir: &'static str,
    },
    /// A Nexus Mods page with no public GitHub binary. `modid` is the numeric id;
    /// `url` is the canonical mod page. Auto-download needs a Nexus **Premium** API
    /// key (see [`LauncherSettings::nexus_api_key`]); without one this is a guided
    /// manual step (a "Get" link to the page).
    Nexus { modid: u32, url: &'static str },
    /// Shipped with Chasm itself (the project's own mod, copied from disk).
    Bundled,
}

impl ModSource {
    /// The "Get" link for this source, or `None` for bundled mods (which show a
    /// "bundled" label instead of a link).
    pub fn url(&self) -> Option<String> {
        match self {
            ModSource::GitHubRelease { repo, .. } => {
                Some(format!("https://github.com/{repo}/releases"))
            }
            ModSource::GitHubRepoDir { repo, .. } => Some(format!("https://github.com/{repo}")),
            ModSource::Nexus { url, .. } => Some((*url).to_string()),
            ModSource::Bundled => None,
        }
    }

    /// Short human label for the source ("GitHub", "Nexus", "Bundled").
    pub fn label(&self) -> &'static str {
        match self {
            ModSource::GitHubRelease { .. } | ModSource::GitHubRepoDir { .. } => "GitHub",
            ModSource::Nexus { .. } => "Nexus",
            ModSource::Bundled => "Bundled",
        }
    }

    /// Whether the auto-setup can install this mod with NO Nexus key and no manual
    /// step: GitHub releases, GitHub repo subfolders, and bundled mods. Nexus mods
    /// are excluded (they need a Premium key or a manual download).
    pub fn auto_installable(&self) -> bool {
        matches!(
            self,
            ModSource::GitHubRelease { .. }
                | ModSource::GitHubRepoDir { .. }
                | ModSource::Bundled
        )
    }
}

/// One mod Chasm needs for a working headless FNV launch. The registry is
/// the single source of truth: Phase 1 detects + links, Phase 2 auto-downloads
/// from `source`.
#[derive(Debug, Clone)]
pub struct RequiredMod {
    /// Stable id used in routes / status maps (e.g. `nvse`, `jip_ln`).
    pub id: &'static str,
    /// Display name shown in the list (e.g. `xNVSE (script extender)`).
    pub display: &'static str,
    /// How its presence is detected.
    pub detect_kind: DetectKind,
    /// The relative path probed: a game-dir file for [`DetectKind::GameFile`], or
    /// the mod-folder name under `<instance>/mods/` for [`DetectKind::ModFolder`].
    pub detect_path: &'static str,
    /// Where it comes from (for the Phase 2 downloader + the Phase 1 link).
    pub source: ModSource,
    /// Whether a missing copy blocks launch.
    pub required: bool,
}

/// The required mods for a headless FNV launch through MO2. Order is the display
/// order in the settings list (script extender first, then NVBridge, then the
/// supporting plugins).
pub const REQUIRED_MODS: &[RequiredMod] = &[
    RequiredMod {
        id: "nvse",
        display: "xNVSE (script extender)",
        detect_kind: DetectKind::GameFile,
        detect_path: NVSE_LOADER,
        // xNVSE ships the loader as a .7z; the auto-setup extracts it straight into
        // the GAME dir (not an MO2 mod) — that's where nvse_loader.exe must live.
        source: ModSource::GitHubRelease {
            repo: "xNVSE/NVSE",
            asset_hint: ".7z",
        },
        required: true,
    },
    RequiredMod {
        id: "nvbridge",
        display: "NVBridge",
        detect_kind: DetectKind::ModFolder,
        detect_path: "NVBridge",
        // The project's own bridge mod: a ready-to-drop MO2 mod folder living at
        // `mo2-mod/NVBridge` in the plugin repo (its own meta.ini + nvse/plugins).
        source: ModSource::GitHubRepoDir {
            repo: "chasmlol/chasm-fnv",
            git_ref: "main",
            subdir: "mo2-mod/NVBridge",
        },
        required: true,
    },
    RequiredMod {
        id: "johnnyguitar",
        display: "JohnnyGuitar NVSE",
        detect_kind: DetectKind::ModFolder,
        detect_path: "JohnnyGuitar NVSE",
        source: ModSource::GitHubRelease {
            repo: "carxt/JohnnyGuitarNVSE",
            asset_hint: ".zip",
        },
        required: true,
    },
    RequiredMod {
        id: "jip_ln",
        display: "JIP LN NVSE Plugin",
        detect_kind: DetectKind::ModFolder,
        detect_path: "JIP LN NVSE Plugin",
        // Nexus-only (no GitHub release). Needs a Premium API key to auto-download;
        // otherwise a guided manual step.
        source: ModSource::Nexus {
            modid: 58277,
            url: "https://www.nexusmods.com/newvegas/mods/58277",
        },
        required: true,
    },
    RequiredMod {
        id: "showoff",
        display: "ShowOff xNVSE",
        detect_kind: DetectKind::ModFolder,
        detect_path: "ShowOff xNVSE",
        source: ModSource::GitHubRelease {
            repo: "Demorome/ShowOff-NVSE",
            asset_hint: ".zip",
        },
        required: true,
    },
    RequiredMod {
        id: "nvtf",
        display: "NVTF (New Vegas Tick Fix)",
        detect_kind: DetectKind::ModFolder,
        detect_path: "NVTF",
        // carxt/New-Vegas-Tick-Fix publishes NO GitHub releases (only git tags +
        // CI artifacts); the distributed binary lives on Nexus, so this is a Nexus
        // source like JIP LN — auto-download needs a Premium key, else manual.
        source: ModSource::Nexus {
            modid: 66537,
            url: "https://www.nexusmods.com/newvegas/mods/66537",
        },
        required: true,
    },
];

/// Detected install status of one required mod.
#[derive(Debug, Clone, Serialize)]
pub struct ModStatus {
    pub id: String,
    pub display: String,
    pub installed: bool,
    pub required: bool,
    /// `"GameFile"` / `"ModFolder"` rendered for the UI.
    pub detect_kind: DetectKind,
    /// The absolute path that was probed (so the UI can show what's missing).
    pub detect_path: String,
    /// `"GitHub"` / `"Nexus"` / `"Bundled"`.
    pub source_label: String,
    /// "Get" link, or `None` for bundled mods.
    pub source_url: Option<String>,
}

// ---------------------------------------------------------------------------
// Launcher config (auto-detected + overridable)
// ---------------------------------------------------------------------------

/// The resolved launcher configuration: every value is either an explicit
/// override from [`LauncherSettings`] or auto-detected from the environment.
#[derive(Debug, Clone, Serialize)]
pub struct LauncherConfig {
    /// Path to `ModOrganizer.exe`.
    pub mo2_exe: PathBuf,
    /// MO2 instance name (the dir under `%LOCALAPPDATA%\ModOrganizer`).
    pub instance: String,
    /// MO2 profile to launch under.
    pub profile: String,
    /// MO2 "executables" entry title to run (e.g. `NVSE`).
    pub executable: String,
    /// Base game directory (contains `nvse_loader.exe`).
    pub game_dir: PathBuf,
    /// Root of the resolved MO2 instance (`%LOCALAPPDATA%\ModOrganizer\<instance>`).
    pub instance_dir: PathBuf,
}

impl LauncherConfig {
    /// Resolves the launcher config from persisted settings + the environment.
    ///
    /// Resolution order for each field:
    /// - `mo2_exe`: settings → [`DEFAULT_MO2_EXE`] → `ModOrganizer.exe` on PATH.
    /// - `instance`: settings → the single dir under `%LOCALAPPDATA%\ModOrganizer`.
    /// - `profile` / `executable`: settings → the defaults.
    /// - `game_dir`: settings → MO2's `ModOrganizer.ini` `[General] gamePath` →
    ///   [`DEFAULT_GAME_DIR`].
    pub fn resolve(settings: &LauncherSettings) -> Self {
        let local_app_data = local_app_data();
        Self::resolve_with(settings, local_app_data.as_deref())
    }

    /// Like [`LauncherConfig::resolve`] but with an explicit `%LOCALAPPDATA%`
    /// root, so tests can point detection at a temp dir.
    pub fn resolve_with(settings: &LauncherSettings, local_app_data: Option<&Path>) -> Self {
        let mo2_exe = resolve_mo2_exe(settings);
        let mo2_root = local_app_data.map(|root| root.join("ModOrganizer"));
        let instance = resolve_instance(settings, mo2_root.as_deref());
        let instance_dir = mo2_root
            .map(|root| root.join(&instance))
            .unwrap_or_else(|| PathBuf::from(&instance));
        let profile = first_non_empty(&settings.profile, DEFAULT_PROFILE);
        let executable = first_non_empty(&settings.executable, DEFAULT_EXECUTABLE);
        let game_dir = resolve_game_dir(settings, &mo2_exe);

        Self {
            mo2_exe,
            instance,
            profile,
            executable,
            game_dir,
            instance_dir,
        }
    }

    /// The `mods/` directory of the resolved instance.
    pub fn mods_dir(&self) -> PathBuf {
        self.instance_dir.join("mods")
    }

    /// The `moshortcut://` argument MO2 takes to launch `executable` under this
    /// instance, e.g. `moshortcut://New Vegas:NVSE`.
    pub fn moshortcut_arg(&self) -> String {
        moshortcut_arg(self)
    }
}

/// `moshortcut://<instance>:<executable>` — the single arg MO2 takes to boot a
/// named executable inside a named instance (the headless launch).
pub fn moshortcut_arg(cfg: &LauncherConfig) -> String {
    format!("moshortcut://{}:{}", cfg.instance, cfg.executable)
}

/// Whether `ModOrganizer.exe` exists at the resolved path.
pub fn mo2_detected(cfg: &LauncherConfig) -> bool {
    cfg.mo2_exe.is_file()
}

/// Whether `nvse_loader.exe` exists in the resolved game dir (xNVSE installed).
pub fn nvse_detected(cfg: &LauncherConfig) -> bool {
    cfg.game_dir.join(NVSE_LOADER).is_file()
}

/// Whether `FalloutNV.exe` exists in the resolved game dir — i.e. the configured
/// installation path really is a Fallout: New Vegas install. This is the signal
/// for the "Fallout: New Vegas installation" field's Detected/Not-found status
/// (and what later no-MO2 provisioning will rely on).
pub fn falloutnv_detected(cfg: &LauncherConfig) -> bool {
    cfg.game_dir.join(FALLOUT_NV_EXE).is_file()
}

/// Detection status of every required mod against `cfg`.
///
/// `GameFile` mods are probed under the game dir; `ModFolder` mods are probed as
/// `<instance>/mods/<name>/`.
pub fn mod_status(cfg: &LauncherConfig) -> Vec<ModStatus> {
    let mods_dir = cfg.mods_dir();
    REQUIRED_MODS
        .iter()
        .map(|m| {
            let probe = match m.detect_kind {
                DetectKind::GameFile => cfg.game_dir.join(m.detect_path),
                DetectKind::ModFolder => mods_dir.join(m.detect_path),
            };
            let installed = match m.detect_kind {
                DetectKind::GameFile => probe.is_file(),
                DetectKind::ModFolder => probe.is_dir(),
            };
            ModStatus {
                id: m.id.to_string(),
                display: m.display.to_string(),
                installed,
                required: m.required,
                detect_kind: m.detect_kind,
                detect_path: probe.display().to_string(),
                source_label: m.source.label().to_string(),
                source_url: m.source.url(),
            }
        })
        .collect()
}

/// Count of required mods the auto-setup can install with no Nexus key + no manual
/// step (GitHub releases, GitHub repo subfolders, bundled), and the count that
/// still needs a Nexus key or a manual download. Used for the settings summary
/// ("auto-setup will install N of M; K need a manual step").
pub fn auto_setup_counts() -> (usize, usize) {
    let auto = REQUIRED_MODS
        .iter()
        .filter(|m| m.source.auto_installable())
        .count();
    (auto, REQUIRED_MODS.len() - auto)
}

// ---------------------------------------------------------------------------
// Auto-setup plan (consumed by scripts/setup-mo2.ps1)
// ---------------------------------------------------------------------------

/// MO2's `gameName` value, written into the generated `ModOrganizer.ini`.
pub const MO2_GAME_NAME: &str = FNV_GAME_NAME;

/// xNVSE GitHub repo, downloaded straight into the GAME dir (not a mod) because
/// `nvse_loader.exe` must sit next to `FalloutNV.exe`.
pub const XNVSE_REPO: &str = "xNVSE/NVSE";

/// One mod entry in the [`SetupPlan`]: everything `setup-mo2.ps1` needs to fetch
/// + place a single mod, with the source flattened into plain fields so the
/// PowerShell side never has to understand the Rust enum.
#[derive(Debug, Clone, Serialize)]
pub struct SetupModEntry {
    pub id: String,
    pub display: String,
    /// `github_release` | `github_repo_dir` | `nexus` | `bundled`.
    pub source_kind: String,
    /// Where the extracted/placed files land: `mod` (a folder under `mods/`) or
    /// `game` (extracted into the game dir, for xNVSE).
    pub install_target: String,
    /// The `mods/<name>` folder name (for `install_target == "mod"`).
    pub mod_folder: String,
    /// GitHub `owner/repo` for `github_release` / `github_repo_dir`.
    pub repo: String,
    /// Asset-name substring to pick the release asset (`github_release`).
    pub asset_hint: String,
    /// Git ref + subdir for `github_repo_dir`.
    pub git_ref: String,
    pub subdir: String,
    /// Nexus numeric mod id + page URL for `nexus`.
    pub nexus_modid: u32,
    pub url: String,
    /// Whether this entry can be fetched without a Nexus key / manual step.
    pub auto_installable: bool,
    /// Whether the entry should be enabled in the profile's `modlist.txt`.
    pub enabled: bool,
}

/// The full instruction set `setup-mo2.ps1` executes: where MO2 + the game live,
/// the instance/profile to build, and every mod to fetch. Serialized to a temp
/// JSON file that the script reads, so the registry stays the single source of
/// truth and the script holds none of the URLs/ids.
#[derive(Debug, Clone, Serialize)]
pub struct SetupPlan {
    /// Resolved `ModOrganizer.exe` path (may not exist yet → the script installs
    /// MO2 next to it / at its parent dir).
    pub mo2_exe: String,
    /// Game dir (must contain `FalloutNV.exe`).
    pub game_dir: String,
    pub instance: String,
    pub profile: String,
    pub executable: String,
    pub game_name: String,
    /// xNVSE GitHub repo (extracted into the game dir).
    pub xnvse_repo: String,
    /// MO2 download (GitHub releases) repo — `ModOrganizer2/modorganizer`.
    pub mo2_repo: String,
    /// Whether a Nexus key is present (the script attempts Nexus auto-downloads
    /// only when true; otherwise it logs a skip + leaves the manual step).
    pub has_nexus_key: bool,
    pub mods: Vec<SetupModEntry>,
}

/// MO2's own GitHub repo (its releases host the MO2 archive the setup downloads
/// when `ModOrganizer.exe` isn't already present).
pub const MO2_REPO: &str = "ModOrganizer2/modorganizer";

impl SetupPlan {
    /// Builds the plan from the resolved [`LauncherConfig`] + whether a Nexus key
    /// is saved. The mod order is the registry order (the load order the profile
    /// is written with).
    pub fn build(cfg: &LauncherConfig, has_nexus_key: bool) -> Self {
        let mods = REQUIRED_MODS.iter().map(SetupModEntry::from_mod).collect();
        Self {
            mo2_exe: cfg.mo2_exe.display().to_string(),
            game_dir: cfg.game_dir.display().to_string(),
            instance: cfg.instance.clone(),
            profile: cfg.profile.clone(),
            executable: cfg.executable.clone(),
            game_name: MO2_GAME_NAME.to_string(),
            xnvse_repo: XNVSE_REPO.to_string(),
            mo2_repo: MO2_REPO.to_string(),
            has_nexus_key,
            mods,
        }
    }
}

impl SetupModEntry {
    /// Flattens a registry [`RequiredMod`] into a script-friendly entry.
    fn from_mod(m: &RequiredMod) -> Self {
        // xNVSE installs into the game dir (its loader sits beside FalloutNV.exe);
        // every other mod is an MO2 mod folder.
        let install_target = if m.id == "nvse" { "game" } else { "mod" };
        let mut entry = SetupModEntry {
            id: m.id.to_string(),
            display: m.display.to_string(),
            source_kind: String::new(),
            install_target: install_target.to_string(),
            mod_folder: if install_target == "mod" {
                m.detect_path.to_string()
            } else {
                String::new()
            },
            repo: String::new(),
            asset_hint: String::new(),
            git_ref: String::new(),
            subdir: String::new(),
            nexus_modid: 0,
            url: m.source.url().unwrap_or_default(),
            auto_installable: m.source.auto_installable(),
            enabled: true,
        };
        match &m.source {
            ModSource::GitHubRelease { repo, asset_hint } => {
                entry.source_kind = "github_release".to_string();
                entry.repo = (*repo).to_string();
                entry.asset_hint = (*asset_hint).to_string();
            }
            ModSource::GitHubRepoDir {
                repo,
                git_ref,
                subdir,
            } => {
                entry.source_kind = "github_repo_dir".to_string();
                entry.repo = (*repo).to_string();
                entry.git_ref = (*git_ref).to_string();
                entry.subdir = (*subdir).to_string();
            }
            ModSource::Nexus { modid, .. } => {
                entry.source_kind = "nexus".to_string();
                entry.nexus_modid = *modid;
            }
            ModSource::Bundled => {
                entry.source_kind = "bundled".to_string();
            }
        }
        entry
    }
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

/// `%LOCALAPPDATA%` as a path, read from the environment.
pub fn local_app_data() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
}

/// First of `value` (trimmed) / `fallback` that is non-empty.
fn first_non_empty(value: &str, fallback: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Resolves the MO2 exe: settings override → default path → PATH lookup.
fn resolve_mo2_exe(settings: &LauncherSettings) -> PathBuf {
    let configured = settings.mo2_exe.trim();
    if !configured.is_empty() {
        return PathBuf::from(configured);
    }
    let default = PathBuf::from(DEFAULT_MO2_EXE);
    if default.is_file() {
        return default;
    }
    if let Some(found) = which_on_path("ModOrganizer.exe") {
        return found;
    }
    default
}

/// Resolves the MO2 instance name: settings override → the single instance under
/// `%LOCALAPPDATA%\ModOrganizer` → (when several exist) the one whose
/// `ModOrganizer.ini` declares `gameName=New Vegas` → empty when undecidable.
fn resolve_instance(settings: &LauncherSettings, mo2_root: Option<&Path>) -> String {
    let configured = settings.instance.trim();
    if !configured.is_empty() {
        return configured.to_string();
    }
    mo2_root.and_then(detect_instance_name).unwrap_or_default()
}

/// Picks the MO2 instance dir under `root`. With exactly one subdirectory that's
/// the instance. With several, prefer the one whose `ModOrganizer.ini` says
/// `gameName=New Vegas` (so a user with multiple instances still resolves FNV);
/// if none match, it's ambiguous → `None` (require an explicit setting).
fn detect_instance_name(root: &Path) -> Option<String> {
    let subdirs: Vec<String> = fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .collect();
    match subdirs.as_slice() {
        [] => None,
        [only] => Some(only.clone()),
        many => many
            .iter()
            .find(|name| {
                ini_general_value(&root.join(name).join("ModOrganizer.ini"), "gameName")
                    .is_some_and(|game| game.eq_ignore_ascii_case(FNV_GAME_NAME))
            })
            .cloned(),
    }
}

/// Reads a raw `[General]` key from a `ModOrganizer.ini` (trimmed value), if
/// present. Used to read `gameName` when disambiguating instances.
fn ini_general_value(ini: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(ini).ok()?;
    let mut in_general = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_general = line.eq_ignore_ascii_case("[General]");
            continue;
        }
        if !in_general {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim().eq_ignore_ascii_case(key) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

/// Resolves the base game dir: settings override → MO2's `ModOrganizer.ini`
/// `[General] gamePath` (read relative to the MO2 exe) → the Steam default.
fn resolve_game_dir(settings: &LauncherSettings, mo2_exe: &Path) -> PathBuf {
    let configured = settings.game_dir.trim();
    if !configured.is_empty() {
        return PathBuf::from(configured);
    }
    if let Some(parent) = mo2_exe.parent() {
        if let Some(path) = game_path_from_ini(&parent.join("ModOrganizer.ini")) {
            return path;
        }
    }
    PathBuf::from(DEFAULT_GAME_DIR)
}

/// Parses `[General] gamePath` out of a `ModOrganizer.ini`, if readable. MO2
/// writes the value as a possibly-quoted, possibly `@ByteArray(...)`-wrapped,
/// `\\`-escaped Windows path; we unwrap and unescape it.
fn game_path_from_ini(ini: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(ini).ok()?;
    let mut in_general = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_general = line.eq_ignore_ascii_case("[General]");
            continue;
        }
        if !in_general {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("gamePath") {
            let cleaned = clean_ini_path(value.trim());
            if !cleaned.is_empty() {
                return Some(PathBuf::from(cleaned));
            }
        }
    }
    None
}

/// Cleans an MO2 INI path value: strips an `@ByteArray(...)` wrapper and
/// surrounding quotes, then collapses `\\` escapes to single backslashes.
fn clean_ini_path(value: &str) -> String {
    let mut v = value.trim();
    if let Some(inner) = v
        .strip_prefix("@ByteArray(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        v = inner.trim();
    }
    let v = v.trim_matches('"');
    v.replace("\\\\", "\\")
}

/// Finds `name` on the `PATH`, returning the first existing match.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> LauncherSettings {
        LauncherSettings::default()
    }

    #[test]
    fn moshortcut_arg_format() {
        let cfg = LauncherConfig {
            mo2_exe: PathBuf::from(DEFAULT_MO2_EXE),
            instance: "New Vegas".to_string(),
            profile: "Default".to_string(),
            executable: "NVSE".to_string(),
            game_dir: PathBuf::from(DEFAULT_GAME_DIR),
            instance_dir: PathBuf::from(r"C:\x\New Vegas"),
        };
        assert_eq!(cfg.moshortcut_arg(), "moshortcut://New Vegas:NVSE");
        assert_eq!(moshortcut_arg(&cfg), "moshortcut://New Vegas:NVSE");
    }

    #[test]
    fn settings_override_everything() {
        let mut s = settings();
        s.mo2_exe = r"D:\MO2\ModOrganizer.exe".to_string();
        s.instance = "Custom".to_string();
        s.profile = "Speedrun".to_string();
        s.executable = "FOSE".to_string();
        s.game_dir = r"D:\Games\FNV".to_string();
        // No %LOCALAPPDATA% needed: every field is overridden.
        let cfg = LauncherConfig::resolve_with(&s, None);
        assert_eq!(cfg.mo2_exe, PathBuf::from(r"D:\MO2\ModOrganizer.exe"));
        assert_eq!(cfg.instance, "Custom");
        assert_eq!(cfg.profile, "Speedrun");
        assert_eq!(cfg.executable, "FOSE");
        assert_eq!(cfg.game_dir, PathBuf::from(r"D:\Games\FNV"));
        assert_eq!(cfg.moshortcut_arg(), "moshortcut://Custom:FOSE");
    }

    #[test]
    fn instance_autodetected_from_single_subdir() {
        let tmp = std::env::temp_dir().join(format!("sb-launcher-inst-{}", std::process::id()));
        let mo2 = tmp.join("ModOrganizer").join("New Vegas");
        fs::create_dir_all(&mo2).unwrap();
        let cfg = LauncherConfig::resolve_with(&settings(), Some(&tmp));
        assert_eq!(cfg.instance, "New Vegas");
        assert_eq!(cfg.instance_dir, mo2);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn instance_ambiguous_when_multiple_subdirs() {
        let tmp = std::env::temp_dir().join(format!("sb-launcher-amb-{}", std::process::id()));
        let root = tmp.join("ModOrganizer");
        // Two instances with no game-identifying INI → can't auto-pick.
        fs::create_dir_all(root.join("InstanceA")).unwrap();
        fs::create_dir_all(root.join("InstanceB")).unwrap();
        let cfg = LauncherConfig::resolve_with(&settings(), Some(&tmp));
        assert_eq!(cfg.instance, "");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn instance_picks_fnv_among_multiple_by_ini() {
        // The real machine: New Vegas + Skyrim instances side by side. We pick
        // the one whose ModOrganizer.ini declares gameName=New Vegas.
        let tmp = std::env::temp_dir().join(format!("sb-launcher-fnvpick-{}", std::process::id()));
        let root = tmp.join("ModOrganizer");
        let nv = root.join("New Vegas");
        let sky = root.join("Skyrim Special Edition");
        fs::create_dir_all(&nv).unwrap();
        fs::create_dir_all(&sky).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();
        fs::write(
            nv.join("ModOrganizer.ini"),
            "[General]\ngameName=New Vegas\n",
        )
        .unwrap();
        fs::write(
            sky.join("ModOrganizer.ini"),
            "[General]\ngameName=Skyrim Special Edition\n",
        )
        .unwrap();
        let cfg = LauncherConfig::resolve_with(&settings(), Some(&tmp));
        assert_eq!(cfg.instance, "New Vegas");
        assert_eq!(cfg.instance_dir, nv);
        assert_eq!(cfg.moshortcut_arg(), "moshortcut://New Vegas:NVSE");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn mod_status_detects_present_and_absent() {
        let tmp = std::env::temp_dir().join(format!("sb-launcher-mods-{}", std::process::id()));
        let game = tmp.join("game");
        let instance = tmp.join("instance");
        let mods = instance.join("mods");
        fs::create_dir_all(&mods).unwrap();
        fs::create_dir_all(&game).unwrap();
        // Present: nvse_loader.exe (GameFile) + two mod folders.
        fs::write(game.join(NVSE_LOADER), b"x").unwrap();
        fs::create_dir_all(mods.join("NVBridge")).unwrap();
        fs::create_dir_all(mods.join("JohnnyGuitar NVSE")).unwrap();
        // Absent: JIP LN, ShowOff, NVTF folders not created.

        let cfg = LauncherConfig {
            mo2_exe: PathBuf::from(DEFAULT_MO2_EXE),
            instance: "instance".to_string(),
            profile: "Default".to_string(),
            executable: "NVSE".to_string(),
            game_dir: game.clone(),
            instance_dir: instance.clone(),
        };
        let statuses = mod_status(&cfg);
        assert_eq!(statuses.len(), REQUIRED_MODS.len());
        let by_id = |id: &str| statuses.iter().find(|s| s.id == id).unwrap();
        assert!(by_id("nvse").installed, "nvse_loader.exe should be found");
        assert!(by_id("nvbridge").installed);
        assert!(by_id("johnnyguitar").installed);
        assert!(!by_id("jip_ln").installed, "JIP LN folder absent");
        assert!(!by_id("showoff").installed);
        assert!(!by_id("nvtf").installed);
        assert!(nvse_detected(&cfg));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn registry_sources_and_links() {
        // nvse → GitHub releases link.
        let nvse = REQUIRED_MODS.iter().find(|m| m.id == "nvse").unwrap();
        assert_eq!(
            nvse.source.url().as_deref(),
            Some("https://github.com/xNVSE/NVSE/releases")
        );
        // nvbridge → GitHub repo subfolder (the project's own mod), auto-installable.
        let nb = REQUIRED_MODS.iter().find(|m| m.id == "nvbridge").unwrap();
        match nb.source {
            ModSource::GitHubRepoDir {
                repo,
                git_ref,
                subdir,
            } => {
                assert_eq!(repo, "chasmlol/chasm-fnv");
                assert_eq!(git_ref, "main");
                assert_eq!(subdir, "mo2-mod/NVBridge");
            }
            _ => panic!("nvbridge should be a GitHubRepoDir source"),
        }
        assert!(nb.source.auto_installable());
        // jip_ln → Nexus page link with the right modid; NOT auto-installable.
        let jip = REQUIRED_MODS.iter().find(|m| m.id == "jip_ln").unwrap();
        match jip.source {
            ModSource::Nexus { modid, url } => {
                assert_eq!(modid, 58277);
                assert_eq!(url, "https://www.nexusmods.com/newvegas/mods/58277");
            }
            _ => panic!("jip_ln should be a Nexus source"),
        }
        assert!(!jip.source.auto_installable());
        // All six are required.
        assert_eq!(REQUIRED_MODS.len(), 6);
        assert!(REQUIRED_MODS.iter().all(|m| m.required));
        // Exactly two Nexus-only mods (JIP LN + NVTF); the other four auto-install.
        let auto = REQUIRED_MODS
            .iter()
            .filter(|m| m.source.auto_installable())
            .count();
        assert_eq!(auto, 4, "nvse, nvbridge, johnnyguitar, showoff auto-install");
    }

    #[test]
    fn parses_game_path_from_ini() {
        let tmp = std::env::temp_dir().join(format!("sb-launcher-ini-{}", std::process::id()));
        let mo2dir = tmp.join("MO2");
        fs::create_dir_all(&mo2dir).unwrap();
        fs::write(
            mo2dir.join("ModOrganizer.ini"),
            "[General]\ngameName=New Vegas\ngamePath=@ByteArray(D:\\\\Games\\\\FNV)\n",
        )
        .unwrap();
        let mut s = settings();
        s.mo2_exe = mo2dir.join("ModOrganizer.exe").display().to_string();
        // game_dir not set → resolved from the ini next to the exe.
        let cfg = LauncherConfig::resolve_with(&s, None);
        assert_eq!(cfg.game_dir, PathBuf::from(r"D:\Games\FNV"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn falloutnv_exe_drives_game_dir_detection() {
        let tmp = std::env::temp_dir().join(format!("sb-launcher-fnv-{}", std::process::id()));
        let game = tmp.join("game");
        fs::create_dir_all(&game).unwrap();
        let cfg = LauncherConfig {
            mo2_exe: PathBuf::from(DEFAULT_MO2_EXE),
            instance: "i".to_string(),
            profile: "Default".to_string(),
            executable: "NVSE".to_string(),
            game_dir: game.clone(),
            instance_dir: tmp.join("instance"),
        };
        // Empty dir → not a real install.
        assert!(!falloutnv_detected(&cfg));
        fs::write(game.join(FALLOUT_NV_EXE), b"x").unwrap();
        assert!(falloutnv_detected(&cfg));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn setup_written_ini_round_trips_through_resolver() {
        // The auto-setup (setup-mo2.ps1) writes gamePath as @ByteArray(<path with
        // each backslash DOUBLED>). Confirm the existing resolver reads that exact
        // form back to the original path — the format contract between the two.
        let tmp = std::env::temp_dir().join(format!("sb-launcher-rt-{}", std::process::id()));
        let mo2dir = tmp.join("MO2");
        fs::create_dir_all(&mo2dir).unwrap();
        // Mirror the script's escaping: C:\Games\FNV -> C:\\Games\\FNV.
        let game = r"C:\Games\FNV";
        let doubled = game.replace('\\', r"\\");
        fs::write(
            mo2dir.join("ModOrganizer.ini"),
            format!("[General]\ngameName=New Vegas\ngamePath=@ByteArray({doubled})\nselected_profile=@ByteArray(Default)\n"),
        )
        .unwrap();
        let mut s = settings();
        s.mo2_exe = mo2dir.join("ModOrganizer.exe").display().to_string();
        let cfg = LauncherConfig::resolve_with(&s, None);
        assert_eq!(cfg.game_dir, PathBuf::from(game));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn auto_setup_counts_match_registry() {
        let (auto, manual) = auto_setup_counts();
        assert_eq!(auto, 4);
        assert_eq!(manual, 2);
        assert_eq!(auto + manual, REQUIRED_MODS.len());
    }

    #[test]
    fn setup_plan_flattens_every_source_kind() {
        let cfg = LauncherConfig {
            mo2_exe: PathBuf::from(r"C:\MO2\ModOrganizer.exe"),
            instance: "New Vegas".to_string(),
            profile: "Default".to_string(),
            executable: "NVSE".to_string(),
            game_dir: PathBuf::from(r"C:\Games\FNV"),
            instance_dir: PathBuf::from(r"C:\MO2\New Vegas"),
        };
        let plan = SetupPlan::build(&cfg, false);
        assert_eq!(plan.mods.len(), REQUIRED_MODS.len());
        assert!(!plan.has_nexus_key);
        assert_eq!(plan.game_name, "New Vegas");
        assert_eq!(plan.executable, "NVSE");

        let by_id = |id: &str| plan.mods.iter().find(|m| m.id == id).unwrap();
        // xNVSE → github_release, .7z hint, extracted into the GAME dir.
        let nvse = by_id("nvse");
        assert_eq!(nvse.source_kind, "github_release");
        assert_eq!(nvse.asset_hint, ".7z");
        assert_eq!(nvse.install_target, "game");
        assert!(nvse.mod_folder.is_empty());
        // NVBridge → github_repo_dir with the right subdir, an MO2 mod folder.
        let nb = by_id("nvbridge");
        assert_eq!(nb.source_kind, "github_repo_dir");
        assert_eq!(nb.repo, "chasmlol/chasm-fnv");
        assert_eq!(nb.subdir, "mo2-mod/NVBridge");
        assert_eq!(nb.install_target, "mod");
        assert_eq!(nb.mod_folder, "NVBridge");
        // ShowOff → github_release into a mod folder.
        let so = by_id("showoff");
        assert_eq!(so.source_kind, "github_release");
        assert_eq!(so.mod_folder, "ShowOff xNVSE");
        // JIP LN + NVTF → nexus, not auto-installable.
        assert_eq!(by_id("jip_ln").source_kind, "nexus");
        assert_eq!(by_id("jip_ln").nexus_modid, 58277);
        assert!(!by_id("jip_ln").auto_installable);
        assert_eq!(by_id("nvtf").source_kind, "nexus");
        assert_eq!(by_id("nvtf").nexus_modid, 66537);
        assert!(!by_id("nvtf").auto_installable);
    }

    #[test]
    fn falls_back_to_steam_default_game_dir() {
        // No setting, no readable ini → Steam default.
        let mut s = settings();
        s.mo2_exe = r"C:\nope\ModOrganizer.exe".to_string();
        let cfg = LauncherConfig::resolve_with(&s, None);
        assert_eq!(cfg.game_dir, PathBuf::from(DEFAULT_GAME_DIR));
    }
}
