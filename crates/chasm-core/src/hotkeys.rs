//! In-game hotkey bindings: the canonical key-name ↔ Win32 virtual-key mapping
//! and the bridge delivery file the NVSE plugin polls.
//!
//! WIRE FORMAT (`<bridge_root>\control\hotkeys.cfg`, written by chasm, read by
//! the plugin's `LoadHotkeysConfigIfNeeded`):
//!
//! ```text
//! # NVBridge hotkeys v1 -- written by chasm; do not edit while chasm runs.
//! version=1
//! chat_vk=13
//! voice_vk=18
//! admin_chat_vk=79
//! admin_voice_vk=72
//! ```
//!
//! Values are DECIMAL Win32 virtual-key codes (the plugin polls
//! `GetAsyncKeyState(vk)`). The plugin re-reads the file when its mtime changes
//! (1s poll, same pattern as `native_debug.cfg`), so a save in the UI takes
//! effect in a running game within about a second. A missing file or an
//! out-of-range value leaves the plugin on its built-in defaults
//! (Enter / Alt / O / H).
//!
//! Settings store canonical key NAMES (human-readable, stable across UIs); the
//! name→code mapping lives here so it is unit-testable on the Rust side.

use std::{io, path::Path};

use crate::settings::HotkeysSettings;

/// File name of the delivery file under `<bridge_root>/control/`.
pub const BRIDGE_HOTKEYS_FILE_NAME: &str = "hotkeys.cfg";

/// Canonical key names → Win32 virtual-key codes.
///
/// The name set intentionally mirrors what the web UI's capture control can
/// produce from `KeyboardEvent.code`: letters `A`..`Z`, digits `0`..`9`,
/// `F1`..`F24`, modifiers as plain keys (`Alt`, `Ctrl`, `Shift` — left/right
/// collapsed, matching the plugin's generic `VK_MENU`-style polling), numpad
/// keys, arrows, navigation cluster, and common punctuation. `Escape` is
/// deliberately absent: the plugin reserves it for cancel.
pub fn virtual_key_code(name: &str) -> Option<u16> {
    let name = name.trim();

    // Single letter A..Z or digit 0..9: VK == ASCII uppercase.
    if name.len() == 1 {
        let ch = name.chars().next().unwrap().to_ascii_uppercase();
        if ch.is_ascii_uppercase() || ch.is_ascii_digit() {
            return Some(ch as u16);
        }
    }

    // F1..F24 → 0x70..0x87.
    if let Some(num) = name
        .strip_prefix('F')
        .or_else(|| name.strip_prefix('f'))
        .and_then(|n| n.parse::<u16>().ok())
    {
        if (1..=24).contains(&num) {
            return Some(0x70 + num - 1);
        }
    }

    // Numpad0..Numpad9 → 0x60..0x69.
    if let Some(num) = name
        .strip_prefix("Numpad")
        .and_then(|n| n.parse::<u16>().ok())
    {
        if num <= 9 {
            return Some(0x60 + num);
        }
    }

    let code: u16 = match name {
        "Enter" => 0x0D,
        "Space" => 0x20,
        "Tab" => 0x09,
        "Backspace" => 0x08,
        "Alt" => 0x12,   // VK_MENU (either Alt; the plugin's original PTT key)
        "Ctrl" => 0x11,  // VK_CONTROL
        "Shift" => 0x10, // VK_SHIFT
        "CapsLock" => 0x14,
        "Left" => 0x25,
        "Up" => 0x26,
        "Right" => 0x27,
        "Down" => 0x28,
        "Home" => 0x24,
        "End" => 0x23,
        "PageUp" => 0x21,
        "PageDown" => 0x22,
        "Insert" => 0x2D,
        "Delete" => 0x2E,
        "Pause" => 0x13,
        "ScrollLock" => 0x91,
        "NumLock" => 0x90,
        "NumpadMultiply" => 0x6A,
        "NumpadAdd" => 0x6B,
        "NumpadSubtract" => 0x6D,
        "NumpadDecimal" => 0x6E,
        "NumpadDivide" => 0x6F,
        "Semicolon" => 0xBA,     // ;:
        "Equals" => 0xBB,        // =+
        "Comma" => 0xBC,         // ,<
        "Minus" => 0xBD,         // -_
        "Period" => 0xBE,        // .>
        "Slash" => 0xBF,         // /?
        "Backquote" => 0xC0,     // `~
        "LeftBracket" => 0xDB,   // [{
        "Backslash" => 0xDC,     // \|
        "RightBracket" => 0xDD,  // ]}
        "Quote" => 0xDE,         // '"
        _ => return None,
    };
    Some(code)
}

/// A binding resolved to its wire code, falling back to the built-in default
/// when the stored name is unknown/empty (so a hand-edited settings file can
/// never deliver a dead binding to the game).
fn code_or_default(name: &str, default_name: &str) -> u16 {
    virtual_key_code(name)
        .or_else(|| virtual_key_code(default_name))
        .expect("built-in default key names always map")
}

/// The exact `hotkeys.cfg` contents for a settings snapshot (see the module
/// docs for the format). CRLF line endings to match the plugin's other bridge
/// files.
pub fn bridge_hotkeys_file_contents(hotkeys: &HotkeysSettings) -> String {
    let defaults = HotkeysSettings::default();
    format!(
        "# NVBridge hotkeys v1 -- written by chasm; do not edit while chasm runs.\r\n\
         # Values are decimal Win32 virtual-key codes.\r\n\
         version=1\r\n\
         chat_vk={}\r\n\
         voice_vk={}\r\n\
         admin_chat_vk={}\r\n\
         admin_voice_vk={}\r\n\
         reflect_vk={}\r\n",
        code_or_default(&hotkeys.enter_text, &defaults.enter_text),
        code_or_default(&hotkeys.push_to_talk, &defaults.push_to_talk),
        code_or_default(&hotkeys.todd_enter_text, &defaults.todd_enter_text),
        code_or_default(&hotkeys.todd_push_to_talk, &defaults.todd_push_to_talk),
        // The reflect key has NO built-in default: unbound (empty / unknown) is a
        // legitimate state, delivered as 0 so the plugin ignores it.
        virtual_key_code(&hotkeys.reflect).unwrap_or(0),
    )
}

/// Writes `hotkeys.cfg` under `<bridge_root>/control/`, creating the directory
/// if needed. Called on every hotkeys save and at bridge startup so the file
/// always reflects the persisted settings.
pub fn write_bridge_hotkeys_file(
    bridge_root: &Path,
    hotkeys: &HotkeysSettings,
) -> io::Result<()> {
    let control = bridge_root.join("control");
    std::fs::create_dir_all(&control)?;
    std::fs::write(
        control.join(BRIDGE_HOTKEYS_FILE_NAME),
        bridge_hotkeys_file_contents(hotkeys),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_and_digits_map_to_ascii_vk() {
        assert_eq!(virtual_key_code("A"), Some(0x41));
        assert_eq!(virtual_key_code("a"), Some(0x41)); // case-insensitive
        assert_eq!(virtual_key_code("Z"), Some(0x5A));
        assert_eq!(virtual_key_code("O"), Some(0x4F));
        assert_eq!(virtual_key_code("H"), Some(0x48));
        assert_eq!(virtual_key_code("0"), Some(0x30));
        assert_eq!(virtual_key_code("9"), Some(0x39));
    }

    #[test]
    fn function_keys_map_to_vk_f_range() {
        assert_eq!(virtual_key_code("F1"), Some(0x70));
        assert_eq!(virtual_key_code("F12"), Some(0x7B));
        assert_eq!(virtual_key_code("F24"), Some(0x87));
        assert_eq!(virtual_key_code("F25"), None);
        assert_eq!(virtual_key_code("F0"), None);
    }

    #[test]
    fn named_keys_map_to_expected_vk() {
        assert_eq!(virtual_key_code("Enter"), Some(0x0D));
        assert_eq!(virtual_key_code("Alt"), Some(0x12));
        assert_eq!(virtual_key_code("Space"), Some(0x20));
        assert_eq!(virtual_key_code("Numpad0"), Some(0x60));
        assert_eq!(virtual_key_code("Numpad9"), Some(0x69));
        assert_eq!(virtual_key_code("Backquote"), Some(0xC0));
        assert_eq!(virtual_key_code("Left"), Some(0x25));
    }

    #[test]
    fn escape_and_unknown_names_are_rejected() {
        assert_eq!(virtual_key_code("Escape"), None); // reserved for cancel
        assert_eq!(virtual_key_code(""), None);
        assert_eq!(virtual_key_code("NoSuchKey"), None);
        assert_eq!(virtual_key_code("Numpad10"), None);
    }

    #[test]
    fn default_bindings_round_trip_to_original_plugin_codes() {
        // The plugin's original hardcoded keys: VK_RETURN, VK_MENU, 'O', 'H'.
        let contents = bridge_hotkeys_file_contents(&HotkeysSettings::default());
        assert!(contents.contains("chat_vk=13\r\n"));
        assert!(contents.contains("voice_vk=18\r\n"));
        assert!(contents.contains("admin_chat_vk=79\r\n"));
        assert!(contents.contains("admin_voice_vk=72\r\n"));
        assert!(contents.contains("version=1\r\n"));
    }

    #[test]
    fn unknown_stored_name_falls_back_to_default_code() {
        let hotkeys = HotkeysSettings {
            push_to_talk: "TotallyBogus".to_string(),
            ..HotkeysSettings::default()
        };
        let contents = bridge_hotkeys_file_contents(&hotkeys);
        assert!(contents.contains("voice_vk=18\r\n")); // Alt, the default
    }

    #[test]
    fn rebound_keys_serialize_their_codes() {
        let hotkeys = HotkeysSettings {
            push_to_talk: "F".to_string(),
            enter_text: "T".to_string(),
            todd_push_to_talk: "F6".to_string(),
            todd_enter_text: "Numpad5".to_string(),
            ..HotkeysSettings::default()
        };
        let contents = bridge_hotkeys_file_contents(&hotkeys);
        assert!(contents.contains("voice_vk=70\r\n"));
        assert!(contents.contains("chat_vk=84\r\n"));
        assert!(contents.contains("admin_voice_vk=117\r\n"));
        assert!(contents.contains("admin_chat_vk=101\r\n"));
    }

    #[test]
    fn reflect_key_is_unbound_by_default_and_binds_when_set() {
        // Unbound by default → delivered as 0 (the plugin ignores it).
        let contents = bridge_hotkeys_file_contents(&HotkeysSettings::default());
        assert!(contents.contains("reflect_vk=0\r\n"));
        // A bound reflect key serializes its VK code (F10 = 0x79 = 121).
        let bound = HotkeysSettings {
            reflect: "F10".to_string(),
            ..HotkeysSettings::default()
        };
        assert!(bridge_hotkeys_file_contents(&bound).contains("reflect_vk=121\r\n"));
    }

    #[test]
    fn write_creates_control_dir_and_file() {
        let dir = std::env::temp_dir().join(format!(
            "chasm-hotkeys-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        write_bridge_hotkeys_file(&dir, &HotkeysSettings::default()).unwrap();
        let written =
            std::fs::read_to_string(dir.join("control").join(BRIDGE_HOTKEYS_FILE_NAME)).unwrap();
        assert_eq!(written, bridge_hotkeys_file_contents(&HotkeysSettings::default()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
