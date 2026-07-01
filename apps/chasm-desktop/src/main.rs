// Hide the extra console window on Windows release builds (the tray is the UI).
// In dev/debug the console stays so the backend + bridge + lifecycle logs show.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    chasm_desktop_lib::run();
}
