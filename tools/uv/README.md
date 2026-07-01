# Bundled `uv` slot

`scripts/ensure-uv.ps1` resolves the [uv](https://github.com/astral-sh/uv) Python
package manager in this order:

1. `uv` already on `PATH`.
2. `tools/uv/uv.exe` bundled beside the app (this folder).
3. A per-user bootstrap install (the official `astral.sh/uv` installer) into
   `%LOCALAPPDATA%\chasm\uv\uv.exe`, used on subsequent runs.

To ship a pinned `uv.exe` with the installer (the "just works, no network for the
toolchain" path), drop the Windows `uv.exe` here as `tools/uv/uv.exe`. It is wired
into the Tauri bundle via `apps/chasm-desktop/tauri.conf.json`
(`bundle.resources` → `"../../tools": "tools"`), so it lands in the install's
resource dir and `ensure-uv.ps1` finds it automatically.

It is intentionally NOT committed to git here (a ~63 MB binary would bloat every
clone). The bootstrap path in `ensure-uv.ps1` makes a missing `uv.exe` self-heal on
first use; the bundled slot is the preferred zero-network alternative for release
builds.
