<#
.SYNOPSIS
    Resolves an absolute path to a usable `uv.exe` (the Astral Python package
    manager), so the engine-install scripts never depend on a system `uv`/Python
    being on PATH. Dot-source this and call `Resolve-Uv`:

        . "$PSScriptRoot\ensure-uv.ps1"
        $uv = Resolve-Uv -Log { param($m) Log $m }

.DESCRIPTION
    Resolution order (first hit wins):
      1. `uv` already on PATH.
      2. A bundled `tools\uv\uv.exe` shipped beside the app. This script lives in
         the install's `<resource>\scripts` dir, so the sibling is `..\tools\uv`.
      3. A per-user bootstrap: download + run the official Astral installer into
         `%LOCALAPPDATA%\chasm\uv`, then use the `uv.exe` it produces. Reused on
         later runs (no repeat download).

    Returns the absolute path to `uv.exe`, or throws if none can be obtained (e.g.
    no bundled binary AND the bootstrap download failed because there's no network).
    The caller turns that throw into a `.failed` marker the UI surfaces.
#>

function Resolve-Uv {
    param(
        # Optional logger scriptblock: & $Log "message".
        [scriptblock]$Log = { param($m) }
    )

    # 1) uv on PATH.
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) {
        & $Log "[uv] using PATH uv ($($onPath.Source))"
        return $onPath.Source
    }

    # 2) Bundled tools\uv\uv.exe (sibling of this scripts dir in the resource dir).
    $bundled = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\uv\uv.exe'
    if (Test-Path $bundled) {
        & $Log "[uv] using bundled uv ($bundled)"
        return $bundled
    }

    # 3) Per-user bootstrap into %LOCALAPPDATA%\chasm\uv.
    $home = if ($env:CHASM_HOME) { $env:CHASM_HOME }
            elseif ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'chasm' }
            elseif ($env:APPDATA) { Join-Path $env:APPDATA 'chasm' }
            else { Join-Path $env:TEMP 'chasm' }
    $uvHome = Join-Path $home 'uv'
    $uvExe = Join-Path $uvHome 'uv.exe'
    if (Test-Path $uvExe) {
        & $Log "[uv] using bootstrapped uv ($uvExe)"
        return $uvExe
    }

    & $Log "[uv] no uv on PATH or bundled; bootstrapping the official installer"
    New-Item -ItemType Directory -Force -Path $uvHome | Out-Null
    # The official installer respects UV_INSTALL_DIR for the binary location and
    # UV_NO_MODIFY_PATH so it doesn't touch the user's PATH (we call it by abs path).
    $env:UV_INSTALL_DIR = $uvHome
    $env:UV_NO_MODIFY_PATH = '1'
    try {
        Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -UseBasicParsing | Invoke-Expression
    }
    catch {
        throw "uv not found and bootstrap failed (no network?): $($_.Exception.Message)"
    }
    if (-not (Test-Path $uvExe)) {
        # Some installer versions land it directly under UV_INSTALL_DIR; re-probe.
        $found = Get-ChildItem -Path $uvHome -Recurse -Filter 'uv.exe' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { $uvExe = $found.FullName }
    }
    if (-not (Test-Path $uvExe)) {
        throw "uv bootstrap completed but uv.exe was not found under $uvHome"
    }
    & $Log "[uv] bootstrapped uv ($uvExe)"
    return $uvExe
}
