@echo off
title chasm backend
rem Run from wherever this script lives, so there are no machine-specific paths.
cd /d "%~dp0"

rem Run the FNV bridge IN-PROCESS (native Rust). The old Node helper is retired and
rem archived (see docs\retired-node-bridge.md); this flag makes chasm run the bridge
rem itself. Without it, Play would look for the now-archived Node helper and find no
rem bridge. To roll back to Node: restore the helper from the archive + remove this line.
set "CHASM_FNV_BRIDGE=1"

echo ============================================================
echo  Starting chasm backend...
echo  When it prints "listening on http://127.0.0.1:7341",
echo  open that address in your browser, then hit Play in the UI.
echo ============================================================
echo.

"target\release\chasm.exe"

echo.
echo chasm backend exited.
pause
