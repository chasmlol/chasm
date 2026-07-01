<#
.SYNOPSIS
    Clones the active profile's character voices with a specific TTS engine.

    1. Extract a shared reference clip per character (game profile extractor).
    2. Run the chosen engine to clone each reference -> voices/<name>/<engine>/sample.wav.

    Status is per-engine: every TTS engine must clone each character separately.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProfileDir,
    [Parameter(Mandatory = $true)][string]$EnginesDir,
    [Parameter(Mandatory = $true)][string]$VoicesDir,
    [string]$Engine = "pockettts"
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

$profileJson = Join-Path $ProfileDir 'profile.json'
$extractor = Join-Path $ProfileDir 'extract_voices.py'
$extractPy = Join-Path $EnginesDir 'pockettts\.venv\Scripts\python.exe'   # has soundfile + numpy
$enginePy = Join-Path $EnginesDir "$Engine\.venv\Scripts\python.exe"
$clonePy = Join-Path $PSScriptRoot 'tts_clone.py'
New-Item -ItemType Directory -Force -Path $VoicesDir | Out-Null
$log = Join-Path $VoicesDir 'clone.log'
"[clone] start $(Get-Date -Format o) engine=$Engine" | Out-File -FilePath $log -Encoding utf8

# 1. Extract shared reference clips (engine-agnostic).
if (Test-Path $extractPy) {
    & $extractPy $extractor --profile $profileJson --out $VoicesDir *>> $log
} else {
    "[clone] warn: extractor python missing ($extractPy)" | Out-File -FilePath $log -Append -Encoding utf8
}

# 2. Clone with the chosen engine (loads the model once, clones every reference).
if (Test-Path $enginePy) {
    & $enginePy $clonePy --engine $Engine --voices-dir $VoicesDir *>> $log
} else {
    "[clone] error: engine python missing ($enginePy) - install $Engine first" | Out-File -FilePath $log -Append -Encoding utf8
}

# 3. Reconcile per-engine markers from the profile character list.
try {
    $chars = (Get-Content $profileJson -Raw | ConvertFrom-Json).characters
    foreach ($c in $chars) {
        $dir = Join-Path (Join-Path $VoicesDir $c.name) $Engine
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Remove-Item (Join-Path $dir '.cloning') -ErrorAction SilentlyContinue
        if (Test-Path (Join-Path $dir 'sample.wav')) {
            Remove-Item (Join-Path $dir '.failed') -ErrorAction SilentlyContinue
        } else {
            Set-Content -Path (Join-Path $dir '.failed') -Value "no sample produced" -Encoding utf8
        }
    }
} catch {
    "[clone] reconcile failed: $($_.Exception.Message)" | Out-File -FilePath $log -Append -Encoding utf8
}
"[clone] done $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
