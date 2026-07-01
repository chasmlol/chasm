<#
.SYNOPSIS
    Downloads a retrieval embedder/reranker model's weights and writes the status
    markers the Chasm app reads (.downloading / .failed under
    <CacheDir>/.markers/<ModelId>). The weights themselves land in the
    fastembed-style <CacheDir>/models--<org>--<repo> dir, whose presence is the
    "downloaded" signal — so there is no .done marker to write.

.DESCRIPTION
    Reuses the app's own embed crate: invokes `chasm download-embed-model
    <id>`, which constructs the fastembed loader for that model, forcing the
    hf-hub download into CacheDir. No Python / extra deps required.

.NOTES
    Windows-oriented. Invoked by the app's download endpoint, or directly:
        pwsh scripts/download-embed-model.ps1 -ModelId bge-small -CacheDir .\models\embed
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [Parameter(Mandatory = $true)][string]$CacheDir
)

# hf-hub / progress bars write to stderr; don't let that abort the run. Detect
# real failure via the process exit code instead.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

$markers = Join-Path (Join-Path $CacheDir '.markers') $ModelId
New-Item -ItemType Directory -Force -Path $markers | Out-Null

$log = Join-Path $markers 'download.log'
$downloading = Join-Path $markers '.downloading'
$failed = Join-Path $markers '.failed'

Remove-Item $failed -ErrorAction SilentlyContinue
Set-Content -Path $downloading -Value "started $(Get-Date -Format o)" -Encoding utf8

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

try {
    Log "[download] model=$ModelId cacheDir=$CacheDir"

    # The model weights download into $CacheDir; point the app there so it
    # matches what the runtime retriever loads.
    $env:CHASM_EMBED_DIR = $CacheDir

    # Resolve the app binary next to this script's workspace. The release build
    # is preferred; fall back to debug, then to `cargo run` for dev checkouts.
    $root = Split-Path -Parent $PSScriptRoot
    $exeRelease = Join-Path $root 'target\release\chasm.exe'
    $exeDebug = Join-Path $root 'target\debug\chasm.exe'

    if (Test-Path $exeRelease) {
        & $exeRelease download-embed-model $ModelId *>> $log
    }
    elseif (Test-Path $exeDebug) {
        & $exeDebug download-embed-model $ModelId *>> $log
    }
    else {
        & cargo run --quiet --manifest-path (Join-Path $root 'Cargo.toml') -p chasm -- download-embed-model $ModelId *>> $log
    }

    if ($LASTEXITCODE -ne 0) { throw "download-embed-model exited $LASTEXITCODE" }

    Log "[done] $ModelId downloaded"
}
catch {
    Log "[error] $($_.Exception.Message)"
    Set-Content -Path $failed -Value "$($_.Exception.Message)" -Encoding utf8
}
finally {
    Remove-Item $downloading -ErrorAction SilentlyContinue
}
