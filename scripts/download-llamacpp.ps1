<#
.SYNOPSIS
    Downloads the llama.cpp llama-server runtime (Windows CUDA build) from the
    latest ggml-org/llama.cpp GitHub release into the folder chasm's launcher
    expects, writing the status markers the app reads (llamacpp.downloading /
    llamacpp.done / llamacpp.failed) beside it.

.NOTES
    Release assets are zips (llama-<tag>-bin-win-cuda-<ver>-x64.zip); the CUDA
    builds additionally need the runtime DLLs from the matching
    cudart-llama-bin-win-cuda-<ver>-x64.zip, so both are fetched and extracted
    into the same folder. Windows-oriented (curl.exe, falling back to
    Invoke-WebRequest). Invoked by the app's runtime-ensure path, or directly:
        pwsh scripts/download-llamacpp.ps1 -ExePath .\llamacpp\llama-server.exe
#>
[CmdletBinding()]
param(
    # Absolute (or workspace-relative) path llama-server.exe should land at. Its
    # parent dir receives the whole extracted build + the status markers.
    [Parameter(Mandatory = $true)][string]$ExePath
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

$dir = Split-Path -Parent $ExePath
if (-not $dir) { $dir = '.' }
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$log = Join-Path $dir 'llamacpp.log'
$downloading = Join-Path $dir 'llamacpp.downloading'
$done = Join-Path $dir 'llamacpp.done'
$failed = Join-Path $dir 'llamacpp.failed'

# Fresh attempt: clear terminal markers, mark as downloading.
Remove-Item $done, $failed -ErrorAction SilentlyContinue
Set-Content -Path $downloading -Value "started $(Get-Date -Format o)" -Encoding utf8

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

# The installed NVIDIA driver's CUDA version (e.g. 13.0), or $null without a GPU.
function Get-DriverCuda {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }
    $match = (& nvidia-smi 2>$null | Select-String 'CUDA Version:\s*([\d.]+)')
    if ($match) { return [version]$match.Matches[0].Groups[1].Value }
    return $null
}

# Picks the Windows x64 CUDA llama.cpp asset best suited to the driver: the
# newest CUDA build whose major version the driver supports, falling back to the
# oldest CUDA build, then the CPU build (never Vulkan/HIP/SYCL — untested here).
function Select-LlamaAsset($assets, $driverCuda) {
    $names = $assets | ForEach-Object { $_.name }
    Log "[assets] $($names -join ', ')"
    $cuda = $assets |
        Where-Object { $_.name -match '^llama-.*-bin-win-cuda-([\d.]+)-x64\.zip$' } |
        ForEach-Object {
            [pscustomobject]@{ Asset = $_; Cuda = [version]($_.name -replace '^llama-.*-bin-win-cuda-([\d.]+)-x64\.zip$', '$1') }
        } |
        Sort-Object Cuda -Descending
    if ($cuda) {
        foreach ($candidate in $cuda) {
            if (-not $driverCuda -or $driverCuda.Major -ge $candidate.Cuda.Major) {
                return $candidate.Asset
            }
        }
        # Driver older than every offered build: take the oldest CUDA build and
        # let minor-version compatibility try its luck (better than no GPU).
        return ($cuda | Select-Object -Last 1).Asset
    }
    return ($assets | Where-Object { $_.name -match '^llama-.*-bin-win-cpu-x64\.zip$' } | Select-Object -First 1)
}

# The cudart zip matching a chosen CUDA asset (ships the CUDA runtime DLLs the
# CUDA build needs beside llama-server.exe). $null for the CPU build.
function Select-CudartAsset($assets, $llamaAssetName) {
    if ($llamaAssetName -notmatch 'bin-win-cuda-([\d.]+)-x64\.zip$') { return $null }
    $ver = $Matches[1]
    return $assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$ver-x64.zip" } | Select-Object -First 1
}

function Download-File($url, $destination) {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & curl.exe -L -f -C - -o $destination $url *>> $log
        if ($LASTEXITCODE -ne 0) { throw "curl failed (exit $LASTEXITCODE) for $url" }
    }
    else {
        Log "[fallback] curl.exe not found; using Invoke-WebRequest"
        Invoke-WebRequest -Uri $url -OutFile $destination -UseBasicParsing
    }
    if (-not (Test-Path $destination)) { throw "download produced no file: $url" }
}

try {
    Log "[download] llama.cpp (llama-server) -> $ExePath"

    if (Test-Path $ExePath) {
        Log "[skip] llama-server.exe already exists"
        Set-Content -Path $done -Value "present $(Get-Date -Format o)" -Encoding utf8
        Remove-Item $downloading -ErrorAction SilentlyContinue
        return
    }

    $api = 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'
    Log "[api] $api"
    $release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'chasm-llamacpp-downloader' } -UseBasicParsing
    Log "[api] latest tag=$($release.tag_name)"

    $driverCuda = Get-DriverCuda
    Log "[gpu] driver CUDA = $driverCuda"
    $asset = Select-LlamaAsset $release.assets $driverCuda
    if (-not $asset) { throw 'no Windows x64 llama.cpp asset in the latest release' }
    Log "[asset] $($asset.name) -> $($asset.browser_download_url)"

    $zip = Join-Path $dir $asset.name
    Download-File $asset.browser_download_url $zip
    Expand-Archive -Path $zip -DestinationPath $dir -Force
    Remove-Item $zip -ErrorAction SilentlyContinue

    $cudart = Select-CudartAsset $release.assets $asset.name
    if ($cudart) {
        Log "[asset] $($cudart.name) -> $($cudart.browser_download_url)"
        $cudartZip = Join-Path $dir $cudart.name
        Download-File $cudart.browser_download_url $cudartZip
        Expand-Archive -Path $cudartZip -DestinationPath $dir -Force
        Remove-Item $cudartZip -ErrorAction SilentlyContinue
    }

    # Some releases extract flat into $dir, some into a subfolder; if flat
    # extraction didn't produce the exe at $ExePath, hoist it (plus DLLs).
    if (-not (Test-Path $ExePath)) {
        $found = Get-ChildItem -Path $dir -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
        if (-not $found) { throw 'archive did not contain llama-server.exe' }
        Log "[layout] hoisting $($found.DirectoryName) -> $dir"
        Get-ChildItem -Path $found.DirectoryName | Move-Item -Destination $dir -Force
    }
    if (-not (Test-Path $ExePath)) { throw "llama-server.exe not at $ExePath after extraction" }

    Log "[done] llama.cpp $($release.tag_name) downloaded to $dir"
    Set-Content -Path $done -Value "downloaded $($release.tag_name) $(Get-Date -Format o)" -Encoding utf8
}
catch {
    Log "[error] $($_.Exception.Message)"
    Set-Content -Path $failed -Value "$($_.Exception.Message)" -Encoding utf8
}
finally {
    Remove-Item $downloading -ErrorAction SilentlyContinue
}
