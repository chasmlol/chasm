<#
.SYNOPSIS
    Downloads the koboldcpp runtime exe (the binary that serves the local LLM AND
    Whisper STT) from the latest LostRuins/koboldcpp GitHub release into the path
    chasm's launcher expects, writing the status markers the app reads
    (koboldcpp.downloading / koboldcpp.done / koboldcpp.failed) beside it.

.NOTES
    Windows-oriented (uses curl.exe, falling back to Invoke-WebRequest). Invoked by
    the app's runtime-ensure path, or run directly:
        pwsh scripts/download-koboldcpp.ps1 -ExePath .\koboldcpp\koboldcpp.exe
#>
[CmdletBinding()]
param(
    # Absolute (or workspace-relative) path the koboldcpp.exe should land at. Its
    # parent dir holds the status markers.
    [Parameter(Mandatory = $true)][string]$ExePath
)

# curl writes progress to stderr; under PowerShell 7.4+ that gets promoted to a
# terminating error when ErrorActionPreference is 'Stop'. Keep going and detect
# real failures via $LASTEXITCODE / the downloaded file instead.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

$dir = Split-Path -Parent $ExePath
if (-not $dir) { $dir = '.' }
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$log = Join-Path $dir 'koboldcpp.log'
$downloading = Join-Path $dir 'koboldcpp.downloading'
$done = Join-Path $dir 'koboldcpp.done'
$failed = Join-Path $dir 'koboldcpp.failed'
$partial = "$ExePath.part"

# Fresh attempt: clear terminal markers, mark as downloading.
Remove-Item $done, $failed -ErrorAction SilentlyContinue
Set-Content -Path $downloading -Value "started $(Get-Date -Format o)" -Encoding utf8

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

# Picks the right Windows CUDA koboldcpp asset from a release's asset list,
# matching the exact main build first and falling back to a CUDA build, but NEVER
# a _nocuda / _oldpc / non-Windows asset (a loose match once grabbed the wrong one).
function Select-KoboldAsset($assets) {
    $names = $assets | ForEach-Object { $_.name }
    Log "[assets] $($names -join ', ')"
    # 1) Exact main Windows CUDA build.
    $exact = $assets | Where-Object { $_.name -match '^koboldcpp\.exe$' } | Select-Object -First 1
    if ($exact) { return $exact }
    # 2) A CUDA build (koboldcpp_cu*.exe), excluding nocuda / oldpc and non-exe.
    $cuda = $assets |
        Where-Object {
            $_.name -match '^koboldcpp_cu.*\.exe$' -and
            $_.name -notmatch 'nocuda' -and $_.name -notmatch 'oldpc'
        } |
        Select-Object -First 1
    if ($cuda) { return $cuda }
    return $null
}

try {
    Log "[download] koboldcpp -> $ExePath"

    if (Test-Path $ExePath) {
        Log "[skip] koboldcpp.exe already exists"
        Set-Content -Path $done -Value "present $(Get-Date -Format o)" -Encoding utf8
        Remove-Item $downloading -ErrorAction SilentlyContinue
        return
    }

    # Resolve the latest release + its CUDA Windows asset via the GitHub API.
    $api = 'https://api.github.com/repos/LostRuins/koboldcpp/releases/latest'
    Log "[api] $api"
    $release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'chasm-koboldcpp-downloader' } -UseBasicParsing
    Log "[api] latest tag=$($release.tag_name)"

    $asset = Select-KoboldAsset $release.assets
    if (-not $asset) { throw 'no koboldcpp.exe / koboldcpp_cu*.exe asset in the latest release' }
    $url = $asset.browser_download_url
    Log "[asset] $($asset.name) -> $url"

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        # -L follow redirects, -C - resume a partial, -f fail on HTTP errors.
        & curl.exe -L -f -C - -o $partial $url *>> $log
        if ($LASTEXITCODE -ne 0) { throw "curl failed (exit $LASTEXITCODE)" }
    }
    else {
        Log "[fallback] curl.exe not found; using Invoke-WebRequest"
        Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
    }
    if (-not (Test-Path $partial)) { throw 'download produced no file' }
    Move-Item -Force $partial $ExePath

    Log "[done] koboldcpp downloaded to $ExePath"
    Set-Content -Path $done -Value "downloaded $(Get-Date -Format o)" -Encoding utf8
}
catch {
    Log "[error] $($_.Exception.Message)"
    Set-Content -Path $failed -Value "$($_.Exception.Message)" -Encoding utf8
}
finally {
    Remove-Item $downloading -ErrorAction SilentlyContinue
}
