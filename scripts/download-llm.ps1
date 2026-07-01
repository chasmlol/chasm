<#
.SYNOPSIS
    Downloads a single LLM GGUF from Hugging Face into the models directory and
    writes status markers the Chasm app reads (<id>.downloading /
    <id>.done / <id>.failed).

.NOTES
    Windows-oriented (uses curl.exe, falling back to Invoke-WebRequest). Invoked
    by the app's download endpoint, or run directly:
        pwsh scripts/download-llm.ps1 -Id gemma-4-e2b `
            -Url "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q4_K_XL.gguf?download=true" `
            -File gemma-4-E2B-it-UD-Q4_K_XL.gguf -ModelsDir .\data\default-user\models\llm
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Id,
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$File,
    [Parameter(Mandatory = $true)][string]$ModelsDir
)

# curl writes progress to stderr; under PowerShell 7.4+ that gets promoted to a
# terminating error when ErrorActionPreference is 'Stop'. Keep going and detect
# real failures via $LASTEXITCODE / the downloaded file instead.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$log = Join-Path $ModelsDir "$Id.log"
$downloading = Join-Path $ModelsDir "$Id.downloading"
$done = Join-Path $ModelsDir "$Id.done"
$failed = Join-Path $ModelsDir "$Id.failed"
$target = Join-Path $ModelsDir $File
$partial = "$target.part"

# Fresh attempt: clear terminal markers, mark as downloading.
Remove-Item $done, $failed -ErrorAction SilentlyContinue
Set-Content -Path $downloading -Value "started $(Get-Date -Format o)" -Encoding utf8

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

try {
    Log "[download] id=$Id file=$File url=$Url"

    if (Test-Path $target) {
        Log "[skip] $File already exists"
    }
    else {
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            # -L follow redirects (HF resolve), -C - resume a partial, -f fail on HTTP errors.
            & curl.exe -L -f -C - -o $partial $Url *>> $log
            if ($LASTEXITCODE -ne 0) { throw "curl failed (exit $LASTEXITCODE)" }
        }
        else {
            Log "[fallback] curl.exe not found; using Invoke-WebRequest"
            Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        }
        if (-not (Test-Path $partial)) { throw "download produced no file" }
        Move-Item -Force $partial $target
    }

    Log "[done] $Id downloaded to $target"
    Set-Content -Path $done -Value "downloaded $(Get-Date -Format o)" -Encoding utf8
}
catch {
    Log "[error] $($_.Exception.Message)"
    Set-Content -Path $failed -Value "$($_.Exception.Message)" -Encoding utf8
}
finally {
    Remove-Item $downloading -ErrorAction SilentlyContinue
}
