<#
.SYNOPSIS
    Installs a local TTS engine into an isolated uv virtualenv and writes status
    markers the Chasm app reads (.installing / .installed / .failed).

.NOTES
    Windows-oriented (uses uv + PowerShell). Invoked by the app's install
    endpoint, or run directly:
        pwsh scripts/install-engine.ps1 -Engine pockettts -EnginesDir .\engines
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Engine,
    [Parameter(Mandatory = $true)][string]$EnginesDir
)

# uv/pip/git write progress to stderr; under PowerShell 7.4+ that gets promoted
# to a terminating error when ErrorActionPreference is 'Stop'. Keep going and
# detect real failures via $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

# The HuggingFace xet backend symlinks blobs->snapshots and bypasses hf_hub's
# "are symlinks supported?" fallback, so on Windows without admin / Developer Mode
# snapshot_download hard-fails (WinError 1314). Disabling xet restores the plain
# hf_hub path, which copies instead of symlinking. Downloads then work for everyone.
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

$dir = Join-Path $EnginesDir $Engine
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$log = Join-Path $dir 'install.log'
$installing = Join-Path $dir '.installing'
$installed = Join-Path $dir '.installed'
$failed = Join-Path $dir '.failed'
$venv = Join-Path $dir '.venv'
$py = Join-Path $venv 'Scripts\python.exe'

# Fresh attempt: clear terminal markers, mark as installing.
Remove-Item $installed, $failed -ErrorAction SilentlyContinue
Set-Content -Path $installing -Value "started $(Get-Date -Format o)" -Encoding utf8

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

# Ordered torch CUDA wheel tags suited to the installed NVIDIA driver, newest
# first. Empty when no NVIDIA GPU is present (→ stay on CPU). Blackwell
# (RTX 50-series, sm_120) needs cu128+, so we never offer older tags to new
# drivers; the real-op check below is the final arbiter regardless.
function Get-CudaWheelTags {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Log "[gpu] no nvidia-smi; using CPU torch"
        return @()
    }
    $name = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    $match = (& nvidia-smi 2>$null | Select-String 'CUDA Version:\s*([\d.]+)')
    $cuda = if ($match) { [version]$match.Matches[0].Groups[1].Value } else { $null }
    Log "[gpu] detected '$($name.Trim())' driver-CUDA=$cuda"
    $tags = @()
    if (-not $cuda -or $cuda -ge [version]'13.0') { $tags += 'cu130' }
    if (-not $cuda -or $cuda -ge [version]'12.8') { $tags += 'cu128' }
    if ($cuda -and $cuda -ge [version]'12.6' -and $cuda -lt [version]'12.8') { $tags += 'cu126' }
    if ($cuda -and $cuda -ge [version]'12.4' -and $cuda -lt [version]'12.6') { $tags += 'cu124' }
    return ($tags | Select-Object -Unique)
}

# Verifies torch can actually run a kernel on the GPU. is_available() alone lies
# on Blackwell with an older wheel (reports True, then "no kernel image" at launch).
function Test-TorchGpu($python) {
    & $python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(256,256,device='cuda'); v=float((x@x).sum()); torch.cuda.synchronize(); print('GPU_OK', torch.cuda.get_device_name(0))" *>> $log
    return ($LASTEXITCODE -eq 0)
}

# Swaps the CPU torch a package pulled in for a CUDA build of the SAME version,
# verifying each candidate with a real GPU op and reverting to CPU if none work.
function Install-TorchCuda($python) {
    $tags = Get-CudaWheelTags
    if ($tags.Count -eq 0) { return }
    $ver = (& $python -c "import torch; print(torch.__version__.split('+')[0])").Trim()
    foreach ($tag in $tags) {
        # Pin the FULL local version (e.g. 2.12.1+cu128). Without the +cuXXX tag,
        # `==2.12.1` matches the already-installed +cpu build and uv skips the swap.
        # Use --index-url (sole source): uv's dependency-confusion guard ignores a
        # +cuXXX build on an --extra-index-url when torch also exists on PyPI.
        Log "[gpu] trying torch==$ver+$tag"
        & $script:uv pip install --python $python --reinstall-package torch "torch==$ver+$tag" --index-url "https://download.pytorch.org/whl/$tag" *>> $log
        if ($LASTEXITCODE -eq 0 -and (Test-TorchGpu $python)) {
            Log "[gpu] torch $ver+$tag verified on GPU"
            return
        }
        Log "[gpu] $tag unusable (install or GPU op failed); trying next"
    }
    Log "[gpu] no working CUDA wheel for torch $ver; reverting to CPU"
    & $script:uv pip install --python $python --reinstall-package torch "torch==$ver+cpu" --extra-index-url "https://download.pytorch.org/whl/cpu" *>> $log
}

try {
    Log "[install] engine=$Engine dir=$dir"

    # Resolve uv (PATH -> bundled tools\uv -> bootstrap), so we never depend on a
    # system uv/Python being installed. A failure here throws -> .failed marker.
    . "$PSScriptRoot\ensure-uv.ps1"
    $script:uv = Resolve-Uv -Log { param($m) Log $m }

    # Python 3.12, not 3.13: several engine deps (notably librosa -> numba ->
    # llvmlite, pulled in by faster-qwen3-tts) only ship wheels up to 3.12. On
    # 3.13 the resolver backtracks to an ancient numba 0.53.1 that requires
    # Python <3.10 and then fails to build -> the whole install dies.
    $pyver = '3.12'
    Remove-Item $venv -Recurse -Force -ErrorAction SilentlyContinue
    # `uv venv --python 3.13` downloads a managed CPython if none is present, so no
    # system Python is required.
    & $script:uv venv $venv --python $pyver *>> $log
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed (exit $LASTEXITCODE)" }

    switch ($Engine) {
        'pockettts' {
            # Pulls its own PyTorch (CPU). Model downloads on first use.
            & $script:uv pip install --python $py pocket-tts soundfile fastapi uvicorn *>> $log
            if ($LASTEXITCODE -ne 0) { throw "pip install pocket-tts failed (exit $LASTEXITCODE)" }
            # If an NVIDIA GPU is present, swap the CPU torch for a verified CUDA build.
            Install-TorchCuda $py
        }
        'faster-qwen3-tts' {
            # The streaming Qwen3 TTS engine + the server's runtime deps. faster-qwen3-tts
            # pulls its own PyTorch (CPU by default); we swap it for a verified CUDA build
            # below. The server (scripts/qwen3_tts_server.py) imports
            # `from faster_qwen3_tts import FasterQwen3TTS` and serves via fastapi/uvicorn.
            # Floor numba/llvmlite: without this the resolver picks numba 0.53.1
            # (via librosa) which has no wheel for modern Python and fails to
            # build. numba>=0.60 / llvmlite>=0.43 have 3.12 wheels and satisfy
            # librosa's constraint.
            & $script:uv pip install --python $py faster-qwen3-tts soundfile numpy fastapi uvicorn 'numba>=0.60' 'llvmlite>=0.43' *>> $log
            if ($LASTEXITCODE -ne 0) { throw "pip install faster-qwen3-tts failed (exit $LASTEXITCODE)" }
            # If an NVIDIA GPU is present, swap the CPU torch for a verified CUDA build.
            Install-TorchCuda $py
        }
        'parakeet' {
            # Parakeet TDT 0.6B v3 STT via nano-parakeet (pure-PyTorch TDT
            # inference; deps: torch, numpy, soundfile, sentencepiece,
            # huggingface-hub). nano-parakeet pulls a CPU torch by default; we
            # swap it for a verified CUDA build below. torchaudio only for
            # resampling non-16kHz clips; python-multipart for the OpenAI
            # multipart transcription form.
            & $script:uv pip install --python $py nano-parakeet torchaudio soundfile numpy fastapi uvicorn python-multipart *>> $log
            if ($LASTEXITCODE -ne 0) { throw "pip install nano-parakeet failed (exit $LASTEXITCODE)" }
            # If an NVIDIA GPU is present, swap the CPU torch for a verified CUDA build.
            Install-TorchCuda $py
            # torchaudio must match the (possibly swapped) torch build or it fails to
            # import; reinstall it against the same index torch came from.
            $torchVer = (& $py -c "import torch; print(torch.__version__)").Trim()
            if ($torchVer -match '\+(cu\d+)$') {
                $tag = $Matches[1]
                & $script:uv pip install --python $py --reinstall-package torchaudio torchaudio --index-url "https://download.pytorch.org/whl/$tag" *>> $log
                if ($LASTEXITCODE -ne 0) { Log "[warn] torchaudio $tag reinstall failed; resampling may fall back to CPU torch import errors" }
            }
        }
        default { throw "unknown engine: $Engine" }
    }

    # Pre-fetch the engine's model weights so ".installed" means fully ready (not
    # just the venv/code). hf_xet makes the HuggingFace download fast. Prefetch is
    # ENGINE-SPECIFIC and deliberately NOT a blind snapshot_download of the repo:
    # some repos hold far more than the engine loads.
    & $script:uv pip install --python $py hf_xet *>> $log
    switch ($Engine) {
        'pockettts' {
            # kyutai/pocket-tts is ~9.8 GB because languages/ holds EVERY language
            # plus dated checkpoints. A full snapshot pulled all of it just to speak
            # English. The library's own loader fetches only the default (English)
            # model's files via per-file hf_hub_download (~a few hundred MB) and
            # loads on CPU, so it neither over-downloads nor fights the GPU.
            Log "[model] prefetch pockettts (English) via TTSModel.load_model()"
            & $py -c "from pocket_tts import TTSModel; TTSModel.load_model()" *>> $log
            if ($LASTEXITCODE -ne 0) { throw "pockettts model prefetch failed (exit $LASTEXITCODE)" }
        }
        'faster-qwen3-tts' {
            # Qwen/Qwen3-TTS-12Hz-1.7B-Base is 13 files / ~4.5 GB and it is ALL the
            # model (weights + speech tokenizer + configs) — nothing to trim, so a
            # plain snapshot is correct here.
            Log "[model] prefetch Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            & $py -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')" *>> $log
            if ($LASTEXITCODE -ne 0) { throw "model prefetch failed for Qwen3-TTS (exit $LASTEXITCODE)" }
        }
        'parakeet' {
            # nano-parakeet loads exactly one file: the .nemo build of the repo.
            # Prefetch it (NOT a full snapshot — the repo also carries safetensors
            # etc. the runtime never reads) so ".installed" means fully ready.
            Log "[model] prefetch nvidia/parakeet-tdt-0.6b-v3 (.nemo)"
            & $py -c "from huggingface_hub import hf_hub_download; hf_hub_download('nvidia/parakeet-tdt-0.6b-v3', 'parakeet-tdt-0.6b-v3.nemo')" *>> $log
            if ($LASTEXITCODE -ne 0) { throw "model prefetch failed for Parakeet (exit $LASTEXITCODE)" }
        }
    }

    Log "[done] $Engine installed"
    Set-Content -Path $installed -Value "installed $(Get-Date -Format o)" -Encoding utf8
}
catch {
    Log "[error] $($_.Exception.Message)"
    Set-Content -Path $failed -Value "$($_.Exception.Message)" -Encoding utf8
}
finally {
    Remove-Item $installing -ErrorAction SilentlyContinue
}
