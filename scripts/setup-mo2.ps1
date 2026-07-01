<#
.SYNOPSIS
    Builds a working Mod Organizer 2 instance for Fallout: New Vegas from a plan
    JSON the chasm backend writes: installs MO2 if missing, downloads + extracts
    the required mods (GitHub releases, the NVBridge repo subfolder, and - only
    with a Nexus Premium API key - the Nexus mods), and writes ModOrganizer.ini +
    the profile's modlist/plugins/loadorder so `moshortcut://<instance>:NVSE`
    launches the modded game.

.DESCRIPTION
    Invoked by chasm's POST /setup/mo2 (see crates/chasm-web/src/setup.rs),
    or run directly:
        pwsh scripts/setup-mo2.ps1 -PlanPath .\plan.json -MarkerDir .\setup\mo2

    The plan JSON is the serialized `SetupPlan` (see game_launcher.rs): it carries
    the MO2 exe path, the game dir, the instance/profile/executable names, and the
    full mod list with each mod's source flattened into plain fields. The Nexus API
    key is passed via the NEXUS_API_KEY env var (never on the command line).

    Status markers (read by the backend):
        <MarkerDir>\.running  - in progress; holds the latest progress line
        <MarkerDir>\.done     - finished OK
        <MarkerDir>\.failed   - finished with an error (holds the message)
        <MarkerDir>\setup.log - full transcript

.NOTES
    Windows-oriented (Expand-Archive, 7-Zip for .7z, tar for tarballs). Verified
    MO2 formats: gameName "New Vegas", gamePath as @ByteArray with doubled
    backslashes, [customExecutables] 1-indexed, plugins.txt has NO '*' prefix for
    FNV, mod-root == virtual Data/ (strip a redundant Data/ wrapper).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PlanPath,
    [Parameter(Mandatory = $true)][string]$MarkerDir
)

# Native tools (curl/tar/7z) write progress to stderr; under PowerShell 7.4+ that
# is promoted to a terminating error when ErrorActionPreference is 'Stop'. Keep
# going and detect real failures via $LASTEXITCODE / file existence instead.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

New-Item -ItemType Directory -Force -Path $MarkerDir | Out-Null
$log = Join-Path $MarkerDir 'setup.log'
$running = Join-Path $MarkerDir '.running'
$doneMk = Join-Path $MarkerDir '.done'
$failedMk = Join-Path $MarkerDir '.failed'

Remove-Item $doneMk, $failedMk -ErrorAction SilentlyContinue

function Log($message) {
    "$(Get-Date -Format o) $message" | Out-File -FilePath $log -Append -Encoding utf8
}

# Update the .running marker so the backend's /setup/status surfaces live progress.
function Progress($message) {
    Set-Content -Path $running -Value $message -Encoding utf8
    Log "[progress] $message"
}

# Download a URL to a file. Prefers curl.exe (resume + redirects), falls back to
# Invoke-WebRequest. Adds the GitHub token header when fetching api.github.com so
# we don't get rate-limited (the token is optional). Throws on failure.
function Get-File($url, $outFile, $headers = @{}) {
    $dir = Split-Path -Parent $outFile
    if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        $args = @('-L', '-f', '-S', '-s', '-o', $outFile)
        foreach ($k in $headers.Keys) { $args += @('-H', "$k`: $($headers[$k])") }
        $args += $url
        & curl.exe @args *>> $log
        if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE): $url" }
    }
    else {
        Invoke-WebRequest -Uri $url -OutFile $outFile -Headers $headers -UseBasicParsing
    }
    if (-not (Test-Path $outFile)) { throw "download produced no file: $url" }
}

# Fetch JSON from a URL (GitHub API). Returns the parsed object.
function Get-Json($url, $headers = @{}) {
    $tmp = Join-Path $env:TEMP ("sb-json-" + [System.Guid]::NewGuid().ToString('N') + '.json')
    Get-File $url $tmp $headers
    $obj = Get-Content -Raw -Path $tmp | ConvertFrom-Json
    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $obj
}

# Find a usable 7-Zip executable (for .7z archives). Checks PATH then common
# install locations. Returns $null when none is installed.
function Find-SevenZip {
    $cmd = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
            "$env:ProgramFiles\7-Zip\7z.exe",
            "${env:ProgramFiles(x86)}\7-Zip\7z.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

# Extract an archive (.zip / .7z) into a fresh temp dir and return that dir.
# .zip uses Expand-Archive; .7z needs 7-Zip (throws a clear error if absent).
function Expand-Any($archive) {
    $dest = Join-Path $env:TEMP ("sb-extract-" + [System.Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $ext = [System.IO.Path]::GetExtension($archive).ToLowerInvariant()
    if ($ext -eq '.zip') {
        Expand-Archive -Path $archive -DestinationPath $dest -Force
    }
    elseif ($ext -eq '.7z') {
        $sevenZip = Find-SevenZip
        if (-not $sevenZip) {
            throw "7-Zip is required to extract $([System.IO.Path]::GetFileName($archive)) but was not found. Install 7-Zip from https://www.7-zip.org and re-run setup."
        }
        & $sevenZip x "-o$dest" -y $archive *>> $log
        if ($LASTEXITCODE -ne 0) { throw "7-Zip extraction failed (exit $LASTEXITCODE): $archive" }
    }
    else {
        throw "unsupported archive type: $ext"
    }
    return $dest
}

# Within an extracted tree, find the directory that should be mapped onto the mod
# root (the virtual Data/). MO2 maps the mod root onto the game's Data/, so:
#   - if the tree contains a 'Data' folder, its CONTENTS are the mod root;
#   - else if there is a single top-level folder that holds the payload, descend;
#   - else the tree itself is the mod root.
# We look for the marker subfolders an FNV mod has at its data root.
function Resolve-ModRoot($extracted) {
    $markers = @('NVSE', 'Data', 'meshes', 'textures', 'MCM', 'menus', 'sound', 'music')
    # 1) An explicit Data folder wins: its contents are the mod root.
    $dataDir = Join-Path $extracted 'Data'
    if (Test-Path $dataDir -PathType Container) { return $dataDir }
    # 2) The tree already looks like a data root (has NVSE/, *.esp, etc.).
    $hasMarker = Get-ChildItem -Path $extracted -Force -ErrorAction SilentlyContinue |
    Where-Object { $markers -contains $_.Name -or $_.Extension -in '.esp', '.esm', '.bsa' }
    if ($hasMarker) { return $extracted }
    # 3) A single wrapper folder (e.g. "ModName-1.2\") - descend once and retry.
    $children = @(Get-ChildItem -Path $extracted -Force -ErrorAction SilentlyContinue)
    $dirs = @($children | Where-Object { $_.PSIsContainer })
    if ($dirs.Count -eq 1 -and $children.Count -eq 1) {
        return (Resolve-ModRoot $dirs[0].FullName)
    }
    return $extracted
}

# Copy the resolved mod root into mods/<folder>/, replacing any existing copy.
function Install-ModFolder($extracted, $modsDir, $folder) {
    $root = Resolve-ModRoot $extracted
    $target = Join-Path $modsDir $folder
    if (Test-Path $target) { Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Path (Join-Path $root '*') -Destination $target -Recurse -Force
    Log "[mod] installed '$folder' (root: $root)"
}

# Download + install a Nexus-hosted mod via the Nexus API. Requires a Nexus
# **Premium** API key: the files list works on any key, but the download_link
# endpoint only returns a CDN url for Premium accounts (free accounts must use the
# website's "slow download" + an nxm:// handshake, which we can't automate here).
# Throws a clear message on the non-Premium 403 so it surfaces as a manual step.
function Install-NexusMod($mod, $modsDir, $apiKey) {
    $headers = @{ 'apikey' = $apiKey; 'Accept' = 'application/json'; 'User-Agent' = 'chasm-setup' }
    $base = "https://api.nexusmods.com/v1/games/newvegas/mods/$($mod.nexus_modid)"
    # 1) List files; pick the MAIN file (category_name == 'MAIN FILES'), newest.
    $files = (Get-Json "$base/files.json" $headers).files
    $main = $files | Where-Object { $_.category_name -eq 'MAIN FILES' } |
    Sort-Object -Property uploaded_timestamp -Descending | Select-Object -First 1
    if (-not $main) { $main = $files | Sort-Object -Property uploaded_timestamp -Descending | Select-Object -First 1 }
    if (-not $main) { throw "no files listed for Nexus mod $($mod.nexus_modid)" }
    # 2) Get a download link (Premium-only). A 403 here == free account.
    $dlInfo = Get-Json "$base/files/$($main.file_id)/download_link.json" $headers
    $url = $dlInfo[0].URI
    if (-not $url) { throw "Nexus returned no download link (a Premium account is required to auto-download)" }
    # 3) Download + extract into the mod folder.
    $dl = Join-Path $env:TEMP $main.file_name
    Get-File $url $dl
    $ex = Expand-Any $dl
    Install-ModFolder $ex $modsDir $mod.mod_folder
    Remove-Item $ex, $dl -Recurse -Force -ErrorAction SilentlyContinue
}

# The first release asset whose name contains $hint (case-insensitive); when $hint
# is empty, the first .zip/.7z asset. $null when none match.
function Select-Asset($assets, $hint) {
    if ($hint) {
        $m = $assets | Where-Object { $_.name -like "*$hint*" } | Select-Object -First 1
        if ($m) { return $m }
    }
    return ($assets | Where-Object { $_.name -match '\.(zip|7z)$' } | Select-Object -First 1)
}

# ---- main ----------------------------------------------------------------------
try {
    Progress 'Reading setup plan...'
    if (-not (Test-Path $PlanPath)) { throw "plan file not found: $PlanPath" }
    $plan = Get-Content -Raw -Path $PlanPath | ConvertFrom-Json

    $gameDir = $plan.game_dir
    if (-not (Test-Path (Join-Path $gameDir 'FalloutNV.exe'))) {
        throw "no FalloutNV.exe in '$gameDir' - set the game folder first"
    }

    # GitHub API headers (token optional, lifts the unauthenticated rate limit).
    $ghHeaders = @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'chasm-setup' }
    if ($env:GITHUB_TOKEN) { $ghHeaders['Authorization'] = "Bearer $($env:GITHUB_TOKEN)" }

    # --- 1) Mod Organizer 2 -----------------------------------------------------
    $mo2Exe = $plan.mo2_exe
    if (Test-Path $mo2Exe) {
        Log "[mo2] using existing ModOrganizer.exe at $mo2Exe"
    }
    else {
        Progress 'Downloading Mod Organizer 2...'
        # MO2 ships its release as Mod.Organizer-<ver>.7z (a portable archive).
        $rel = Get-Json "https://api.github.com/repos/$($plan.mo2_repo)/releases/latest" $ghHeaders
        # Pick the portable APP archive (Mod.Organizer-<ver>.7z) only - NOT the
        # -pdbs (debug symbols) / -src / -uibase archives that share the prefix.
        $asset = $rel.assets | Where-Object { $_.name -match '^Mod\.Organizer-[0-9][0-9.]*\.(7z|zip)$' } | Select-Object -First 1
        if (-not $asset) { $asset = $rel.assets | Where-Object { $_.name -match '\.(7z|zip)$' -and $_.name -notmatch '(pdbs?|-src|uibase|debug|-dev)' } | Select-Object -First 1 }
        if (-not $asset) { $asset = Select-Asset $rel.assets '' }
        if (-not $asset) { throw "no MO2 archive asset found in the latest release" }
        $mo2Dir = Split-Path -Parent $mo2Exe
        New-Item -ItemType Directory -Force -Path $mo2Dir | Out-Null
        $dl = Join-Path $env:TEMP $asset.name
        Get-File $asset.browser_download_url $dl $ghHeaders
        Progress 'Extracting Mod Organizer 2...'
        $ex = Expand-Any $dl
        # MO2's archive root contains ModOrganizer.exe directly (or one wrapper).
        $exeFound = Get-ChildItem -Path $ex -Recurse -Filter 'ModOrganizer.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $exeFound) { throw "ModOrganizer.exe not found in the downloaded MO2 archive" }
        Copy-Item -Path (Join-Path $exeFound.Directory.FullName '*') -Destination $mo2Dir -Recurse -Force
        Remove-Item $ex, $dl -Recurse -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $mo2Exe)) { throw "MO2 install did not place ModOrganizer.exe at $mo2Exe" }
        Log "[mo2] installed to $mo2Dir"
    }

    # --- 2) Instance + profile skeleton -----------------------------------------
    # Global instance under %LOCALAPPDATA%\ModOrganizer\<instance> (matches what
    # the backend's LauncherConfig resolves + the seamless moshortcut launch).
    $instance = $plan.instance
    if (-not $instance) { $instance = 'New Vegas' }
    $profile = $plan.profile
    if (-not $profile) { $profile = 'Default' }
    $instanceDir = Join-Path (Join-Path $env:LOCALAPPDATA 'ModOrganizer') $instance
    $modsDir = Join-Path $instanceDir 'mods'
    $profileDir = Join-Path (Join-Path $instanceDir 'profiles') $profile
    foreach ($d in @($modsDir, $profileDir, (Join-Path $instanceDir 'overwrite'), (Join-Path $instanceDir 'downloads'))) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
    Log "[instance] $instanceDir (profile: $profile)"

    # --- 3) Mods ----------------------------------------------------------------
    $manual = @()    # mods left for the user (Nexus, no key, etc.)
    $enabledMods = @()    # mod-folder names to enable in modlist.txt (load order)

    foreach ($mod in $plan.mods) {
        $name = $mod.display
        switch ($mod.source_kind) {

            'github_release' {
                Progress "Downloading $name..."
                try {
                    $rel = Get-Json "https://api.github.com/repos/$($mod.repo)/releases/latest" $ghHeaders
                    $asset = Select-Asset $rel.assets $mod.asset_hint
                    if (-not $asset) { throw "no matching asset (hint '$($mod.asset_hint)') in $($mod.repo) latest release" }
                    $dl = Join-Path $env:TEMP $asset.name
                    Get-File $asset.browser_download_url $dl $ghHeaders
                    $ex = Expand-Any $dl
                    if ($mod.install_target -eq 'game') {
                        # xNVSE: copy the loaders loose into the GAME dir (NOT a mod,
                        # NOT under Data/). The loaders + any bundled Data/ sit at the
                        # archive root; descend through a single wrapper folder if the
                        # release zips everything under one top-level dir.
                        $src = $ex
                        $kids = @(Get-ChildItem -Path $ex -Force)
                        if ($kids.Count -eq 1 -and $kids[0].PSIsContainer) { $src = $kids[0].FullName }
                        Copy-Item -Path (Join-Path $src '*') -Destination $gameDir -Recurse -Force
                        Log "[game] installed $name into $gameDir"
                    }
                    else {
                        Install-ModFolder $ex $modsDir $mod.mod_folder
                        $enabledMods += $mod.mod_folder
                    }
                    Remove-Item $ex, $dl -Recurse -Force -ErrorAction SilentlyContinue
                }
                catch {
                    Log "[warn] $name (github_release) failed: $($_.Exception.Message)"
                    $manual += "$name (download failed - get it from $($mod.url))"
                }
            }

            'github_repo_dir' {
                Progress "Installing $name..."
                try {
                    # Download the repo tarball at the ref, then copy out the subdir.
                    $tarUrl = "https://codeload.github.com/$($mod.repo)/tar.gz/refs/heads/$($mod.git_ref)"
                    $tar = Join-Path $env:TEMP ("sb-repo-" + [System.Guid]::NewGuid().ToString('N') + '.tar.gz')
                    Get-File $tarUrl $tar $ghHeaders
                    $ex = Join-Path $env:TEMP ("sb-repo-" + [System.Guid]::NewGuid().ToString('N'))
                    New-Item -ItemType Directory -Force -Path $ex | Out-Null
                    & tar.exe -xzf $tar -C $ex *>> $log
                    if ($LASTEXITCODE -ne 0) { throw "tar extraction failed (exit $LASTEXITCODE)" }
                    # Tarball top-level is "<repo>-<ref>/"; the subdir lives under it.
                    $top = @(Get-ChildItem -Path $ex -Directory)[0].FullName
                    $sub = Join-Path $top ($mod.subdir -replace '/', '\')
                    if (-not (Test-Path $sub)) { throw "subfolder '$($mod.subdir)' not found in repo $($mod.repo)" }
                    Install-ModFolder $sub $modsDir $mod.mod_folder
                    $enabledMods += $mod.mod_folder
                    Remove-Item $ex, $tar -Recurse -Force -ErrorAction SilentlyContinue
                }
                catch {
                    Log "[warn] $name (github_repo_dir) failed: $($_.Exception.Message)"
                    $manual += "$name (install failed - get it from $($mod.url))"
                }
            }

            'nexus' {
                if ($env:NEXUS_API_KEY) {
                    Progress "Downloading $name from Nexus..."
                    try {
                        Install-NexusMod $mod $modsDir $env:NEXUS_API_KEY
                        $enabledMods += $mod.mod_folder
                    }
                    catch {
                        Log "[warn] $name (nexus) failed: $($_.Exception.Message)"
                        $manual += "$name (Nexus auto-download failed - get it from $($mod.url))"
                    }
                }
                else {
                    Log "[manual] $name needs a Nexus download (no API key set)"
                    $manual += "$name (Nexus - download from $($mod.url), or add a Nexus Premium API key in Settings)"
                }
            }

            'bundled' {
                Log "[skip] $name is bundled; handled elsewhere"
            }

            default {
                Log "[warn] unknown source kind '$($mod.source_kind)' for $name"
            }
        }
    }

    # --- 4) ModOrganizer.ini ----------------------------------------------------
    Progress 'Writing Mod Organizer configuration...'
    $gamePathEsc = $gameDir -replace '\\', '\\'      # double each backslash for @ByteArray
    $loaderPath = (Join-Path $gameDir 'nvse_loader.exe') -replace '\\', '/'   # MO2 stores fwd slashes
    $gameFwd = $gameDir -replace '\\', '/'
    $iniLines = @(
        '[General]',
        "gameName=$($plan.game_name)",
        "gamePath=@ByteArray($gamePathEsc)",
        "selected_profile=@ByteArray($profile)",
        'first_start=false',
        '',
        '[customExecutables]',
        'size=1',
        '1\arguments=',
        "1\binary=$loaderPath",
        '1\hide=false',
        '1\ownicon=true',
        '1\steamAppID=',
        "1\title=$($plan.executable)",
        '1\toolbar=false',
        "1\workingDirectory=$gameFwd"
    )
    Set-Content -Path (Join-Path $instanceDir 'ModOrganizer.ini') -Value $iniLines -Encoding utf8
    Log "[ini] wrote ModOrganizer.ini"

    # --- 5) Profile files (modlist / plugins / loadorder) -----------------------
    Progress 'Writing load order...'
    $header = '# This file was automatically generated by Mod Organizer.'

    # modlist.txt: FIRST line = LOWEST priority, LAST = HIGHEST (reverse of GUI).
    # Our registry order is the intended load order (xNVSE-adjacent libs first,
    # then NVBridge); reverse it so the file's last line wins conflicts last.
    $modlist = @($header)
    $rev = @($enabledMods); [array]::Reverse($rev)
    foreach ($m in $rev) { $modlist += "+$m" }
    Set-Content -Path (Join-Path $profileDir 'modlist.txt') -Value $modlist -Encoding utf8

    # plugins.txt (FNV): header + enabled plugins, NO '*' prefix. The required mods
    # are NVSE DLL plugins (no .esp/.esm), so only the base game masters are listed.
    $masters = @(
        'FalloutNV.esm', 'DeadMoney.esm', 'HonestHearts.esm', 'OldWorldBlues.esm',
        'LonesomeRoad.esm', 'GunRunnersArsenal.esm', 'ClassicPack.esm',
        'MercenaryPack.esm', 'TribalPack.esm', 'CaravanPack.esm'
    )
    # Only list masters whose .esm actually exists in the game's Data folder.
    $dataDir = Join-Path $gameDir 'Data'
    $presentMasters = @($masters | Where-Object { Test-Path (Join-Path $dataDir $_) })
    if ($presentMasters.Count -eq 0) { $presentMasters = @('FalloutNV.esm') }
    $plugins = @($header) + $presentMasters
    Set-Content -Path (Join-Path $profileDir 'plugins.txt') -Value $plugins -Encoding utf8
    # loadorder.txt: header + ALL plugins in order (same set here; no disabled).
    Set-Content -Path (Join-Path $profileDir 'loadorder.txt') -Value $plugins -Encoding utf8
    Log "[profile] wrote modlist/plugins/loadorder ($($enabledMods.Count) mods, $($presentMasters.Count) masters)"

    # --- done -------------------------------------------------------------------
    $summary = "Setup complete. Installed $($enabledMods.Count) mod(s)."
    if ($manual.Count -gt 0) {
        $summary += " Manual steps needed: " + ($manual -join '; ')
    }
    Progress $summary
    Set-Content -Path $doneMk -Value $summary -Encoding utf8
    Log "[done] $summary"
}
catch {
    $msg = $_.Exception.Message
    Log "[error] $msg"
    Set-Content -Path $failedMk -Value $msg -Encoding utf8
}
finally {
    Remove-Item $running -ErrorAction SilentlyContinue
}
