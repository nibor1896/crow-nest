<#
.SYNOPSIS
Builds serve.exe for shipping and packs it, then proves the package carries no trace of
the machine it was built on (#131) and no NVIDIA file. NVRTC is NOT in the package: the
user fetches it from NVIDIA's own PyPI wheel with tools\fetch-nvrtc.ps1.

.DESCRIPTION
A plain `cargo build` serve.exe is not portable (#131, measured 2026-10-01 at f4a3bd8):
it carried the builder's profile path 405 times (404 panic locations under
%USERPROFILE%\.cargo\registry plus the compiled-in lock path) and imported
VCRUNTIME140.dll. This script:

1. builds `serve` with `--remap-path-prefix` for %USERPROFILE% and the repo root
   (the last matching remap wins, so the repo root, which lies inside the profile,
   comes second) and `-C target-feature=+crt-static`, through
   CARGO_ENCODED_RUSTFLAGS so a path with spaces stays one argument;
2. refuses when serve.exe still imports the VC++ runtime (dumpbin /dependents);
3. stages serve.exe and LICENSE, NOTICE and THIRD-PARTY-NOTICES.txt (the crates'
   licence texts), refusing first when `tools/engine_notices.py` says
   THIRD-PARTY-NOTICES.txt is not what engine/Cargo.lock produces, and refusing when
   the stage holds any NVIDIA file (nvrtc, nvidia, cuda, cublas, cudart in a name) or
   anything else than those four plus tools\fetch-nvrtc.ps1, which is staged beside
   serve.exe so that a user without a checkout can run it. NVRTC (nvrtc64_130_0.dll and
   nvrtc-builtins64_133.dll, the CUDA 13.3 names) is placed beside serve.exe by that
   script;
4. scans every staged file (the notices included), as UTF-8 and as UTF-16LE at both byte alignments, for
   the builder's profile path, any `\Users\<name>\`, the bare user name (any
   context, case-insensitive) and the computer name, and REFUSES on any hit;
5. writes MANIFEST.json ({path, bytes, sha256}, sha256 upper-case hex, backslash
   paths - the shape of nibor1896/Crow tools/pack-release.ps1) and zips the stage as
   crow-nest-engine-<version>-win-x64.zip.

.PARAMETER Version
The version in the zip name. Default: `git describe --tags --always --dirty`
without the leading `v` (exactly `0.7.2` on the tag, `0.7.2-N-g<sha>` after it).

.PARAMETER OutDir
Where the stage and the zip go. Default: dist\ in the repo root (gitignored).

.PARAMETER Selftest
Run the checks on synthetic inputs, including ones that must fail, without building.
#>
[CmdletBinding()]
param(
    [string] $Version = "",
    [string] $OutDir  = "",
    [switch] $Selftest
)

$ErrorActionPreference = "Stop"

# from the repo root: crow-nest's licence, NOTICE and the crates' texts. The package holds no
# NVIDIA file: nothing is staged from a CUDA install.
$SHIP_FILES = @('LICENSE', 'NOTICE', 'THIRD-PARTY-NOTICES.txt')
# crow-nest's own script that fetches NVRTC from NVIDIA; staged from tools\ beside serve.exe
$TOOL_FILES = @('fetch-nvrtc.ps1')
# the C runtime that +crt-static links in; any of these in the imports means it did not
$VC_RUNTIME = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'ucrtbase.dll')

# ---------------------------------------------------------------------------
# Checks (pure functions: the selftest calls exactly these)
# ---------------------------------------------------------------------------

# One byte <-> one char, so a regex over the string is a regex over the bytes.
$LATIN1 = [Text.Encoding]::GetEncoding(28591)

function Get-PrivacyNeedles {
    param([string] $UserProfile, [string] $UserName, [string] $ComputerName)
    $n = @()
    if ($UserProfile) {
        $up = $UserProfile.TrimEnd('\', '/')
        $n += [pscustomobject]@{ label = "USERPROFILE $up"; regex = [regex]::Escape($up) }
        $n += [pscustomobject]@{ label = "USERPROFILE $($up.Replace('\', '/'))"; regex = [regex]::Escape($up.Replace('\', '/')) }
    }
    # any \Users\<name>\ - another builder's profile is as much a leak as this one's
    $n += [pscustomobject]@{ label = '\Users\<name>\'; regex = '[\\/]Users[\\/][^\\/\x00-\x1f"<>|:*?]{1,64}[\\/]' }
    if ($UserName) {
        # the bare name, anywhere (prose like "<name>'s decision" too, not only a path segment)
        $n += [pscustomobject]@{ label = "USERNAME $UserName (bare)"; regex = [regex]::Escape($UserName) }
    }
    if ($ComputerName) {
        $n += [pscustomobject]@{ label = "COMPUTERNAME $ComputerName"; regex = [regex]::Escape($ComputerName) }
    }
    return $n
}

# Every hit of every needle in one byte array: UTF-8 (= the bytes) and UTF-16LE at
# offset 0 and 1, case-insensitive (Windows paths come in any case).
function Find-PrivacyHits {
    param([byte[]] $Bytes, [object[]] $Needles, [string] $Name)
    $hits = @()
    $views = @(
        @{ enc = 'utf-8';      text = $LATIN1.GetString($Bytes) },
        @{ enc = 'utf-16le';   text = [Text.Encoding]::Unicode.GetString($Bytes) }
    )
    if ($Bytes.Length -gt 1) {
        $views += @{ enc = 'utf-16le+1'; text = [Text.Encoding]::Unicode.GetString($Bytes, 1, $Bytes.Length - 1) }
    }
    foreach ($v in $views) {
        foreach ($nd in $Needles) {
            $ms = [regex]::Matches($v.text, $nd.regex, 'IgnoreCase')
            if ($ms.Count -gt 0) {
                $hits += [pscustomobject]@{
                    file = $Name; encoding = $v.enc; needle = $nd.label; count = $ms.Count
                    first = $ms[0].Value
                }
            }
        }
    }
    return $hits
}

# Every hit in every file of a stage directory (recursive): what the build refuses on.
function Find-StageHits {
    param([string] $Stage, [object[]] $Needles)
    $hits = @()
    foreach ($f in Get-ChildItem -LiteralPath $Stage -Recurse -File) {
        $hits += @(Find-PrivacyHits -Bytes ([IO.File]::ReadAllBytes($f.FullName)) -Needles $Needles -Name $f.Name)
    }
    return $hits
}

# The files a package holds, MANIFEST.json aside.
function Get-PackageFiles {
    return @(@('serve.exe') + $TOOL_FILES + $SHIP_FILES)
}

# Every file under a directory whose name looks like NVIDIA's (nvrtc, nvidia, cuda, cublas,
# cudart; any case), except crow-nest's own fetch script: what the build refuses on.
function Find-NvidiaFiles {
    param([string] $Dir)
    return @(Get-ChildItem -LiteralPath $Dir -Recurse -File -Force | Where-Object { $TOOL_FILES -notcontains $_.Name -and $_.Name -match '(?i)nvrtc|nvidia|cuda|cublas|cudart' })
}

function Get-Manifest {
    param([string] $Stage)
    $Stage = (Resolve-Path -LiteralPath $Stage).Path.TrimEnd('\')
    return @(Get-ChildItem -LiteralPath $Stage -Recurse -File | Where-Object { $_.Name -ne 'MANIFEST.json' } |
        Sort-Object FullName | ForEach-Object {
            [pscustomobject]@{
                path   = $_.FullName.Substring($Stage.Length + 1)
                bytes  = $_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
            }
        })
}

# dumpbin /dependents prints each imported name indented by four spaces
function Get-ImportsFromDumpbinText {
    param([string[]] $Lines)
    return @($Lines | Where-Object { $_ -match '^    \S+\.dll\s*$' } | ForEach-Object { $_.Trim() })
}

function Get-VcRuntimeImports {
    param([string[]] $Imports)
    return @($Imports | Where-Object { $VC_RUNTIME -contains $_.ToLowerInvariant() })
}

function Get-ZipName {
    param([string] $Ver)
    return "crow-nest-engine-$Ver-win-x64.zip"
}

function Find-Dumpbin {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $hit = & $vswhere -latest -products * -find "**\Hostx64\x64\dumpbin.exe" 2>$null | Select-Object -First 1
        if (-not $hit) { $hit = & $vswhere -latest -products * -find "**\dumpbin.exe" 2>$null | Select-Object -First 1 }
        if ($hit) { return $hit }
    }
    $cmd = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "dumpbin.exe not found (no Visual Studio through vswhere, none on PATH) - the VC++ runtime check cannot run"
}

# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

$script:ok = 0
$script:red = 0
function Check {
    param([string] $Name, [bool] $Passed)
    if ($Passed) { Write-Host "  ok   $Name"; $script:ok++ }
    else { Write-Host "  FAIL $Name" -ForegroundColor Red; $script:red++ }
}

function Invoke-Selftest {
    Write-Host "pack-engine selftest (synthetic identity: C:\Users\builder, builder, BUILDBOX)"
    $nd = Get-PrivacyNeedles -UserProfile 'C:\Users\builder' -UserName 'builder' -ComputerName 'BUILDBOX'
    $u8 = { param($s) [Text.Encoding]::UTF8.GetBytes($s) }
    $u16 = { param($s) [Text.Encoding]::Unicode.GetBytes($s) }
    $scan = { param($b) @(Find-PrivacyHits -Bytes $b -Needles $nd -Name 'x') }

    Check "a clean binary has no hit" ((& $scan (& $u8 "panicked at ~\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\serde-1.0\src\de.rs")).Count -eq 0)
    Check "the profile path in UTF-8 is a hit" ((& $scan (& $u8 "x C:\Users\builder\.cargo\registry y")).Count -gt 0)
    Check "the profile path with forward slashes is a hit" ((& $scan (& $u8 "C:/Users/builder/dev/crow-nest")).Count -gt 0)
    Check "the profile path in another case is a hit" ((& $scan (& $u8 "c:\users\BUILDER\x")).Count -gt 0)
    Check "the profile path in UTF-16LE is a hit" ((& $scan (& $u16 "C:\Users\builder\AppData")).Count -gt 0)
    $odd = [byte[]](@(0x41) + [byte[]](& $u16 "C:\Users\builder\AppData"))
    Check "UTF-16LE at an odd offset is a hit" ((& $scan $odd).Count -gt 0)
    Check "ANOTHER user's \Users\<name>\ is a hit" ((& $scan (& $u8 "D:\Users\someone\src\lib.rs")).Count -gt 0)
    Check "the user name as a path segment is a hit" ((& $scan (& $u8 "/home/builder/.cargo")).Count -gt 0)
    Check "the user name inside a word is a hit too (bare-name rule)" ((& $scan (& $u8 "rebuilder and builders")).Count -gt 0)
    Check "text without the user name is NOT a hit" ((& $scan (& $u8 "the owner's decision 2026-09-25")).Count -eq 0)
    Check "the bare user name in prose is a hit (UTF-8)" ((& $scan (& $u8 "default since 2026-09-09 (Builder's decision)")).Count -gt 0)
    Check "the bare user name in prose is a hit (UTF-16LE)" ((& $scan (& $u16 "decided by BUILDER")).Count -gt 0)
    Check "the computer name is a hit" ((& $scan (& $u8 "host=buildbox.local")).Count -gt 0)
    Check "an empty file has no hit" ((& $scan ([byte[]]@())).Count -eq 0)

    # manifest shape: backslash paths, upper-case hex, MANIFEST.json itself not listed
    $st = Join-Path ([IO.Path]::GetTempPath()) ("crow-nest-pack-selftest-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path (Join-Path $st 'sub') | Out-Null
    try {
        [IO.File]::WriteAllBytes((Join-Path $st 'sub\abc.txt'), [Text.Encoding]::ASCII.GetBytes('abc'))
        [IO.File]::WriteAllBytes((Join-Path $st 'MANIFEST.json'), [Text.Encoding]::ASCII.GetBytes('[]'))
        $m = @(Get-Manifest -Stage $st)
        Check "the manifest lists the file, not itself" ($m.Count -eq 1)
        Check "the manifest path uses backslashes" ($m[0].path -eq 'sub\abc.txt')
        Check "the manifest carries the byte count" ($m[0].bytes -eq 3)
        Check "the manifest sha256 is upper-case hex" ($m[0].sha256 -ceq 'BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD')
    } finally { Remove-Item -LiteralPath $st -Recurse -Force }

    $dump = @('Dump of file serve.exe', '', '  Image has the following dependencies:', '',
              '    KERNEL32.dll', '    VCRUNTIME140.dll', '    api-ms-win-crt-runtime-l1-1-0.dll', '', '  Summary')
    $imp = Get-ImportsFromDumpbinText -Lines $dump
    Check "dumpbin text parses to three imports" ($imp.Count -eq 3)
    Check "VCRUNTIME140.dll is flagged" ((Get-VcRuntimeImports -Imports $imp).Count -eq 1)
    Check "a static-CRT import list is not flagged" ((Get-VcRuntimeImports -Imports @('KERNEL32.dll', 'ntdll.dll', 'WS2_32.dll')).Count -eq 0)
    Check "the zip name" ((Get-ZipName -Ver '0.7.2') -eq 'crow-nest-engine-0.7.2-win-x64.zip')

    # the notices travel in the package and the privacy scan covers them
    $pf = Get-PackageFiles
    Check "the package carries LICENSE" ($pf -contains 'LICENSE')
    Check "the package carries NOTICE" ($pf -contains 'NOTICE')
    Check "the package carries THIRD-PARTY-NOTICES.txt" ($pf -contains 'THIRD-PARTY-NOTICES.txt')
    Check "the package is 5 distinct files + MANIFEST.json" ($pf.Count -eq 5 -and @($pf | Sort-Object -Unique).Count -eq 5)
    # the package holds no NVIDIA file
    Check "the package list names no NVIDIA file (nvrtc, nvidia, cuda in a name, fetch-nvrtc.ps1 aside)" (@($pf | Where-Object { $_ -ne 'fetch-nvrtc.ps1' -and $_ -match '(?i)nvrtc|nvidia|cuda' }).Count -eq 0)
    Check "the package is exactly serve.exe, fetch-nvrtc.ps1, LICENSE, NOTICE, THIRD-PARTY-NOTICES.txt (+ MANIFEST.json)" ((($pf | Sort-Object) -join ',') -ceq 'fetch-nvrtc.ps1,LICENSE,NOTICE,serve.exe,THIRD-PARTY-NOTICES.txt')
    Check "the package carries fetch-nvrtc.ps1" ($pf -contains 'fetch-nvrtc.ps1')
    $nv = Join-Path ([IO.Path]::GetTempPath()) ("crow-nest-pack-selftest-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $nv | Out-Null
    try {
        foreach ($n in $pf) { [IO.File]::WriteAllBytes((Join-Path $nv $n), [byte[]]@()) }
        Check "the stage check passes the five package files (fetch-nvrtc.ps1 included)" ((Find-NvidiaFiles -Dir $nv).Count -eq 0)
        foreach ($bad in @('nvrtc64_130_0.dll', 'nvrtc-builtins64_133.dll', 'NVRTC64_130_0.DLL', 'libnvrtc.so', 'cudart64_13.dll')) {
            [IO.File]::WriteAllBytes((Join-Path $nv $bad), [byte[]]@())
            Check "the stage check refuses a staged $bad" ((Find-NvidiaFiles -Dir $nv).Count -ge 1)
            Remove-Item -LiteralPath (Join-Path $nv $bad) -Force
        }
        New-Item -ItemType Directory -Force -Path (Join-Path $nv 'sub') | Out-Null
        [IO.File]::WriteAllBytes((Join-Path $nv 'sub\nvrtc64_130_0.dll'), [byte[]]@())
        Check "the stage check refuses an NVIDIA file in a subfolder" ((Find-NvidiaFiles -Dir $nv).Count -eq 1)
    } finally { Remove-Item -LiteralPath $nv -Recurse -Force }
    $st = Join-Path ([IO.Path]::GetTempPath()) ("crow-nest-pack-selftest-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $st | Out-Null
    try {
        [IO.File]::WriteAllBytes((Join-Path $st 'NOTICE'), [byte[]](& $u8 "see C:\Users\builder\dev\NOTICE"))
        [IO.File]::WriteAllBytes((Join-Path $st 'THIRD-PARTY-NOTICES.txt'), [byte[]](& $u16 "built on BUILDBOX"))
        [IO.File]::WriteAllBytes((Join-Path $st 'LICENSE'), [byte[]](& $u8 "Apache License"))
        $sh = @(Find-StageHits -Stage $st -Needles $nd)
        Check "the stage scan finds a leak in NOTICE" (@($sh | Where-Object file -eq 'NOTICE').Count -gt 0)
        Check "the stage scan finds a leak in THIRD-PARTY-NOTICES.txt (UTF-16LE)" (@($sh | Where-Object file -eq 'THIRD-PARTY-NOTICES.txt').Count -gt 0)
        Check "the stage scan leaves a clean LICENSE alone" (@($sh | Where-Object file -eq 'LICENSE').Count -eq 0)
    } finally { Remove-Item -LiteralPath $st -Recurse -Force }

    # the repo's real texts, scanned with THIS machine's identity, as the build will
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $real = Get-PrivacyNeedles -UserProfile $env:USERPROFILE -UserName $env:USERNAME -ComputerName $env:COMPUTERNAME
    foreach ($f in $SHIP_FILES) {
        $p = Join-Path $repoRoot $f
        $present = Test-Path -LiteralPath $p
        Check "$f exists in the repo root" $present
        if ($present) {
            Check "$f passes this machine's privacy scan" (@(Find-PrivacyHits -Bytes ([IO.File]::ReadAllBytes($p)) -Needles $real -Name $f).Count -eq 0)
        }
    }

    foreach ($f in $TOOL_FILES) {
        $p = Join-Path $repoRoot "tools\$f"
        $present = Test-Path -LiteralPath $p
        Check "tools\$f exists" $present
        if ($present) {
            Check "tools\$f passes this machine's privacy scan" (@(Find-PrivacyHits -Bytes ([IO.File]::ReadAllBytes($p)) -Needles $real -Name $f).Count -eq 0)
        }
    }

    Write-Host ("selftest: {0} ok, {1} failed" -f $script:ok, $script:red)
    if ($script:red -gt 0) { exit 1 }
    exit 0
}

if ($Selftest) { Invoke-Selftest }

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
if (-not $Version) {
    $d = (& git -C $repo describe --tags --always --dirty 2>$null)
    if (-not $d) { throw "git describe gave no version - pass -Version" }
    $Version = ($d | Select-Object -First 1).Trim() -replace '^v', ''
}
if (-not $OutDir) { $OutDir = Join-Path $repo 'dist' }
# the crates' licence texts must be the ones this Cargo.lock compiles in
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "python not found - tools/engine_notices.py cannot check THIRD-PARTY-NOTICES.txt" }
& $py.Source (Join-Path $repo 'tools\engine_notices.py') $repo
if ($LASTEXITCODE -ne 0) { throw "tools/engine_notices.py failed (exit $LASTEXITCODE) - THIRD-PARTY-NOTICES.txt or NOTICE is stale; nothing was built" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path.TrimEnd('\')
if (-not $env:USERPROFILE) { throw "USERPROFILE is not set - the remap and the privacy scan need it" }
$profileDir = $env:USERPROFILE.TrimEnd('\')

Write-Host "packing crow-nest engine $Version"
Write-Host "  repo   : $repo"
Write-Host "  out    : $OutDir"

$target = Join-Path $repo 'engine\target_pack'
$flags = @(
    "--remap-path-prefix=$profileDir=~",
    "--remap-path-prefix=$repo=crow-nest",
    "-Ctarget-feature=+crt-static"
)
$savedFlags = $env:CARGO_ENCODED_RUSTFLAGS
$savedPlain = $env:RUSTFLAGS
$env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
Push-Location (Join-Path $repo 'engine')
try {
    & cargo build --release --bin serve --target-dir $target
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed: exit $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:CARGO_ENCODED_RUSTFLAGS = $savedFlags
    if ($savedPlain) { $env:RUSTFLAGS = $savedPlain }
}
$serve = Join-Path $target 'release\serve.exe'
if (-not (Test-Path -LiteralPath $serve)) { throw "no serve.exe at $serve" }

$dumpbin = Find-Dumpbin
$out = & $dumpbin /nologo /dependents $serve
if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on ${serve}: exit $LASTEXITCODE" }
$imports = Get-ImportsFromDumpbinText -Lines $out
$vc = Get-VcRuntimeImports -Imports $imports
Write-Host ("  imports: " + ($imports -join ', '))
if ($vc.Count -gt 0) { throw ("serve.exe still imports the VC++ runtime (" + ($vc -join ', ') + ") - +crt-static did not take") }

# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

$stage = Join-Path $OutDir "crow-nest-engine-$Version-win-x64"
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath $serve -Destination $stage
foreach ($f in $SHIP_FILES) { Copy-Item -LiteralPath (Join-Path $repo $f) -Destination $stage }
foreach ($f in $TOOL_FILES) { Copy-Item -LiteralPath (Join-Path $repo "tools\$f") -Destination $stage }
$nvidia = @(Find-NvidiaFiles -Dir $stage)
if ($nvidia.Count -gt 0) { throw ("the stage holds an NVIDIA file (" + (($nvidia | ForEach-Object Name) -join ', ') + ") - crow-nest does not distribute NVRTC; tools\fetch-nvrtc.ps1 fetches it from NVIDIA") }
$staged = @(Get-ChildItem -LiteralPath $stage -File | ForEach-Object Name | Sort-Object)
$want = @(Get-PackageFiles | Sort-Object)
if (Compare-Object $staged $want) { throw ("the stage holds " + ($staged -join ', ') + " - expected " + ($want -join ', ')) }

# ---------------------------------------------------------------------------
# Privacy scan: refuse before anything is zipped
# ---------------------------------------------------------------------------

$needles = Get-PrivacyNeedles -UserProfile $profileDir -UserName $env:USERNAME -ComputerName $env:COMPUTERNAME
$hits = @(Find-StageHits -Stage $stage -Needles $needles)
if ($hits.Count -gt 0) {
    $hits | Format-Table file, encoding, needle, count, first -AutoSize | Out-String | Write-Host
    throw ("privacy scan: {0} hit(s) in the staged files - nothing was zipped" -f ($hits | Measure-Object count -Sum).Sum)
}
Write-Host ("  privacy scan: 0 hits in {0} files ({1} needles, UTF-8 and UTF-16LE)" -f (Get-ChildItem -LiteralPath $stage -File).Count, $needles.Count)

# ---------------------------------------------------------------------------
# Manifest, then archive
# ---------------------------------------------------------------------------

$manifest = @(Get-Manifest -Stage $stage)
ConvertTo-Json -InputObject $manifest -Depth 3 | Set-Content -LiteralPath (Join-Path $stage 'MANIFEST.json') -Encoding utf8
$zip = Join-Path $OutDir (Get-ZipName -Ver $Version)
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal

$total = ($manifest | Measure-Object bytes -Sum).Sum
# the manifest count, not the directory count: MANIFEST.json cannot list itself
Write-Host ("RESULT: {0} files in the manifest (+ MANIFEST.json = {1} in the package), {2:N1} MB staged, {3:N1} MB zipped" -f $manifest.Count, ($manifest.Count + 1), ($total / 1MB), ((Get-Item -LiteralPath $zip).Length / 1MB)) -ForegroundColor Green
Write-Host "  $zip"
exit 0
