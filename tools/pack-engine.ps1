<#
.SYNOPSIS
Builds serve.exe for shipping and packs it with the NVRTC runtime it loads, then
proves the package carries no trace of the machine it was built on (#131).

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
3. stages serve.exe, nvrtc64_130_0.dll and nvrtc-builtins64_133.dll from
   $env:CUDA_PATH\bin\x64 (the CUDA 13.3 names; there is no nvrtc64_133_0.dll)
   and LICENSE;
4. scans every staged file, as UTF-8 and as UTF-16LE at both byte alignments, for
   the builder's profile path, any `\Users\<name>\`, the user name as a path
   segment and the computer name, and REFUSES on any hit;
5. writes MANIFEST.json ({path, bytes, sha256}, sha256 upper-case hex, backslash
   paths - the shape of nibor1896/Crow tools/pack-release.ps1) and zips the stage as
   crow-nest-engine-<version>-win-x64.zip.

.PARAMETER Version
The version in the zip name. Default: `git describe --tags --always --dirty`
without the leading `v` (exactly `0.7.2` on the tag, `0.7.2-N-g<sha>` after it).

.PARAMETER CudaBin
Where the NVRTC DLLs are taken from. Default: $env:CUDA_PATH\bin\x64.

.PARAMETER OutDir
Where the stage and the zip go. Default: dist\ in the repo root (gitignored).

.PARAMETER Selftest
Run the checks on synthetic inputs, including ones that must fail, without building.
#>
[CmdletBinding()]
param(
    [string] $Version = "",
    [string] $CudaBin = "",
    [string] $OutDir  = "",
    [switch] $Selftest
)

$ErrorActionPreference = "Stop"

$NVRTC_DLLS = @('nvrtc64_130_0.dll', 'nvrtc-builtins64_133.dll')
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
        $n += [pscustomobject]@{ label = "USERNAME $UserName as a path segment"; regex = '[\\/]' + [regex]::Escape($UserName) + '[\\/]' }
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
    Check "the user name inside a word is NOT a hit" ((& $scan (& $u8 "rebuilder and builders")).Count -eq 0)
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
if (-not $CudaBin) {
    if (-not $env:CUDA_PATH) { throw "CUDA_PATH is not set - pass -CudaBin" }
    $CudaBin = Join-Path $env:CUDA_PATH 'bin\x64'
}
if (-not $OutDir) { $OutDir = Join-Path $repo 'dist' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path.TrimEnd('\')
if (-not $env:USERPROFILE) { throw "USERPROFILE is not set - the remap and the privacy scan need it" }
$profileDir = $env:USERPROFILE.TrimEnd('\')

Write-Host "packing crow-nest engine $Version"
Write-Host "  repo   : $repo"
Write-Host "  nvrtc  : $CudaBin"
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
foreach ($dll in $NVRTC_DLLS) {
    $src = Join-Path $CudaBin $dll
    if (-not (Test-Path -LiteralPath $src)) {
        $have = (Get-ChildItem -LiteralPath $CudaBin -Filter 'nvrtc*' -ErrorAction SilentlyContinue | ForEach-Object Name) -join ', '
        throw "missing $src (nvrtc files there: $have)"
    }
    Copy-Item -LiteralPath $src -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE') -Destination $stage

# ---------------------------------------------------------------------------
# Privacy scan: refuse before anything is zipped
# ---------------------------------------------------------------------------

$needles = Get-PrivacyNeedles -UserProfile $profileDir -UserName $env:USERNAME -ComputerName $env:COMPUTERNAME
$hits = @()
foreach ($f in Get-ChildItem -LiteralPath $stage -Recurse -File) {
    $hits += @(Find-PrivacyHits -Bytes ([IO.File]::ReadAllBytes($f.FullName)) -Needles $needles -Name $f.Name)
}
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
