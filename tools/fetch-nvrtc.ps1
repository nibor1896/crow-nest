<#
.SYNOPSIS
Fetches NVIDIA's NVRTC runtime for the crow-nest engine straight from NVIDIA's own
PyPI wheel and places it beside serve.exe. crow-nest does not distribute NVRTC.

.DESCRIPTION
serve loads NVRTC at run time to compile its CUDA kernels. The engine package holds no
NVIDIA file (tools/pack-engine.ps1 stages none), so this script gets them from the
source NVIDIA publishes: the `nvidia-cuda-nvrtc` 13.3.33 wheel on PyPI, the exact
NVRTC build the engine's PTX manifest was recorded with. It is NVIDIA's software
under NVIDIA's licence; fetching it is your own act, not a redistribution by crow-nest.

1. downloads the pinned wheel from files.pythonhosted.org (or takes -WheelFile);
2. refuses unless the wheel's size and sha256 are the pinned ones;
3. reads only the two members it needs (never extracts a path from the archive);
4. refuses unless each member's sha256 and size equal the entry in the wheel's own
   *.dist-info/RECORD (urlsafe base64, no padding);
5. only when every check passed, writes nvrtc64_130_0.dll and nvrtc-builtins64_133.dll
   into -Target. On any mismatch nothing is written.

.PARAMETER Target
Directory the two DLLs go to: the folder holding serve.exe. Created if missing.

.PARAMETER WheelFile
Use this already downloaded wheel instead of downloading. It is verified the same way.

.PARAMETER Selftest
Run the checks offline on a synthetic wheel, including ones that must refuse.
#>
[CmdletBinding()]
param(
    [string] $Target = "",
    [string] $WheelFile = "",
    [switch] $Selftest
)

$ErrorActionPreference = "Stop"

# nvidia-cuda-nvrtc 13.3.33, win_amd64, as published by NVIDIA on PyPI
$WHEEL_URL    = 'https://files.pythonhosted.org/packages/a1/42/edce72f2c5a0f587168109c867f25f4a9a6cd7289ecf0d68ed2b1070f273/nvidia_cuda_nvrtc-13.3.33-py3-none-win_amd64.whl'
$WHEEL_SHA256 = '7d2af818851c0c224d5f92221e9226e51ee23c236df4b51f9194563979c888be'
$WHEEL_SIZE   = 45319163
# wheel member -> file name beside serve.exe (the CUDA 13.3 names)
$NVRTC_MEMBERS = [ordered]@{
    'nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll'        = 'nvrtc64_130_0.dll'
    'nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll' = 'nvrtc-builtins64_133.dll'
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

# ---------------------------------------------------------------------------
# Checks (the selftest calls exactly these)
# ---------------------------------------------------------------------------

function Get-Sha256Hex {
    param([byte[]] $Bytes)
    $h = [Security.Cryptography.SHA256]::Create()
    try { return (-join ($h.ComputeHash($Bytes) | ForEach-Object { $_.ToString('x2') })) } finally { $h.Dispose() }
}

# the RECORD form of a digest: urlsafe base64, no padding
function Get-RecordHash {
    param([byte[]] $Bytes)
    $h = [Security.Cryptography.SHA256]::Create()
    try { return [Convert]::ToBase64String($h.ComputeHash($Bytes)).TrimEnd('=').Replace('+', '-').Replace('/', '_') } finally { $h.Dispose() }
}

function Read-ZipEntry {
    param($Entry)
    $s = $Entry.Open()
    try {
        $ms = New-Object IO.MemoryStream
        $s.CopyTo($ms)
        return , $ms.ToArray()
    } finally { $s.Dispose() }
}

# RECORD lines are `path,sha256=<b64>,size`; returns path -> @{ hash; size }
function Read-Record {
    param([byte[]] $Bytes)
    $map = @{}
    foreach ($line in ([Text.Encoding]::UTF8.GetString($Bytes) -split "`r?`n")) {
        if ($line -match '^(.*),([^,]*),([^,]*)$') {
            $h = $Matches[2]
            if ($h.StartsWith('sha256=')) { $h = $h.Substring(7) } else { $h = '' }   # an entry without sha256 never verifies
            $map[$Matches[1].Trim('"')] = @{ hash = $h; size = $Matches[3] }
        }
    }
    return $map
}

# Verify the wheel and its members, then write them. Throws (and writes nothing) on any
# mismatch. Returns the written file names.
function Install-NvrtcFromWheel {
    param([string] $WheelPath, [string] $Sha256, [long] $Size, $Members, [string] $TargetDir)
    if (-not (Test-Path -LiteralPath $WheelPath -PathType Leaf)) { throw "no wheel at $WheelPath" }
    $len = (Get-Item -LiteralPath $WheelPath).Length
    if ($len -ne $Size) { throw "REFUSED: the wheel is $len bytes, expected $Size" }
    $sha = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha -ne $Sha256.ToLowerInvariant()) { throw "REFUSED: the wheel's sha256 is $sha, expected $Sha256" }

    $zip = [IO.Compression.ZipFile]::OpenRead($WheelPath)
    $verified = [ordered]@{}
    try {
        $recs = @($zip.Entries | Where-Object { $_.FullName -match '^[^/]+\.dist-info/RECORD$' })
        if ($recs.Count -ne 1) { throw "REFUSED: the wheel has $($recs.Count) dist-info/RECORD files, expected exactly 1" }
        $record = Read-Record -Bytes (Read-ZipEntry $recs[0])
        foreach ($member in $Members.Keys) {
            $entry = @($zip.Entries | Where-Object { $_.FullName -ceq $member })
            if ($entry.Count -ne 1) { throw "REFUSED: the wheel has no member $member" }
            if (-not $record.ContainsKey($member)) { throw "REFUSED: RECORD has no entry for $member" }
            $bytes = Read-ZipEntry $entry[0]
            $want = $record[$member]
            $got = Get-RecordHash -Bytes $bytes
            if ($got -cne $want.hash) { throw "REFUSED: $member sha256=$got, RECORD says $($want.hash)" }
            if ("$($bytes.Length)" -ne $want.size) { throw "REFUSED: $member is $($bytes.Length) bytes, RECORD says $($want.size)" }
            $verified[$Members[$member]] = $bytes
        }
    } finally { $zip.Dispose() }

    # everything verified: only now touch the target
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    $TargetDir = (Resolve-Path -LiteralPath $TargetDir).Path
    foreach ($name in $verified.Keys) {
        $dest = Join-Path $TargetDir $name
        $part = "$dest.part"
        [IO.File]::WriteAllBytes($part, [byte[]]$verified[$name])
        Move-Item -LiteralPath $part -Destination $dest -Force
    }
    return @($verified.Keys)
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

# a small wheel: the members, plus a RECORD (generated from the members unless overridden)
function New-SyntheticWheel {
    param([string] $Path, $Members, $Record = $null, [string] $RecordName = 'synth-1.0.dist-info/RECORD')
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    if ($null -eq $Record) {
        $Record = (($Members.Keys | ForEach-Object {
            $b = [byte[]]$Members[$_]
            "$_,sha256=$(Get-RecordHash -Bytes $b),$($b.Length)"
        }) -join "`n") + "`nsynth-1.0.dist-info/RECORD,,`n"
    }
    $fs = [IO.File]::Create($Path)
    try {
        $za = New-Object IO.Compression.ZipArchive($fs, [IO.Compression.ZipArchiveMode]::Create)
        try {
            $all = [ordered]@{}
            foreach ($k in $Members.Keys) { $all[$k] = [byte[]]$Members[$k] }
            if ($RecordName) { $all[$RecordName] = [Text.Encoding]::UTF8.GetBytes($Record) }
            foreach ($k in $all.Keys) {
                $e = $za.CreateEntry($k)
                $s = $e.Open()
                try { $s.Write($all[$k], 0, $all[$k].Length) } finally { $s.Dispose() }
            }
        } finally { $za.Dispose() }
    } finally { $fs.Dispose() }
}

function Invoke-Selftest {
    Write-Host "fetch-nvrtc selftest (synthetic wheel, offline)"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("crow-nest-fetch-selftest-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        $a = [Text.Encoding]::ASCII.GetBytes('synthetic nvrtc payload')
        $b = [Text.Encoding]::ASCII.GetBytes('synthetic builtins payload')
        $members = [ordered]@{ 'p/bin/a.dll' = $a; 'p/bin/b.dll' = $b; 'p/bin/unrelated.txt' = [Text.Encoding]::ASCII.GetBytes('x') }
        $want = [ordered]@{ 'p/bin/a.dll' = 'a.dll'; 'p/bin/b.dll' = 'b.dll' }
        $wheel = Join-Path $tmp 'synth.whl'
        New-SyntheticWheel -Path $wheel -Members $members
        $wsha = (Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
        $wsize = (Get-Item -LiteralPath $wheel).Length

        # the pins themselves
        Check "the pinned URL is on files.pythonhosted.org over https" ($WHEEL_URL -match '^https://files\.pythonhosted\.org/packages/')
        Check "the pinned sha256 is 64 lower-case hex" ($WHEEL_SHA256 -cmatch '^[0-9a-f]{64}$')
        Check "the pin names the two NVRTC 13.3 DLLs" ((@($NVRTC_MEMBERS.Values) -join ',') -eq 'nvrtc64_130_0.dll,nvrtc-builtins64_133.dll')
        Check "the RECORD hash of 'abc' is urlsafe base64 without padding" ((Get-RecordHash -Bytes ([Text.Encoding]::ASCII.GetBytes('abc'))) -ceq 'ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0')

        # happy path: both members land, byte for byte, and nothing else
        $t1 = Join-Path $tmp 't1'
        $names = Install-NvrtcFromWheel -WheelPath $wheel -Sha256 $wsha -Size $wsize -Members $want -TargetDir $t1
        Check "a valid wheel installs both members" ($names.Count -eq 2)
        Check "the placed files are the member bytes" ((Get-Sha256Hex ([IO.File]::ReadAllBytes((Join-Path $t1 'a.dll')))) -eq (Get-Sha256Hex $a) -and (Get-Sha256Hex ([IO.File]::ReadAllBytes((Join-Path $t1 'b.dll')))) -eq (Get-Sha256Hex $b))
        Check "only the two members are placed (no .part left, no unrelated member)" (@(Get-ChildItem -LiteralPath $t1 -File | ForEach-Object Name | Sort-Object) -join ',' -eq 'a.dll,b.dll')

        # refusals; each must leave the target untouched
        function Refuses($label, [scriptblock] $act, $dir) {
            $threw = $false
            try { & $act } catch { $threw = $true }
            $empty = -not (Test-Path -LiteralPath $dir) -or @(Get-ChildItem -LiteralPath $dir -Force).Count -eq 0
            Check "$label (refused, nothing written)" ($threw -and $empty)
        }
        $t2 = Join-Path $tmp 't2'
        Refuses "a wheel with the wrong sha256" { Install-NvrtcFromWheel -WheelPath $wheel -Sha256 ('0' * 64) -Size $wsize -Members $want -TargetDir $t2 } $t2
        Refuses "a wheel with the wrong size" { Install-NvrtcFromWheel -WheelPath $wheel -Sha256 $wsha -Size ($wsize + 1) -Members $want -TargetDir $t2 } $t2
        Refuses "a missing wheel file" { Install-NvrtcFromWheel -WheelPath (Join-Path $tmp 'nope.whl') -Sha256 $wsha -Size $wsize -Members $want -TargetDir $t2 } $t2

        # a RECORD that lies about the second member (the wheel's own hash is pinned to what we built)
        $lie = "p/bin/a.dll,sha256=$(Get-RecordHash -Bytes $a),$($a.Length)`np/bin/b.dll,sha256=$(Get-RecordHash -Bytes $a),$($b.Length)`n"
        $w2 = Join-Path $tmp 'lie.whl'; New-SyntheticWheel -Path $w2 -Members $members -Record $lie
        Refuses "a member whose sha256 differs from RECORD" { Install-NvrtcFromWheel -WheelPath $w2 -Sha256 ((Get-FileHash -LiteralPath $w2 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w2).Length -Members $want -TargetDir $t2 } $t2
        $badsize = "p/bin/a.dll,sha256=$(Get-RecordHash -Bytes $a),$($a.Length)`np/bin/b.dll,sha256=$(Get-RecordHash -Bytes $b),$($b.Length + 1)`n"
        $w3 = Join-Path $tmp 'size.whl'; New-SyntheticWheel -Path $w3 -Members $members -Record $badsize
        Refuses "a member whose size differs from RECORD" { Install-NvrtcFromWheel -WheelPath $w3 -Sha256 ((Get-FileHash -LiteralPath $w3 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w3).Length -Members $want -TargetDir $t2 } $t2
        $w4 = Join-Path $tmp 'norec.whl'; New-SyntheticWheel -Path $w4 -Members $members -Record "p/bin/a.dll,sha256=$(Get-RecordHash -Bytes $a),$($a.Length)`n"
        Refuses "a member RECORD does not list" { Install-NvrtcFromWheel -WheelPath $w4 -Sha256 ((Get-FileHash -LiteralPath $w4 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w4).Length -Members $want -TargetDir $t2 } $t2
        $w5 = Join-Path $tmp 'nomember.whl'; New-SyntheticWheel -Path $w5 -Members ([ordered]@{ 'p/bin/a.dll' = $a })
        Refuses "a wheel without the second member" { Install-NvrtcFromWheel -WheelPath $w5 -Sha256 ((Get-FileHash -LiteralPath $w5 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w5).Length -Members $want -TargetDir $t2 } $t2
        $w6 = Join-Path $tmp 'nodist.whl'; New-SyntheticWheel -Path $w6 -Members $members -RecordName ''
        Refuses "a wheel without a RECORD" { Install-NvrtcFromWheel -WheelPath $w6 -Sha256 ((Get-FileHash -LiteralPath $w6 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w6).Length -Members $want -TargetDir $t2 } $t2
        $other = [Text.Encoding]::ASCII.GetBytes('swapped after RECORD was made')
        $swapped = [ordered]@{ 'p/bin/a.dll' = $other; 'p/bin/b.dll' = $b }
        $w7 = Join-Path $tmp 'swap.whl'
        New-SyntheticWheel -Path $w7 -Members $swapped -Record "p/bin/a.dll,sha256=$(Get-RecordHash -Bytes $a),$($a.Length)`np/bin/b.dll,sha256=$(Get-RecordHash -Bytes $b),$($b.Length)`n"
        Refuses "a swapped member (right wheel pin, RECORD from the original)" { Install-NvrtcFromWheel -WheelPath $w7 -Sha256 ((Get-FileHash -LiteralPath $w7 -Algorithm SHA256).Hash) -Size (Get-Item -LiteralPath $w7).Length -Members $want -TargetDir $t2 } $t2
    } finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }

    Write-Host ("selftest: {0} ok, {1} failed" -f $script:ok, $script:red)
    if ($script:red -gt 0) { exit 1 }
    exit 0
}

if ($Selftest) { Invoke-Selftest }

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

if (-not $Target) { throw "pass -Target <the folder holding serve.exe>" }

$tmpWheel = $null
try {
    if ($WheelFile) {
        $wheel = $WheelFile
    } else {
        $tmpWheel = Join-Path ([IO.Path]::GetTempPath()) ("nvidia_cuda_nvrtc-" + [Guid]::NewGuid().ToString('N') + ".whl")
        Write-Host "downloading $WHEEL_URL"
        $ProgressPreference = 'SilentlyContinue'
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $WHEEL_URL -OutFile $tmpWheel -UseBasicParsing
        $wheel = $tmpWheel
    }
    $names = Install-NvrtcFromWheel -WheelPath $wheel -Sha256 $WHEEL_SHA256 -Size $WHEEL_SIZE -Members $NVRTC_MEMBERS -TargetDir $Target
    $dir = (Resolve-Path -LiteralPath $Target).Path
    foreach ($n in $names) {
        $p = Join-Path $dir $n
        Write-Host ("  {0}  {1:N0} bytes  sha256 {2}  (matches the wheel's RECORD)" -f $n, (Get-Item -LiteralPath $p).Length, (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant())
    }
    Write-Host "RESULT: NVRTC 13.3.33 from NVIDIA's wheel placed in $dir" -ForegroundColor Green
} finally {
    if ($tmpWheel -and (Test-Path -LiteralPath $tmpWheel)) { Remove-Item -LiteralPath $tmpWheel -Force }
}
exit 0
