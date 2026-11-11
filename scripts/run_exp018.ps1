[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Preflight', 'Audit', 'Pilot', 'CV', 'Confirm', 'Dev', 'Verify')]
    [string]$Stage,
    [string]$RunDir = 'runs/exp018_polarity_captions/prepared',
    [string]$DataDir = 'data/upstream',
    [string]$Exp017Dir = 'runs/exp017_text_supervision',
    [string]$CacheDir = 'cache',
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [ValidateRange(0, 86400)]
    [double]$MaxSeconds = 0,
    [switch]$DeepVerify
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($DeepVerify -and $Stage -ne 'Verify') {
    Write-Error '-DeepVerify applies only to the Verify stage.' -ErrorAction Continue
    exit 2
}

function Resolve-ProjectPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $projectRoot '.venv/bin/python'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error 'The pinned .venv is missing. Prepare the project environment first.' -ErrorAction Continue
    exit 2
}

$runPath = Resolve-ProjectPath $RunDir
$logDirectory = Join-Path $runPath 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$logPath = Join-Path $logDirectory ($Stage.ToLowerInvariant() + '-' + $logId + '.log')
$arguments = @(
    (Join-Path $projectRoot 'scripts/run_st3.py'),
    ('polarity-' + $Stage.ToLowerInvariant()),
    '--run-dir', $runPath,
    '--data-dir', (Resolve-ProjectPath $DataDir),
    '--exp017-dir', (Resolve-ProjectPath $Exp017Dir),
    '--cache-dir', (Resolve-ProjectPath $CacheDir),
    '--device', $Device,
    '--max-seconds', $MaxSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($DeepVerify) { $arguments += '--deep-verify' }

$availableMemoryBefore = $null
if ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT) {
    $systemMemory = Get-CimInstance Win32_OperatingSystem
    $availableMemoryBefore = [long]$systemMemory.FreePhysicalMemory * 1024
}
$timer = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location -LiteralPath $projectRoot
try {
    Write-Host "Running EXP-018 $Stage. Log $logPath"
    & $python @arguments 2>&1 | Tee-Object -FilePath $logPath
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
$timer.Stop()
$record = [ordered]@{
    stage = $Stage
    exit_code = $resultCode
    process_seconds = $timer.Elapsed.TotalSeconds
    log = $logPath
    run_directory = $runPath
    available_ram_bytes_before = $availableMemoryBefore
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDirectory ($Stage.ToLowerInvariant() + '-' + $logId + '.json')) -Encoding utf8
switch ($resultCode) {
    0 { Write-Host "Stage finished in $($timer.Elapsed.TotalSeconds.ToString('F2')) seconds." }
    3 { Write-Warning 'A required prior stage is missing or unpublished. Inspect the log before continuing.' }
    75 { Write-Warning 'Time budget reached. Repeat the same stage to reuse completed sealed seeds or replayed folds.' }
    default { Write-Warning 'Stage failed validation or execution. Prior successful artifacts were retained.' }
}
exit $resultCode
