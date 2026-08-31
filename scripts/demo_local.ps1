param(
    [string]$OutputDir = "output/local_dev_demo",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location -LiteralPath $repoRoot

$repoPythonPath = (Join-Path $repoRoot "src") + [IO.Path]::PathSeparator + $repoRoot
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $repoPythonPath
}
else {
    $env:PYTHONPATH = $repoPythonPath + [IO.Path]::PathSeparator + $env:PYTHONPATH
}

$rawJobs = Join-Path $repoRoot "data\fixtures\raw_jobs_llm_intern_skill_scout_sample.json"
$sessionDir = Join-Path $OutputDir "sessions\bytedance_posttraining"
$routeDecisionPath = Join-Path $OutputDir "demo_route_decision.json"
$executionPath = Join-Path $OutputDir "demo_next_action_execution.txt"
$trackerExecutionPath = Join-Path $OutputDir "demo_tracker_execution.txt"

& $PythonExe -m job_agent.cli `
    --raw-jobs $rawJobs `
    --selected-job-id raw_002 `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Job agent demo flow failed with exit code $LASTEXITCODE."
}

$routeJson = & $PythonExe -m job_agent.cli `
    --session-dir $sessionDir `
    --plan-next-action "continue mock interview"
if ($LASTEXITCODE -ne 0) {
    throw "Session route planning failed with exit code $LASTEXITCODE."
}

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$routeDecisionFullPath = [System.IO.Path]::GetFullPath($routeDecisionPath)
[System.IO.File]::WriteAllText(
    $routeDecisionFullPath,
    (($routeJson -join [Environment]::NewLine) + [Environment]::NewLine),
    $utf8NoBom
)

$executionText = & $PythonExe -m job_agent.cli `
    --session-dir $sessionDir `
    --run-next-action "continue mock interview"
if ($LASTEXITCODE -ne 0) {
    throw "Session next-action execution failed with exit code $LASTEXITCODE."
}

$executionFullPath = [System.IO.Path]::GetFullPath($executionPath)
[System.IO.File]::WriteAllText(
    $executionFullPath,
    (($executionText -join [Environment]::NewLine) + [Environment]::NewLine),
    $utf8NoBom
)

$trackerExecutionText = & $PythonExe -m job_agent.cli `
    --session-dir $sessionDir `
    --run-next-action "track application"
if ($LASTEXITCODE -ne 0) {
    throw "Session tracker execution failed with exit code $LASTEXITCODE."
}

$trackerExecutionFullPath = [System.IO.Path]::GetFullPath($trackerExecutionPath)
[System.IO.File]::WriteAllText(
    $trackerExecutionFullPath,
    (($trackerExecutionText -join [Environment]::NewLine) + [Environment]::NewLine),
    $utf8NoBom
)

Write-Output $routeJson
Write-Output $executionText
Write-Output $trackerExecutionText
Write-Output "Demo artifacts written to $OutputDir"
Write-Output "Route decision written to $routeDecisionPath"
Write-Output "Next-action execution written to $executionPath"
Write-Output "Tracker execution written to $trackerExecutionPath"
