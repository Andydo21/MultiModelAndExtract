param(
    [ValidateSet('Ai','Ledger','All')][string]$Mode = 'Ai',
    [switch]$SkipLocalInference,
    [switch]$AllowLowMemory, # Compatibility only; startup has no free-RAM threshold.
    [switch]$Logs
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.env')) { throw 'Configure the root .env first.' }
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop first.' }
$taskCompose = @('compose','-f','docker-compose.yml','-f','docker-compose.web-local.yml')
& docker @taskCompose up -d --no-build postgres
if ($LASTEXITCODE -ne 0) { throw 'Cannot start local PostgreSQL.' }
for ($taskAttempt = 0; $taskAttempt -lt 30; $taskAttempt++) {
    $taskPgHealth = & docker inspect --format '{{.State.Health.Status}}' contract_postgres
    if ($taskPgHealth -eq 'healthy') { break }
    Start-Sleep -Seconds 2
}
if ($taskPgHealth -ne 'healthy') { throw 'Local PostgreSQL did not become healthy.' }
$taskAi = @('web','qdrant','knowledge-worker')
$taskLedger = @('workflow-service','blockchain-service','orderer.example.com','peer0.org1.example.com','fabric-chaincode','fabric-gateway','ledger-worker')
$taskLegacy = @('ai-service','ai-summary','gateway')
$taskManaged = $taskAi + $taskLedger + $taskLegacy
$taskWanted = switch ($Mode) {
    'Ai' { $taskAi }
    'Ledger' { @('web') + $taskLedger }
    'All' { $taskManaged }
}
# Read the project inventory once rather than launching Docker for each service.
$taskInventory = @(& docker @taskCompose ps -a --format json | ForEach-Object { $_ | ConvertFrom-Json })
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Compose service inventory.' }
foreach ($taskGroup in @('Ai','Other')) {
    $taskIds = @($taskInventory | Where-Object {
        $_.Service -in $taskManaged -and (($_.Service -in $taskAi) -eq ($taskGroup -eq 'Ai'))
    } | ForEach-Object { $_.ID })
    if ($taskIds.Count) {
        $taskPolicy = if ($taskGroup -eq 'Ai') { 'unless-stopped' } else { 'no' }
        & docker update --restart $taskPolicy @taskIds | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Cannot set $taskGroup restart policy." }
    }
}
$taskStopIds = @($taskInventory | Where-Object {
    $_.Service -in $taskManaged -and $_.Service -notin $taskWanted -and $_.State -eq 'running'
} | ForEach-Object { $_.ID })
if ($taskStopIds.Count) {
    & docker stop @taskStopIds | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Cannot stop services outside the selected mode.' }
}
if ($Mode -eq 'Ledger') {
    $taskRuntime = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m uvicorn ai_extract\.views\.inference_app:app'
    }
    foreach ($taskProcess in $taskRuntime) { Stop-Process -Id $taskProcess.ProcessId -ErrorAction Stop }
    $taskLlamaRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'scratch\a3-models\llama')) + '\'
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'llama-server.exe' -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($taskLlamaRoot,[StringComparison]::OrdinalIgnoreCase)
    } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction Stop }
}
& docker @taskCompose up -d --no-build --no-deps @taskWanted
if ($LASTEXITCODE -ne 0) { throw 'Cannot start selected services from existing images.' }
$taskGraph = @(& docker ps -a --filter 'name=^newidea-neo4j$' --format '{{.ID}}')
if ($Mode -eq 'Ledger') {
    if ($taskGraph.Count) { & docker stop @taskGraph | Out-Null }
} else {
    if (-not $taskGraph.Count) { throw 'Existing Neo4j container newidea-neo4j is required; do not create a second graph database.' }
    & docker update --restart unless-stopped @taskGraph | Out-Null
    & docker start @taskGraph | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Neo4j startup failed.' }
    $taskNetworks = & docker inspect --format '{{json .NetworkSettings.Networks}}' @taskGraph
    if ($taskNetworks -notmatch '"riskdl-net"') {
        & docker network connect riskdl-net @taskGraph
        if ($LASTEXITCODE -ne 0) { throw 'Cannot connect Neo4j to riskdl-net.' }
    }
}
Write-Host "Mode: $Mode. Web: http://localhost:8000/contract-intelligence/"
if ($Mode -ne 'Ledger' -and -not $SkipLocalInference) {
    $taskListening = Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
    if (-not $taskListening) {
        $taskArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+(Join-Path $PSScriptRoot 'start-a3-inference.ps1')+'"'))
        New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'scratch') -Force | Out-Null
        Start-Process powershell.exe -ArgumentList $taskArgs -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput scratch\inference-out.log -RedirectStandardError scratch\inference-error.log | Out-Null
        Write-Host 'Local inference requested; check http://localhost:8010/rag/health and scratch/inference-error.log.'
    }
}
Write-Host 'A2: http://localhost:8000/newidea/ | A3: http://localhost:8000/ai-extract/'
Write-Host 'Logs: .\show-logs.ps1 (Ctrl+C stops the log viewer only)'
if ($Logs) { & (Join-Path $PSScriptRoot 'show-logs.ps1') }
