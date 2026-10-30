$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$taskLog = Join-Path $PSScriptRoot 'a3_docker_activation.log'
function Write-TaskLog([string]$Message) { Add-Content -LiteralPath $taskLog -Value "$(Get-Date -Format o) $Message" }
try {
    Write-TaskLog 'Waiting for installation.'
    $result = docker wait contract_a3_runtime_setup
    if ($LASTEXITCODE -ne 0 -or "$result".Trim() -ne '0') { throw "Runtime installation failed: exit $result" }
    Write-TaskLog 'Waiting for any active batch to finish before switching runtime.'
    do {
        $batch = docker top riskdl_web
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the web container.' }
        $batchRunning = ($batch | Select-String -SimpleMatch '/app/evaluation/run_180_pdf_main.py') -ne $null
        if ($batchRunning) { Start-Sleep -Seconds 30 }
    } while ($batchRunning)
    $compose = @('compose','-f','docker-compose.yml','-f','docker-compose.web-local.yml')
    & docker @compose up -d --no-deps a3-inference
    if ($LASTEXITCODE -ne 0) { throw 'Cannot start inference container.' }
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        $health = docker inspect -f '{{.State.Health.Status}}' contract_a3_inference
        if ($health -eq 'healthy') { $ready = $true; break }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) { throw 'Inference container did not become healthy.' }
    Write-TaskLog 'Testing real-page OCR before switching the web service.'
    docker exec riskdl_web python /app/evaluation/probe_a3_docker_ocr.py
    if ($LASTEXITCODE -ne 0) { throw 'Real-page OCR probe failed; web remains on the existing runtime.' }
    & docker @compose up -d --no-deps web knowledge-worker
    if ($LASTEXITCODE -ne 0) { throw 'Could not switch web and worker to Docker inference.' }
    $webReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        docker exec riskdl_web python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/admin/login/', timeout=3)" 2>$null
        if ($LASTEXITCODE -eq 0) { $webReady = $true; break }
        Start-Sleep -Seconds 5
    }
    if (-not $webReady) { throw 'Web did not become ready; batch not resumed.' }
    Write-TaskLog 'Docker inference activated; resuming batch.'
    docker exec -d -e PYTHONPATH=/app -e PYTHONDONTWRITEBYTECODE=1 riskdl_web python /app/evaluation/run_180_pdf_main.py
    if ($LASTEXITCODE -ne 0) { throw 'Could not resume the batch.' }
    Write-TaskLog 'Batch started.'
} catch {
    Write-TaskLog "FAILED: $($_.Exception.Message)"
    exit 1
}
