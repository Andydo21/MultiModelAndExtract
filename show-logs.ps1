param(
    [ValidateSet('All','Web','Worker','Models','Surya','Neo4j','Qdrant')][string]$Service = 'All',
    [ValidateRange(1,10000)][int]$Tail = 50,
    [ValidateRange(0,86400)][int]$Seconds = 0,
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $OutputEncoding
Set-Location $PSScriptRoot
$taskContainers = [ordered]@{
    Web='riskdl_web'; Worker='contract_knowledge_worker'; Neo4j='newidea-neo4j'; Qdrant='contract_qdrant'
}
$taskSources = @()
foreach ($taskName in $taskContainers.Keys) {
    if ($Service -eq 'All' -or $Service -eq $taskName) {
        $taskSources += [pscustomobject]@{Name=$taskName; Type='Docker'; Path=$taskContainers[$taskName]}
    }
}
if ($Service -in @('All','Models')) {
    # Previous launchers used different filenames; follow the latest live family.
    foreach ($taskStream in @('out','error')) {
        $taskFile = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'scratch') -Filter "inference*-$taskStream.log" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($taskFile) { $taskSources += [pscustomobject]@{Name="Models/$taskStream"; Type='File'; Path=$taskFile.FullName} }
    }
}
if ($Service -in @('All','Surya')) {
    $taskManagedLog = Join-Path $PSScriptRoot 'scratch\surya-managed.log'
    if (Test-Path -LiteralPath $taskManagedLog) { $taskSources += [pscustomobject]@{Name='Surya/managed'; Type='File'; Path=$taskManagedLog} }
    foreach ($taskStream in @('out','error')) {
        $taskPath = Join-Path $PSScriptRoot "scratch\surya-$taskStream.log"
        if (Test-Path -LiteralPath $taskPath) { $taskSources += [pscustomobject]@{Name="Surya/$taskStream"; Type='File'; Path=$taskPath} }
    }
}
if (-not $taskSources.Count) { Write-Host 'No logs found yet. Start the project first.'; return }
if ($Once) {
    foreach ($taskSource in $taskSources) {
        Write-Host "[$($taskSource.Name)] $($taskSource.Path)"
        if ($taskSource.Type -eq 'Docker') {
            # Docker writes ordinary application logs to stderr too.
            $taskPreviousPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & docker logs --timestamps --tail $Tail $taskSource.Path 2>&1 | ForEach-Object { "$_" }
                if ($LASTEXITCODE -ne 0) { throw "Docker log reader failed: $($taskSource.Path)" }
            } finally { $ErrorActionPreference = $taskPreviousPreference }
        }
        else { Get-Content -LiteralPath $taskSource.Path -Tail $Tail }
    }
    return
}
$taskJobs = @()
$taskDeadline = if ($Seconds) { (Get-Date).AddSeconds($Seconds) } else { [datetime]::MaxValue }
try {
    foreach ($taskSource in $taskSources) {
        Write-Host "Following [$($taskSource.Name)] $($taskSource.Path)"
        $taskJobs += Start-Job -ArgumentList $taskSource,$Tail -ScriptBlock {
            param($source,$lines)
            $ErrorActionPreference = 'Continue'
            $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
            [Console]::OutputEncoding = $OutputEncoding
            if ($source.Type -eq 'Docker') {
                & docker logs --timestamps --tail $lines --follow $source.Path 2>&1 |
                    ForEach-Object { "[$($source.Name)] $_" }
                if ($LASTEXITCODE -ne 0) { throw "Docker log reader failed: $($source.Path)" }
            } else {
                Get-Content -LiteralPath $source.Path -Tail $lines -Wait |
                    ForEach-Object { "[$($source.Name)] $_" }
            }
        }
    }
    Write-Host 'Live logs. Ctrl+C closes log readers; services keep running.'
    while ((Get-Date) -lt $taskDeadline -and @($taskJobs | Where-Object { $_.State -in @('Running','NotStarted') }).Count) {
        foreach ($taskJob in $taskJobs) { Receive-Job -Job $taskJob }
        Start-Sleep -Milliseconds 200
    }
    foreach ($taskJob in $taskJobs) { Receive-Job -Job $taskJob }
} finally {
    foreach ($taskJob in $taskJobs) {
        Stop-Job -Job $taskJob -ErrorAction SilentlyContinue
        Remove-Job -Job $taskJob -Force -ErrorAction SilentlyContinue
    }
}
