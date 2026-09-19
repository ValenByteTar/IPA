param(
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$StartOrchestrator,
    [switch]$NoWatchdog
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PythonW = Join-Path $Root ".venv\Scripts\pythonw.exe"
$Dashboard = Join-Path $Root "scripts\operations\web_dashboard.py"
$Watchdog = Join-Path $Root "scripts\operations\dashboard_watchdog.py"
$Url = "http://127.0.0.1:$Port"

# Use pythonw.exe (no console window) if available, to prevent accidental closure
$DashboardPython = if (Test-Path $PythonW) { $PythonW } else { $Python }

if (-not (Test-Path $Python)) {
    Write-Host "No se encontró el entorno virtual en $Python" -ForegroundColor Red
    Write-Host "Ejecuta: py -3.12 -m venv .venv" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $Dashboard)) {
    Write-Host "No se encontró el dashboard en $Dashboard" -ForegroundColor Red
    exit 1
}

# Ensure Ollama is running (LLM backend for the agent; also the CPU fallback target)
$ollamaUp = $false
try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
    $ollamaUp = $true
} catch { }

if (-not $ollamaUp) {
    $ollamaExe = (Get-Command ollama.exe -ErrorAction SilentlyContinue).Source
    if (-not $ollamaExe) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $candidate) { $ollamaExe = $candidate }
    }
    if ($ollamaExe) {
        Write-Host "Iniciando Ollama (backend LLM)..." -ForegroundColor Cyan
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
                $ollamaUp = $true; break
            } catch { }
        }
        if ($ollamaUp) {
            Write-Host "Ollama listo." -ForegroundColor Green
        } else {
            Write-Host "Ollama no respondio en 127.0.0.1:11434; el chat del agente puede fallar." -ForegroundColor Yellow
        }
    } else {
        Write-Host "No se encontro ollama.exe; el chat del agente no tendra backend LLM." -ForegroundColor Yellow
    }
}

# Ensure local SearXNG (preferred web-search backend for research_topic).
# The watchdog also keeps it alive; this makes it available from boot.
if (-not $env:IPA_SEARXNG_URL) { $env:IPA_SEARXNG_URL = "http://127.0.0.1:8888" }
$searxngUp = $false
try {
    $null = Invoke-WebRequest -Uri $env:IPA_SEARXNG_URL -UseBasicParsing -TimeoutSec 2
    $searxngUp = $true
} catch { }

$searxngLocal = $env:IPA_SEARXNG_URL -match "127\.0\.0\.1|localhost|\[::1\]"
if (-not $searxngUp -and $searxngLocal) {
    $compose = Join-Path $Root ".devin\searxng\docker-compose.yml"
    $dockerUp = $false
    try {
        $null = docker version --format "{{.Server.Version}}" 2>$null
        $dockerUp = ($LASTEXITCODE -eq 0)
    } catch { }
    if (-not $dockerUp) {
        $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerDesktop) {
            Write-Host "Iniciando Docker Desktop para SearXNG..." -ForegroundColor Cyan
            Start-Process -FilePath $dockerDesktop
            for ($attempt = 0; $attempt -lt 90; $attempt++) {
                Start-Sleep -Seconds 1
                $null = docker version --format "{{.Server.Version}}" 2>$null
                if ($LASTEXITCODE -eq 0) { $dockerUp = $true; break }
            }
        }
    }
    if ($dockerUp -and (Test-Path $compose)) {
        Write-Host "Levantando SearXNG (backend de búsqueda web)..." -ForegroundColor Cyan
        docker compose -f $compose up -d 2>$null | Out-Null
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            Start-Sleep -Milliseconds 500
            try {
                $null = Invoke-WebRequest -Uri $env:IPA_SEARXNG_URL -UseBasicParsing -TimeoutSec 2
                $searxngUp = $true; break
            } catch { }
        }
    }
    if ($searxngUp) {
        Write-Host "SearXNG listo en $env:IPA_SEARXNG_URL." -ForegroundColor Green
    } else {
        Write-Host "SearXNG no disponible; la investigación web caerá al fallback DDG." -ForegroundColor Yellow
    }
}

$existing = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "El dashboard ya está ejecutándose en $Url" -ForegroundColor Green
} else {
    $env:PYTHONPATH = Join-Path $Root "src"
    $env:PYTHONIOENCODING = "utf-8"
    Write-Host "Levantando IPA Control Room..." -ForegroundColor Cyan
    Start-Process -FilePath $DashboardPython `
        -ArgumentList @("-u", $Dashboard, "--host", "127.0.0.1", "--port", $Port) `
        -WorkingDirectory $Root

    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 300
        $connection = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connection) { $ready = $true; break }
    }
    if (-not $ready) {
        Write-Host "El dashboard no respondió en el puerto $Port" -ForegroundColor Red
        exit 1
    }
    Write-Host "Dashboard listo." -ForegroundColor Green
}

# Launch watchdog to keep dashboard alive (unless disabled)
if (-not $NoWatchdog -and (Test-Path $Watchdog)) {
    $watchdogRunning = Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*dashboard_watchdog*' }
    if ($watchdogRunning) {
        Write-Host "Watchdog ya está corriendo (PID $($watchdogRunning.ProcessId))." -ForegroundColor Green
    } else {
        Write-Host "Iniciando watchdog (auto-restart si el dashboard cae)..." -ForegroundColor Cyan
        Start-Process -FilePath $DashboardPython `
            -ArgumentList @("-u", $Watchdog, "--host", "127.0.0.1", "--port", $Port, "--interval", "5") `
            -WorkingDirectory $Root `
            -WindowStyle Minimized
        Write-Host "Watchdog activo." -ForegroundColor Green
    }
}

if ($StartOrchestrator) {
    $Orchestrator = Join-Path $Root "scripts\orchestrator.py"
    if (Test-Path $Orchestrator) {
        Write-Host "Iniciando orquestador del corpus principal..." -ForegroundColor Cyan
        Start-Process -FilePath $Python `
            -ArgumentList @("-u", $Orchestrator) `
            -WorkingDirectory $Root
    } else {
        Write-Host "No se encontró orchestrator.py; se continúa solo con el dashboard." -ForegroundColor Yellow
    }
}

if (-not $NoBrowser) {
    Start-Process $Url
}

Write-Host ""
Write-Host "IPA Control Room: $Url" -ForegroundColor Green
Write-Host "Desde la interfaz puedes lanzar scraper, Reporter, ingestas, deep dives y revisiones." -ForegroundColor Gray
Write-Host "El dashboard, el watchdog y Ollama corren sin ventana de consola." -ForegroundColor Gray
Write-Host "Para detener todo: taskkill /F /IM pythonw.exe /IM python.exe" -ForegroundColor Gray
Write-Host "Si el dashboard cae, el watchdog lo reinicia automaticamente." -ForegroundColor Gray
