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
    Write-Host "Iniciando watchdog (auto-restart si el dashboard cae)..." -ForegroundColor Cyan
    Start-Process -FilePath $DashboardPython `
        -ArgumentList @("-u", $Watchdog, "--host", "127.0.0.1", "--port", $Port, "--interval", "5") `
        -WorkingDirectory $Root `
        -WindowStyle Minimized
    Write-Host "Watchdog activo." -ForegroundColor Green
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
Write-Host "El dashboard y el watchdog corren sin ventana de consola (pythonw)." -ForegroundColor Gray
Write-Host "Para detener todo: taskkill /F /IM pythonw.exe /IM python.exe" -ForegroundColor Gray
Write-Host "Si el dashboard cae, el watchdog lo reinicia automaticamente." -ForegroundColor Gray
