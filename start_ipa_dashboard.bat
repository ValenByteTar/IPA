@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ipa_dashboard.ps1" %*
if errorlevel 1 (
  echo.
  echo No se pudo iniciar IPA Control Room.
  pause
)
endlocal
