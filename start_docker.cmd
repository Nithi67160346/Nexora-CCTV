@echo off
setlocal
title NEXORA QA - Docker
if not exist "%~dp0scripts\windows\run_docker.ps1" (
  echo NEXORA files are incomplete. Extract the complete QA package before starting.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\run_docker.ps1" %*
set "nexoraExit=%ERRORLEVEL%"
echo.
pause
exit /b %nexoraExit%
