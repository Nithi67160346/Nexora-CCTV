@echo off
setlocal
title NEXORA QA - Stop Docker
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\run_docker.ps1" -Action stop %*
set "nexoraExit=%ERRORLEVEL%"
echo.
pause
exit /b %nexoraExit%
