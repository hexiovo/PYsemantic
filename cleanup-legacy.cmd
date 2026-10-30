@echo off
setlocal
set "CLEANUP_SCRIPT=%~dp0scripts\cleanup_legacy.ps1"
if not exist "%CLEANUP_SCRIPT%" (
    echo Missing cleanup script: "%CLEANUP_SCRIPT%"
    pause
    exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CLEANUP_SCRIPT%" -Apply
set "CLEANUP_EXIT=%ERRORLEVEL%"
echo.
if not "%CLEANUP_EXIT%"=="0" echo Cleanup stopped or failed with exit code %CLEANUP_EXIT%.
pause
exit /b %CLEANUP_EXIT%
