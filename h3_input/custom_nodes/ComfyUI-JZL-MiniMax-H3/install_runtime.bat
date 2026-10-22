@echo off
setlocal
cd /d "%~dp0"

REM Detect Python: prefer the ComfyUI portable build, otherwise fall back to PATH.
set "PY="
if exist "..\..\..\python_embeded\python.exe" (
    set "PY=..\..\..\python_embeded\python.exe"
)

if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [ERROR] Python not found. Please run: python install_runtime.py
    pause
    exit /b 1
)

echo Installing the local llama-server runtime ^(llama.cpp b10436^)...
echo Python used: %PY%
echo.

"%PY%" install_runtime.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo [OK] Runtime installed. Restart ComfyUI to use local LLM/VLM nodes.
pause
