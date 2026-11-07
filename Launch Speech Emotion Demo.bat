@echo off
setlocal
title Speech Emotion Detection Demo

pushd "%~dp0"

if not exist "app.py" (
    echo ERROR: app.py was not found in this folder.
    echo Keep this launcher in the main UCCD3094_Assignment2 application folder.
    pause
    popd
    exit /b 1
)

rem Prefer the project's own .venv (see README); fall back to Python 3.13.
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=py -3.13"

%PYTHON% -c "import gradio, numpy, torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python or one of the required packages is unavailable.
    echo Set up the project first, as described in README.md:
    echo     python -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    popd
    exit /b 1
)

if not exist "checkpoints\crema_d_partial_top4_best.pt" (
    echo ERROR: The trained checkpoint is missing:
    echo     checkpoints\crema_d_partial_top4_best.pt
    echo Download it with:
    echo     %PYTHON% scripts\download_checkpoint.py
    pause
    popd
    exit /b 1
)

echo Starting Speech Emotion Detection Demo...
echo The browser will open automatically when the model is ready.
echo Keep this window open while using the demo.
echo Press Ctrl+C in this window to stop the application.
echo.

%PYTHON% app.py --inbrowser %*
set "DEMO_EXIT_CODE=%ERRORLEVEL%"

if not "%DEMO_EXIT_CODE%"=="0" (
    echo.
    echo The application stopped with error code %DEMO_EXIT_CODE%.
    pause
)

popd
endlocal & exit /b %DEMO_EXIT_CODE%
