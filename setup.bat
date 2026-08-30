@echo off
rem One-time setup: create the virtual environment and install dependencies.
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ is required on PATH. & pause & exit /b 1)
if not exist ".venv" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
echo.
echo Setup complete. Run run.bat to start the application.
pause
