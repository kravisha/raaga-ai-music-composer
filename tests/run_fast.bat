@echo off
rem Unit and regression tests only - a few seconds.
setlocal
cd /d "%~dp0\.."
set QT_QPA_PLATFORM=offscreen
".venv\Scripts\python.exe" -m pytest tests -q -m "not slow"
pause
