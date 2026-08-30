@echo off
rem Run every test suite: unit, integration and regression.
setlocal
cd /d "%~dp0\.."
set QT_QPA_PLATFORM=offscreen
echo === unit ===
".venv\Scripts\python.exe" -m pytest tests\unit -q
echo.
echo === regression ===
".venv\Scripts\python.exe" -m pytest tests\regression -q
echo.
echo === integration ===
".venv\Scripts\python.exe" -m pytest tests\integration -q
echo.
echo Done.
pause
