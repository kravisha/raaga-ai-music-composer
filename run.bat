@echo off
rem Launch the Raaga AI Music Composer desktop application.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m raagacomposer
) else (
    echo Virtual environment not found. Run setup.bat first.
    pause
)
