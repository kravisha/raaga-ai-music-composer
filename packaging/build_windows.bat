@echo off
rem Build the packaged Windows application into dist\RaagaComposer\.
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller>=6.4
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean packaging\RaagaComposer.spec
echo.
echo Built: dist\RaagaComposer\RaagaComposer.exe
echo Create a desktop shortcut to that file, or run packaging\make_shortcut.ps1.
pause
