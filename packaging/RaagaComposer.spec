# PyInstaller spec for the Windows desktop build.
#
#   .venv\Scripts\python.exe -m PyInstaller packaging\RaagaComposer.spec
#
# Produces dist\RaagaComposer\RaagaComposer.exe - a windowed application the
# creator launches from a desktop or start-menu shortcut, with no terminal.
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "raagacomposer" / "raaga" / "data" / "raagas.json"),
     "raagacomposer/raaga/data"),
]

binaries = collect_dynamic_libs("sounddevice") + collect_dynamic_libs("soundfile")

hiddenimports = [
    "sounddevice",
    "soundfile",
    "scipy.signal",
    "scipy.special",
    "numpy",
]

a = Analysis(
    [str(ROOT / "raagacomposer" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RaagaComposer",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # a real windowed desktop app, no terminal
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RaagaComposer",
)
