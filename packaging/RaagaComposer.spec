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

# Claude is optional, and the adapter imports it inside a try block so the
# application runs without it.  But if it is installed when the bundle is
# built it has to be carried into the bundle, or adding a key to the packaged
# application would silently do nothing - and switching Claude on by adding a
# key, with no code change, is the point of the provider layer.  llama-cpp is
# deliberately not listed: it is large, it ships its own binaries, and the
# creator who wants it installs it themselves.
try:
    import anthropic                                            # noqa: F401
except ImportError:
    pass
else:
    hiddenimports.append("anthropic")

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
