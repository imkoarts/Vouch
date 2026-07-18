# Build on Windows with BUILD_EXE.bat.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

root = Path.cwd()

datas = [
    (str(root / "app" / "prompts"), "app/prompts"),
    (str(root / "app" / "web" / "static"), "app/web/static"),
    (str(root / "config"), "config"),
    (str(root / "alembic"), "alembic"),
    (str(root / "alembic.ini"), "."),
]

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "sqlalchemy.dialects.sqlite",
]
hiddenimports += collect_submodules("webview")

a = Analysis(
    [str(root / "desktop.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "tkinter.test"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vouch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Vouch",
)
