# -*- mode: python ; coding: utf-8 -*-
# Executável único da Panografia:  pyinstaller panografia.spec
# Sem argumentos abre a interface no navegador; com argumentos funciona como CLI.
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["app/launcher.py"],
    pathex=["."],
    datas=[("web", "web"), ("panografia", "panografia")],
    hiddenimports=collect_submodules("uvicorn") + ["app.server", "multipart", "python_multipart"],
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="panografia", console=True, upx=False)
