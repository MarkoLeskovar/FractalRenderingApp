# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
import inspect
import colormaps
from PyInstaller.utils.hooks import collect_dynamic_libs


# O------------------------------------------------------------------------------O
# | SETUP DYNAMIC APP DATA                                                       |
# O------------------------------------------------------------------------------O

# Find hidden dynamic libs
dynamic_libs = collect_dynamic_libs('glfw') + collect_dynamic_libs('freetype')

# Add colormaps data
colormaps_src = os.path.join(os.path.dirname(inspect.getfile(colormaps.__init__)), 'colormaps')
colormaps_dst = os.path.join(os.curdir, 'colormaps', 'colormaps')

# Add shaders data
shaders_src = os.path.join(os.pardir, 'fractals', 'shaders')
shaders_dst = os.path.join(os.curdir, 'fractals', 'shaders')

# Add shaders data
assets_src = os.path.join(os.pardir, 'fractals', 'assets')
assets_dst = os.path.join(DISTPATH, 'main_app')

# Icon data
icon_src = os.path.join(os.pardir, 'fractals', 'assets', 'mandelbrot.ico')


# O------------------------------------------------------------------------------O
# | PYINSTALLER SPEC FILE                                                        |
# O------------------------------------------------------------------------------O

a = Analysis(
    [os.path.join(os.pardir, 'main_app.py')],
    pathex=[],
    binaries=dynamic_libs,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
a.datas += Tree(colormaps_src, prefix=colormaps_dst)
a.datas += Tree(shaders_src, prefix=shaders_dst)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FractalRendering',
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
    icon=icon_src
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_app',
)


# O------------------------------------------------------------------------------O
# | COPY ASSETS                                                                  |
# O------------------------------------------------------------------------------O
shutil.copytree(assets_src, assets_dst, dirs_exist_ok=True)