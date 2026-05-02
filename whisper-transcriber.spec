# whisper-transcriber.spec
from PyInstaller.compat import is_darwin, is_win

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'mpv',
        'PyQt6.QtOpenGL',
        'PyQt6.QtOpenGLWidgets',
        'OpenGL',
        'OpenGL.GL',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    icon='assets/icon.icns' if is_darwin else
         'assets/icon.ico'  if is_win else
         'assets/icon.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='WhisperTranscriber',
)

# macOS: wrap in a .app bundle
if is_darwin:
    app = BUNDLE(
        coll,
        name='WhisperTranscriber.app',
        icon='assets/icon.icns',
        bundle_identifier='com.vladt.WhisperTranscriber',
        info_plist={
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
        },
    )