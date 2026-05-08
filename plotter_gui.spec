# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for xDraw A4 G-code Sender GUI.

ビルド: Windows 上で `pyinstaller plotter_gui.spec` を実行。
出力: dist/plotter_gui.exe (単一ファイル, ポータブル)。

データファイル (data/report_paper.jpg) は exe に同梱され、実行時に
sys._MEIPASS 配下へ展開される。src/plotter_gui/_resources.py の
resource_path がそれを透過的に解決する。
"""

block_cipher = None


a = Analysis(
    ['scripts/run_plotter_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/report_paper.jpg', 'data'),
    ],
    hiddenimports=[
        # Tkinter + matplotlib のバックエンドを明示。Tk は標準ライブラリだが
        # FigureCanvasTkAgg の遅延 import を PyInstaller が拾えないことがある。
        'matplotlib.backends.backend_tkagg',
        # tkinter サブモジュールは PyInstaller の自動検出で漏れることがある。
        # 起動時 ModuleNotFoundError を回避するため明示。
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 配布物を絞るため、ML 系・Web UI 系の重い依存を除外する。
        # GUI のみで使う依存だけを含める。
        'torch',
        'torchvision',
        'gradio',
        'scipy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='plotter_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,           # UPX があれば圧縮、無ければ警告のみで続行
    upx_exclude=[],
    runtime_tmpdir=None,
    # 配布版は --windowed (subsystem=windows): コンソール窓を出さず
    # ダブルクリック起動時の見た目を整える。クラッシュ詳細は
    # scripts/run_plotter_gui.py の sys.excepthook 経由で
    # plotter_gui_error.log に保存されるため、エラー追跡には支障なし。
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,          # アイコン未指定 (将来 .ico を入れたら指定)
)
