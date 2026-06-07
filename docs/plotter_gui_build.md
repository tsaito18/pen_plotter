# plotter_gui.exe ビルド手順 (Windows)

xDraw A4 G-code 送信 GUI を単一ポータブル exe として配布する手順。
通常の送信経路は Gradio Web UI の WebSerial 送信。`plotter_gui.exe` は WebSerial が使えない環境向けの
legacy fallback として扱う。

## 前提

- Windows ネイティブ Python (3.12 推奨、IPEX 互換のため)
- `uv` インストール済み
- リポジトリルートで実行
- ビルドは Windows 上でのみ可能 (PyInstaller はクロスコンパイル非対応)

## 手順

```pwsh
git pull
uv sync --extra build
.\scripts\build_exe.bat
```

成果物: `dist\plotter_gui.exe`

## 配布先での起動

`plotter_gui.exe` をコピーして配置するだけ。Python インストール不要。

ただし xDraw A4 を接続する PC には **CH340 USB-Serial ドライバ** が必要 (exe には同梱できない)。CH340 ドライバは https://www.wch-ic.com/downloads/CH341SER_EXE.html から入手。

## トラブルシューティング

- **起動時に画面が白いまま**: 初回起動は数秒の展開待ちあり、しばらく待つ。
- **アンチウイルスがブロック**: PyInstaller 製 exe は誤検知されやすい。配布前に SmartScreen 通過のため自己署名するか、Windows Defender に除外指定を案内する。
- **FigureCanvasTkAgg の ImportError**: `plotter_gui.spec` の `hiddenimports` に `matplotlib.backends.backend_tkagg` が入っていることを確認。
- **データファイルが見つからない**: `plotter_gui.spec` の `datas` 行と `_resources.resource_path` 経由のパス解決を確認。
