# Gradio WebSerial xDraw A4 送信手順

Gradio Web UI の「プロッタ送信 (WebSerial)」を通常の送信経路として使う。
旧 Tkinter GUI は WebSerial が使えない環境向けの予備。

## 前提

- Windows PC に xDraw A4 を USB 接続する。
- Chrome または Edge で `http://localhost` の Gradio UI を開く。
- CH340 USB-Serial ドライバを Windows に入れる。
- Firefox / Safari は WebSerial 非対応。
- `share=True`、LAN 共有、Hugging Face Spaces では Secure Context や USB 権限の制約で対象外。

## 起動

```sh
python scripts/run_ui.py
```

ブラウザで表示されたローカル URL を開く。WebSerial はブラウザ側で USB シリアルへ接続するため、
Python サーバ側では COM ポートを保持しない。

## 操作

「プロッタ送信」タブは Step 1〜4 のカードで構成される。上から順に進める。

### Step 1 接続

1. xDraw A4 を USB 接続する。
2. Web UI の「プロッタ送信」タブを開く。
3. 「接続」を押し、ブラウザのポート選択ダイアログで CH340 / xDraw の COM ポートを選ぶ。
4. 状態バッジが「未接続」→「接続中」に変わる（送信中は「送信中」）。

### Step 2 原点・ペン確認

5. 「Home」で `$H`, `G4 P1`, `G92 X0 Y297 Z0`, `G90` を送る。
6. 「ペン Up」「ペン Down」で Z 軸ペン制御を確認する。

### Step 3 送信データ選択

7. 送信元を選ぶ。
   - 生成済み G-code: 「生成」タブの「G-code 生成」で作ったファイル（既定）。
   - アップロード G-code: 選ぶとファイル欄が表示される。`.gcode`, `.nc`, `.txt` をアップロード。

### Step 4 送信実行

8. 「送信開始」を押す。確認ダイアログは出ず、ファイル読込後そのまま送信を開始する（行数は送信ログに `[start] stream N lines` として残る）。
9. 通常停止は「停止」。次の行送信前に止まる。
10. 緊急時は「緊急停止」。GRBL realtime `!` と soft reset `0x18` を即送信する。
11. 進捗は状態カードの進捗バー、詳細は「送信ログ」アコーディオンで確認する。

## 安全仕様

- ボーレートは 115200。
- ペン Up は `G1G90 Z0.5 F5000`。
- ペン Down は `G1G90 Z5 F5000`。
- G-code は空行、`;` コメント、括弧コメントを除外して送信する。
- 各行送信後、`ok` を待って次行へ進む。
- `error:*`、`ALARM:*`、応答タイムアウト時は停止し、対象行と応答をログに出す。
- 送信中は接続、切断、Home、ペン操作をロックする。

## 実機チェック

- [ ] Chrome / Edge で WebSerial 使用可能ログが出る。
- [ ] 接続ダイアログで CH340 / xDraw の COM ポートを選べる。
- [ ] Home 実行後、紙座標が `(0,0)=左下`, `(210,297)=右上` になる。
- [ ] ペン Up / Down が Z 軸で動く。
- [ ] 短い安全 G-code をアップロード送信できる。
- [ ] Web UI 生成 G-code を送信できる。
- [ ] 停止で次行送信前に止まる。
- [ ] 緊急停止で物理動作が止まり、再 Home 後に復帰できる。
- [ ] `error:*` または `ALARM:*` を返す行で停止し、ログに行番号と内容が出る。
