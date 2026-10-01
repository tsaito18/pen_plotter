"""Web UI の静的コンテンツ（例文・ヘルプ・HTML 断片・ブラウザ側 JS）。"""

from __future__ import annotations

EXAMPLE_REPORT_HEADER = """\
# 物理学実験レポート
## 実験目的
オームの法則 $V = IR$ を検証し、抵抗値の温度依存性について考察する。
## 実験方法
直流電源を用いて回路に電圧を印加し、電流計と電圧計の読み取り値を記録した。測定は室温 $T = 25$ ℃の環境で行った。
"""

EXAMPLE_MATH_REPORT = """\
# 微分方程式の解法
## 問題
次の二階線形微分方程式を解け。
$$
\\frac{d^2y}{dx^2} + 4y = 0
$$
## 解法
特性方程式 $\\lambda^2 + 4 = 0$ より $\\lambda = \\pm 2i$ を得る。
したがって一般解は
$$y = C_1 \\cos 2x + C_2 \\sin 2x$$
ここで $C_1$, $C_2$ は任意定数である。
"""

EXAMPLE_ESSAY = """\
近年、人工知能の発展は目覚ましく、私たちの生活に大きな変化をもたらしている。特に自然言語処理の分野では、大規模言語モデルの登場により、文章の生成や翻訳の精度が飛躍的に向上した。
しかしながら、技術の進歩には常に倫理的な課題が伴う。個人情報の保護やバイアスの問題など、解決すべき課題は多い。
今後は技術と倫理のバランスを取りながら、社会全体で議論を深めていく必要があるだろう。
"""

EXAMPLE_TABLE = """\
# 引張試験結果
各供試材の機械的性質を表に示す。

: 表1 各材料の機械的性質
| 材料 | 降伏応力 | 引張強さ | 伸び |
|---|---|---|---|
| SS400 | 245 | 400 | 28 |
| S35C | 305 | 510 | 23 |
| SUS304 | 205 | 520 | 40 |

降伏応力は SUS304 が最も低く、引張強さは S35C と SUS304 が高い。
"""

# 例文ボタンのラベル → 本文
EXAMPLES: dict[str, str] = {
    "レポートヘッダー": EXAMPLE_REPORT_HEADER,
    "数式レポート": EXAMPLE_MATH_REPORT,
    "小論文": EXAMPLE_ESSAY,
    "表サンプル": EXAMPLE_TABLE,
}

HELP_MARKDOWN = """\
### 書式リファレンス

| 書式 | 入力例 | 説明 |
|------|--------|------|
| 見出し | `# 大見出し` / `## 中見出し` | 最大3段階 |
| インライン数式 | `$V = IR$` | 文中に数式を挿入 |
| ブロック数式 | `$$E = mc^2$$` | 独立行に数式を配置 |
| 分数 | `$\\frac{a}{b}$` | 分子/分母を上下に配置 |
| 上付き・下付き | `$x^2$` / `$f_0$` | 指数・添字 |
| ギリシャ文字 | `$\\alpha$` `$\\beta$` `$\\omega$` | 主要なギリシャ文字に対応 |
| 表 | `\\| 列1 \\| 列2 \\|`<br>`\\|---\\|---\\|`<br>`\\| a \\| b \\|` | パイプ表（2行目の区切り必須）。中央寄せで描画 |
| 表キャプション | 表の直後に `: 表1 タイトル` | 表の下に中央寄せで配置 |
| 段落区切り | 空行 | 空行で段落を分割 |
| 段落字下げなし | `\\noindent 本文` | 段落先頭の字下げを抑止 |

### 対応文字

ひらがな・カタカナ・漢字（常用）・英数字・数式記号

手書きサンプルが収集済みの文字はユーザー筆跡で描画され、それ以外は KanjiVG データをベースに生成されます。プレビュー後「文字カバレッジ」で各文字の描画方式を確認できます。

### ヒント

- 設定パネルでフォントサイズや余白を調整できます
- 温度を上げると文字の揺らぎが増し、下げると整った字になります
- G-code 生成は全ページ分が自動ダウンロードされます（初回はブラウザの許可ダイアログを承認してください）
- 設定を変更したらプレビューを再生成してください（黄色の警告が出ます）
"""

STALE_BANNER_HTML = """\
<div style="padding:8px 12px;background:#fff7e6;border-left:4px solid #faad14;border-radius:4px;color:#874d00;">設定が変更されました。プレビューを再生成してください。</div>"""

FONT_HEAD = """\
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;600;700&display=swap" rel="stylesheet">
"""

WEBSERIAL_STATUS_HTML = """\
<div class="pp-status-card" id="webserial-status-panel">
  <div class="pp-status-head">
    <span id="webserial-status-badge" class="pp-badge pp-badge--idle">初期化中</span>
    <span id="webserial-status-value" class="pp-status-detail">初期化中</span>
  </div>
</div>
"""

WEBSERIAL_PROGRESS_HTML = """\
<div class="pp-progress-card" id="webserial-progress-panel">
  <div class="pp-progress-track">
    <div id="webserial-progress-bar" class="pp-progress-fill"></div>
  </div>
  <div id="webserial-progress-text" class="pp-progress-text">0 / 0 行 (0%)</div>
  <div id="webserial-current-line" class="pp-current-line">現在行: -</div>
  <div id="webserial-paper-change"
       style="display:none; margin-top:8px; padding:10px 12px; border-radius:8px;
              background:#fff7e6; border:1px solid #ffd591; color:#874d00; font-weight:600;">
  </div>
</div>
"""

WEBSERIAL_PREVIEW_HTML = """\
<canvas id="webserial-preview-canvas" class="pp-preview-canvas"></canvas>
<div id="webserial-preview-info" class="pp-preview-info">対象なし</div>
"""

WEBSERIAL_LOG_HTML = """\
<div id="webserial-log-entries" class="pp-log">
</div>
"""

# Gradio の gr.Files が JS に渡す FileData[] を順にダウンロードする（Chrome の複数
# ダウンロード許可は初回のみ。400ms 間隔でクリックする）。
TRIGGER_MULTI_DOWNLOAD_JS = r"""
(files) => {
    if (!files) { return; }
    const list = Array.isArray(files) ? files : [files];
    list.forEach((f, i) => {
        if (!f) { return; }
        const url = f.url || f.path || (typeof f === 'string' ? f : null);
        if (!url) { return; }
        const name = f.orig_name || (typeof url === 'string' ? url.split('/').pop() : 'file');
        setTimeout(() => {
            const a = document.createElement('a');
            a.href = url;
            a.download = name;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }, i * 400);
    });
}
"""
