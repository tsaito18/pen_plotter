from __future__ import annotations

import json
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.stroke_recorder import StrokeRecorder

GUIDED_CHARS: list[str] = list(
    # ひらがな (46)
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    # カタカナ (46)
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    # 常用漢字 (150)
    "一二三四五六七八九十百千万円年月日時分秒"
    "人口目手足心力山川田木林森火水土金石雨雪"
    "風空天気花草虫魚鳥犬猫牛馬車道町村市国王"
    "玉文字学校先生男女子父母兄弟姉妹友家族店"
    "会社食飲休走歩行来出入立見聞読書話言語計"
    "算数理科体音楽画色白黒赤青緑上下左右中前"
    "後内外東西南北大小高長新古多少強弱明暗早"
    "遅近遠広深重軽正反対同合開閉始終起動止使作持送届受取売買切払落記名"
    # 英数字・記号 (20)
    "0123456789ABCDEFabcdef+-×÷="
)

# レポート頻出文字の優先度（高→低の3段階）
# Tier 1: レポートで非常に頻出するひらがな・漢字（最優先で収集）
_TIER1_CHARS: set[str] = set(
    "のをにはでがとるたしいてれかなまうもこさよりおくえあわけせすみつね"  # 頻出ひらがな
    "実験測定結果値回路電圧周波数特性図表示"  # 理工系レポート頻出
    "的方法用使用変化比較大小高低"  # 説明文頻出
    "アイウエオカコサセタテナニノラルロン"  # 頻出カタカナ
    "0123456789"  # 数字
)
# Tier 2: 中程度の頻度（基本漢字・残りカタカナ）
_TIER2_CHARS: set[str] = set(
    "キクケシスソチツトヌネハヘホマミムメモヤユリレワヲ"  # 残りカタカナ
    "一二三四五六七八九十百千万年月日時分秒"  # 基本数量
    "人学校先生会社体力行来出入上下左右中前後内外"  # 基本漢字
    "ABCDEFabcdef+-×÷="  # 英字・記号
)


def select_next_char(
    saved_counts: dict[str, int],
    target_samples: int = 3,
    seed: int | None = None,
) -> str | None:
    """学習効率を最大化する次の文字を選択する。

    優先度ロジック:
    1. Tier優先（Tier1 > Tier2 > Tier3）— レポート頻出文字を先に完成させる
    2. 同一Tier内ではサンプル数が少ない文字を優先（0 > 1 > 2）
    3. 同一優先度内ではランダム選択（偏りを防ぐ）
    """
    import random

    rng = random.Random(seed)

    remaining = [
        c for c in GUIDED_CHARS if saved_counts.get(c, 0) < target_samples
    ]
    if not remaining:
        return None

    def _priority(ch: str) -> tuple[int, int]:
        """(Tier, サンプル数) — 小さいほど優先"""
        count = saved_counts.get(ch, 0)
        if ch in _TIER1_CHARS:
            tier = 0
        elif ch in _TIER2_CHARS:
            tier = 1
        else:
            tier = 2
        return (tier, count)

    remaining.sort(key=_priority)
    best_priority = _priority(remaining[0])
    candidates = [c for c in remaining if _priority(c) == best_priority]

    return rng.choice(candidates)


class StrokeCollectorApp:
    def __init__(self, output_dir: Path, port: int = 8080, target_samples: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.target_samples = target_samples
        self._recorder = StrokeRecorder(output_dir=self.output_dir)
        self._next_char_cache: str | None = None

    def build_html(self) -> str:
        return _HTML_PAGE

    def parse_stroke_data(self, data: dict) -> StrokeSample:
        return StrokeSample(
            character=data["character"],
            strokes=[[StrokePoint.from_dict(pt) for pt in stroke] for stroke in data["strokes"]],
        )

    def save_stroke(self, sample: StrokeSample) -> Path:
        return self._recorder.save_sample(sample)

    def list_saved_characters(self) -> list[str]:
        return self._recorder.list_characters()

    def get_progress(self) -> dict:
        saved_chars: dict[str, int] = {}
        for char in GUIDED_CHARS:
            char_dir = self.output_dir / char
            if char_dir.is_dir():
                saved_chars[char] = len(list(char_dir.glob(f"{char}_*.json")))
            else:
                saved_chars[char] = 0

        completed = [c for c in GUIDED_CHARS if saved_chars.get(c, 0) >= self.target_samples]

        # キャッシュされた next_char があり、まだ未完了ならそれを使う
        if (
            self._next_char_cache is not None
            and saved_chars.get(self._next_char_cache, 0) < self.target_samples
        ):
            current_char = self._next_char_cache
            self._next_char_cache = None
        else:
            current_char = select_next_char(saved_chars, self.target_samples)
            self._next_char_cache = None
        current_index = GUIDED_CHARS.index(current_char) if current_char else 0

        # 次の文字を先読みしてキャッシュ
        next_counts = dict(saved_chars)
        if current_char:
            next_counts[current_char] = self.target_samples
        next_char = select_next_char(next_counts, self.target_samples)
        self._next_char_cache = next_char

        # Tier情報を付与
        tier_label = ""
        if current_char:
            if current_char in _TIER1_CHARS:
                tier_label = "★ 最優先"
            elif current_char in _TIER2_CHARS:
                tier_label = "☆ 優先"
            else:
                tier_label = "通常"

        return {
            "total": len(GUIDED_CHARS),
            "completed": len(completed),
            "current_char": current_char,
            "current_index": current_index,
            "samples_for_current": saved_chars.get(current_char, 0) if current_char else 0,
            "target_samples": self.target_samples,
            "tier": tier_label,
            "next_char": next_char,
        }

    def serve(self) -> None:
        handler = partial(_RequestHandler, self)
        server = HTTPServer(("0.0.0.0", self.port), handler)
        if self.port == 0:
            self.port = server.server_address[1]
        print(f"Stroke Collector: http://0.0.0.0:{self.port}/")
        server.serve_forever()


class _RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, app: StrokeCollectorApp, *args, **kwargs) -> None:
        self.app = app
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "":
            body = self.app.build_html().encode("utf-8")
            self._respond(200, "text/html; charset=utf-8", body)
        elif self.path == "/api/characters":
            chars = self.app.list_saved_characters()
            body = json.dumps(chars, ensure_ascii=False).encode("utf-8")
            self._respond(200, "application/json", body)
        elif self.path == "/api/progress":
            progress = self.app.get_progress()
            body = json.dumps(progress, ensure_ascii=False).encode("utf-8")
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain", b"Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/stroke":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            sample = self.app.parse_stroke_data(data)
            path = self.app.save_stroke(sample)
            resp = json.dumps({"status": "ok", "path": str(path)}).encode("utf-8")
            self._respond(200, "application/json", resp)
        else:
            self._respond(404, "text/plain", b"Not Found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        pass


_HTML_PAGE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Stroke Collector</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100svh; overflow: hidden; }
  body {
    font-family: -apple-system, 'SF Pro Display', 'Hiragino Sans', sans-serif;
    background: #f2f2f7; color: #1c1c1e;
    display: flex; flex-direction: row; padding: 12px; gap: 12px;
  }

  /* === 左パネル（左手操作エリア） === */
  .left-panel {
    display: flex; flex-direction: column; gap: 10px;
    width: 220px; flex-shrink: 0;
  }
  .char-display {
    background: #fff; border-radius: 14px; padding: 16px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  #guidedChar {
    font-size: 6rem; line-height: 1; color: #1c1c1e;
    font-weight: 300; letter-spacing: -2px;
  }
  #tierBadge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px; margin-top: 6px;
  }
  .tier-high { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
  .tier-mid { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
  .tier-low { background: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }
  #sampleDots { display: flex; gap: 5px; justify-content: center; margin-top: 8px; }
  .dot {
    width: 10px; height: 10px; border-radius: 50%; background: #d1d1d6;
    transition: background 0.3s;
  }
  .dot.filled { background: #007aff; }

  /* 次の文字プレビュー */
  .next-char {
    background: #fff; border-radius: 14px; padding: 10px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .next-label { font-size: 0.7rem; color: #8e8e93; margin-bottom: 2px; }
  #nextChar { font-size: 2.5rem; color: #c7c7cc; font-weight: 300; line-height: 1.1; }

  /* ボタン群（左手で押す） */
  .controls {
    display: flex; flex-direction: column; gap: 6px;
  }
  .controls button {
    width: 100%; font-size: 0.95rem; padding: 14px 0; border-radius: 10px;
    border: none; cursor: pointer; font-weight: 500;
    transition: opacity 0.15s, transform 0.1s;
  }
  .controls button:active { transform: scale(0.97); }
  .controls button:disabled { opacity: 0.4; }
  #sendBtn { background: #007aff; color: #fff; font-size: 1.1rem; }
  #clearBtn { background: #e5e5ea; color: #1c1c1e; }
  #undoBtn { background: #e5e5ea; color: #1c1c1e; }
  #skipBtn { background: #e5e5ea; color: #8e8e93; font-size: 0.85rem; }

  /* 進捗・ステータス */
  .progress-area {
    margin-top: auto;
    background: #fff; border-radius: 14px; padding: 10px 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  #progressInfo { font-size: 0.8rem; color: #8e8e93; text-align: center; }
  .progress-track {
    width: 100%; height: 4px; background: #e5e5ea; border-radius: 2px; margin-top: 6px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, #007aff, #5856d6);
    border-radius: 2px; transition: width 0.4s ease;
  }
  #status {
    font-size: 0.8rem; color: #8e8e93; text-align: center;
    margin-top: 6px; min-height: 1.1em;
  }
  #status.success { color: #34c759; }
  #status.error { color: #ff3b30; }
  #status.warn { color: #ff9500; }

  /* === 右エリア（キャンバス） === */
  #canvas {
    display: block; flex: 1 1 0; min-width: 0;
    height: 100%; aspect-ratio: 1; max-height: 100%;
    background: #fff; border-radius: 12px; touch-action: none;
    border: 1px solid #d1d1d6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  #charInput { display: none; }
  #charList { display: none; }
  #categoryProgress { display: none; }

  /* 縦画面フォールバック */
  @media (orientation: portrait) {
    body { flex-direction: column; align-items: center; }
    .left-panel { flex-direction: row; flex-wrap: wrap; width: 100%; max-width: 560px; }
    .char-display { flex: 1; min-width: 120px; }
    .next-char { flex: 0; min-width: 80px; }
    .controls { flex-direction: row; flex: 1; }
    .controls button { flex: 1; padding: 10px 0; }
    .progress-area { width: 100%; }
    #canvas { flex: 1 1 0; width: 100%; max-width: 560px; min-height: 0; }
    #guidedChar { font-size: 4rem; }
    #nextChar { font-size: 2rem; }
  }
</style>
</head>
<body>

<!-- 左パネル: お手本 + ボタン（左手操作） -->
<div class="left-panel">
  <div class="char-display">
    <div id="guidedChar"></div>
    <div id="tierBadge"></div>
    <div id="sampleDots"></div>
  </div>
  <div class="next-char">
    <div class="next-label">次</div>
    <div id="nextChar"></div>
  </div>
  <div class="controls">
    <button id="sendBtn">送信</button>
    <button id="undoBtn">戻す</button>
    <button id="clearBtn">消去</button>
    <button id="skipBtn">Skip</button>
  </div>
  <div class="progress-area">
    <div id="progressInfo"></div>
    <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
    <div id="status"></div>
  </div>
</div>

<!-- 右: キャンバス（右手ペン操作） -->
<input id="charInput" type="text" maxlength="1">
<canvas id="canvas" width="512" height="512"></canvas>
<div id="charList"></div>
<div id="categoryProgress"></div>

<script>
(function() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const status = document.getElementById('status');
  let strokes = [];
  let currentStroke = null;
  let drawing = false;
  let guidedChars = [];
  let progressData = null;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    redraw();
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function canvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      pressure: e.pressure || 0.5,
      timestamp: e.timeStamp
    };
  }

  function drawPoint(prev, pt) {
    ctx.beginPath();
    ctx.lineWidth = 1 + pt.pressure * 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000';
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();
  }

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const stroke of strokes) {
      for (let i = 1; i < stroke.length; i++) {
        drawPoint(stroke[i-1], stroke[i]);
      }
    }
  }

  canvas.addEventListener('pointerdown', function(e) {
    e.preventDefault();
    drawing = true;
    currentStroke = [canvasCoords(e)];
  });

  canvas.addEventListener('pointermove', function(e) {
    if (!drawing || !currentStroke) return;
    e.preventDefault();
    const pt = canvasCoords(e);
    currentStroke.push(pt);
    if (currentStroke.length >= 2) {
      drawPoint(currentStroke[currentStroke.length-2], pt);
    }
  });

  canvas.addEventListener('pointerup', function(e) {
    if (!drawing) return;
    drawing = false;
    if (currentStroke && currentStroke.length > 0) {
      strokes.push(currentStroke);
    }
    currentStroke = null;
  });

  canvas.addEventListener('pointerleave', function(e) {
    if (!drawing) return;
    drawing = false;
    if (currentStroke && currentStroke.length > 0) {
      strokes.push(currentStroke);
    }
    currentStroke = null;
  });

  document.getElementById('clearBtn').addEventListener('click', function() {
    strokes = [];
    redraw();
    status.textContent = '';
  });

  document.getElementById('undoBtn').addEventListener('click', function() {
    strokes.pop();
    redraw();
  });

  document.getElementById('skipBtn').addEventListener('click', function() {
    if (progressData && progressData.current_char) {
      const idx = progressData.current_index;
      if (idx + 1 < progressData.total) {
        loadProgress();
      }
    }
  });

  const sendBtn = document.getElementById('sendBtn');
  sendBtn.addEventListener('click', function() {
    const ch = document.getElementById('charInput').value.trim();
    if (!ch) { status.textContent = '文字を入力してください'; status.className = 'warn'; return; }
    if (strokes.length === 0) { status.textContent = '文字を書いてください'; status.className = 'warn'; return; }
    if (progressData && progressData.current_char && ch !== progressData.current_char) {
      status.textContent = '「' + progressData.current_char + '」を書いてください';
      status.className = 'warn';
      return;
    }

    sendBtn.disabled = true;
    sendBtn.textContent = '...';
    const payload = { character: ch, strokes: strokes };
    fetch('/api/stroke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(d => {
      status.textContent = '保存しました';
      status.className = 'success';
      strokes = [];
      redraw();
      sendBtn.disabled = false;
      sendBtn.textContent = '送信';
      setTimeout(() => { status.textContent = ''; status.className = ''; loadProgress(); }, 500);
    })
    .catch(err => {
      status.textContent = 'エラー: ' + err;
      status.className = 'error';
      sendBtn.disabled = false;
      sendBtn.textContent = '送信';
    });
  });

  function loadCharacters() {
    fetch('/api/characters')
      .then(r => r.json())
      .then(chars => {
        document.getElementById('charList').textContent =
          '保存済み: ' + (chars.length ? chars.join(', ') : 'なし');
      });
  }

  function loadProgress() {
    fetch('/api/progress')
      .then(r => r.json())
      .then(data => {
        progressData = data;
        const guidedEl = document.getElementById('guidedChar');
        const progressEl = document.getElementById('progressInfo');
        const charInput = document.getElementById('charInput');
        const progressFill = document.getElementById('progressFill');
        const tierBadge = document.getElementById('tierBadge');
        const dotsEl = document.getElementById('sampleDots');
        const nextCharEl = document.getElementById('nextChar');

        if (data.total > 0) {
          progressFill.style.width = (data.completed / data.total * 100) + '%';
        }

        if (data.current_char) {
          guidedEl.textContent = data.current_char;
          charInput.value = data.current_char;

          // Tier badge
          if (data.tier === '★ 最優先') {
            tierBadge.textContent = 'レポート頻出';
            tierBadge.className = 'tier-high';
          } else if (data.tier === '☆ 優先') {
            tierBadge.textContent = '基本文字';
            tierBadge.className = 'tier-mid';
          } else {
            tierBadge.textContent = '標準';
            tierBadge.className = 'tier-low';
          }

          // Sample dots
          let dots = '';
          for (let i = 0; i < data.target_samples; i++) {
            dots += '<span class="dot' + (i < data.samples_for_current ? ' filled' : '') + '"></span>';
          }
          dotsEl.innerHTML = dots;

          // Next char preview
          nextCharEl.textContent = data.next_char || '';

          const pct = Math.round(data.completed / data.total * 100);
          progressEl.textContent = data.completed + ' / ' + data.total + ' (' + pct + '%)';
        } else {
          guidedEl.textContent = '✓';
          tierBadge.textContent = '';
          dotsEl.innerHTML = '';
          nextCharEl.textContent = '';
          progressEl.textContent = '完了！';
        }
      });
  }

  loadProgress();
})();
</script>
</body>
</html>
"""
