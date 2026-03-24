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


class StrokeCollectorApp:
    def __init__(self, output_dir: Path, port: int = 8080, target_samples: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.target_samples = target_samples
        self._recorder = StrokeRecorder(output_dir=self.output_dir)

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
        import random

        saved_chars: dict[str, int] = {}
        for char in GUIDED_CHARS:
            char_dir = self.output_dir / char
            if char_dir.is_dir():
                saved_chars[char] = len(list(char_dir.glob(f"{char}_*.json")))
            else:
                saved_chars[char] = 0

        completed = [c for c in GUIDED_CHARS if saved_chars.get(c, 0) >= self.target_samples]
        remaining = [c for c in GUIDED_CHARS if saved_chars.get(c, 0) < self.target_samples]

        current_char = random.choice(remaining) if remaining else None
        current_index = GUIDED_CHARS.index(current_char) if current_char else 0

        return {
            "total": len(GUIDED_CHARS),
            "completed": len(completed),
            "current_char": current_char,
            "current_index": current_index,
            "samples_for_current": saved_chars.get(current_char, 0) if current_char else 0,
            "target_samples": self.target_samples,
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
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 16px; }
  #guidedChar {
    font-size: 8rem; text-align: center; line-height: 1.1;
    margin-bottom: 8px; color: #222;
  }
  #progressInfo { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 12px; }
  .controls { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
  #charInput {
    font-size: 2rem; width: 3em; text-align: center;
    border: 2px solid #333; border-radius: 8px; padding: 4px;
  }
  button {
    font-size: 1rem; padding: 8px 16px; border-radius: 8px;
    border: 1px solid #999; background: #fff; cursor: pointer;
  }
  button.primary { background: #007aff; color: #fff; border: none; }
  #canvas {
    display: block; width: 100%; max-width: 512px;
    aspect-ratio: 1; background: #fff; border: 2px solid #333;
    border-radius: 8px; touch-action: none;
  }
  #status { margin-top: 8px; font-size: 0.9rem; color: #666; }
  #charList { margin-top: 12px; font-size: 0.9rem; color: #333; }
  #categoryProgress { margin-top: 12px; font-size: 0.95rem; color: #444; }
  #categoryProgress div { margin: 2px 0; }
</style>
</head>
<body>
<div id="guidedChar"></div>
<div id="progressInfo"></div>
<div class="controls">
  <input id="charInput" type="text" maxlength="1" placeholder="字">
  <button class="primary" id="sendBtn">送信</button>
  <button id="clearBtn">消去</button>
  <button id="undoBtn">戻す</button>
  <button id="skipBtn">スキップ</button>
</div>
<canvas id="canvas" width="512" height="512"></canvas>
<div id="status"></div>
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

  document.getElementById('sendBtn').addEventListener('click', function() {
    const ch = document.getElementById('charInput').value.trim();
    if (!ch) { status.textContent = '文字を入力してください'; return; }
    if (strokes.length === 0) { status.textContent = 'ストロークを描いてください'; return; }

    const payload = { character: ch, strokes: strokes };
    fetch('/api/stroke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(d => {
      status.textContent = '保存しました: ' + ch;
      strokes = [];
      redraw();
      loadCharacters();
      loadProgress();
    })
    .catch(err => { status.textContent = 'エラー: ' + err; });
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

        if (data.current_char) {
          guidedEl.textContent = data.current_char;
          charInput.value = data.current_char;
          const sampleNum = data.samples_for_current + 1;
          progressEl.textContent = data.completed + ' / ' + data.total
            + ' 文字完了 (' + data.current_char + ': '
            + sampleNum + '/' + data.target_samples + '回目)';
        } else {
          guidedEl.textContent = '✓';
          progressEl.textContent = '全文字の収集が完了しました！';
        }
      });
  }

  loadCharacters();
  loadProgress();
})();
</script>
</body>
</html>
"""
