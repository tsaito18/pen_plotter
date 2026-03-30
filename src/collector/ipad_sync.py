from __future__ import annotations

import json
import urllib.parse
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.stroke_recorder import StrokeRecorder

GUIDED_CHARS: list[str] = list(
    # ひらがな (51: 基本46 + レポート頻出5)
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    "げじでびべ"
    # カタカナ (53: 基本46 + レポート頻出7)
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "グジダッデピプ"
    # 常用漢字 (150 + レポート頻出69)
    "一二三四五六七八九十百千万円年月日時分秒"
    "人口目手足心力山川田木林森火水土金石雨雪"
    "風空天気花草虫魚鳥犬猫牛馬車道町村市国王"
    "玉文字学校先生男女子父母兄弟姉妹友家族店"
    "会社食飲休走歩行来出入立見聞読書話言語計"
    "算数理科体音楽画色白黒赤青緑上下左右中前"
    "後内外東西南北大小高長新古多少強弱明暗早"
    "遅近遠広深重軽正反対同合開閉始終起動止使作持送届受取売買切払落記名"
    # 理工系レポート頻出漢字（追加）
    "実験測定結果値回路電圧周波特性比位相差交流直列基本的抵抗"
    "働理解深通機能慣得検討論較組伝処原座確等線義習考術述認誤"
    "構成描曲標片似容両修察"
    # 英数字・記号 (28)
    "0123456789ABCDEFMRabcdeforsu+-×÷="
    # 句読点・記号・括弧
    "、。・ー（）"
)

# レポート頻出文字の優先度（高→低の3段階）
# Tier 1: レポートで非常に頻出するひらがな・漢字（最優先で収集）
_TIER1_CHARS: set[str] = set(
    "のをにはでがとるたしいてれかなまうもこさよりおくえあわけせすみつねげじびべ"  # 頻出ひらがな
    "実験測定結果値回路電圧周波数特性図表示"  # 理工系レポート頻出
    "位相差交流直列基本的抵抗理解深通機能得検討論較組伝"  # 理工系レポート追加
    "処原座確等線義習考術述認誤構成描曲標片似容両修察働慣"  # レポート文章頻出
    "的方法用使変化比較大小高低"  # 説明文頻出
    "アイウエオカコサセタテナニノラルロングジダッデピプ"  # 頻出カタカナ
    "0123456789"  # 数字
    "、。・ー（）"  # 句読点・括弧
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

    remaining = [c for c in GUIDED_CHARS if saved_counts.get(c, 0) < target_samples]
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
    def __init__(
        self,
        output_dir: Path,
        port: int = 8080,
        target_samples: int = 3,
        kanjivg_dir: Path | str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.target_samples = target_samples
        self._recorder = StrokeRecorder(output_dir=self.output_dir)
        self._next_char_cache: str | None = None
        self._kanjivg_dir = Path(kanjivg_dir) if kanjivg_dir else None

    def get_kanjivg_strokes(self, char: str) -> list[list[dict[str, float]]] | None:
        """KanjiVGのストロークデータを返す。"""
        if self._kanjivg_dir is None:
            return None
        char_dir = self._kanjivg_dir / char
        if not char_dir.is_dir():
            return None
        json_files = sorted(char_dir.glob(f"{char}_*.json"))
        if not json_files:
            return None
        try:
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            return data.get("strokes", None)
        except (json.JSONDecodeError, KeyError):
            return None

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

    def get_samples_for_char(self, char: str) -> list[dict]:
        return self._recorder.get_sample_info(char)

    def delete_sample(self, char: str, filename: str) -> bool:
        return self._recorder.delete_sample(char, filename)

    def delete_all_samples(self, char: str) -> int:
        return self._recorder.delete_all_samples(char)

    def get_anomalies(self) -> list[dict]:
        return self._recorder.find_anomalies()

    def get_stroke_mismatches(self) -> list[dict]:
        return self._recorder.find_stroke_mismatches()

    def set_sample_metadata(self, character: str, filename: str, key: str, value: object) -> bool:
        return self._recorder.set_metadata(character, filename, key, value)

    def get_collection_stats(self) -> dict:
        tier_stats = {
            "tier1": {"total": 0, "completed": 0, "samples": 0},
            "tier2": {"total": 0, "completed": 0, "samples": 0},
            "tier3": {"total": 0, "completed": 0, "samples": 0},
        }
        total_samples = 0
        char_counts: dict[str, int] = {}
        for char in GUIDED_CHARS:
            char_dir = self.output_dir / char
            count = len(list(char_dir.glob(f"{char}_*.json"))) if char_dir.is_dir() else 0
            total_samples += count
            char_counts[char] = count
            if char in _TIER1_CHARS:
                key = "tier1"
            elif char in _TIER2_CHARS:
                key = "tier2"
            else:
                key = "tier3"
            tier_stats[key]["total"] += 1
            tier_stats[key]["samples"] += count
            if count >= self.target_samples:
                tier_stats[key]["completed"] += 1
        return {"tiers": tier_stats, "total_samples": total_samples, "char_counts": char_counts}

    def get_progress(self, forced_char: str | None = None) -> dict:
        saved_chars: dict[str, int] = {}
        for ch in GUIDED_CHARS:
            char_dir = self.output_dir / ch
            if char_dir.is_dir():
                saved_chars[ch] = len(list(char_dir.glob(f"{ch}_*.json")))
            else:
                saved_chars[ch] = 0

        completed = [c for c in GUIDED_CHARS if saved_chars.get(c, 0) >= self.target_samples]

        if forced_char is not None:
            current_char = forced_char
            self._next_char_cache = None
        elif (
            self._next_char_cache is not None
            and saved_chars.get(self._next_char_cache, 0) < self.target_samples
        ):
            current_char = self._next_char_cache
            self._next_char_cache = None
        else:
            current_char = select_next_char(saved_chars, self.target_samples)
            self._next_char_cache = None
        current_index = (
            GUIDED_CHARS.index(current_char) if current_char and current_char in GUIDED_CHARS else 0
        )

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

    def _parse_path(self) -> tuple[str, dict[str, str]]:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        params = {k: v[0] for k, v in qs.items()}
        return parsed.path, params

    def _json_respond(self, code: int, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._respond(code, "application/json", body)

    def do_GET(self) -> None:
        path, params = self._parse_path()
        if path == "/" or path == "":
            body = self.app.build_html().encode("utf-8")
            self._respond(200, "text/html; charset=utf-8", body)
        elif path == "/api/characters":
            chars = self.app.list_saved_characters()
            self._json_respond(200, chars)
        elif path == "/api/progress":
            forced_char = params.get("char")
            progress = self.app.get_progress(forced_char=forced_char)
            self._json_respond(200, progress)
        elif path == "/api/samples":
            char = params.get("char", "")
            samples = self.app.get_samples_for_char(char)
            self._json_respond(200, samples)
        elif path == "/api/stats":
            stats = self.app.get_collection_stats()
            self._json_respond(200, stats)
        elif path == "/api/stroke-mismatches":
            mismatches = self.app.get_stroke_mismatches()
            self._json_respond(200, mismatches)
        elif path == "/api/anomalies":
            anomalies = self.app.get_anomalies()
            self._json_respond(200, anomalies)
        elif path == "/api/kanjivg":
            char = params.get("char", "")
            strokes = self.app.get_kanjivg_strokes(char)
            if strokes is not None:
                self._json_respond(200, {"strokes": strokes})
            else:
                self._json_respond(200, {"strokes": []})
        else:
            self._respond(404, "text/plain", b"Not Found")

    def do_POST(self) -> None:
        path, _params = self._parse_path()
        if path == "/api/stroke":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            sample = self.app.parse_stroke_data(data)
            saved = self.app.save_stroke(sample)
            self._json_respond(200, {"status": "ok", "path": str(saved)})
        elif path == "/api/samples/metadata":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            char = data.get("char", "")
            filename = data.get("file", "")
            key = data.get("key", "")
            value = data.get("value")
            try:
                result = self.app.set_sample_metadata(char, filename, key, value)
                if result:
                    self._json_respond(200, {"status": "ok"})
                else:
                    self._json_respond(404, {"error": "sample not found"})
            except ValueError as e:
                self._json_respond(400, {"error": str(e)})
        elif path == "/api/undo-last":
            self._handle_undo_last()
        else:
            self._respond(404, "text/plain", b"Not Found")

    def do_DELETE(self) -> None:
        path, params = self._parse_path()
        if path == "/api/samples":
            char = params.get("char", "")
            filename = params.get("file")
            if filename:
                try:
                    result = self.app.delete_sample(char, filename)
                except ValueError:
                    self._json_respond(404, {"error": "invalid filename"})
                    return
                if result:
                    remaining = (
                        len(list((self.app.output_dir / char).glob("*.json")))
                        if (self.app.output_dir / char).is_dir()
                        else 0
                    )
                    self._json_respond(200, {"status": "ok", "remaining": remaining})
                else:
                    self._json_respond(404, {"error": "not found"})
            else:
                count = self.app.delete_all_samples(char)
                self._json_respond(200, {"deleted_count": count})
        else:
            self._respond(404, "text/plain", b"Not Found")

    def _handle_undo_last(self) -> None:
        latest_path: Path | None = None
        latest_mtime = 0.0
        if self.app.output_dir.exists():
            for json_file in self.app.output_dir.rglob("*.json"):
                mtime = json_file.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = json_file
        if latest_path is not None:
            character = latest_path.parent.name
            latest_path.unlink()
            self._json_respond(
                200, {"status": "ok", "deleted_file": latest_path.name, "character": character}
            )
        else:
            self._json_respond(404, {"error": "no samples to undo"})

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
<title>Stroke Collector — 収集・管理</title>
<style>
  * {
    margin: 0; padding: 0; box-sizing: border-box;
    -webkit-touch-callout: none; -webkit-user-select: none; user-select: none;
  }
  html, body { height: 100svh; overflow: hidden; touch-action: manipulation; }
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
  #guideCanvas {
    width: 100%; max-width: 160px; aspect-ratio: 1;
    display: block; margin: 0 auto;
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

  /* === モードタブ（セグメントコントロール風） === */
  .mode-tabs {
    display: flex; background: #e5e5ea; border-radius: 10px; padding: 2px;
    margin-bottom: 4px;
  }
  .mode-tabs .tab {
    flex: 1; padding: 8px 0; border: none; border-radius: 8px;
    font-size: 0.85rem; font-weight: 600; cursor: pointer;
    background: transparent; color: #8e8e93;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
  }
  .mode-tabs .tab.active {
    background: #fff; color: #1c1c1e;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  }

  /* === 管理モード === */
  .manage-mode {
    display: none; flex-direction: column; gap: 10px;
    flex: 1; overflow: hidden;
  }
  .manage-mode.visible { display: flex; }
  .collect-mode { display: contents; }
  .collect-mode.hidden .left-panel,
  .collect-mode.hidden #canvas,
  .collect-mode.hidden #charInput,
  .collect-mode.hidden #charList,
  .collect-mode.hidden #categoryProgress { display: none; }

  /* 統計ダッシュボード */
  .stats-dashboard {
    background: #fff; border-radius: 14px; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .stats-dashboard.collapsed .stats-body { display: none; }
  .stats-header {
    display: flex; justify-content: space-between; align-items: center;
    cursor: pointer;
  }
  .stats-header h3 { font-size: 0.9rem; font-weight: 600; color: #1c1c1e; }
  .stats-toggle {
    font-size: 0.8rem; color: #007aff; background: none; border: none;
    cursor: pointer; font-weight: 500;
  }
  .stats-body { margin-top: 10px; }
  .stats-overall {
    font-size: 1.1rem; font-weight: 600; color: #1c1c1e;
    margin-bottom: 10px; text-align: center;
  }
  .tier-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    font-size: 0.8rem;
  }
  .tier-label { width: 50px; font-weight: 600; flex-shrink: 0; }
  .tier-track {
    flex: 1; height: 6px; background: #e5e5ea; border-radius: 3px;
    overflow: hidden;
  }
  .tier-fill { height: 100%; border-radius: 3px; transition: width 0.4s; }
  .tier-fill.t1 { background: linear-gradient(90deg, #ff9500, #ff6723); }
  .tier-fill.t2 { background: linear-gradient(90deg, #007aff, #5856d6); }
  .tier-fill.t3 { background: linear-gradient(90deg, #34c759, #30b050); }
  .tier-count { width: 80px; text-align: right; color: #8e8e93; flex-shrink: 0; }

  /* フィルターバー */
  .filter-bar {
    display: flex; gap: 8px; align-items: center;
  }
  .filter-tabs {
    display: flex; background: #e5e5ea; border-radius: 10px; padding: 2px;
  }
  .filter-tabs .filter-btn {
    padding: 6px 14px; border: none; border-radius: 8px;
    font-size: 0.8rem; font-weight: 500; cursor: pointer;
    background: transparent; color: #8e8e93;
    transition: background 0.2s, color 0.2s;
  }
  .filter-tabs .filter-btn.active {
    background: #fff; color: #1c1c1e;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  }
  .char-search {
    flex: 1; max-width: 180px; padding: 7px 12px;
    border: 1px solid #d1d1d6; border-radius: 10px;
    font-size: 0.85rem; background: #fff;
    font-family: inherit;
  }
  .char-search::placeholder { color: #c7c7cc; }

  /* 文字ギャラリーグリッド */
  .char-grid-wrapper {
    flex: 1; overflow-y: auto; -webkit-overflow-scrolling: touch;
  }
  .char-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
    gap: 6px; padding: 2px;
  }
  .char-cell {
    aspect-ratio: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #fff; border-radius: 10px; cursor: pointer;
    border: 2px solid #d1d1d6;
    transition: transform 0.1s, border-color 0.2s;
    position: relative;
  }
  .char-cell:active { transform: scale(0.95); }
  .char-cell.complete { border-color: #34c759; }
  .char-cell.partial { border-color: #ff9500; }
  .char-cell .cell-char {
    font-size: 1.6rem; font-weight: 400; line-height: 1;
    color: #1c1c1e;
  }
  .char-cell .cell-badge {
    position: absolute; top: 3px; right: 3px;
    font-size: 0.6rem; font-weight: 700; min-width: 16px; height: 16px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 8px; color: #fff;
  }
  .char-cell.complete .cell-badge { background: #34c759; }
  .char-cell.partial .cell-badge { background: #ff9500; }

  /* === サンプル詳細パネル（スライドイン） === */
  .sample-detail-overlay {
    display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.3);
  }
  .sample-detail-overlay.visible { display: flex; justify-content: flex-end; }
  .sample-detail {
    width: 380px; max-width: 90vw; height: 100%;
    background: #f2f2f7; padding: 16px;
    box-shadow: -4px 0 16px rgba(0,0,0,0.15);
    display: flex; flex-direction: column; gap: 12px;
    overflow-y: auto; -webkit-overflow-scrolling: touch;
    animation: slideIn 0.25s ease-out;
  }
  @keyframes slideIn {
    from { transform: translateX(100%); } to { transform: translateX(0); }
  }
  .detail-header {
    display: flex; align-items: center; justify-content: space-between;
  }
  .detail-header h2 { font-size: 2rem; font-weight: 300; }
  .detail-close {
    width: 44px; height: 44px; border-radius: 22px;
    background: #e5e5ea; border: none; cursor: pointer;
    font-size: 1.2rem; color: #8e8e93;
    display: flex; align-items: center; justify-content: center;
  }
  .detail-actions {
    display: flex; gap: 8px;
  }
  .recollect-btn {
    flex: 1; padding: 12px; border-radius: 10px; border: none;
    background: #007aff; color: #fff; font-weight: 600;
    font-size: 0.9rem; cursor: pointer;
  }
  .recollect-btn:active { opacity: 0.8; }
  .delete-all-btn {
    padding: 12px 16px; border-radius: 10px; border: none;
    background: #ff3b30; color: #fff; font-weight: 600;
    font-size: 0.9rem; cursor: pointer;
  }
  .delete-all-btn:active { opacity: 0.8; }
  .thumbnail-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
  }
  .thumbnail-card {
    background: #fff; border-radius: 12px; padding: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    position: relative;
  }
  .thumbnail-canvas {
    width: 120px; height: 120px; border-radius: 8px;
    border: 1px solid #e5e5ea; background: #fafafa;
  }
  .thumbnail-info {
    font-size: 0.7rem; color: #8e8e93; text-align: center;
  }
  .thumbnail-warn {
    color: #ff9500; font-weight: 600;
  }
  .delete-btn {
    width: 28px; height: 28px; border-radius: 14px;
    background: #ff3b30; color: #fff; border: none;
    font-size: 0.85rem; cursor: pointer;
    position: absolute; top: 4px; right: 4px;
    display: flex; align-items: center; justify-content: center;
  }
  .delete-btn:active { opacity: 0.7; }

  /* === 異常サンプル === */
  .anomaly-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }
  .anomaly-card {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    background: #fff; border-radius: 12px; padding: 10px 10px 10px 13px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    position: relative; overflow: hidden;
  }
  .anomaly-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: #ff3b30;
  }
  .anomaly-card .thumbnail-canvas {
    width: 100px; height: 100px;
  }
  .anomaly-char { font-size: 1.4rem; font-weight: 300; text-align: center; }
  .anomaly-info { text-align: center; }
  .anomaly-meta { font-size: 0.7rem; color: #8e8e93; }
  .anomaly-reason {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    padding: 2px 6px; border-radius: 6px;
    background: #ffebee; color: #ff3b30; border: 1px solid #ffcdd2;
    margin-right: 4px;
  }
  .anomaly-empty {
    text-align: center; color: #34c759; font-size: 0.95rem;
    padding: 40px 0; font-weight: 500;
  }
  .anomaly-delete-btn {
    width: 28px; height: 28px; border-radius: 14px;
    background: #ff3b30; color: #fff; border: none;
    font-size: 0.85rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
  }
  .anomaly-delete-btn:active { opacity: 0.7; }
  .anomaly-ignore-btn {
    padding: 4px 10px; border-radius: 8px; border: 1px solid #34c759;
    background: #f0faf3; color: #34c759; font-weight: 600;
    font-size: 0.7rem; cursor: pointer; white-space: nowrap;
  }
  .anomaly-ignore-btn:active { opacity: 0.7; }
  .anomaly-actions {
    display: flex; gap: 6px;
    align-items: center;
  }

  /* === 画数不一致グループ === */
  .mismatch-section-title {
    font-size: 0.9rem; font-weight: 600; color: #1c1c1e;
    padding: 12px 0 6px;
    border-top: 1px solid #e5e5ea;
    margin-top: 12px;
  }
  .mismatch-card {
    background: #fff; border-radius: 12px; padding: 12px 12px 12px 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 8px;
    position: relative; overflow: hidden;
  }
  .mismatch-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: #ff9500;
  }
  .mismatch-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 8px;
  }
  .mismatch-char { font-size: 2rem; font-weight: 300; }
  .mismatch-label { font-size: 0.8rem; color: #8e8e93; }
  .mismatch-samples {
    display: flex; flex-wrap: wrap; gap: 8px;
  }
  .mismatch-thumb {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
    padding: 6px; border-radius: 8px; border: 2px solid transparent;
  }
  .mismatch-thumb.outlier {
    border-color: #ff3b30; background: #fff5f5;
  }
  .mismatch-thumb canvas {
    width: 90px; height: 90px; border-radius: 6px;
    border: 1px solid #e5e5ea; background: #fafafa;
  }
  .mismatch-thumb .thumb-label {
    font-size: 0.7rem; color: #8e8e93;
  }
  .mismatch-thumb.outlier .thumb-label { color: #ff3b30; font-weight: 600; }
  .mismatch-actions {
    display: flex; gap: 6px; margin-top: 8px; justify-content: flex-end;
  }

  /* === 取り消しトースト === */
  .undo-toast {
    display: none; position: fixed; bottom: 24px; left: 50%;
    transform: translateX(-50%); z-index: 200;
    background: #1c1c1e; color: #fff; border-radius: 12px;
    padding: 12px 20px; font-size: 0.9rem; font-weight: 500;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    cursor: pointer; white-space: nowrap;
    animation: toastIn 0.25s ease-out;
  }
  .undo-toast.visible { display: block; }
  @keyframes toastIn {
    from { opacity: 0; transform: translateX(-50%) translateY(20px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
  .undo-toast .undo-link {
    color: #0a84ff; font-weight: 700; margin-left: 8px;
  }

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
    .manage-mode { max-width: 560px; width: 100%; }
    .sample-detail { width: 100%; max-width: 100vw; }
  }
</style>
</head>
<body>

<!-- 収集モード -->
<div class="collect-mode" id="collectMode">
  <!-- 左パネル: お手本 + ボタン（左手操作） -->
  <div class="left-panel">
    <div class="mode-tabs">
      <button id="collectTab" class="tab active" onclick="switchMode('collect')">収集</button>
      <button id="manageTab" class="tab" onclick="switchMode('manage')">管理</button>
    </div>
    <div class="char-display">
      <canvas id="guideCanvas" width="160" height="160"></canvas>
      <div id="guidedChar" style="display:none;"></div>
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
</div>

<!-- 管理モード -->
<div class="manage-mode" id="manageMode">
  <div class="mode-tabs">
    <button class="tab" onclick="switchMode('collect')">収集</button>
    <button class="tab active" onclick="switchMode('manage')">管理</button>
  </div>

  <!-- 統計ダッシュボード -->
  <div class="stats-dashboard" id="statsDashboard">
    <div class="stats-header" onclick="toggleStats()">
      <h3>統計</h3>
      <button class="stats-toggle" id="statsToggle">折りたたむ</button>
    </div>
    <div class="stats-body" id="statsBody">
      <div class="stats-overall" id="statsOverall"></div>
      <div class="tier-row">
        <span class="tier-label">Tier 1</span>
        <div class="tier-track"><div class="tier-fill t1" id="tier1Fill"></div></div>
        <span class="tier-count" id="tier1Count"></span>
      </div>
      <div class="tier-row">
        <span class="tier-label">Tier 2</span>
        <div class="tier-track"><div class="tier-fill t2" id="tier2Fill"></div></div>
        <span class="tier-count" id="tier2Count"></span>
      </div>
      <div class="tier-row">
        <span class="tier-label">Tier 3</span>
        <div class="tier-track"><div class="tier-fill t3" id="tier3Fill"></div></div>
        <span class="tier-count" id="tier3Count"></span>
      </div>
    </div>
  </div>

  <!-- フィルターバー -->
  <div class="filter-bar">
    <div class="filter-tabs">
      <button class="filter-btn active" onclick="filterGallery('all')">全て</button>
      <button class="filter-btn" onclick="filterGallery('incomplete')">未完了</button>
      <button class="filter-btn" onclick="filterGallery('complete')">完了</button>
      <button class="filter-btn" onclick="filterGallery('anomaly')">異常</button>
    </div>
    <input type="text" class="char-search" id="charSearch" placeholder="文字検索..." maxlength="1">
  </div>

  <!-- 文字グリッド -->
  <div class="char-grid-wrapper">
    <div class="char-grid" id="charGrid"></div>
  </div>
</div>

<!-- サンプル詳細パネル -->
<div class="sample-detail-overlay" id="sampleDetailOverlay">
  <div class="sample-detail" id="sampleDetail">
    <div class="detail-header">
      <h2 id="detailChar"></h2>
      <button class="detail-close" onclick="closeSampleDetail()">✕</button>
    </div>
    <div class="detail-actions">
      <button class="recollect-btn" id="recollectBtn">再収集</button>
      <button class="delete-all-btn" id="deleteAllBtn">全削除</button>
    </div>
    <div class="thumbnail-grid" id="thumbnailGrid"></div>
  </div>
</div>

<!-- 取り消しトースト -->
<div class="undo-toast" id="undoToast">
  保存しました<span class="undo-link">取り消す</span>
</div>

<script>
(function() {
  // ロングプレスによるコンテキストメニュー・テキスト選択を無効化
  document.addEventListener('contextmenu', function(e) { e.preventDefault(); });

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

  let kvgCache = {};

  function drawKanjiVGGuide(char) {
    const gc = document.getElementById('guideCanvas');
    const gctx = gc.getContext('2d');
    const guidedEl = document.getElementById('guidedChar');
    const rect = gc.getBoundingClientRect();
    gc.width = rect.width * window.devicePixelRatio;
    gc.height = rect.height * window.devicePixelRatio;
    gctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const strokeColors = [
      '#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
      '#264653','#a855f7','#06b6d4','#84cc16','#f97316',
      '#ec4899','#6366f1','#14b8a6','#eab308','#ef4444',
      '#8b5cf6','#0ea5e9','#22c55e','#f59e0b','#d946ef',
    ];

    function renderGuide(guideStrokes) {
      gctx.clearRect(0, 0, gc.width, gc.height);
      if (!guideStrokes || guideStrokes.length === 0) {
        guidedEl.style.display = 'block';
        gc.style.display = 'none';
        return;
      }
      guidedEl.style.display = 'none';
      gc.style.display = 'block';

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const s of guideStrokes) {
        for (const p of s) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
      }
      const kvgW = maxX - minX || 1;
      const kvgH = maxY - minY || 1;
      const cw = rect.width;
      const ch2 = rect.height;
      const margin = cw * 0.1;
      const drawW = cw - margin * 2;
      const drawH = ch2 - margin * 2;
      const scale = Math.min(drawW / kvgW, drawH / kvgH);
      const offX = margin + (drawW - kvgW * scale) / 2 - minX * scale;
      // Y反転: KanjiVGデータはY大=上、Canvas座標はY大=下
      const offY = margin + (drawH - kvgH * scale) / 2 + maxY * scale;

      gctx.lineCap = 'round';
      gctx.lineJoin = 'round';
      for (let si = 0; si < guideStrokes.length; si++) {
        const s = guideStrokes[si];
        if (s.length < 2) continue;
        const color = strokeColors[si % strokeColors.length];
        gctx.strokeStyle = color;
        gctx.lineWidth = 2;
        gctx.beginPath();
        gctx.moveTo(s[0].x * scale + offX, -s[0].y * scale + offY);
        for (let i = 1; i < s.length; i++) {
          gctx.lineTo(s[i].x * scale + offX, -s[i].y * scale + offY);
        }
        gctx.stroke();

        // 書き順番号をストロークの始点に表示
        const numX = s[0].x * scale + offX;
        const numY = -s[0].y * scale + offY;
        gctx.fillStyle = color;
        gctx.font = 'bold 11px -apple-system, sans-serif';
        gctx.textAlign = 'center';
        gctx.textBaseline = 'bottom';
        gctx.fillText(String(si + 1), numX, numY - 3);
      }
    }

    if (kvgCache[char]) {
      renderGuide(kvgCache[char]);
      return;
    }
    fetch('/api/kanjivg?char=' + encodeURIComponent(char))
      .then(r => r.json())
      .then(data => {
        kvgCache[char] = data.strokes || [];
        renderGuide(kvgCache[char]);
      });
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
      status.textContent = '';
      status.className = '';
      strokes = [];
      redraw();
      sendBtn.disabled = false;
      sendBtn.textContent = '送信';
      showUndoToast();
      loadProgress();
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

          // KanjiVGお手本を描画
          drawKanjiVGGuide(data.current_char);

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

  window.loadProgress = loadProgress;
  loadProgress();
})();

/* === 管理モード === */
let galleryData = {};
let currentFilter = 'all';

function switchMode(mode) {
  const collectMode = document.getElementById('collectMode');
  const manageMode = document.getElementById('manageMode');
  if (mode === 'manage') {
    collectMode.classList.add('hidden');
    manageMode.classList.add('visible');
    document.querySelectorAll('.mode-tabs .tab').forEach(function(t) {
      t.classList.toggle('active', t.textContent === '管理');
    });
    loadGallery();
  } else {
    collectMode.classList.remove('hidden');
    manageMode.classList.remove('visible');
    document.querySelectorAll('.mode-tabs .tab').forEach(function(t) {
      t.classList.toggle('active', t.textContent === '収集');
    });
  }
}

function toggleStats() {
  var dash = document.getElementById('statsDashboard');
  var btn = document.getElementById('statsToggle');
  dash.classList.toggle('collapsed');
  btn.textContent = dash.classList.contains('collapsed') ? '展開' : '折りたたむ';
}

function loadGallery() {
  fetch('/api/stats')
    .then(function(r) { return r.json(); })
    .then(function(stats) {
      galleryData = stats.char_counts || {};
      var tiers = stats.tiers;

      // 統計更新
      var total = stats.total_samples || 0;
      var totalChars = (tiers.tier1.total + tiers.tier2.total + tiers.tier3.total);
      var completedChars = (tiers.tier1.completed + tiers.tier2.completed + tiers.tier3.completed);
      var pct = totalChars > 0 ? Math.round(completedChars / totalChars * 100) : 0;
      document.getElementById('statsOverall').textContent =
        total + ' / ' + (totalChars * 3) + ' サンプル (' + pct + '%)';

      function setTier(id, tier) {
        var fillPct = tier.total > 0 ? (tier.completed / tier.total * 100) : 0;
        document.getElementById(id + 'Fill').style.width = fillPct + '%';
        document.getElementById(id + 'Count').textContent =
          tier.completed + ' / ' + tier.total + ' 文字';
      }
      setTier('tier1', tiers.tier1);
      setTier('tier2', tiers.tier2);
      setTier('tier3', tiers.tier3);

      renderGallery();
    });
}

function renderGallery() {
  var grid = document.getElementById('charGrid');
  grid.style.display = '';
  grid.style.flexDirection = '';
  grid.style.gap = '';
  var searchVal = (document.getElementById('charSearch').value || '').trim();
  var html = '';

  var keys = Object.keys(galleryData);
  if (keys.length === 0) return;

  keys.forEach(function(ch) {
    var count = galleryData[ch];
    if (searchVal && ch !== searchVal) return;
    if (currentFilter === 'complete' && count < 3) return;
    if (currentFilter === 'incomplete' && count >= 3) return;

    var cls = 'char-cell';
    if (count >= 3) cls += ' complete';
    else if (count > 0) cls += ' partial';

    html += '<div class="' + cls + '" onclick="showSampleDetail(\\'' + ch + '\\')">';
    html += '<span class="cell-char">' + ch + '</span>';
    if (count > 0) {
      html += '<span class="cell-badge">' + count + '</span>';
    }
    html += '</div>';
  });

  grid.innerHTML = html;
}

function filterGallery(filter) {
  currentFilter = filter;
  var labels = {all: '全て', incomplete: '未完了', complete: '完了', anomaly: '異常'};
  document.querySelectorAll('.filter-tabs .filter-btn').forEach(function(btn) {
    btn.classList.toggle('active', btn.textContent === labels[filter]);
  });
  if (filter === 'anomaly') {
    loadAnomalies();
  } else {
    renderGallery();
  }
}

function ignoreSample(ch, filename, cardEl) {
  fetch('/api/samples/metadata', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({char: ch, file: filename, key: 'ignore_anomaly', value: true})
  })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    if (d.status === 'ok' && cardEl) {
      cardEl.style.transition = 'opacity 0.2s';
      cardEl.style.opacity = '0';
      setTimeout(function() { cardEl.remove(); checkAnomalyEmpty(); }, 200);
    }
  });
}

function checkAnomalyEmpty() {
  var grid = document.getElementById('charGrid');
  var remaining = grid.querySelectorAll('.anomaly-card, .mismatch-card');
  if (remaining.length === 0) {
    grid.innerHTML = '<div class="anomaly-empty">異常なサンプルはありません ✓</div>';
  }
}

function loadAnomalies() {
  var grid = document.getElementById('charGrid');
  grid.innerHTML = '<div style="color:#8e8e93;font-size:0.85rem;">読み込み中...</div>';
  grid.style.display = 'block';
  grid.style.flexDirection = '';
  grid.style.gap = '';

  Promise.all([
    fetch('/api/anomalies').then(function(r) { return r.json(); }),
    fetch('/api/stroke-mismatches').then(function(r) { return r.json(); })
  ]).then(function(results) {
    var anomalies = results[0];
    var mismatches = results[1];
    grid.innerHTML = '';

    if (anomalies.length === 0 && mismatches.length === 0) {
      grid.innerHTML = '<div class="anomaly-empty">異常なサンプルはありません ✓</div>';
      return;
    }

    if (anomalies.length > 0) {
      var listDiv = document.createElement('div');
      listDiv.className = 'anomaly-list';
      anomalies.forEach(function(item) {
        var card = document.createElement('div');
        card.className = 'anomaly-card';

        var cvs = document.createElement('canvas');
        cvs.className = 'thumbnail-canvas';
        cvs.width = 200; cvs.height = 200;
        card.appendChild(cvs);

        var charSpan = document.createElement('div');
        charSpan.className = 'anomaly-char';
        charSpan.textContent = item.character;
        card.appendChild(charSpan);

        var reasons = document.createElement('div');
        item.reasons.forEach(function(r) {
          var badge = document.createElement('span');
          badge.className = 'anomaly-reason';
          badge.textContent = r;
          reasons.appendChild(badge);
        });
        card.appendChild(reasons);

        var meta = document.createElement('div');
        meta.className = 'anomaly-meta';
        meta.textContent = item.stroke_count + '画 / ' + item.point_count + '点';
        card.appendChild(meta);

        var actions = document.createElement('div');
        actions.className = 'anomaly-actions';

        var delBtn = document.createElement('button');
        delBtn.className = 'anomaly-delete-btn';
        delBtn.textContent = '✕';
        delBtn.onclick = (function(cardRef) {
          return function(e) {
            e.stopPropagation();
            if (!confirm('このサンプルを削除しますか？')) return;
            fetch('/api/samples?char=' + encodeURIComponent(item.character) + '&file=' + encodeURIComponent(item.filename), {
              method: 'DELETE'
            })
            .then(function(r) { return r.json(); })
            .then(function(d) {
              if (d.status === 'ok') {
                cardRef.style.transition = 'opacity 0.2s';
                cardRef.style.opacity = '0';
                setTimeout(function() { cardRef.remove(); checkAnomalyEmpty(); }, 200);
              }
            });
          };
        })(card);
        actions.appendChild(delBtn);

        var ignoreBtn = document.createElement('button');
        ignoreBtn.className = 'anomaly-ignore-btn';
        ignoreBtn.textContent = '異常ではない';
        ignoreBtn.onclick = (function(cardRef) {
          return function(e) {
            e.stopPropagation();
            ignoreSample(item.character, item.filename, cardRef);
          };
        })(card);
        actions.appendChild(ignoreBtn);

        card.appendChild(actions);
        listDiv.appendChild(card);
        renderStrokeThumbnail(cvs, item.strokes, 200);
      });
      grid.appendChild(listDiv);
    }

    if (mismatches.length > 0) {
      loadStrokeMismatches(grid, mismatches);
    }
  });
}

function loadStrokeMismatches(container, mismatches) {
  var title = document.createElement('div');
  title.className = 'mismatch-section-title';
  title.textContent = '画数不一致 (' + mismatches.length + '文字)';
  container.appendChild(title);

  mismatches.forEach(function(group) {
    var card = document.createElement('div');
    card.className = 'mismatch-card';

    var header = document.createElement('div');
    header.className = 'mismatch-header';
    var charEl = document.createElement('span');
    charEl.className = 'mismatch-char';
    charEl.textContent = group.character;
    header.appendChild(charEl);
    var labelEl = document.createElement('span');
    labelEl.className = 'mismatch-label';
    labelEl.textContent = '画数不一致（最頻: ' + group.mode_count + '画）';
    header.appendChild(labelEl);
    card.appendChild(header);

    var samplesDiv = document.createElement('div');
    samplesDiv.className = 'mismatch-samples';
    group.samples.forEach(function(sample) {
      var thumb = document.createElement('div');
      thumb.className = 'mismatch-thumb' + (sample.is_outlier ? ' outlier' : '');

      var cvs = document.createElement('canvas');
      cvs.width = 180; cvs.height = 180;
      thumb.appendChild(cvs);

      var label = document.createElement('div');
      label.className = 'thumb-label';
      label.textContent = sample.stroke_count + '画' + (sample.is_outlier ? ' ⚠' : '');
      thumb.appendChild(label);

      if (sample.is_outlier) {
        var delBtn = document.createElement('button');
        delBtn.className = 'anomaly-delete-btn';
        delBtn.textContent = '✕';
        delBtn.onclick = (function(thumbRef, cardRef) {
          return function(e) {
            e.stopPropagation();
            if (!confirm('このサンプルを削除しますか？')) return;
            fetch('/api/samples?char=' + encodeURIComponent(group.character) + '&file=' + encodeURIComponent(sample.filename), {
              method: 'DELETE'
            })
            .then(function(r) { return r.json(); })
            .then(function(d) {
              if (d.status === 'ok') {
                thumbRef.style.transition = 'opacity 0.2s';
                thumbRef.style.opacity = '0';
                setTimeout(function() {
                  thumbRef.remove();
                  var remaining = cardRef.querySelectorAll('.mismatch-thumb.outlier');
                  if (remaining.length === 0) {
                    cardRef.style.transition = 'opacity 0.2s';
                    cardRef.style.opacity = '0';
                    setTimeout(function() { cardRef.remove(); checkAnomalyEmpty(); }, 200);
                  }
                }, 200);
              }
            });
          };
        })(thumb, card);
        thumb.appendChild(delBtn);
      }

      samplesDiv.appendChild(thumb);
      renderStrokeThumbnail(cvs, sample.strokes, 180);
    });
    card.appendChild(samplesDiv);

    var hasOutliers = group.samples.some(function(s) { return s.is_outlier; });
    if (hasOutliers) {
      var actionsDiv = document.createElement('div');
      actionsDiv.className = 'mismatch-actions';
      var ignoreBtn = document.createElement('button');
      ignoreBtn.className = 'anomaly-ignore-btn';
      ignoreBtn.textContent = '異常ではない';
      ignoreBtn.onclick = (function(cardRef) {
        return function(e) {
          e.stopPropagation();
          var outliers = group.samples.filter(function(s) { return s.is_outlier; });
          var promises = outliers.map(function(s) {
            return fetch('/api/samples/metadata', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({char: group.character, file: s.filename, key: 'ignore_stroke_mismatch', value: true})
            }).then(function(r) { return r.json(); });
          });
          Promise.all(promises).then(function() {
            cardRef.style.transition = 'opacity 0.2s';
            cardRef.style.opacity = '0';
            setTimeout(function() { cardRef.remove(); checkAnomalyEmpty(); }, 200);
          });
        };
      })(card);
      actionsDiv.appendChild(ignoreBtn);
      card.appendChild(actionsDiv);
    }

    container.appendChild(card);
  });
}

var detailChar = '';

function showSampleDetail(ch) {
  detailChar = ch;
  document.getElementById('detailChar').textContent = ch;
  document.getElementById('sampleDetailOverlay').classList.add('visible');
  document.getElementById('recollectBtn').onclick = function() { recollect(ch); };
  document.getElementById('deleteAllBtn').onclick = function() { deleteAllSamples(ch); };
  loadSampleThumbnails(ch);
}

function closeSampleDetail() {
  document.getElementById('sampleDetailOverlay').classList.remove('visible');
  detailChar = '';
}

document.getElementById('sampleDetailOverlay').addEventListener('click', function(e) {
  if (e.target === this) closeSampleDetail();
});

function loadSampleThumbnails(ch) {
  var grid = document.getElementById('thumbnailGrid');
  grid.innerHTML = '<div style="color:#8e8e93;font-size:0.85rem;">読み込み中...</div>';
  fetch('/api/samples?char=' + encodeURIComponent(ch))
    .then(function(r) { return r.json(); })
    .then(function(samples) {
      grid.innerHTML = '';
      if (samples.length === 0) {
        grid.innerHTML = '<div style="color:#8e8e93;font-size:0.85rem;">サンプルなし</div>';
        return;
      }
      samples.forEach(function(sample) {
        var card = document.createElement('div');
        card.className = 'thumbnail-card';

        var cvs = document.createElement('canvas');
        cvs.className = 'thumbnail-canvas';
        cvs.width = 240; cvs.height = 240;
        card.appendChild(cvs);

        var info = document.createElement('div');
        info.className = 'thumbnail-info';
        var infoText = sample.stroke_count + ' 画 / ' + sample.point_count + ' 点';
        if (sample.point_count < 5) {
          infoText += ' <span class="thumbnail-warn">⚠ 点数少</span>';
        } else if (sample.stroke_count > 10) {
          infoText += ' <span class="thumbnail-warn">⚠ 画数多</span>';
        }
        info.innerHTML = infoText;
        card.appendChild(info);

        var delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '✕';
        delBtn.onclick = function(e) { e.stopPropagation(); deleteSample(ch, sample.filename); };
        card.appendChild(delBtn);

        grid.appendChild(card);
        renderStrokeThumbnail(cvs, sample.strokes, 240);
      });
    });
}

function renderStrokeThumbnail(canvasEl, strokeData, size) {
  var ctx2 = canvasEl.getContext('2d');
  ctx2.clearRect(0, 0, size, size);
  if (!strokeData || strokeData.length === 0) return;

  var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  strokeData.forEach(function(stroke) {
    stroke.forEach(function(pt) {
      if (pt.x < minX) minX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y > maxY) maxY = pt.y;
    });
  });

  var w = maxX - minX || 1;
  var h = maxY - minY || 1;
  var pad = size * 0.1;
  var drawSize = size - pad * 2;
  var scale = Math.min(drawSize / w, drawSize / h);
  var offX = pad + (drawSize - w * scale) / 2;
  var offY = pad + (drawSize - h * scale) / 2;

  ctx2.lineWidth = 2;
  ctx2.lineCap = 'round';
  ctx2.lineJoin = 'round';
  ctx2.strokeStyle = '#1c1c1e';

  strokeData.forEach(function(stroke) {
    if (stroke.length < 2) return;
    ctx2.beginPath();
    ctx2.moveTo((stroke[0].x - minX) * scale + offX, (stroke[0].y - minY) * scale + offY);
    for (var i = 1; i < stroke.length; i++) {
      ctx2.lineTo((stroke[i].x - minX) * scale + offX, (stroke[i].y - minY) * scale + offY);
    }
    ctx2.stroke();
  });
}

function deleteSample(ch, filename) {
  if (!confirm('このサンプルを削除しますか？')) return;
  fetch('/api/samples?char=' + encodeURIComponent(ch) + '&file=' + encodeURIComponent(filename), {
    method: 'DELETE'
  })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    if (d.status === 'ok') {
      loadSampleThumbnails(ch);
      loadGallery();
    }
  });
}

function deleteAllSamples(ch) {
  if (!confirm('「' + ch + '」の全サンプルを削除しますか？')) return;
  fetch('/api/samples?char=' + encodeURIComponent(ch), { method: 'DELETE' })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    closeSampleDetail();
    loadGallery();
  });
}

function recollect(ch) {
  closeSampleDetail();
  switchMode('collect');
  fetch('/api/progress?char=' + encodeURIComponent(ch))
    .then(function(r) { return r.json(); })
    .then(function() { location.href = '/?char=' + encodeURIComponent(ch); });
}

/* === 取り消しトースト === */
var undoTimer = null;

function showUndoToast() {
  var toast = document.getElementById('undoToast');
  toast.classList.add('visible');
  if (undoTimer) clearTimeout(undoTimer);
  undoTimer = setTimeout(function() {
    toast.classList.remove('visible');
    undoTimer = null;
  }, 5000);
}

document.getElementById('undoToast').addEventListener('click', function() {
  var toast = document.getElementById('undoToast');
  toast.classList.remove('visible');
  if (undoTimer) { clearTimeout(undoTimer); undoTimer = null; }
  fetch('/api/undo-last', { method: 'POST' })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      var st = document.getElementById('status');
      if (d.status === 'ok') {
        st.textContent = '「' + d.character + '」を取り消しました';
        st.className = 'warn';
        loadProgress();
      } else {
        st.textContent = '取り消し失敗';
        st.className = 'error';
      }
      setTimeout(function() { st.textContent = ''; st.className = ''; }, 2000);
    });
});

document.getElementById('charSearch').addEventListener('input', function() {
  renderGallery();
});
</script>
</body>
</html>
"""
