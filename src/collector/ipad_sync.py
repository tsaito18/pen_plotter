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
        template_path = Path(__file__).parent / "templates" / "collector.html"
        return template_path.read_text(encoding="utf-8")

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


