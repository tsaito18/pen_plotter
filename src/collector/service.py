"""手書きサンプル収集のサービス層（Web UI の「筆跡」画面から使う）。

プロファイルはリクエストごとに指定する（iPad と PC で別の人を同時に扱える）。
保存先は ``<root>/<profile>/<文字>/<文字>_<時刻>.json``。削除は
``<root>/.trash/<profile>/<文字>/`` へ移すだけなので、取り消しで戻せる。
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from src.collector.data_format import StrokePoint, StrokeSample
from src.collector.profiles import StrokeProfile, ensure_profile, list_profiles, validate_profile_id
from src.collector.stroke_recorder import StrokeRecorder

# 収集セット（重複は除く。順序は表示順）
GUIDED_CHARS: list[str] = list(
    dict.fromkeys(
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


TIER_LABELS = {0: "レポート頻出", 1: "基本", 2: "標準"}


def tier_of(char: str) -> int:
    """収集の優先度（0=最優先 / 1=優先 / 2=通常）。"""
    if char in _TIER1_CHARS:
        return 0
    if char in _TIER2_CHARS:
        return 1
    return 2


def validate_char(char: str) -> str:
    """ディレクトリ名に使える 1 文字か検証する（パス区切り・制御文字・空白は不可）。"""
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("character must be exactly one glyph")
    if char in {"/", "\\", "."} or char.isspace() or unicodedata.category(char).startswith("C"):
        raise ValueError(f"invalid character: {char!r}")
    return char


class CollectorService:
    """筆跡サンプルの保存・見直し・収集順の決定。

    Args:
        root_dir: プロファイルのルート（``data/user_strokes``）。
        kanjivg_dir: お手本に使う KanjiVG 字形ディレクトリ。
        target_samples: 1 文字あたりの目標サンプル数。
    """

    def __init__(
        self,
        root_dir: Path | str,
        *,
        kanjivg_dir: Path | str | None = None,
        target_samples: int = 3,
    ) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.kanjivg_dir = Path(kanjivg_dir) if kanjivg_dir else None
        self.target_samples = target_samples
        # サンプルが変わるたびに増える（ML のスタイル推定を読み直す合図）
        self.revision = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ プロファイル

    def profiles(self) -> list[StrokeProfile]:
        return list_profiles(self.root)

    def create_profile(self, profile_id: str) -> StrokeProfile:
        path = ensure_profile(self.root, profile_id)
        return StrokeProfile(id=path.name, path=path, character_count=0, sample_count=0)

    def _recorder(self, profile: str) -> StrokeRecorder:
        # 読むだけの操作でプロファイルのディレクトリを作らない（保存時に作られる）
        return StrokeRecorder(output_dir=self.root / validate_profile_id(profile))

    def _char_dir(self, profile: str, char: str) -> Path:
        return self.root / validate_profile_id(profile) / validate_char(char)

    def _trash_dir(self, profile: str, char: str) -> Path:
        return self.root / ".trash" / validate_profile_id(profile) / validate_char(char)

    # ------------------------------------------------------------------ 収集

    def counts(self, profile: str) -> dict[str, int]:
        """収集セットの全文字（＋セット外で書いた字）のサンプル数。"""
        base = self.root / validate_profile_id(profile)
        counts = {c: 0 for c in GUIDED_CHARS}
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if d.is_dir():
                    n = sum(1 for _ in d.glob("*.json"))
                    if n or d.name in counts:
                        counts[d.name] = n
        return counts

    def char_info(self, profile: str, char: str, counts: dict[str, int] | None = None) -> dict:
        counts = counts if counts is not None else self.counts(profile)
        validate_char(char)
        return {
            "char": char,
            "count": counts.get(char, 0),
            "target": self.target_samples,
            "tier": tier_of(char),
            "tier_label": TIER_LABELS[tier_of(char)],
        }

    def next_char(self, profile: str, prefer: str | None = None) -> dict:
        """次に書く字と、その次の字（先読み表示用）。

        ``prefer`` に前回の「次の字」を渡すと、まだ目標数に届いていない限りそれを返す
        （画面に出した「次」と実際の次がずれない）。1 字は 1 回ずつ書いて次へ進む。
        """
        counts = self.counts(profile)
        guided = {c: counts[c] for c in GUIDED_CHARS}
        if prefer in guided and guided[prefer] < self.target_samples:
            current = prefer
        else:
            current = select_next_char(guided, self.target_samples)
        completed = sum(1 for c in GUIDED_CHARS if guided[c] >= self.target_samples)
        result = {"completed": completed, "total": len(GUIDED_CHARS), "char": None, "next": None}
        if current is None:
            return result
        lookahead = dict(guided)
        lookahead[current] = self.target_samples
        result.update(self.char_info(profile, current, counts))
        result["next"] = select_next_char(lookahead, self.target_samples)
        return result

    def save(self, profile: str, character: str, strokes: list[list[dict]]) -> dict:
        char_dir = self._char_dir(profile, character)
        if not strokes or not any(strokes):
            raise ValueError("strokes are empty")
        sample = StrokeSample(
            character=character,
            strokes=[[StrokePoint.from_dict(p) for p in stroke] for stroke in strokes if stroke],
        )
        with self._lock:
            path = self._recorder(profile).save_sample(sample)
            self.revision += 1
            queue = self.queue(profile)
            if character in queue:
                queue.remove(character)
                self._write_queue(profile, queue)
        return {
            "filename": path.name,
            "character": character,
            "count": sum(1 for _ in char_dir.glob("*.json")),
        }

    # ------------------------------------------------------------------ 見直し

    def samples(self, profile: str, char: str) -> list[dict]:
        self._char_dir(profile, char)
        return self._recorder(profile).get_sample_info(char)

    def set_metadata(self, profile: str, char: str, filename: str, key: str, value: object) -> bool:
        self._char_dir(profile, char)
        return self._recorder(profile).set_metadata(char, filename, key, value)

    def delete(self, profile: str, char: str, filename: str) -> int:
        """1 サンプルをゴミ箱へ移し、残り数を返す。"""
        _check_filename(filename)
        src = self._char_dir(profile, char) / filename
        if not src.exists():
            raise FileNotFoundError(filename)
        self._move(src, self._trash_dir(profile, char) / filename)
        return sum(1 for _ in src.parent.glob("*.json")) if src.parent.exists() else 0

    def delete_all(self, profile: str, char: str) -> list[str]:
        char_dir = self._char_dir(profile, char)
        names = sorted(p.name for p in char_dir.glob("*.json")) if char_dir.is_dir() else []
        for name in names:
            self._move(char_dir / name, self._trash_dir(profile, char) / name)
        return names

    def restore(self, profile: str, char: str, filenames: Iterable[str]) -> int:
        restored = 0
        for name in filenames:
            _check_filename(name)
            src = self._trash_dir(profile, char) / name
            if src.exists():
                self._move(src, self._char_dir(profile, char) / name)
                restored += 1
        return restored

    def undo_last(self, profile: str) -> dict | None:
        """最後に保存したサンプルを取り消す（ゴミ箱へ）。"""
        base = self.root / validate_profile_id(profile)
        files = list(base.glob("*/*.json")) if base.is_dir() else []
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime_ns)
        char = latest.parent.name
        self.delete(profile, char, latest.name)
        return {"character": char, "filename": latest.name}

    def _move(self, src: Path, dst: Path) -> None:
        with self._lock:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if src.parent.is_dir() and not any(src.parent.iterdir()):
                src.parent.rmdir()
            self.revision += 1

    def stats(self, profile: str) -> dict:
        counts = self.counts(profile)
        tiers = {f"tier{t + 1}": {"total": 0, "completed": 0, "samples": 0} for t in range(3)}
        for c in GUIDED_CHARS:
            t = tiers[f"tier{tier_of(c) + 1}"]
            t["total"] += 1
            t["samples"] += counts[c]
            t["completed"] += counts[c] >= self.target_samples
        return {
            "char_counts": counts,
            "tiers": tiers,
            "total_samples": sum(counts.values()),
            "target": self.target_samples,
        }

    def issues(self, profile: str) -> dict:
        """要確認のサンプル（形の異常・画数のばらつき）。"""
        recorder = self._recorder(profile)
        return {
            "anomalies": recorder.find_anomalies(),
            "mismatches": recorder.find_stroke_mismatches(),
        }

    def glyph(self, char: str) -> list[list[dict]]:
        """お手本（KanjiVG・Y-UP・0..10）の画。無ければ空。"""
        if self.kanjivg_dir is None:
            return []
        validate_char(char)
        files = sorted((self.kanjivg_dir / char).glob(f"{char}_*.json"))
        if not files:
            return []
        try:
            return json.loads(files[0].read_text(encoding="utf-8")).get("strokes", [])
        except (OSError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------ スタジオからの依頼

    def _queue_path(self, profile: str) -> Path:
        return self.root / ".state" / f"queue-{validate_profile_id(profile)}.json"

    def queue(self, profile: str) -> list[str]:
        """スタジオから「書いて教えて」と頼まれた字（保存すると消える）。"""
        path = self._queue_path(profile)
        try:
            return list(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return []

    def set_queue(self, profile: str, chars: Iterable[str]) -> list[str]:
        unique: list[str] = []
        for c in chars:
            if validate_char(c) not in unique:
                unique.append(c)
        self._write_queue(profile, unique)
        return unique

    def _write_queue(self, profile: str, chars: list[str]) -> None:
        path = self._queue_path(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")


def _check_filename(filename: str) -> None:
    if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".json"):
        raise ValueError(f"invalid filename: {filename}")
