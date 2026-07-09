from __future__ import annotations

import re
from dataclasses import dataclass, field

# ブロック図 DSL の開始/終了フェンス。既存のコードブロックや $$ 数式・表検出と
# 衝突しないよう明示トークン ``blockdiagram`` で判定する。
_FENCE_START = "```blockdiagram"
_FENCE_END = "```"

_FEEDBACK_RE = re.compile(r"^feedback\s*:\s*(.+)$", re.IGNORECASE)
_SUM_LABELS = {"σ", "+", "sum"}

# デンドログラム DSL の開始フェンス。終了フェンスは blockdiagram と共通(_FENCE_END)。
_DENDRO_FENCE_START = "```dendrogram"
_DENDRO_ORDER_RE = re.compile(r"^order\s*:\s*(.+)$", re.IGNORECASE)
_DENDRO_MERGE_RE = re.compile(r"^(\S+)\s*\+\s*(\S+)\s*:\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class DiagramNode:
    """ブロック図の1ノード。

    ``kind``: ``"box"``（矩形）/ ``"circle"``（円形ノード）/ ``"signal"``（裸の信号ラベル）。
    ``is_sum``: circle かつラベルが Σ/+/sum 相当のとき True（加算点）。
    """

    kind: str
    label: str
    is_sum: bool = False


@dataclass
class DiagramSpec:
    """``parse_blockdiagram`` の結果。メインチェーンと任意の帰還経路。"""

    nodes: list[DiagramNode] = field(default_factory=list)
    feedback: list[DiagramNode] = field(default_factory=list)

    @property
    def is_closed_loop(self) -> bool:
        return len(self.feedback) > 0


def _parse_node_token(token: str) -> DiagramNode | None:
    """``-->`` で区切られた1トークンを ``DiagramNode`` へ変換する。空トークンは None。"""
    token = token.strip()
    if not token:
        return None
    if token.startswith("[") and token.endswith("]") and len(token) >= 2:
        return DiagramNode(kind="box", label=token[1:-1].strip())
    if token.startswith("(") and token.endswith(")") and len(token) >= 2:
        inner = token[1:-1].strip()
        if inner.lower() in _SUM_LABELS:
            return DiagramNode(kind="circle", label="Σ", is_sum=True)
        return DiagramNode(kind="circle", label=inner)
    return DiagramNode(kind="signal", label=token)


def parse_blockdiagram(lines: list[str]) -> DiagramSpec:
    """フェンス内の行群からブロック図を解析する。

    最初に見つかった非 ``feedback:`` 行をメインチェーンとして採用する
    （``-->`` 区切りのノード列）。``feedback: ...`` 行があれば帰還経路として
    別途パースする。不正入力（空行のみ・``-->`` のみ等）は空の
    ``DiagramSpec`` を返す（例外を投げない）。
    """
    chain_line: str | None = None
    feedback_line: str | None = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = _FEEDBACK_RE.match(s)
        if m:
            if feedback_line is None:
                feedback_line = m.group(1)
            continue
        if chain_line is None:
            chain_line = s

    def _parse_chain(text: str | None) -> list[DiagramNode]:
        if not text:
            return []
        return [n for tok in text.split("-->") if (n := _parse_node_token(tok)) is not None]

    return DiagramSpec(nodes=_parse_chain(chain_line), feedback=_parse_chain(feedback_line))


def detect_blockdiagram(paragraphs: list[str], start: int) -> tuple[DiagramSpec, int] | None:
    """``paragraphs[start]`` から始まる ``` ```blockdiagram ``` フェンスを検出する。

    Returns:
        ``(spec, consumed)``。``consumed`` は開始/終了フェンスを含む段落数。
        フェンス開始でない、または終端フェンスが見つからない場合は ``None``。
    """
    n = len(paragraphs)
    if start >= n or paragraphs[start].strip() != _FENCE_START:
        return None
    body: list[str] = []
    i = start + 1
    while i < n and paragraphs[i].strip() != _FENCE_END:
        body.append(paragraphs[i])
        i += 1
    if i >= n:
        return None  # 終端フェンスが見つからない（未終端） → 図として扱わない
    consumed = i - start + 1
    return parse_blockdiagram(body), consumed


@dataclass
class DendrogramMerge:
    """デンドログラムの1結合。

    ``left``/``right`` は構成要素（葉ラベル文字）の集合として正規化したクラスタ
    識別子（順不同）。``height`` は結合距離。
    """

    left: frozenset[str]
    right: frozenset[str]
    height: float


@dataclass
class DendrogramSpec:
    """``parse_dendrogram`` の結果。``order`` は葉の左→右配置順、``merges`` は結合順。"""

    order: list[str] = field(default_factory=list)
    merges: list[DendrogramMerge] = field(default_factory=list)


def parse_dendrogram(lines: list[str]) -> DendrogramSpec:
    """フェンス内の行群からデンドログラムを解析する。

    ``order: <葉ラベル群>`` 行が葉の並び順、``X + Y : h`` 行が結合を表す
    （``X``/``Y`` は文字集合として正規化、``h`` は結合距離）。不正な行
    （数値変換不可・パターン不一致等）は無視して例外を投げない。
    """
    order: list[str] = []
    merges: list[DendrogramMerge] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = _DENDRO_ORDER_RE.match(s)
        if m:
            if not order:
                order = m.group(1).split()
            continue
        m = _DENDRO_MERGE_RE.match(s)
        if m:
            left_tok, right_tok, height_s = m.groups()
            try:
                height = float(height_s)
            except ValueError:
                continue
            merges.append(
                DendrogramMerge(left=frozenset(left_tok), right=frozenset(right_tok), height=height)
            )
    return DendrogramSpec(order=order, merges=merges)


def detect_dendrogram(paragraphs: list[str], start: int) -> tuple[DendrogramSpec, int] | None:
    """``paragraphs[start]`` から始まる ``` ```dendrogram ``` フェンスを検出する。

    Returns:
        ``(spec, consumed)``。``consumed`` は開始/終了フェンスを含む段落数。
        フェンス開始でない、または終端フェンスが見つからない場合は ``None``。
    """
    n = len(paragraphs)
    if start >= n or paragraphs[start].strip() != _DENDRO_FENCE_START:
        return None
    body: list[str] = []
    i = start + 1
    while i < n and paragraphs[i].strip() != _FENCE_END:
        body.append(paragraphs[i])
        i += 1
    if i >= n:
        return None  # 終端フェンスが見つからない（未終端） → 図として扱わない
    consumed = i - start + 1
    return parse_dendrogram(body), consumed
