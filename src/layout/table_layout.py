"""Markdown パイプ表（``| a | b |`` ＋区切り行）の検出。"""

from __future__ import annotations

import re

# パイプ表の区切り行セル（---, :--, --:, :-: のような形）。
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")


def split_pipe_row(line: str) -> list[str]:
    """Markdown パイプ表の 1 行をセル文字列のリストへ分解する。

    端の ``|`` は任意。``"| a | b |"`` も ``"a | b"`` も ``["a", "b"]``。各セルは
    前後空白を除去する。
    """
    s = line.strip().removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in s.split("|")]


def is_table_separator(line: str) -> bool:
    """行が表の区切り行（``|---|---|`` 等）かを判定する。

    ``|`` を含み、全セルが ``-`` 主体（``:`` 揃え指定可）であること。
    """
    if "|" not in line:
        return False
    cells = split_pipe_row(line)
    if not cells or any(c == "" for c in cells):
        return False
    return all(_TABLE_SEP_CELL_RE.match(c) is not None for c in cells)


def detect_pipe_table(paragraphs: list[str], start: int) -> tuple[list[list[str]], int] | None:
    """``paragraphs[start]`` から始まるパイプ表を検出して (行データ, 消費行数) を返す。

    表の条件: ``start`` 行がパイプ行、``start+1`` 行が区切り行。以降 ``|`` を含む行を
    データ行として取り込む。行データはヘッダ＋データ（区切りは除く）。列数が揃わない
    行は最大列数まで空セルでパディングする。表でなければ ``None``。

    Returns:
        ``(rows, consumed)``。``rows`` は ``list[list[str]]``（先頭がヘッダ）、
        ``consumed`` は表が占める段落数（区切り行を含む）。
    """
    n = len(paragraphs)
    if start + 1 >= n:
        return None
    header = paragraphs[start]
    if "|" not in header or not is_table_separator(paragraphs[start + 1]):
        return None
    rows: list[list[str]] = [split_pipe_row(header)]
    consumed = 2  # ヘッダ + 区切り
    i = start + 2
    while i < n and "|" in paragraphs[i] and not is_table_separator(paragraphs[i]):
        rows.append(split_pipe_row(paragraphs[i]))
        consumed += 1
        i += 1
    ncols = max(len(r) for r in rows)
    for r in rows:
        if len(r) < ncols:
            r.extend([""] * (ncols - len(r)))
    return rows, consumed
