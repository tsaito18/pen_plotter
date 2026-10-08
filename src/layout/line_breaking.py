"""禁則処理付きの改行。"""

from collections.abc import Callable

LINE_START_PROHIBITED: set[str] = set("。、，．）」』】〉》〕!?！？ー")
LINE_END_PROHIBITED: set[str] = set("（「『【〈《〔")


def is_halfwidth(ch: str) -> bool:
    """欧文の幅で組む文字か（ASCII とギリシャ文字）。"""
    return ord(ch) < 128 or 0x0391 <= ord(ch) <= 0x03F5


def break_paragraph_by_width(
    text: str,
    max_width: float,
    char_width: Callable[[str], float],
) -> list[str]:
    """段落を ``max_width`` 以内で改行する（``char_width`` は 1 文字の幅）。

    行末禁止文字（開き括弧）で終わる行は 1 文字前で切り、次行頭が行頭禁止文字
    （句読点・閉じ括弧）なら現在行へ追い込む。
    """
    lines: list[str] = []
    i = 0

    while i < len(text):
        width = 0.0
        end = i

        while end < len(text):
            next_width = char_width(text[end])
            if end > i and width + next_width > max_width:
                break

            width += next_width
            end += 1

            if width > max_width:
                break

        if end >= len(text):
            lines.append(text[i:end])
            break

        # 行末禁止文字チェック: 行末が行末禁止文字なら1文字前で切る
        if text[end - 1] in LINE_END_PROHIBITED:
            end -= 1

        # 行頭禁止文字チェック: 次行の先頭が行頭禁止文字なら現在行に含める
        elif end < len(text) and text[end] in LINE_START_PROHIBITED:
            end += 1

        if end == i:
            end += 1

        lines.append(text[i:end])
        i = end

    return lines
