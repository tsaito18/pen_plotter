"""禁則処理付き改行モジュール。

日本語組版の禁則処理ルールに従い、行頭・行末禁止文字を考慮した改行を行う。
半角文字は0.6文字幅として計算する。
"""

LINE_START_PROHIBITED: set[str] = set("。、，．）」』】〉》〕!?！？ー")
LINE_END_PROHIBITED: set[str] = set("（「『【〈《〔")


def is_line_start_prohibited(ch: str) -> bool:
    return ch in LINE_START_PROHIBITED


def is_line_end_prohibited(ch: str) -> bool:
    return ch in LINE_END_PROHIBITED


def is_halfwidth(ch: str) -> bool:
    return ord(ch) < 128


def _char_width(ch: str) -> float:
    return 0.6 if is_halfwidth(ch) else 1.0


def _text_width(text: str) -> float:
    return sum(_char_width(ch) for ch in text)


def break_lines(text: str, chars_per_line: int) -> list[str]:
    """禁則処理付きで改行する。

    Args:
        text: 入力テキスト
        chars_per_line: 1行あたりの全角文字数上限

    Returns:
        行のリスト
    """
    if not text:
        return [""]

    paragraphs = text.split("\n")
    result: list[str] = []

    for paragraph in paragraphs:
        if not paragraph:
            result.append("")
            continue
        result.extend(_break_paragraph(paragraph, chars_per_line))

    return result


def _break_paragraph(text: str, chars_per_line: int) -> list[str]:
    lines: list[str] = []
    max_width = float(chars_per_line)
    i = 0

    while i < len(text):
        width = 0.0
        end = i

        while end < len(text) and width + _char_width(text[end]) <= max_width:
            width += _char_width(text[end])
            end += 1

        if end >= len(text):
            lines.append(text[i:end])
            break

        # 行末禁止文字チェック: 行末が行末禁止文字なら1文字前で切る
        if is_line_end_prohibited(text[end - 1]):
            end -= 1

        # 行頭禁止文字チェック: 次行の先頭が行頭禁止文字なら現在行に含める
        elif end < len(text) and is_line_start_prohibited(text[end]):
            end += 1

        lines.append(text[i:end])
        i = end

    return lines
