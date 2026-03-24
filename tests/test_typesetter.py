import pytest
from src.layout.typesetter import CharPlacement, Typesetter
from src.layout.page_layout import PageConfig, PageLayout
from src.model.augmentation import AugmentConfig, HandwritingAugmenter


class TestCharPlacement:
    def test_creation(self):
        p = CharPlacement(char="あ", x=10.0, y=20.0, font_size=7.0)
        assert p.char == "あ"
        assert p.x == 10.0
        assert p.y == 20.0
        assert p.font_size == 7.0


class TestTypesetter:
    def test_simple_text(self):
        ts = Typesetter(PageConfig())
        pages = ts.typeset("あいうえお")
        assert len(pages) >= 1
        assert len(pages[0]) == 5

    def test_placement_positions_differ(self):
        ts = Typesetter(PageConfig())
        pages = ts.typeset("あいう")
        placements = pages[0]
        xs = [p.x for p in placements]
        # 各文字のx座標は異なる（横書き）
        assert len(set(xs)) == 3

    def test_line_wrap(self):
        cfg = PageConfig()
        ts = Typesetter(cfg)
        layout = PageLayout(cfg)
        area = layout.content_area()
        # area幅から1行の文字数を計算し、それを超えるテキストを入力
        chars_per_line = int(area.width / ts.font_size)
        text = "あ" * (chars_per_line + 5)
        pages = ts.typeset(text)
        placements = pages[0]
        ys = set(p.y for p in placements)
        assert len(ys) >= 2  # 少なくとも2行

    def test_newline(self):
        ts = Typesetter(PageConfig())
        pages = ts.typeset("あ\nい")
        placements = pages[0]
        assert placements[0].y != placements[1].y

    def test_halfwidth_chars(self):
        ts = Typesetter(PageConfig())
        pages = ts.typeset("ab")
        placements = pages[0]
        # 半角文字は全角の半分の幅
        width = placements[1].x - placements[0].x
        assert width < ts.font_size  # 全角幅より小さい

    def test_font_size_default(self):
        ts = Typesetter(PageConfig())
        # デフォルトのfont_sizeは行間隔(8mm)に合わせる
        assert ts.font_size > 0
        assert ts.font_size <= 8.0

    def test_multipage(self):
        cfg = PageConfig()
        ts = Typesetter(cfg)
        layout = PageLayout(cfg)
        lines = layout.line_positions()
        area = layout.content_area()
        chars_per_line = int(area.width / ts.font_size)
        # 1ページの行数 * 行あたり文字数 + α でオーバーフロー
        text = "あ" * (len(lines) * chars_per_line + 10)
        pages = ts.typeset(text)
        assert len(pages) >= 2

    def test_empty_text(self):
        ts = Typesetter(PageConfig())
        pages = ts.typeset("")
        assert len(pages) == 1
        assert len(pages[0]) == 0


class TestTypesetterAugmentation:
    def _make_augmented_typesetter(self, seed: int = 42) -> Typesetter:
        aug = HandwritingAugmenter(AugmentConfig(), seed=seed)
        return Typesetter(PageConfig(), augmenter=aug)

    def test_augmentation_changes_placement(self):
        """augmentation有効時、配置座標が元の等間隔配置から変動する。"""
        ts_plain = Typesetter(PageConfig())
        ts_aug = self._make_augmented_typesetter()
        text = "あいうえお"

        plain = ts_plain.typeset(text)[0]
        augmented = ts_aug.typeset(text)[0]

        plain_xs = [p.x for p in plain]
        aug_xs = [p.x for p in augmented]
        assert plain_xs != aug_xs

    def test_augmentation_disabled_preserves_original(self):
        """enabled=False時、augmentなしと完全に同じ配置になる。"""
        ts_plain = Typesetter(PageConfig())
        cfg = AugmentConfig(enabled=False)
        aug = HandwritingAugmenter(cfg)
        ts_disabled = Typesetter(PageConfig(), augmenter=aug)
        text = "あいうえお"

        plain = ts_plain.typeset(text)[0]
        disabled = ts_disabled.typeset(text)[0]

        for p, d in zip(plain, disabled):
            assert p.x == d.x
            assert p.y == d.y
            assert p.font_size == d.font_size

    def test_baseline_consistent_within_line(self):
        """同一行内の全文字は同じベースラインシフトを持つ。"""
        ts = self._make_augmented_typesetter()
        pages = ts.typeset("あいうえお")
        placements = pages[0]
        ys = [p.y for p in placements]
        assert len(set(ys)) == 1

    def test_baseline_varies_across_lines(self):
        """異なる行のベースラインシフトは異なる（高確率）。"""
        ts = self._make_augmented_typesetter()
        cfg = PageConfig()
        layout = PageLayout(cfg)
        area = layout.content_area()
        chars_per_line = int(area.width / ts.font_size)
        text = "あ" * (chars_per_line + 5)

        pages = ts.typeset(text)
        placements = pages[0]
        line1_y = placements[0].y
        line2_y = placements[chars_per_line].y

        ts_plain = Typesetter(cfg)
        plain = ts_plain.typeset(text)[0]
        plain_line1_y = plain[0].y
        plain_line2_y = plain[chars_per_line].y

        drift1 = line1_y - plain_line1_y
        drift2 = line2_y - plain_line2_y
        assert drift1 != drift2

    def test_font_size_varies(self):
        """augmentation有効時、文字サイズが変動する。"""
        ts = self._make_augmented_typesetter()
        pages = ts.typeset("あいうえおかきくけこ")
        sizes = [p.font_size for p in pages[0]]
        assert len(set(sizes)) > 1

    def test_line_density_affects_spacing(self):
        """augmentation有効時、行ごとの平均文字間隔が異なる（density_scaleが行ごとに変わる）。"""
        ts = self._make_augmented_typesetter(seed=42)
        cfg = PageConfig()
        layout = PageLayout(cfg)
        area = layout.content_area()
        chars_per_line = int(area.width / ts.font_size)
        text = "あ" * (chars_per_line * 3)

        pages = ts.typeset(text)
        placements = pages[0]

        lines: dict[float, list[float]] = {}
        for p in placements:
            lines.setdefault(p.y, []).append(p.x)

        avg_spacings = []
        for y in sorted(lines.keys()):
            xs = sorted(lines[y])
            if len(xs) >= 2:
                spacings = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
                avg_spacings.append(sum(spacings) / len(spacings))

        assert len(avg_spacings) >= 2, "Need at least 2 lines to compare"
        assert len(set(avg_spacings)) > 1, (
            "Average spacing should differ across lines due to density variation"
        )

    def test_no_augmenter_backward_compatible(self):
        """augmenter未指定時は従来通りの動作。"""
        ts = Typesetter(PageConfig())
        pages = ts.typeset("あいう")
        assert len(pages[0]) == 3


class TestParagraphIndent:
    """段落インデントのテスト。"""

    def test_page_first_line_no_indent(self):
        """ページ冒頭の行はインデントしない。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=6.0)
        layout = PageLayout(cfg)
        area = layout.content_area()

        pages = ts.typeset("あいうえお")
        placements = pages[0]
        # ページ冒頭 → インデントなし
        assert placements[0].x == pytest.approx(area.x)

    def test_second_paragraph_indented(self):
        """2段落目以降の先頭行はfont_size分インデントされる。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=6.0)
        layout = PageLayout(cfg)
        area = layout.content_area()

        pages = ts.typeset("あいう\nかきく")
        placements = pages[0]

        # 段落1（ページ冒頭）→ インデントなし
        assert placements[0].x == pytest.approx(area.x)

        # 段落2の先頭（改行後）→ インデントあり
        para2_chars = [p for p in placements if p.char == "か"]
        assert len(para2_chars) == 1
        assert para2_chars[0].x == pytest.approx(area.x + 6.0)

    def test_no_indent_on_wrapped_lines(self):
        """行折り返しで生まれた行にはインデントしない。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=6.0)
        layout = PageLayout(cfg)
        area = layout.content_area()

        chars_per_line = int(area.width / ts.font_size)
        # 折り返しが発生する長さ（改行なし = 1段落）
        text = "あ" * (chars_per_line + 5)
        pages = ts.typeset(text)
        placements = pages[0]

        # 2行目先頭はarea.xから開始（インデントなし）
        line2_chars = [p for p in placements if p.y != placements[0].y]
        assert len(line2_chars) > 0
        assert line2_chars[0].x == pytest.approx(area.x)

    def test_multiple_paragraphs_each_indented(self):
        """複数段落（3つ以上）それぞれの先頭行がインデントされる（ページ冒頭除く）。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=6.0)
        layout = PageLayout(cfg)
        area = layout.content_area()

        pages = ts.typeset("あいう\nかきく\nさしす")
        placements = pages[0]

        # 段落1（ページ冒頭）→ インデントなし
        assert placements[0].x == pytest.approx(area.x)

        # 段落2 → インデントあり
        para2_chars = [p for p in placements if p.char == "か"]
        assert len(para2_chars) == 1
        assert para2_chars[0].x == pytest.approx(area.x + 6.0)

        # 段落3 → インデントあり
        para3_chars = [p for p in placements if p.char == "さ"]
        assert len(para3_chars) == 1
        assert para3_chars[0].x == pytest.approx(area.x + 6.0)
