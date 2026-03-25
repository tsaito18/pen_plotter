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
        # 半角文字は_char_size_scaleにより0.6倍幅
        width = placements[1].x - placements[0].x
        assert width == pytest.approx(ts.font_size * 0.6)

    def test_halfwidth_width_ratio(self):
        """半角文字幅は漢字の0.6倍。"""
        ts = Typesetter(PageConfig(), font_size=6.0)
        pages = ts.typeset("a漢")
        p = pages[0]
        half_w = p[1].x - p[0].x  # 'a'の幅
        pages2 = ts.typeset("漢字")
        p2 = pages2[0]
        full_w = p2[1].x - p2[0].x  # '漢'の幅（スケール1.0）
        assert half_w == pytest.approx(full_w * 0.6)

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


class TestHalfwidthSpacingReduction:
    """連続する半角文字間ではaugmentationのspacing variationが減衰する。"""

    def _make_typesetter(self, seed: int = 42) -> Typesetter:
        cfg = AugmentConfig(spacing_variation=1.0, baseline_drift=0.0,
                            size_variation=0.0, slant_variation=0.0,
                            jitter_amplitude=0.0, line_density_variation=0.0)
        aug = HandwritingAugmenter(cfg, seed=seed)
        return Typesetter(PageConfig(), augmenter=aug)

    def test_consecutive_halfwidth_spacing_reduced(self):
        """連続半角文字間のspacing変動は、全角文字間より小さい。"""
        half_deviations = []
        full_deviations = []
        for seed in range(20):
            ts_half = self._make_typesetter(seed=seed)
            pages_half = ts_half.typeset("abcdefgh")
            p_half = pages_half[0]

            ts_full = self._make_typesetter(seed=seed)
            pages_full = ts_full.typeset("漢字書道文章表現技")
            p_full = pages_full[0]

            # 半角: 2文字目以降のspacing deviation（期待幅からの差）
            font = ts_half.font_size
            for i in range(1, len(p_half) - 1):
                actual_w = p_half[i + 1].x - p_half[i].x
                expected_w = font * 0.6  # _char_size_scale による半角幅
                half_deviations.append(abs(actual_w - expected_w))

            # 漢字: スケール1.0
            for i in range(1, len(p_full) - 1):
                actual_w = p_full[i + 1].x - p_full[i].x
                expected_w = font  # 漢字のスケール1.0
                full_deviations.append(abs(actual_w - expected_w))

        avg_half_dev = sum(half_deviations) / len(half_deviations)
        avg_full_dev = sum(full_deviations) / len(full_deviations)
        # 半角連続のspacing変動は全角の概ね半分程度になる
        assert avg_half_dev < avg_full_dev * 0.75

    def test_halfwidth_after_fullwidth_no_reduction(self):
        """全角→半角の遷移ではspacing減衰しない。"""
        ts_plain = Typesetter(PageConfig(), font_size=6.0)
        ts_aug = self._make_typesetter(seed=42)

        pages_aug = ts_aug.typeset("あa")
        p = pages_aug[0]
        # 全角→半角の遷移: 減衰なしなのでaugmentationフル適用
        # （減衰が適用されないことのテスト — 厳密な値ではなくフル適用を確認）
        actual_w = p[1].x - p[0].x
        # augmentなしの幅
        pages_plain = ts_plain.typeset("あa")
        pp = pages_plain[0]
        plain_w = pp[1].x - pp[0].x
        # augmentありで値が変動している（減衰なし=フルaugment）
        # NOTE: seed次第で偶然一致する可能性はあるが、spacing_variation=1.0なら非常に低い
        assert actual_w != pytest.approx(plain_w, abs=0.01)


class TestCharSizeScale:
    """文字種別サイズスケールのテスト。"""

    def test_kanji_scale_is_1(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("漢") == 1.0
        assert _char_size_scale("字") == 1.0

    def test_hiragana_scale(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("あ") == 0.88
        assert _char_size_scale("ん") == 0.88

    def test_katakana_scale(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("ア") == 0.88
        assert _char_size_scale("ン") == 0.88

    def test_small_kana_scale(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("っ") == 0.55
        assert _char_size_scale("ょ") == 0.55
        assert _char_size_scale("ッ") == 0.55
        assert _char_size_scale("ャ") == 0.55

    def test_halfwidth_scale(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("a") == 0.6
        assert _char_size_scale("1") == 0.6

    def test_typeset_hiragana_smaller_than_kanji(self):
        """組版時、ひらがなは漢字より小さいfont_sizeで配置される。"""
        from src.layout.page_layout import PageConfig
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("漢あ")
        placements = pages[0]
        assert placements[0].font_size > placements[1].font_size
        assert placements[0].font_size == pytest.approx(7.0)
        assert placements[1].font_size == pytest.approx(7.0 * 0.88)

    def test_typeset_small_kana_smallest(self):
        """小書き文字は最小のfont_sizeで配置される。"""
        from src.layout.page_layout import PageConfig
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("漢っ")
        placements = pages[0]
        assert placements[1].font_size == pytest.approx(7.0 * 0.55)

    def test_typeset_with_augmenter_applies_scale(self):
        """augmenter有効時もスケールが適用される。"""
        from src.layout.page_layout import PageConfig
        from src.model.augmentation import AugmentConfig, HandwritingAugmenter
        aug = HandwritingAugmenter(AugmentConfig(), seed=42)
        ts = Typesetter(PageConfig(), font_size=7.0, augmenter=aug)
        pages = ts.typeset("漢あ")
        placements = pages[0]
        assert placements[1].font_size < placements[0].font_size
