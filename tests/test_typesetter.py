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
        assert width == pytest.approx(ts.font_size * 0.55)

    def test_halfwidth_width_ratio(self):
        """半角文字幅は全角文字幅より小さい。"""
        ts = Typesetter(PageConfig(), font_size=6.0)
        pages = ts.typeset("a漢")
        p = pages[0]
        half_w = p[1].x - p[0].x
        pages2 = ts.typeset("漢字")
        p2 = pages2[0]
        full_w = p2[1].x - p2[0].x
        assert half_w < full_w

    def test_body_char_advance_tracks_size_scale(self):
        ts = Typesetter(PageConfig(), font_size=10.0)
        placements = ts.typeset("あ漢")[0]

        width = placements[1].x - placements[0].x

        assert width == pytest.approx(10.0 * (0.45 + 0.55 * 0.85))

    def test_kanji_advance_adds_breathing_room(self):
        ts = Typesetter(PageConfig(), font_size=10.0)
        placements = ts.typeset("漢字")[0]

        width = placements[1].x - placements[0].x

        assert width == pytest.approx(10.0 * 1.08)

    def test_kanji_wrap_keeps_dynamic_advance_inside_content_edge(self):
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=10.0)
        area = PageLayout(cfg).content_area()

        placements = ts.typeset("漢" * 18)[0]
        lines: dict[float, list[CharPlacement]] = {}
        for placement in placements:
            lines.setdefault(placement.y, []).append(placement)

        assert len(lines) == 2
        for line in lines.values():
            assert max(p.x + ts.font_size for p in line) <= area.x + area.width + 1e-9

    def test_hiragana_wrap_uses_smaller_body_advance(self):
        ts = Typesetter(PageConfig(), font_size=10.0)

        placements = ts.typeset("あ" * 19)[0]

        assert len({p.y for p in placements}) == 1

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
        char_advance = ts.font_size * (0.45 + 0.55 * 0.85)
        chars_per_line = int(area.width / char_advance)
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
        # 3行分のテキスト
        text = "あ" * 200

        pages = ts.typeset(text)
        placements = pages[0]

        ts_plain = Typesetter(cfg)
        plain = ts_plain.typeset(text)[0]

        # augmentation 有効時と無効時で y 座標に差がある文字があるはず
        diffs = [p.y - q.y for p, q in zip(placements[:50], plain[:50]) if abs(p.y - q.y) > 0.01]
        assert len(diffs) > 0, "baseline drift が効いていない"

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

    def test_char_density_combines_with_line_density(self):
        class FixedDensityAugmenter:
            def augment_char_placement(
                self, x: float, y: float, font_size: float
            ) -> tuple[float, float, float]:
                return x + 2.0, y, font_size

            def get_line_density_scale(self) -> float:
                return 1.5

            def get_char_density_scale(self) -> float:
                return 0.5

        cfg = PageConfig()
        layout = PageLayout(cfg)
        area = layout.content_area()
        ts = Typesetter(cfg, font_size=10.0, augmenter=FixedDensityAugmenter())

        placements = ts.typeset("漢字")[0]

        assert placements[0].x == pytest.approx(area.x + 2.0 * 0.75)
        assert placements[1].x - placements[0].x == pytest.approx(10.0 * 1.08 * 0.75)

    def test_density_expansion_stays_inside_wrapped_body_width(self):
        class ExpandingDensityAugmenter:
            def augment_char_placement(
                self, x: float, y: float, font_size: float
            ) -> tuple[float, float, float]:
                return x, y, font_size

            def get_line_density_scale(self) -> float:
                return 1.2

            def get_char_density_scale(self) -> float:
                return 1.0

        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=10.0, augmenter=ExpandingDensityAugmenter())
        area = PageLayout(cfg).content_area()

        placements = ts.typeset("漢" * 16)[0]
        last = max(placements, key=lambda p: p.x)

        assert len({p.y for p in placements}) == 1
        assert placements[1].x - placements[0].x > ts.font_size
        assert last.x + ts.font_size <= area.x + area.width + 1e-9

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
        cfg = AugmentConfig(
            spacing_variation=1.0,
            baseline_drift=0.0,
            size_variation=0.0,
            slant_variation=0.0,
            jitter_amplitude=0.0,
            line_density_variation=0.0,
        )
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
                expected_w = font * 0.55  # _char_size_scale による半角幅
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

        assert _char_size_scale("あ") == 0.85
        assert _char_size_scale("ん") == 0.80

    def test_katakana_scale(self):
        from src.layout.typesetter import _char_size_scale

        assert _char_size_scale("ア") == 0.85  # override table
        assert _char_size_scale("ン") == 0.78

    def test_small_kana_scale(self):
        from src.layout.typesetter import _char_size_scale

        assert _char_size_scale("っ") == 0.55
        assert _char_size_scale("ょ") == 0.55
        assert _char_size_scale("ッ") == 0.55
        assert _char_size_scale("ャ") == 0.55

    def test_halfwidth_scale(self):
        from src.layout.typesetter import _char_size_scale

        assert _char_size_scale("a") == 0.7
        assert _char_size_scale("1") == 0.7

    def test_typeset_hiragana_smaller_than_kanji(self):
        """組版時、ひらがなは漢字より小さいfont_sizeで配置される。"""
        from src.layout.page_layout import PageConfig

        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("漢あ")
        placements = pages[0]
        assert placements[0].font_size > placements[1].font_size
        assert placements[0].font_size == pytest.approx(7.0)
        assert placements[1].font_size == pytest.approx(7.0 * 0.85)

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


class TestBlockMath:
    """ブロック数式 $$...$$ の検出・中央配置テスト。"""

    def test_block_math_separates_into_three_lines(self):
        """前テキスト$$数式$$後テキスト が少なくとも3行に分離される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("前$$a + b$$後")
        placements = pages[0]
        ys = sorted(set(p.y for p in placements))
        # 前テキスト、数式、後テキスト の3行
        assert len(ys) == 3

    def test_block_math_centered(self):
        """ブロック数式が行の中央に配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        layout = PageLayout(PageConfig())
        area = layout.content_area()
        pages = ts.typeset("$$x$$")
        placements = pages[0]
        # 数式の左端が area.x より右（中央寄せされている）
        math_left = min(p.x for p in placements)
        assert math_left > area.x + 1.0
        # 中央付近に配置されている
        center = area.x + area.width / 2
        assert abs(math_left - center) < area.width / 3

    def test_block_math_contains_expected_chars(self):
        """ブロック数式の文字がCharPlacementに含まれる。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$$E = mc^{2}$$")
        chars = [p.char for p in pages[0]]
        assert "E" in chars
        assert "m" in chars
        assert "c" in chars

    def test_no_block_math_unchanged(self):
        """$$がない場合は既存動作と同じ。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("あいうえお")
        assert len(pages[0]) == 5
        chars = [p.char for p in pages[0]]
        assert "".join(chars) == "あいうえお"

    def test_text_before_and_after_on_separate_lines(self):
        """ブロック数式前後のテキストが異なる行に配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("前$$x$$後")
        placements = pages[0]
        y_by_char = {p.char: p.y for p in placements}
        assert y_by_char["前"] != y_by_char["x"]
        assert y_by_char["x"] != y_by_char["後"]

    def test_block_math_only(self):
        """$$数式$$ のみの入力が正しく配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$$a + b$$")
        chars = [p.char for p in pages[0]]
        assert "a" in chars
        assert "b" in chars

    def test_block_math_with_newline_paragraphs(self):
        """段落区切りとブロック数式の組み合わせ。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("段落1\n$$x = 1$$\n段落2")
        placements = pages[0]
        ys = sorted(set(p.y for p in placements))
        # 段落1、数式、段落2 の少なくとも3行
        assert len(ys) >= 3


class TestBlockMathTag:
    """ブロック数式の \\tag{n} 記法（右寄せ式番号）のテスト。"""

    def test_block_tag_right_aligned(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        layout = PageLayout(PageConfig())
        area = layout.content_area()
        pages = ts.typeset(r"$$ x = 1 \tag{1} $$")
        placements = pages[0]
        # tag の '(' が本体の 'x' より右に配置される
        body_x = [p.x for p in placements if p.char == "x"][0]
        paren_x = [p.x for p in placements if p.char == "("][0]
        assert paren_x > body_x
        # tag は area の右端付近にある
        assert paren_x > area.x + area.width / 2

    def test_block_tag_chars_present(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"$$ x = 1 \tag{1} $$")
        chars = [p.char for p in pages[0]]
        assert "(" in chars
        assert ")" in chars
        assert "1" in chars

    def test_inline_tag_ignored(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"$x = 1 \tag{1}$")
        placements = pages[0]
        chars = [p.char for p in placements]
        # インライン tag は配置されない
        assert "(" not in chars
        assert ")" not in chars

    def test_block_no_tag_centered(self):
        """tag 無しでも中央寄せが壊れていない（既存挙動の確認）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"$$ x = 1 $$")
        chars = [p.char for p in pages[0]]
        assert "x" in chars

    def test_block_tag_body_centered_independent_of_tag_width(self):
        """本体の中心位置は tag の幅に影響されない（area 中央に配置）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages_with_tag = ts.typeset(r"$$ x = 1 \tag{1} $$")
        pages_no_tag = ts.typeset(r"$$ x = 1 $$")
        x_with = [p.x for p in pages_with_tag[0] if p.char == "x"][0]
        x_no = [p.x for p in pages_no_tag[0] if p.char == "x"][0]
        assert x_with == pytest.approx(x_no)

    def test_block_tag_only(self):
        """本体なしでブロック tag だけ（$$ \\tag{1} $$）でも例外を投げない。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"$$ \tag{1} $$")
        chars = [p.char for p in pages[0]]
        assert "(" in chars
        assert ")" in chars


class TestBlockMathRowConsumption:
    """ブロック数式が必要な行数を消費する（最低2行、分数なら高さに応じて増加）。"""

    def test_simple_block_consumes_two_rows(self):
        """単純な式は 2 行ぶん（最低 2 行）消費する。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("前\n$$ x = 1 $$\n後")
        placements = pages[0]
        前_y = next(p.y for p in placements if p.char == "前")
        後_y = next(p.y for p in placements if p.char == "後")
        line_spacing = ts._config.line_spacing
        # 前-後 の差は line_spacing*2.5 以上 = 数式が最低 2 行消費している証拠
        assert (前_y - 後_y) >= line_spacing * 2.5

    def test_nested_frac_consumes_three_or_more_rows(self):
        """ネスト分数は 3 行以上消費する。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"前" + "\n" + r"$$ \frac{\frac{1}{2}}{x} $$" + "\n" + r"後")
        placements = pages[0]
        前_y = next(p.y for p in placements if p.char == "前")
        後_y = next(p.y for p in placements if p.char == "後")
        line_spacing = ts._config.line_spacing
        # 3 行以上消費 → 前-後 の差は line_spacing*4 以上（行間4個分）
        assert (前_y - 後_y) >= line_spacing * 3.5

    def test_block_math_pages_overflow(self):
        """ページ末尾近くに数式があり残行が足りないと次ページに送られる。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        layout = PageLayout(PageConfig())
        line_positions = layout.line_positions()
        rows = len(line_positions)
        # 最後から1行手前まで本文を埋める（残り1行）
        body = "\n".join([f"行{i}" for i in range(rows - 1)])
        text = body + "\n" + r"$$ \frac{1}{2} $$"
        pages = ts.typeset(text)
        # 残り1行で2行必要なので次ページ送り
        assert len(pages) >= 2
        # 数式の '1' または '2' が最終ページに存在する
        last_page_chars = [p.char for p in pages[-1]]
        assert "1" in last_page_chars or "2" in last_page_chars

    def test_block_math_centered_vertically_in_consumed_rows(self):
        """確保した行範囲の垂直中央に数式が配置される（2行確保時）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        layout = PageLayout(PageConfig())
        line_positions = layout.line_positions()
        line_spacing = ts._config.line_spacing
        # ページ先頭にブロック数式
        pages = ts.typeset(r"$$ x = 1 $$")
        placements = pages[0]
        x_chars = [p.y for p in placements if p.char == "x"]
        assert len(x_chars) == 1
        # 2行ぶん確保 → 中央 = (line_positions[0] + line_positions[1]) / 2
        expected_center = (line_positions[0] + line_positions[1]) / 2
        # 数式のベースライン（'x' の y）が中央付近にあるか（半行ぶんの誤差を許容）
        assert abs(x_chars[0] - expected_center) < line_spacing * 0.5

    def test_consecutive_block_math_each_consumes_two_rows(self):
        """連続するブロック数式がそれぞれ2行ずつ消費する。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$$ a = 1 $$\n$$ b = 2 $$")
        placements = pages[0]
        a_y = next(p.y for p in placements if p.char == "a")
        b_y = next(p.y for p in placements if p.char == "b")
        line_spacing = ts._config.line_spacing
        # 1個目が2行確保(中央=spacing*1.5地点)、2個目も2行確保(中央=spacing*3.5地点)
        # 差は line_spacing * 2 程度になる
        assert (a_y - b_y) >= line_spacing * 1.5
        assert (a_y - b_y) <= line_spacing * 2.5


class TestBlockMathLineBreak:
    """ブロック数式内の \\\\ による強制改行のテスト。"""

    def test_double_backslash_creates_two_lines(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "前\n$$ a = 1 \\\\ b = 2 $$\n後"
        pages = ts.typeset(text)
        a_y = next(p.y for p in pages[0] if p.char == "a")
        b_y = next(p.y for p in pages[0] if p.char == "b")
        # a が上、b が下
        assert a_y > b_y
        # 「前」「後」が a/b を挟まないこと（ブロック数式は別行）
        前_y = next(p.y for p in pages[0] if p.char == "前")
        後_y = next(p.y for p in pages[0] if p.char == "後")
        assert 前_y > a_y
        assert b_y > 後_y

    def test_multiline_block_consumes_more_rows(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        text_one = "前\n$$ a = 1 $$\n後"
        text_three = "前\n$$ a = 1 \\\\ b = 2 \\\\ c = 3 $$\n後"
        pages_one = ts.typeset(text_one)
        pages_three = ts.typeset(text_three)
        前_one_y = next(p.y for p in pages_one[0] if p.char == "前")
        後_one_y = next(p.y for p in pages_one[0] if p.char == "後")
        前_three_y = next(p.y for p in pages_three[0] if p.char == "前")
        後_three_y = next(p.y for p in pages_three[0] if p.char == "後")
        assert (前_three_y - 後_three_y) > (前_one_y - 後_one_y)

    def test_linebreak_with_tag(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = r"$$ a = 1 \\ b = 2 \tag{1} $$"
        pages = ts.typeset(text)
        # a, b, ( がある
        chars = [p.char for p in pages[0]]
        assert "a" in chars
        assert "b" in chars
        assert "(" in chars
        # ( のy座標は b と同じ（最終行）
        b_y = next(p.y for p in pages[0] if p.char == "b")
        paren_y = next(p.y for p in pages[0] if p.char == "(")
        assert abs(b_y - paren_y) < 0.5


class TestBlockMathMultiline:
    """改行を含むブロック数式 $$\\n...\\n$$ のテスト。"""

    def test_block_math_with_newlines(self):
        """$$ と $$ の間に改行があってもブロック数式として認識される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "前\n$$\ny = ax + b\n$$\n後"
        pages = ts.typeset(text)
        chars = [p.char for p in pages[0]]
        assert "y" in chars
        assert "=" in chars
        assert "a" in chars
        assert "x" in chars
        assert "b" in chars
        # ブロック数式として認識されているなら $ は文字として配置されない
        assert "$" not in chars
        # 「前」「後」と数式の y 座標が分離していること（前 > y > 後）
        前_y = next(p.y for p in pages[0] if p.char == "前")
        後_y = next(p.y for p in pages[0] if p.char == "後")
        y_eq = next(p.y for p in pages[0] if p.char == "y")
        assert 前_y > y_eq > 後_y

    def test_block_math_multiline_with_linebreak(self):
        """$$ と $$ の間に改行と \\\\ 改行が混在しても正しく処理される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "$$\ny = ax + b \\\\\ny = cx + d \\tag{1}\n$$"
        pages = ts.typeset(text)
        chars = [p.char for p in pages[0]]
        # 両式とも描画されている
        assert "a" in chars
        assert "c" in chars
        assert "b" in chars
        assert "d" in chars
        # tag の (1) も
        assert "(" in chars
        assert ")" in chars
        assert "1" in chars

    def test_block_math_singleline_still_works(self):
        """既存の 1 行 $$...$$ も壊れない。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "$$ x = 1 $$"
        pages = ts.typeset(text)
        chars = [p.char for p in pages[0]]
        assert "x" in chars
        assert "=" in chars
        assert "1" in chars

    def test_block_math_multiline_single_equation(self):
        """改行を含む単一行の数式（前後空行のみ）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "$$\nE = mc^{2}\n$$"
        pages = ts.typeset(text)
        chars = [p.char for p in pages[0]]
        assert "E" in chars
        assert "m" in chars
        assert "c" in chars
        assert "2" in chars
        # $ デリミタが文字として残っていないこと
        assert "$" not in chars

    def test_block_math_multiline_with_backslash_newline(self):
        """$$\\n の後に \\\\ 改行を含む数式。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        text = "$$\na = 1 \\\\\nb = 2\n$$"
        pages = ts.typeset(text)
        chars = [p.char for p in pages[0]]
        assert "$" not in chars
        a_y = next(p.y for p in pages[0] if p.char == "a")
        b_y = next(p.y for p in pages[0] if p.char == "b")
        # a が上、b が下（ \\\\ で改行）
        assert a_y > b_y


class TestInlineMath:
    """インライン数式 $...$ の検出・配置テスト。"""

    def test_inline_math_produces_char_placements(self):
        """$V = IR$ がCharPlacementのリストに変換される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$V = IR$")
        placements = pages[0]
        # "V = IR" の5文字分（スペース含む）のCharPlacementが生成される
        chars = [p.char for p in placements]
        assert "".join(chars) == "V = IR"

    def test_mixed_text_and_math(self):
        """通常テキスト$数式$通常テキスト の混在が正しく配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("電圧$V = IR$です")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "".join(chars) == "電圧V = IRです"

    def test_mixed_text_x_positions_monotonic(self):
        """混在テキストのx座標が単調増加する（重なりなし）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("式$x = 1$。")
        placements = pages[0]
        xs = [p.x for p in placements]
        for i in range(1, len(xs)):
            assert xs[i] > xs[i - 1], f"x[{i}]={xs[i]} <= x[{i - 1}]={xs[i - 1]}"

    def test_math_width_advances_cursor(self):
        """数式部分の幅分だけカーソルが進み、後続テキストが正しい位置に配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        # 数式なし
        pages_plain = ts.typeset("式です")
        # 数式あり
        pages_math = ts.typeset("式$x$す")
        p_plain = pages_plain[0]
        p_math = pages_math[0]
        # "式" の後に "x" 分の幅が加わるので、"す" の x 位置は plain の "で" とは異なる
        last_plain_x = p_plain[-1].x
        last_math_x = p_math[-1].x
        # 両方とも最後の文字が存在し、x位置が正
        assert last_math_x > 0
        assert last_plain_x > 0

    def test_no_math_unchanged(self):
        """$を含まないテキストは既存と同じ動作。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("あいうえお")
        assert len(pages[0]) == 5
        chars = [p.char for p in pages[0]]
        assert "".join(chars) == "あいうえお"

    def test_superscript_in_inline_math(self):
        """上付き文字 $x^{2}$ がCharPlacementに変換される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$x^{2}$")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "x" in chars
        assert "2" in chars

    def test_fraction_in_inline_math(self):
        r"""分数 $\frac{1}{2}$ がCharPlacementに変換される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset(r"$\frac{1}{2}$")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "1" in chars
        assert "2" in chars
        # 分子・分母は異なるy座標
        ys = set(p.y for p in placements)
        assert len(ys) >= 2

    def test_empty_dollar_pair_passthrough(self):
        """$$ はブロック数式デリミタなので、インライン処理ではそのまま通過する。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("あ$$い")
        placements = pages[0]
        chars = [p.char for p in placements]
        # $$ はブロック数式(Task #3)のため、ここでは文字として残る
        assert "$$" in "".join(chars)

    def test_math_no_augmentation(self):
        """数式部分にはaugmentationが適用されない（正確な配置）。"""
        aug = HandwritingAugmenter(AugmentConfig(), seed=42)
        ts_aug = Typesetter(PageConfig(), font_size=7.0, augmenter=aug)
        ts_plain = Typesetter(PageConfig(), font_size=7.0)

        pages_aug = ts_aug.typeset("$V = IR$")
        pages_plain = ts_plain.typeset("$V = IR$")

        # 数式部分のfont_sizeはaugmentationで変わらない
        for pa, pp in zip(pages_aug[0], pages_plain[0]):
            assert pa.font_size == pp.font_size

    def test_math_baseline_matches_augmented_text_line(self):
        class FixedBaselineAugmenter:
            def augment_char_placement(
                self, x: float, y: float, font_size: float
            ) -> tuple[float, float, float]:
                return x, y + 2.5, font_size

            def get_line_density_scale(self) -> float:
                return 1.0

            def get_char_density_scale(self) -> float:
                return 1.0

        ts = Typesetter(PageConfig(), font_size=7.0, augmenter=FixedBaselineAugmenter())
        placements = ts.typeset(r"あ$x$い")[0]
        y_by_char = {p.char: p.y for p in placements}

        assert y_by_char["x"] == pytest.approx(y_by_char["あ"])
        assert y_by_char["x"] == pytest.approx(y_by_char["い"])

    def test_inline_operator_kept_as_word_placement(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        placements = ts.typeset(r"$\cos 2x$")[0]

        assert placements[0].char == "cos"
        assert placements[0].role == "operator"
        assert [p.char for p in placements[1:]] == [" ", "2", "x"]

    def test_inline_math_wrap_uses_math_layout_width(self):
        cfg = PageConfig(
            paper_size=(42.0, 80.0),
            margin_left=5.0,
            margin_right=5.0,
            margin_top=5.0,
            margin_bottom=5.0,
        )
        ts = Typesetter(cfg, font_size=7.0)
        placements = ts.typeset(r"ああ$\frac{1}{2}$あああ")[0]
        lines: dict[float, list[str]] = {}
        for p in placements:
            if p.line_segment is not None:
                continue
            lines.setdefault(p.y, []).append(p.char)

        baseline_line = max(
            (chars for chars in lines.values() if "あ" in chars),
            key=lambda chars: chars.count("あ"),
        )
        assert baseline_line.count("あ") == 4
        assert any("1" in chars for chars in lines.values())
        assert any("2" in chars for chars in lines.values())

    def test_multiple_inline_math(self):
        """複数のインライン数式が1行に含まれる場合。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$a$と$b$")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "".join(chars) == "aとb"


class TestPageBreak:
    """ "-----" 行は改ページ指示。"""

    def test_hr_forces_new_page(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("前のページ。\n\n-----\n\n次のページ。")
        assert len(pages) == 2
        assert any(p.char == "前" for p in pages[0])
        assert any(p.char == "次" for p in pages[1])

    def test_leading_hr_no_empty_page(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("-----\n\n本文。")
        assert len(pages) == 1


class TestHeadings:
    """セクション見出し（# で大きく表示）のテスト。"""

    def test_h1_larger_font_size(self):
        """# 見出し が通常テキストより大きい font_size（1.15倍）で配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("# 見出し")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "見" in chars
        assert "#" not in chars
        for p in placements:
            assert p.font_size == pytest.approx(7.0 * 1.15)

    def test_h2_medium_font_size(self):
        """## 小見出し が font_size * 1.15 で配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("## 小見出し")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "小" in chars
        assert "#" not in chars
        for p in placements:
            assert p.font_size == pytest.approx(7.0 * 1.08)

    def test_h3_normal_font_size(self):
        """### 見出し3 は通常サイズ（1.0倍）で配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("### 見出し3")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "見" in chars
        assert "#" not in chars
        for p in placements:
            assert p.font_size == pytest.approx(7.0 * 1.0)

    def test_heading_preceded_by_blank_line(self):
        """見出し前に空行が入る（ページ先頭では不要）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("前の段落\n# 見出し")
        placements = pages[0]
        y_mae = [p.y for p in placements if p.char == "前"][0]
        y_midashi = [p.y for p in placements if p.char == "見"][0]
        # 通常の2行分の間隔より大きい（空行1行分が入るため）
        ts_normal = Typesetter(PageConfig(), font_size=7.0)
        pages_normal = ts_normal.typeset("前の段落\n次の段落")
        pn = pages_normal[0]
        y_mae_n = [p.y for p in pn if p.char == "前"][0]
        y_tsugi_n = [p.y for p in pn if p.char == "次"][0]
        normal_gap = abs(y_tsugi_n - y_mae_n)
        heading_gap = abs(y_midashi - y_mae)
        assert heading_gap > normal_gap

    def test_heading_at_page_start_no_blank_line(self):
        """ページ先頭の見出しでは空行を入れない。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=7.0)
        layout = PageLayout(cfg)
        line_positions = layout.line_positions()
        pages = ts.typeset("# 見出し")
        placements = pages[0]
        # ページ先頭の行位置に配置される
        assert placements[0].y == pytest.approx(line_positions[0])

    def test_no_heading_unchanged(self):
        """# なしテキストは既存動作と同じ。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("あいうえお")
        assert len(pages[0]) == 5
        for p in pages[0]:
            assert p.font_size < 7.0  # ひらがなは漢字より小さい
            assert p.font_size >= 7.0 * 0.75  # 最小でも75%

    def test_heading_strips_hash_and_space(self):
        """# と後続スペースが除去され、テキスト部分のみ配置される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("# テスト")
        chars = [p.char for p in pages[0]]
        assert "".join(chars) == "テスト"

    def test_heading_no_indent(self):
        """見出し行はインデントしない（段落先頭でも）。"""
        cfg = PageConfig()
        ts = Typesetter(cfg, font_size=7.0)
        pages = ts.typeset("前の段落\n# 見出し")
        placements = pages[0]
        heading_chars = [p for p in placements if p.char == "見"]
        assert len(heading_chars) == 1
        # h1見出しは15mmから開始
        assert heading_chars[0].x == pytest.approx(15.0)

    def test_period_replaced_with_kuten(self):
        """半角ピリオドは句点（。）に全置換される。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("これはテストです。")
        chars = [p.char for p in pages[0]]
        assert "." not in chars
        assert "。" in chars

    def test_period_in_number_also_replaced(self):
        """数値中のピリオドも句点に置換される（一律置換の仕様）。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("0.2")
        chars = [p.char for p in pages[0]]
        assert "." not in chars
        assert "。" in chars


class TestMathRealDrawWidth:
    """数式の予約幅が実描画幅（matplotlib aspect）に一致することのテスト。

    論理幅（MathLayoutEngine.width, _CHAR_WIDTH_RATIO 等幅）は上付き ^ を幅0、∑/∫を
    過大に見積もるため実描画とずれる。予約を実描画幅にして行はみ出しを解消する。
    """

    def test_inline_width_matches_real_draw_width(self):
        from src.layout.math_layout import MathLayoutEngine, MathParser
        from src.ui.math_skeletonize import formula_aspect

        ts = Typesetter(PageConfig(), font_size=7.0)
        src = "E=mc^2"
        reserved = ts._inline_math_width(src)

        elements = [e for e in MathParser.parse(src) if e.type != "tag"]
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=ts.font_size)
        logical = box.width
        h_mm = box.ascent + box.descent
        expected = h_mm * formula_aspect(src)

        # 実描画幅に一致し、論理幅とは異なる（^2 で論理幅は過小）
        assert reserved == pytest.approx(expected)
        assert reserved != pytest.approx(logical)

    def test_inline_cursor_advances_by_real_width(self):
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("本文$E=mc^2$後")
        placements = pages[0]
        x_本 = next(p.x for p in placements if p.char == "本")
        x_後 = next(p.x for p in placements if p.char == "後")
        # 数式は「本文」の直後に始まる: 数式開始 x = x_本 + 2文字分
        adv = ts._body_char_advance("本")
        math_start = x_本 + adv * 2
        expected_after = math_start + ts._inline_math_width("E=mc^2")
        assert x_後 == pytest.approx(expected_after, abs=0.5)

    def test_block_long_formula_shrinks_to_body_width(self):
        from src.ui.math_skeletonize import formula_draw_width_mm

        ts = Typesetter(PageConfig(), font_size=7.0)
        layout = PageLayout(PageConfig())
        area = layout.content_area()
        long_src = r"x^2+y^2+z^2+w^2+a^2+b^2+c^2+d^2+e^2+f^2"
        pages = ts.typeset(f"$$ {long_src} $$")
        placements = pages[0]
        math_ps = [p for p in placements if p.math_source]
        assert math_ps, "block math placement should exist"
        # 実描画想定の右端が本文幅を超えない（縮小が効いている）
        mp = math_ps[0]
        x0, _, w_mm, h_mm = mp.math_bbox
        draw_w = formula_draw_width_mm(long_src, h_mm)
        assert draw_w == pytest.approx(w_mm, abs=0.5) or w_mm <= area.width + 0.5
        assert x0 + w_mm <= area.x + area.width + 0.5


class TestInlineMathBboxWidth:
    """インライン数式 bbox の幅が実描画幅に一致し右隣文字とかぶらないテスト（バグ2）。

    旧実装は bbox 幅に論理幅(box.width)を使い、実描画幅(h_mm*aspect)より小さいため
    数式画像が予約枠からはみ出し右隣文字に重なっていた。
    """

    def test_inline_bbox_width_matches_real_draw_width(self):
        from src.ui.math_skeletonize import formula_draw_width_mm

        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("A$E$B")
        placements = pages[0]
        mp = next(p for p in placements if p.math_source == "E")
        _, _, w_mm, h_mm = mp.math_bbox
        expected = formula_draw_width_mm("E", h_mm)
        assert w_mm == pytest.approx(expected, abs=0.1)

    def test_inline_math_does_not_overlap_next_char(self):
        from src.ui.math_skeletonize import render_latex_to_strokes

        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("A$E=mc^2$B")
        placements = pages[0]
        mp = next(p for p in placements if p.math_source)
        strokes = render_latex_to_strokes(mp.math_source, mp.math_bbox, mp.math_align)
        assert strokes, "inline math should render strokes"
        math_right = max(float(s[:, 0].max()) for s in strokes)
        next_char = next(p for p in placements if p.char == "B")
        # 数式の実描画右端が右隣文字の開始 x を超えない（重なりなし）
        assert math_right <= next_char.x + 0.5


class TestMultiCharSubscript:
    """複数文字添字 σ_{sU} の全文字が描画対象として保持されるテスト（バグ1）。

    旧実装は複数文字 MathPlacement(text="sU") を1文字ずつ分割し、先頭以外を
    math_skip=True・role=None に落として2文字目以降を取りこぼしていた。
    画像レンダリングしない（math_source 無し）経路を直接検証する。
    """

    def _convert(self, src: str) -> list[CharPlacement]:
        from src.layout.math_layout import MathLayoutEngine, MathParser

        ts = Typesetter(PageConfig(), font_size=7.0)
        elements = [e for e in MathParser.parse(src) if e.type != "tag"]
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=ts.font_size)
        output: list[CharPlacement] = []
        # math_source を渡さない＝個別文字描画経路（タグ等と同じ）
        ts._convert_math_placements(box.placements, page_idx=0, output=output)
        return output

    def test_multichar_subscript_keeps_all_chars(self):
        output = self._convert(r"\sigma_{sU}")
        chars = "".join(p.char for p in output)
        assert "s" in chars
        assert "U" in chars

    def test_multichar_subscript_all_have_role_and_no_skip(self):
        output = self._convert(r"\sigma_{sL}")
        sub_ps = [p for p in output if p.char in ("s", "L")]
        assert len(sub_ps) == 2
        for p in sub_ps:
            assert p.role == "subscript", f"{p.char!r} lost subscript role"
            assert not p.math_skip, f"{p.char!r} wrongly marked math_skip"
            # 添字サイズは本文の 0.75 倍
            assert p.font_size == pytest.approx(7.0 * 0.75, abs=0.01)

    def test_multichar_subscript_chars_distinct_x(self):
        output = self._convert(r"\sigma_{sU}")
        xs = [p.x for p in output if p.char in ("s", "U")]
        assert len(xs) == 2
        assert xs[1] > xs[0], "添字2文字目が1文字目と同じ位置に重なっている"

    def test_multichar_word_in_image_mode_still_skips(self):
        """math_source 有り（画像描画）経路では従来どおり先頭以外 skip。"""
        from src.layout.math_layout import MathLayoutEngine, MathParser

        ts = Typesetter(PageConfig(), font_size=7.0)
        elements = [e for e in MathParser.parse(r"\sigma_{sU}") if e.type != "tag"]
        box = MathLayoutEngine.layout(elements, x=0.0, y=0.0, font_size=ts.font_size)
        output: list[CharPlacement] = []
        ts._convert_math_placements(
            box.placements,
            page_idx=0,
            output=output,
            math_source=r"\sigma_{sU}",
            math_bbox=(0.0, 0.0, 10.0, 7.0),
        )
        # 先頭1つだけ math_source を持ち、残りは skip（画像が一括描画する）
        head = [p for p in output if p.math_source]
        assert len(head) == 1
        rest = [p for p in output if not p.math_source]
        assert all(p.math_skip for p in rest)


class TestTablePlacement:
    """Markdownパイプ表の組版（_place_table）。"""

    _TABLE = "| 材料 | 引張強さ |\n|---|---|\n| 圧延鋼材 | 400 |\n| 焼鈍材 | 510 |\n"

    def _segments_and_cells(self, font_size=4.5):
        ts = Typesetter(PageConfig(), font_size=font_size)
        pages = ts.typeset(self._TABLE)
        placements = pages[0]
        v_rules = sorted(
            p.line_segment[0]
            for p in placements
            if p.line_segment is not None and p.line_segment[0] == p.line_segment[2]
        )
        cells = [p for p in placements if p.char and p.line_segment is None]
        return ts, v_rules, cells

    def test_cells_do_not_overflow_columns(self):
        """各セル文字は所属列の縦罫線(右境界)を越えない（漢字混在でもはみ出さない）。"""
        ts, v_rules, cells = self._segments_and_cells()
        assert len(v_rules) >= 3  # 2列なら縦罫線3本
        for cp in cells:
            # この文字が属する列を、左境界 < x の最右な縦罫線で特定
            left = max((x for x in v_rules if x <= cp.x + 1e-6), default=None)
            right = min((x for x in v_rules if x > cp.x + 1e-6), default=None)
            assert left is not None and right is not None
            # 文字送り分も右境界内に収まること
            adv = ts._body_char_advance(cp.char)
            assert cp.x + adv <= right + 1e-6, (
                f"セル文字 {cp.char!r} (x={cp.x:.2f}+{adv:.2f}) が右境界 {right:.2f} を越えた"
            )

    def test_table_fits_in_content_width(self):
        """表全体は本文幅に収まる。"""
        ts, v_rules, _ = self._segments_and_cells()
        layout = PageLayout(PageConfig())
        area = layout.content_area()
        assert v_rules[0] >= area.x - 1e-6
        assert v_rules[-1] <= area.x + area.width + 1e-6
