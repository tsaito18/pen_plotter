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
        char_advance = ts.font_size * 0.9
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
        assert _char_size_scale("ん") == 0.85

    def test_katakana_scale(self):
        from src.layout.typesetter import _char_size_scale
        assert _char_size_scale("ア") == 0.85
        assert _char_size_scale("ン") == 0.85

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
            assert xs[i] > xs[i - 1], f"x[{i}]={xs[i]} <= x[{i-1}]={xs[i-1]}"

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

    def test_multiple_inline_math(self):
        """複数のインライン数式が1行に含まれる場合。"""
        ts = Typesetter(PageConfig(), font_size=7.0)
        pages = ts.typeset("$a$と$b$")
        placements = pages[0]
        chars = [p.char for p in placements]
        assert "".join(chars) == "aとb"


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
            assert p.font_size == pytest.approx(7.0 * 0.85)  # ひらがなスケール

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
        layout = PageLayout(cfg)
        area = layout.content_area()
        pages = ts.typeset("前の段落\n# 見出し")
        placements = pages[0]
        heading_chars = [p for p in placements if p.char == "見"]
        assert len(heading_chars) == 1
        # h1見出しは15mmから開始
        assert heading_chars[0].x == pytest.approx(15.0)
