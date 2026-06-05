"""Tests for UISettings dataclass."""

from __future__ import annotations

import pytest

from src.ui.settings import UISettings


class TestUISettingsDefault:
    """UISettings.default() のテスト。"""

    def test_default_returns_uisettings(self):
        """default() は UISettings インスタンスを返す。"""
        s = UISettings.default()
        assert isinstance(s, UISettings)

    def test_default_matches_pipeline_init_values(self):
        """default() は PlotterPipeline.__init__() のデフォルトと一致する。"""
        s = UISettings.default()
        assert s.font_size == 4.5
        assert s.line_spacing == 7.16
        assert s.margin_top == 48.0
        assert s.margin_bottom == 34.0
        assert s.margin_left == 5.0
        assert s.margin_right == 5.0
        assert s.draw_speed == 3000.0
        assert s.travel_speed == 5000.0
        assert s.pen_delay == 0.0
        assert s.temperature == 1.0
        assert s.messiness == 1.0
        assert s.pressure_variation == 0.35
        assert s.instance_variation == 0.5
        assert s.entry_taper == 0.3

    def test_default_paper_size_a4(self):
        """default() の用紙サイズは A4。"""
        s = UISettings.default()
        assert s.paper_width == 210.0
        assert s.paper_height == 297.0


class TestUISettingsFrozen:
    """UISettings は frozen dataclass であるテスト。"""

    def test_frozen_cannot_modify(self):
        """frozen=True なので属性変更不可。"""
        s = UISettings.default()
        with pytest.raises(Exception):
            s.font_size = 10.0  # type: ignore[misc]


class TestUISettingsValidate:
    """UISettings.validate() のテスト。"""

    def test_default_is_valid(self):
        """default() は検証エラーなし。"""
        errs = UISettings.default().validate()
        assert errs == []

    def test_margins_too_large_top_bottom(self):
        """上下余白合計が用紙高を超えるとエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=s.font_size,
            line_spacing=s.line_spacing,
            margin_top=200.0,
            margin_bottom=200.0,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=s.draw_speed,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1
        assert any("余白" in e or "margin" in e.lower() for e in errs)

    def test_margins_too_large_left_right(self):
        """左右余白合計が用紙幅を超えるとエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=s.font_size,
            line_spacing=s.line_spacing,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=150.0,
            margin_right=150.0,
            draw_speed=s.draw_speed,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_zero_font_size_invalid(self):
        """font_size <= 0 はエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=0.0,
            line_spacing=s.line_spacing,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=s.draw_speed,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_negative_messiness_invalid(self):
        """messiness < 0 はエラー。"""
        from dataclasses import replace

        invalid = replace(UISettings.default(), messiness=-0.5)
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_zero_messiness_is_valid(self):
        """messiness == 0（揺らぎなし）は妥当。"""
        from dataclasses import replace

        s = replace(UISettings.default(), messiness=0.0)
        assert s.validate() == []

    def test_line_spacing_too_small(self):
        """line_spacing < font_size * 0.6 はエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=10.0,
            line_spacing=2.0,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=s.draw_speed,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_zero_draw_speed_invalid(self):
        """draw_speed <= 0 はエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=s.font_size,
            line_spacing=s.line_spacing,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=0.0,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_zero_travel_speed_invalid(self):
        """travel_speed <= 0 はエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=s.font_size,
            line_spacing=s.line_spacing,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=s.draw_speed,
            travel_speed=0.0,
            pen_delay=s.pen_delay,
            temperature=s.temperature,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_zero_temperature_invalid(self):
        """temperature <= 0 はエラー。"""
        s = UISettings.default()
        invalid = UISettings(
            font_size=s.font_size,
            line_spacing=s.line_spacing,
            margin_top=s.margin_top,
            margin_bottom=s.margin_bottom,
            margin_left=s.margin_left,
            margin_right=s.margin_right,
            draw_speed=s.draw_speed,
            travel_speed=s.travel_speed,
            pen_delay=s.pen_delay,
            temperature=0.0,
        )
        errs = invalid.validate()
        assert len(errs) >= 1

    def test_validate_returns_list_type(self):
        """validate() は list を返す。"""
        s = UISettings.default()
        result = s.validate()
        assert isinstance(result, list)


class TestUISettingsSerialization:
    """to_dict() / from_dict() による永続化シリアライズのテスト。"""

    def test_to_dict_contains_all_fields(self):
        """to_dict() は全フィールドを含む。"""
        d = UISettings.default().to_dict()
        for field in (
            "font_size",
            "line_spacing",
            "margin_top",
            "margin_bottom",
            "margin_left",
            "margin_right",
            "draw_speed",
            "travel_speed",
            "pen_delay",
            "temperature",
            "messiness",
            "pressure_variation",
            "instance_variation",
            "entry_taper",
            "paper_width",
            "paper_height",
        ):
            assert field in d

    def test_roundtrip_preserves_values(self):
        """to_dict → from_dict で元の値が完全に復元される。"""
        s = UISettings(
            font_size=6.0,
            line_spacing=9.0,
            margin_top=20.0,
            margin_bottom=20.0,
            margin_left=15.0,
            margin_right=15.0,
            draw_speed=1500.0,
            travel_speed=4000.0,
            pen_delay=0.1,
            temperature=1.5,
        )
        assert UISettings.from_dict(s.to_dict()) == s

    def test_from_dict_none_returns_default(self):
        """None からは default() を返す。"""
        assert UISettings.from_dict(None) == UISettings.default()

    def test_from_dict_empty_returns_default(self):
        """空 dict からは default() を返す。"""
        assert UISettings.from_dict({}) == UISettings.default()

    def test_from_dict_partial_fills_defaults(self):
        """部分 dict は欠損フィールドを default 値で補完する。"""
        s = UISettings.from_dict({"font_size": 7.0})
        assert s.font_size == 7.0
        assert s.line_spacing == UISettings.default().line_spacing

    def test_from_dict_ignores_unknown_keys(self):
        """未知キーは無視される（ストレージ破損・将来削除フィールドに強い）。"""
        s = UISettings.from_dict({"font_size": 7.0, "bogus_key": 99})
        assert s.font_size == 7.0
        assert not hasattr(s, "bogus_key")

    def test_from_dict_ignores_invalid_values(self):
        """float 化できない値は無視され default 値が残る。"""
        default_fs = UISettings.default().font_size
        s = UISettings.from_dict({"font_size": "not_a_number"})
        assert s.font_size == default_fs

    def test_from_dict_ignores_none_values(self):
        """値が None のフィールドは無視される。"""
        default_fs = UISettings.default().font_size
        s = UISettings.from_dict({"font_size": None})
        assert s.font_size == default_fs


class TestBuildPipeline:
    """build_pipeline() のテスト。"""

    def test_build_pipeline_returns_pipeline(self):
        """build_pipeline() は PlotterPipeline を返す。"""
        from src.ui.web_app import PlotterPipeline, build_pipeline

        s = UISettings.default()
        pipeline = build_pipeline(s)
        assert isinstance(pipeline, PlotterPipeline)

    def test_build_pipeline_reflects_settings(self):
        """build_pipeline() は UISettings の値を PageConfig/PlotterConfig に反映する。"""
        from src.ui.web_app import build_pipeline

        s = UISettings(
            font_size=6.0,
            line_spacing=9.0,
            margin_top=20.0,
            margin_bottom=20.0,
            margin_left=15.0,
            margin_right=15.0,
            draw_speed=1500.0,
            travel_speed=4000.0,
            pen_delay=0.1,
            temperature=1.5,
        )
        pipeline = build_pipeline(s)

        assert pipeline._page_config.line_spacing == 9.0
        assert pipeline._page_config.margin_top == 20.0
        assert pipeline._page_config.margin_bottom == 20.0
        assert pipeline._page_config.margin_left == 15.0
        assert pipeline._page_config.margin_right == 15.0
        assert pipeline._typesetter.font_size == 6.0
        assert pipeline._plotter_config.draw_speed == 1500.0
        assert pipeline._plotter_config.travel_speed == 4000.0
        assert pipeline._plotter_config.pen_delay == 0.1
        assert pipeline._temperature == 1.5

    def test_build_pipeline_default_matches_pipeline_init_default(self):
        """default() からの build_pipeline は PlotterPipeline() と等価な値を持つ。"""
        from src.ui.web_app import PlotterPipeline, build_pipeline

        from_settings = build_pipeline(UISettings.default())
        from_init = PlotterPipeline()

        assert from_settings._page_config.line_spacing == from_init._page_config.line_spacing
        assert from_settings._page_config.margin_top == from_init._page_config.margin_top
        assert from_settings._page_config.margin_bottom == from_init._page_config.margin_bottom
        assert from_settings._page_config.margin_left == from_init._page_config.margin_left
        assert from_settings._page_config.margin_right == from_init._page_config.margin_right
        assert from_settings._typesetter.font_size == from_init._typesetter.font_size
        assert from_settings._plotter_config.draw_speed == from_init._plotter_config.draw_speed
        assert from_settings._plotter_config.travel_speed == from_init._plotter_config.travel_speed
        assert from_settings._plotter_config.pen_delay == from_init._plotter_config.pen_delay
        assert from_settings._temperature == from_init._temperature

    def test_build_pipeline_messiness_scales_augmenter(self):
        """messiness は augmenter の揺らぎ4項目を一括スケールする。"""
        from dataclasses import replace

        from src.model.augmentation import AugmentConfig
        from src.ui.web_app import build_pipeline

        base = AugmentConfig()
        pipeline = build_pipeline(replace(UISettings.default(), messiness=2.0))
        cfg = pipeline._typesetter.augmenter._config
        assert cfg.baseline_drift == pytest.approx(base.baseline_drift * 2.0)
        assert cfg.spacing_variation == pytest.approx(base.spacing_variation * 2.0)
        assert cfg.size_variation == pytest.approx(base.size_variation * 2.0)
        assert cfg.slant_variation == pytest.approx(base.slant_variation * 2.0)

    def test_build_pipeline_messiness_zero_disables_jitter(self):
        """messiness=0 で揺らぎ4項目が 0 になる（整った字）。"""
        from dataclasses import replace

        from src.ui.web_app import build_pipeline

        pipeline = build_pipeline(replace(UISettings.default(), messiness=0.0))
        cfg = pipeline._typesetter.augmenter._config
        assert cfg.baseline_drift == 0.0
        assert cfg.spacing_variation == 0.0
        assert cfg.size_variation == 0.0
        assert cfg.slant_variation == 0.0

    def test_build_pipeline_default_messiness_unchanged(self):
        """messiness=1.0（デフォルト）は AugmentConfig の素の値と一致。"""
        from src.model.augmentation import AugmentConfig
        from src.ui.web_app import build_pipeline

        base = AugmentConfig()
        pipeline = build_pipeline(UISettings.default())
        cfg = pipeline._typesetter.augmenter._config
        assert cfg.baseline_drift == pytest.approx(base.baseline_drift)
        assert cfg.slant_variation == pytest.approx(base.slant_variation)

    def test_build_pipeline_passes_through_kwargs(self, tmp_path):
        """build_pipeline は user_strokes_dir などを既存シグネチャ通り受け渡す。"""
        from src.ui.web_app import build_pipeline

        empty_dir = tmp_path / "empty_strokes"
        empty_dir.mkdir()

        s = UISettings.default()
        pipeline = build_pipeline(s, user_strokes_dir=empty_dir)
        assert pipeline._user_stroke_db == {}
