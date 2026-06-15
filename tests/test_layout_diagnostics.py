from src.layout.typesetter import CharPlacement
from src.ui import layout_diagnostics as diagnostics
from src.ui.layout_diagnostics import LayoutReport, _y_extent, diagnose_placements


def _cp(char="", x=0.0, y=0.0, fs=4.5, **kw):
    return CharPlacement(char=char, x=x, y=y, font_size=fs, **kw)


class TestHorizontalOverlap:
    def test_no_overlap_for_spaced_chars(self):
        page = [_cp("あ", x=0.0, y=10.0), _cp("い", x=5.0, y=10.0)]
        assert diagnose_placements([page], line_spacing=7.0) == []

    def test_math_wider_than_slot_flagged(self):
        # 数式 bbox 幅(8mm)が次の文字までのスロット(2mm)を超える → 水平かぶり
        page = [
            _cp("", x=0.0, y=10.0, math_source="x^2", math_bbox=(0.0, 8.0, 8.0, 4.0)),
            _cp("あ", x=2.0, y=10.0),
        ]
        ov = diagnose_placements([page], line_spacing=7.0)
        assert len(ov) == 1
        assert ov[0].kind == "horizontal"

    def test_line_segments_ignored(self):
        page = [_cp("", x=0.0, y=10.0, line_segment=(0, 10, 50, 10))]
        assert diagnose_placements([page], line_spacing=7.0) == []


class TestVerticalOverlap:
    def test_tall_fraction_intrudes_next_line(self, monkeypatch):
        # baseline align の math_bbox[1] は bbox 下端でなく本文ベースライン。
        monkeypatch.setattr(diagnostics, "_baseline_frac_from_top", lambda _src: 0.5)
        page = [
            _cp(
                "",
                x=0.0,
                y=10.0,
                math_source="\\frac{a}{b}",
                math_bbox=(0.0, 10.0, 5.0, 14.0),
                math_align="baseline",
            )
        ]
        ov = diagnose_placements([page], line_spacing=7.0)
        assert [(o.kind, o.b, o.amount_mm) for o in ov] == [("vertical", "下の行", 5.75)]

    def test_inline_math_within_line_ok(self):
        # 高さが行間に収まる数式は垂直かぶりなし
        page = [_cp("", x=0.0, y=10.0, math_source="V", math_bbox=(0.0, 10.0, 3.0, 4.0))]
        ov = [o for o in diagnose_placements([page], line_spacing=7.0) if o.kind == "vertical"]
        assert ov == []

    def test_y_extent_baseline_uses_baseline_fraction(self, monkeypatch):
        monkeypatch.setattr(diagnostics, "_baseline_frac_from_top", lambda _src: 0.5)
        p = _cp(
            "",
            y=10.0,
            math_source="x",
            math_bbox=(0.0, 10.0, 3.0, 4.0),
            math_align="baseline",
        )

        assert _y_extent(p, line_spacing=7.0) == (9.25, 13.25)

    def test_y_extent_center_uses_bbox_y_as_bottom(self):
        p = _cp("", y=10.0, math_source="x", math_bbox=(0.0, 4.0, 3.0, 4.0))

        assert _y_extent(p, line_spacing=7.0) == (4.0, 8.0)


class TestReport:
    def test_ok_when_clean(self):
        r = LayoutReport()
        assert r.ok
        assert "なし" in r.summary()
