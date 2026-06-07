from src.layout.typesetter import CharPlacement
from src.ui.layout_diagnostics import LayoutReport, diagnose_placements


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
    def test_tall_fraction_intrudes_next_line(self):
        # 高さ14mm の数式が line_spacing 7mm を大きく超え上下行へ食い込む
        page = [
            _cp(
                "",
                x=0.0,
                y=10.0,
                math_source="\\frac{a}{b}",
                math_bbox=(0.0, 4.0, 5.0, 14.0),
                math_align="baseline",
            )
        ]
        ov = diagnose_placements([page], line_spacing=7.0)
        assert any(o.kind == "vertical" for o in ov)

    def test_inline_math_within_line_ok(self):
        # 高さが行間に収まる数式は垂直かぶりなし
        page = [_cp("", x=0.0, y=10.0, math_source="V", math_bbox=(0.0, 10.0, 3.0, 4.0))]
        ov = [o for o in diagnose_placements([page], line_spacing=7.0) if o.kind == "vertical"]
        assert ov == []


class TestReport:
    def test_ok_when_clean(self):
        r = LayoutReport()
        assert r.ok
        assert "なし" in r.summary()
