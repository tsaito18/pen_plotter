"""手書き生成の設定（用紙・組版・揺らぎ）。UI・CLI・パイプラインの単一ソース。

不変（frozen）にして、UI は設定の snapshot から毎回パイプラインを組み立てる。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

from src.gcode.config import PlotterConfig
from src.layout.page_layout import PageConfig


@dataclass(frozen=True)
class Settings:
    """手書き生成の設定。既定値はレポート用紙（A4・罫線 7.14mm）の実測値。

    Attributes:
        font_size: 漢字の字高 (mm)。
        line_spacing: 行間＝罫線間隔 (mm)。
        margin_top / margin_bottom / margin_left / margin_right: 余白 (mm)。
        temperature: ML 変形の字形揺らぎ（0 で決定的）。
        messiness: 配置の揺らぎ倍率（ベースライン・字間・サイズ・傾き）。
            0=整った字、1=素の値、2=大きく乱れる。既定は揺らぎ控えめ（各画が
            バラバラに動くとストローク間のバランスが崩れるため）。
        pressure_variation: 画内の筆圧（濃淡）変調 ∈[0,1]。【実機注意】描画中に Z を
            振るため単線シャーペンでは点線化する。実機は 0（プレビュー演出用）。
        instance_variation: 同じ字を書くたびに形を変える強さ ∈[0,1]。
        entry_taper: 入筆（始筆を軽く入れる）強さ ∈[0,1]。【実機注意】始筆がかすれ得る。
        connection_strength: 連綿（近い画を薄いつなぎ線で続ける）強さ ∈[0,1]。
        plot_page_numbers: ページ番号を手書きで入れるか。
        paper_width / paper_height: 用紙寸法 (mm)。
    """

    font_size: float = 4.5
    line_spacing: float = 7.14
    margin_top: float = 48.0
    margin_bottom: float = 34.0
    margin_left: float = 5.0
    margin_right: float = 5.0
    temperature: float = 0.2
    messiness: float = 0.4
    pressure_variation: float = 0.0
    instance_variation: float = 0.1
    entry_taper: float = 0.0
    connection_strength: float = 0.0
    plot_page_numbers: bool = True
    paper_width: float = 210.0
    paper_height: float = 297.0

    def page_config(self) -> PageConfig:
        return PageConfig(
            paper_size=(self.paper_width, self.paper_height),
            margin_top=self.margin_top,
            margin_bottom=self.margin_bottom,
            margin_left=self.margin_left,
            margin_right=self.margin_right,
            line_spacing=self.line_spacing,
        )

    def plotter_config(self) -> PlotterConfig:
        return PlotterConfig(
            work_area_width=220.0,
            work_area_height=310.0,
            paper_width=self.paper_width,
            paper_height=self.paper_height,
            pressure_variation=self.pressure_variation,
            entry_taper=self.entry_taper,
        )

    def validate(self) -> list[str]:
        """設定値の問題点（日本語メッセージ）のリスト。問題なければ空。"""
        errors: list[str] = []
        if self.margin_top + self.margin_bottom >= self.paper_height:
            errors.append(
                f"上下余白の合計({self.margin_top + self.margin_bottom:.1f}mm)が"
                f"用紙高({self.paper_height:.1f}mm)以上です"
            )
        if self.margin_left + self.margin_right >= self.paper_width:
            errors.append(
                f"左右余白の合計({self.margin_left + self.margin_right:.1f}mm)が"
                f"用紙幅({self.paper_width:.1f}mm)以上です"
            )
        if self.font_size <= 0:
            errors.append(f"font_size は正の値である必要があります (現在: {self.font_size})")
        # 行間がフォントサイズの 60% 未満では行が重なって読めない
        if self.line_spacing < self.font_size * 0.6:
            errors.append(
                f"line_spacing({self.line_spacing:.2f}mm)が"
                f"font_size の 60% ({self.font_size * 0.6:.2f}mm)未満です"
            )
        if self.temperature < 0:
            errors.append(f"temperature は 0 以上である必要があります (現在: {self.temperature})")
        if self.messiness < 0:
            errors.append(f"messiness は 0 以上である必要があります (現在: {self.messiness})")
        return errors

    def to_dict(self) -> dict[str, object]:
        """JSON 化できる dict（ブラウザ永続化用）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> Settings:
        """dict から復元する。未知キー・不正値は無視し、欠損は既定値で補う。"""
        base = cls()
        if not data:
            return base
        updates: dict[str, object] = {}
        for f in fields(cls):
            value = data.get(f.name)
            if value is None:
                continue
            if isinstance(getattr(base, f.name), bool):
                parsed = _parse_bool(value)
                if parsed is not None:
                    updates[f.name] = parsed
                continue
            try:
                updates[f.name] = float(value)
            except (TypeError, ValueError):
                continue
        return replace(base, **updates)


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None
