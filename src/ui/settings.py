"""UI 設定の不変データクラス。

UI 層から扱いやすい snapshot を提供し、検証ロジックを集約する。
PlotterPipeline は UISettings を介して構築されることで、
副作用を伴う設定差し替えを排除する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UISettings:
    """UI から渡されるパイプライン設定。

    PlotterPipeline.__init__() のデフォルトと一致させた値を default() で返す。
    変更不可（frozen=True）にすることで「設定 snapshot から毎回新しいパイプラインを構築する」
    ステートレス API パターンを支える。

    Attributes:
        font_size: フォントサイズ (mm)。
        line_spacing: 行間隔 (mm)。
        margin_top: 上余白 (mm)。
        margin_bottom: 下余白 (mm)。
        margin_left: 左余白 (mm)。
        margin_right: 右余白 (mm)。
        draw_speed: 描画速度 (mm/min)。
        travel_speed: 移動速度 (mm/min)。
        pen_delay: ペン昇降後の待機 (s)。
        temperature: ML モデル温度（揺らぎの強度）。
        paper_width: 用紙幅 (mm)。デフォルト A4。
        paper_height: 用紙高 (mm)。デフォルト A4。
    """

    font_size: float
    line_spacing: float
    margin_top: float
    margin_bottom: float
    margin_left: float
    margin_right: float
    draw_speed: float
    travel_speed: float
    pen_delay: float
    temperature: float
    paper_width: float = 210.0
    paper_height: float = 297.0

    @classmethod
    def default(cls) -> UISettings:
        """PlotterPipeline.__init__() のデフォルト値と完全一致する設定を返す。

        UI の「リセット」操作はこの値に戻すべき真の初期値である。
        既存の gradio_app.py の _reset_defaults() は font_size=6.0 等の
        誤った値を返していたが、本メソッドへ統一する。
        """
        return cls(
            font_size=4.5,
            line_spacing=7.16,
            margin_top=48.0,
            margin_bottom=34.0,
            margin_left=5.0,
            margin_right=5.0,
            draw_speed=1000.0,
            travel_speed=5000.0,
            pen_delay=0.0,
            temperature=1.0,
        )

    def validate(self) -> list[str]:
        """設定値の妥当性を検証し、エラーメッセージのリストを返す。

        Returns:
            エラーメッセージのリスト。妥当な場合は空リスト。
        """
        errors: list[str] = []

        # 用紙余白: 最低 1 行・1 文字が入る余地を確保
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

        # 行間がフォントサイズの 60% 未満では行が重なって読めなくなる
        if self.line_spacing < self.font_size * 0.6:
            errors.append(
                f"line_spacing({self.line_spacing:.2f}mm)が"
                f"font_size の 60% ({self.font_size * 0.6:.2f}mm)未満です"
            )

        if self.draw_speed <= 0:
            errors.append(f"draw_speed は正の値である必要があります (現在: {self.draw_speed})")
        if self.travel_speed <= 0:
            errors.append(f"travel_speed は正の値である必要があります (現在: {self.travel_speed})")
        if self.temperature <= 0:
            errors.append(f"temperature は正の値である必要があります (現在: {self.temperature})")

        return errors
