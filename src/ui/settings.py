"""UI 設定の不変データクラス。

UI 層から扱いやすい snapshot を提供し、検証ロジックを集約する。
PlotterPipeline は UISettings を介して構築されることで、
副作用を伴う設定差し替えを排除する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace


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
        temperature: ML モデル温度（字形そのものの揺らぎの強度）。
        messiness: レイアウトの汚さ倍率。baseline_drift/字間/サイズ/傾きを
            一括スケール。1.0=標準、0=整った字、2=大きく乱れる。temperature
            （字形の揺らぎ）とは別軸。
        pressure_variation: 画内の筆圧（濃淡）変調の深さ ∈[0,1]。0=均一な定幅
            ペン感、大=人の筆圧リズム（下ろし濃く・上げ薄く）。【実機注意】描画中に
            Z を振るため単線シャーペンでは点線化する。実機では 0、プレビュー演出用。
        instance_variation: 同一字の繰り返しで形を変える強度 ∈[0,1]。0=毎回同じ、
            大=書くたびに微妙に違う（人が同じ字を書いても揺れる）。Z は動かさない。
        entry_taper: 入筆（始筆を軽く入れて立ち上げる）の強度 ∈[0,1]。0=なし。
            【実機注意】始筆で Z を動かすためかすれ得る。実機では 0 推奨。
        connection_strength: 連綿（続け字）の強さ ∈[0,1]。0=なし。同じ字の近い画を、
            近いほど高確率＋乱数で薄いつなぎ線で結ぶ（ペンを上げずに継続）。Z 一定の
            つなぎなので点線化しない。
        plot_page_numbers: ページ番号をプロットするか。
        paper_width: 用紙幅 (mm)。デフォルト A4。
        paper_height: 用紙高 (mm)。デフォルト A4。
    """

    font_size: float
    line_spacing: float
    margin_top: float
    margin_bottom: float
    margin_left: float
    margin_right: float
    temperature: float
    messiness: float = 1.0
    pressure_variation: float = 0.0
    instance_variation: float = 0.5
    entry_taper: float = 0.0
    connection_strength: float = 0.0
    plot_page_numbers: bool = True
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
            line_spacing=7.14,
            margin_top=48.0,
            margin_bottom=34.0,
            margin_left=5.0,
            margin_right=5.0,
            temperature=1.0,
            messiness=1.0,
            pressure_variation=0.0,
            instance_variation=0.5,
            entry_taper=0.0,
            connection_strength=0.0,
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

        if self.temperature <= 0:
            errors.append(f"temperature は正の値である必要があります (現在: {self.temperature})")
        if self.messiness < 0:
            errors.append(f"messiness は 0 以上である必要があります (現在: {self.messiness})")

        return errors

    def to_dict(self) -> dict[str, object]:
        """ブラウザ永続化（gr.BrowserState）用の plain dict へ変換する。

        Returns:
            全フィールドを持つ dict。JSON シリアライズ可能。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> UISettings:
        """to_dict() / 永続化された dict から UISettings を復元する。

        default() をベースに既知フィールドのみを上書きするため、
        フィールドの増減やストレージ破損に強い（前方/後方互換）。

        Args:
            data: 復元元 dict。None・空・未知キーは無視され、欠損は default 値で補完。

        Returns:
            復元された UISettings。data が None/空なら default()。
        """
        base = cls.default()
        if not data:
            return base
        valid = {f.name for f in fields(cls)}
        updates: dict[str, object] = {}
        for key, value in data.items():
            if key not in valid or value is None:
                continue
            current = getattr(base, key)
            if isinstance(current, bool):
                parsed_bool = cls._parse_bool(value)
                if parsed_bool is not None:
                    updates[key] = parsed_bool
                continue
            try:
                updates[key] = float(value)
            except (TypeError, ValueError):
                continue
        return replace(base, **updates)

    @staticmethod
    def _parse_bool(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            return None
        if isinstance(value, int | float):
            return bool(value)
        return None
