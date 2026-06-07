from dataclasses import dataclass


@dataclass
class PlotterConfig:
    """xDraw A4 ペンプロッタの機械パラメータ"""

    # ワーキングエリア (mm) — A4サイズ
    work_area_width: float = 210.0
    work_area_height: float = 297.0

    # A4用紙配置（紙座標系: 左下=(0,0), 右上=(210,297)）
    paper_origin_x: float = 0.0
    paper_origin_y: float = 0.0
    paper_width: float = 210.0
    paper_height: float = 297.0

    # 速度 (mm/min)
    travel_speed: float = 5000.0
    draw_speed: float = 3000.0

    # ペン制御 (Z軸)
    pen_down_command: str = "G1G90 Z5.0 F5000"
    pen_up_command: str = "G1G90 Z0.5 F5000"
    pen_delay: float = 0.0  # Z軸制御は速度指定で完了するため遅延不要

    # 終端Zリフト（払い・はねの接触圧を抜く筆遣い表現）
    # pen_down_z は補間の基準となる接触（最大筆圧）高さ。pen_down_command の Z 値と
    # 一致させること。リフト導入時に一旦 3.5 へ下げたが、通常画まで筆圧不足になり
    # 終端が薄くなったため、元の実機キャリブ値 5.0（しっかり下ろす）へ戻す。
    pen_down_z: float = 5.0
    # 終端で芯を持ち上げる到達高さ。pen_up_z(0.5) < finish_lift_z < pen_down_z(5.0)。
    # 実機計測で芯の浮き始め≈2.7。それより下げると余計に持ち上がる（上がりが急・
    # 過大）だけで線は変わらないため、浮き始め直下の 2.6 に留める＝最小リフト。
    finish_lift_z: float = 2.6
    # 終端のリフト区間長(mm)。終端からこの距離でZを接触→finish_lift_zへ漸減する。
    # 点数でなく実距離なので文字サイズに依らず同じ抜けになる（小さい字でも先細る）。
    finish_lift_length_mm: float = 1.5
    # 払い＝ゆっくり抜く、はね＝速く跳ね上げる、の速度倍率。
    harai_speed_factor: float = 0.5
    hane_speed_factor: float = 1.3
    # 画内の筆圧（濃淡）変調の深さ ∈[0,1]。0=均一(定幅ペン感)、大=人の筆圧リズム。
    # 【実機注意】描画中に Z を上下に振るため、単線シャーペンではペンがバウンドして
    # 線が点線化する。実機では 0 推奨（プレビュー上の濃淡表現のみ）。preview/Z連動。
    pressure_variation: float = 0.0
    # 入筆: 始筆を軽く入れて満額へ立ち上げる強度 ∈[0,1] と、その区間長(mm)。
    # 【実機注意】始筆で Z を動かすため線がかすれ得る。実機では 0 推奨。
    entry_taper: float = 0.0
    entry_length_mm: float = 0.7
    # Z軸移動のフィードレート (mm/min)。
    pen_z_feed: float = 5000.0

    # ストローク簡略化(RDP)の許容偏差(mm)。リサンプリングは曲率に関係なく32点固定で
    # 出力するため、4.5mm字では約0.12mmの極短セグメントが連続し GRBL が加減速しきれず
    # 「ガガガ」と震えて遅くなる。共線に近い冗長点を畳んで最小セグメント長を確保する。
    # 0.05mm: 緩い曲線32→約5点(最小≈0.8mm)・直線→2点、手書き揺らぎ(tremor)は温存。
    # 0=簡略化なし。
    simplify_tolerance_mm: float = 0.05

    # G-code設定
    decimal_places: int = 2

    def pen_delay_gcode(self) -> str:
        """ペン昇降後の待機G-code。xDrawはZ軸速度制御のため通常不要。"""
        if self.pen_delay <= 0:
            return ""
        ms = int(self.pen_delay * 1000)
        return f"G4 P{ms}"
