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
    draw_speed: float = 1000.0

    # ペン制御 (Z軸)
    pen_down_command: str = "G1G90 Z3.5 F5000"
    pen_up_command: str = "G1G90 Z0.5 F5000"
    pen_delay: float = 0.0  # Z軸制御は速度指定で完了するため遅延不要

    # 終端Zリフト（払い・はねの接触圧を抜く筆遣い表現）
    # pen_down_z は補間の基準となる接触高さ。pen_down_command の Z 値と一致させること。
    pen_down_z: float = 3.5
    # 終端で芯を持ち上げる到達高さ。pen_up_z(0.5) < finish_lift_z < pen_down_z(3.5)。
    # 芯が完全に離れない手前。実機キャリブで詰める仮値。
    finish_lift_z: float = 2.0
    # 終端のリフト区間に充てるストローク末尾の点数。
    finish_lift_points: int = 5
    # 払い＝ゆっくり抜く、はね＝速く跳ね上げる、の速度倍率。
    harai_speed_factor: float = 0.5
    hane_speed_factor: float = 1.3
    # Z軸移動のフィードレート (mm/min)。
    pen_z_feed: float = 5000.0

    # G-code設定
    decimal_places: int = 2

    def pen_delay_gcode(self) -> str:
        """ペン昇降後の待機G-code。xDrawはZ軸速度制御のため通常不要。"""
        if self.pen_delay <= 0:
            return ""
        ms = int(self.pen_delay * 1000)
        return f"G4 P{ms}"
