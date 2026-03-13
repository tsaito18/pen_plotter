from dataclasses import dataclass


@dataclass
class PlotterConfig:
    """CoreXY ペンプロッタの機械パラメータ"""

    # 作業エリア (mm)
    work_area_width: float = 300.0
    work_area_height: float = 220.0

    # A4用紙配置（用紙左下の座標）
    paper_origin_x: float = 1.5
    paper_origin_y: float = 5.0
    paper_width: float = 297.0
    paper_height: float = 210.0

    # モーター設定: GT2 20T + 1.8° stepper
    steps_per_mm: float = 80.0

    # 速度 (mm/min)
    travel_speed: float = 3000.0
    draw_speed: float = 1000.0
    max_speed: float = 5000.0
    acceleration: float = 500.0  # mm/s^2

    # ペン制御
    pen_down_command: str = "M3 S255"
    pen_up_command: str = "M5"
    pen_delay: float = 0.15  # ペン昇降後の待機時間 (秒)

    # G-code設定
    decimal_places: int = 3

    def pen_delay_gcode(self) -> str:
        """ペン昇降後の待機G-code (G4 Pミリ秒)"""
        ms = int(self.pen_delay * 1000)
        return f"G4 P{ms}"
