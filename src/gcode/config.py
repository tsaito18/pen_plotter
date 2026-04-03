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

    # G-code設定
    decimal_places: int = 2

    def pen_delay_gcode(self) -> str:
        """ペン昇降後の待機G-code。xDrawはZ軸速度制御のため通常不要。"""
        if self.pen_delay <= 0:
            return ""
        ms = int(self.pen_delay * 1000)
        return f"G4 P{ms}"
