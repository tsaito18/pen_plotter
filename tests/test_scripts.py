"""CLI スクリプト（KanjiVG 変換・訓練・Web UI の構築）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_kanjivg import (
    convert_single_svg,
    convert_xml_to_samples,
    hex_filename_to_char,
)
from src.collector.data_format import StrokeSample

XML = """<?xml version="1.0" encoding="UTF-8"?>
<kanjivg xmlns:kvg='http://kanjivg.tagaini.net'>
<kanji id="kvg:kanji_04e8c">
<g id="kvg:04e8c" kvg:element="二">
  <path kvg:type="㇐" d="M 20,30 L 80,30"/>
  <path kvg:type="㇒" d="M 50,50"/>
  <path kvg:type="㇏" d="M 10,70 L 90,70"/>
</g>
</kanji>
</kanjivg>"""

SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:kvg="http://kanjivg.tagaini.net">
  <g id="kvg:04e00" kvg:element="一">
    <path kvg:type="㇐" d="M 30,20 L 30,90"/>
    <path d="M 70,20 L 70,90"/>
  </g>
</svg>"""


def test_hex_filename_to_char():
    assert hex_filename_to_char("0904e") == "過"
    assert hex_filename_to_char("03042") == "あ"


def test_convert_xml_flips_to_y_up_and_keeps_types_aligned(tmp_path: Path):
    xml = tmp_path / "kanjivg.xml"
    xml.write_text(XML, encoding="utf-8")
    assert convert_xml_to_samples(xml, tmp_path / "out", target_size=10.0, num_points=8) == 1
    sample = StrokeSample.load(next((tmp_path / "out" / "二").glob("二_*.json")))
    # 単一点の 2 画目は捨てられ、筆画タイプも同じ位置で落ちる
    assert sample.stroke_types == ["㇐", "㇏"]
    assert all(len(s) == 8 for s in sample.strokes)
    top, bottom = (np.mean([p.y for p in s]) for s in sample.strokes)
    assert top > bottom  # SVG(Y-DOWN) の上の画が Y-UP で上


def test_convert_single_svg(tmp_path: Path):
    svg = tmp_path / "04e00.svg"
    svg.write_text(SVG, encoding="utf-8")
    sample = convert_single_svg(svg, tmp_path, target_size=10.0, num_points=8)
    assert sample.character == "一" and sample.stroke_types == ["㇐", ""]
    assert list((tmp_path / "一").glob("*.json"))
    bad = tmp_path / "not_hex.svg"
    bad.write_text(SVG, encoding="utf-8")
    assert convert_single_svg(bad, tmp_path, target_size=10.0, num_points=8) is None


def test_train_cli_runs_pretrain(user_strokes_root: Path, kanjivg_dir: Path, tmp_path: Path):
    from scripts.train import main

    args = ["pretrain", "--user-dir", str(user_strokes_root), "--ref-dir", str(kanjivg_dir)]
    args += ["--output-dir", str(tmp_path), "--epochs", "1", "--deformer-type", "offset"]
    losses = main([*args, "--style-dim", "16", "--hidden-dim", "16"])
    assert len(losses) == 1 and (tmp_path / "pretrain_checkpoint.pt").exists()
    with pytest.raises(SystemExit):
        main(["finetune", "--checkpoint", str(tmp_path / "none.pt"), "--ref-dir", str(tmp_path)])
