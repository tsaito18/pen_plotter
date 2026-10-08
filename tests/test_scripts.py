"""CLI スクリプト（KanjiVG 変換・訓練・Web UI の構築）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_kanjivg import (
    COMPONENTS_FILE,
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

XML_WITH_PARTS = """<?xml version="1.0" encoding="UTF-8"?>
<kanjivg xmlns:kvg='http://kanjivg.tagaini.net'>
<kanji id="kvg:kanji_0597d">
<g id="kvg:0597d" kvg:element="好">
  <g id="kvg:0597d-g1" kvg:position="left">
    <g id="kvg:0597d-g2" kvg:element="女">
      <path d="M 20,20 L 30,80"/><path d="M 40,20 L 10,60"/><path d="M 10,50 L 45,50"/>
    </g>
  </g>
  <g id="kvg:0597d-g3" kvg:element="子" kvg:position="right">
    <path d="M 60,20 L 90,20"/><path d="M 50,50"/><path d="M 75,20 L 75,90"/>
    <path d="M 55,55 L 95,55"/>
  </g>
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


def test_convert_xml_writes_the_parts_table(tmp_path: Path):
    """部品（部首など）ごとの画番号表。1 点だけの画は捨てるので番号もそれに合わせる。"""
    xml = tmp_path / "kanjivg.xml"
    xml.write_text(XML_WITH_PARTS, encoding="utf-8")
    convert_xml_to_samples(xml, tmp_path / "out", target_size=10.0, num_points=8)
    table = json.loads((tmp_path / "out" / COMPONENTS_FILE).read_text(encoding="utf-8"))
    assert table["好"] == {
        "strokes": 6,
        "parts": [["女", "left", [0, 1, 2]], ["子", "right", [3, 4, 5]]],
    }
    sample = StrokeSample.load(next((tmp_path / "out" / "好").glob("*.json")))
    assert len(sample.strokes) == table["好"]["strokes"]


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
