from src.layout.diagram_layout import (
    detect_blockdiagram,
    detect_dendrogram,
    parse_blockdiagram,
    parse_dendrogram,
)


class TestParseBlockdiagram:
    def test_parse_open_loop_chain(self):
        spec = parse_blockdiagram(["R --> [制御器] --> [プラント] --> C"])
        assert [n.kind for n in spec.nodes] == ["signal", "box", "box", "signal"]
        assert [n.label for n in spec.nodes] == ["R", "制御器", "プラント", "C"]
        assert spec.feedback == []
        assert spec.is_closed_loop is False

    def test_parse_closed_loop_with_feedback(self):
        spec = parse_blockdiagram(["R --> [制御器] --> [プラント] --> C", "feedback: [センサ]"])
        assert spec.is_closed_loop is True
        assert len(spec.feedback) == 1
        assert spec.feedback[0].kind == "box"
        assert spec.feedback[0].label == "センサ"

    def test_parse_empty_input_returns_empty_spec(self):
        spec = parse_blockdiagram([])
        assert spec.nodes == []
        assert spec.feedback == []

    def test_parse_blank_lines_only(self):
        spec = parse_blockdiagram(["", "   ", ""])
        assert spec.nodes == []

    def test_parse_arrow_only_line_returns_empty_nodes(self):
        spec = parse_blockdiagram(["-->"])
        assert spec.nodes == []

    def test_parse_bare_signal_labels(self):
        spec = parse_blockdiagram(["R --> C"])
        assert [n.kind for n in spec.nodes] == ["signal", "signal"]

    def test_parse_circle_sum_node(self):
        spec = parse_blockdiagram(["R --> (Σ) --> [G] --> C"])
        assert spec.nodes[1].kind == "circle"
        assert spec.nodes[1].is_sum is True
        assert spec.nodes[1].label == "Σ"

    def test_parse_circle_plus_alias(self):
        spec = parse_blockdiagram(["R --> (+) --> [G] --> C"])
        assert spec.nodes[1].is_sum is True

    def test_parse_circle_non_sum_label(self):
        spec = parse_blockdiagram(["R --> (X) --> [G] --> C"])
        assert spec.nodes[1].kind == "circle"
        assert spec.nodes[1].is_sum is False
        assert spec.nodes[1].label == "X"


class TestDetectBlockdiagram:
    def test_detect_fence(self):
        paragraphs = [
            "text before",
            "```blockdiagram",
            "R --> [制御器] --> C",
            "```",
            "text after",
        ]
        result = detect_blockdiagram(paragraphs, 1)
        assert result is not None
        spec, consumed = result
        assert consumed == 3
        assert len(spec.nodes) == 3

    def test_detect_multiline_fence_with_feedback(self):
        paragraphs = [
            "```blockdiagram",
            "R --> [制御器] --> [プラント] --> C",
            "feedback: [センサ]",
            "```",
        ]
        result = detect_blockdiagram(paragraphs, 0)
        assert result is not None
        spec, consumed = result
        assert consumed == 4
        assert spec.is_closed_loop is True

    def test_detect_not_matching(self):
        assert detect_blockdiagram(["not a fence"], 0) is None

    def test_detect_unterminated_fence_returns_none(self):
        paragraphs = ["```blockdiagram", "R --> C"]
        assert detect_blockdiagram(paragraphs, 0) is None

    def test_detect_out_of_range_start(self):
        assert detect_blockdiagram(["a"], 5) is None


class TestParseDendrogram:
    _LINES = [
        "order: B C A D E",
        "B + C : 1",
        "A + D : 2",
        "BC + AD : 3",
        "ABCD + E : 4",
    ]

    def test_parse_order(self):
        spec = parse_dendrogram(self._LINES)
        assert spec.order == ["B", "C", "A", "D", "E"]

    def test_parse_merges_left_right_height(self):
        spec = parse_dendrogram(self._LINES)
        assert len(spec.merges) == 4
        assert spec.merges[0].left == frozenset("B")
        assert spec.merges[0].right == frozenset("C")
        assert spec.merges[0].height == 1
        assert spec.merges[1].left == frozenset("A")
        assert spec.merges[1].right == frozenset("D")
        assert spec.merges[1].height == 2

    def test_parse_merges_normalize_cluster_as_set(self):
        """クラスタは構成要素の集合として正規化される（順不同）。"""
        spec = parse_dendrogram(self._LINES)
        merged_clusters = [m.left | m.right for m in spec.merges]
        assert merged_clusters[0] == frozenset("BC")
        assert merged_clusters[1] == frozenset("AD")
        assert merged_clusters[2] == frozenset("ABCD")
        assert merged_clusters[3] == frozenset("ABCDE")
        assert spec.merges[3].height == 4

    def test_parse_empty_input_returns_empty_spec(self):
        spec = parse_dendrogram([])
        assert spec.order == []
        assert spec.merges == []

    def test_parse_blank_lines_only(self):
        spec = parse_dendrogram(["", "   ", ""])
        assert spec.order == []
        assert spec.merges == []

    def test_parse_malformed_lines_are_skipped_not_raised(self):
        spec = parse_dendrogram(["order: A B", "garbage line", "A + B : notanumber"])
        assert spec.order == ["A", "B"]
        assert spec.merges == []


class TestDetectDendrogram:
    def test_detect_fence(self):
        paragraphs = [
            "text before",
            "```dendrogram",
            "order: B C A D E",
            "B + C : 1",
            "A + D : 2",
            "BC + AD : 3",
            "ABCD + E : 4",
            "```",
            "text after",
        ]
        result = detect_dendrogram(paragraphs, 1)
        assert result is not None
        spec, consumed = result
        assert consumed == 7
        assert spec.order == ["B", "C", "A", "D", "E"]
        assert len(spec.merges) == 4

    def test_detect_not_matching(self):
        assert detect_dendrogram(["not a fence"], 0) is None

    def test_detect_unterminated_fence_returns_none(self):
        paragraphs = ["```dendrogram", "order: A B"]
        assert detect_dendrogram(paragraphs, 0) is None

    def test_detect_out_of_range_start(self):
        assert detect_dendrogram(["a"], 5) is None
