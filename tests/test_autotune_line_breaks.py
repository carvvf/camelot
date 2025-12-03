from camelot.parsers.autotune import Autotune


class DummyCell:
    def __init__(self, text: str):
        self.text = text


class DummyTable:
    def __init__(self, cell_texts):
        self.cells = [
            [DummyCell(text) for text in row]
            for row in cell_texts
        ]


def test_line_breaks_require_spacing_between_markers():
    table = DummyTable(
        [
            ["1\n2\n3", "alpha beta gamma delta\nomega"],
        ]
    )

    assert Autotune._max_line_breaks_per_column(table) == 1


def test_line_breaks_accumulate_only_when_spaced():
    table = DummyTable(
        [
            ["a b c d e f\nx\ny z w v", ""],
            ["q r s t\nu v w x", ""],
        ]
    )

    assert Autotune._max_line_breaks_per_column(table) == 2
