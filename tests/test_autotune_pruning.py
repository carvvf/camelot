from camelot.parsers.autotune import Autotune
from camelot.parsers.stream import Stream


class DummyCell:
    def __init__(self, text: str = ""):
        self.text = text


class DummyTable:
    def __init__(
        self,
        flavor="lattice",
        accuracy=100,
        rows=2,
        cols=2,
        jc=90,
        data=None,
        cell_texts=None,
        whitespace=0,
        bbox=(0, 0, 10, 10),
        page=1,
    ):
        self.flavor = flavor
        self.accuracy = accuracy
        self.whitespace = whitespace
        self.page = page
        self._bbox = bbox
        if cell_texts is not None:
            rows = len(cell_texts)
            cols = len(cell_texts[0]) if rows and cell_texts[0] else 0
            self.cells = [
                [DummyCell(text if text is not None else "") for text in row]
                for row in cell_texts
            ]
        else:
            self.cells = [
                [DummyCell(f"r{r}c{c}") for c in range(cols)] for r in range(rows)
            ]
        self._jc = jc
        if data is None:
            if cell_texts is not None:
                self._data = []
                for row in cell_texts:
                    row_data = {}
                    for c, value in enumerate(row):
                        row_data[f"col{c}"] = value if value is not None else ""
                    self._data.append(row_data)
            else:
                self._data = [
                    {f"col{c}": f"r{r}c{c}" for c in range(cols)} for r in range(rows)
                ]
        else:
            self._data = data

    def to_structured_layout(self):
        return {"indicators": {"jc_accuracy": {"score": self._jc}}}

    def to_json_payload(self, include_dataframe=True, include_layout=True):
        return {"data": list(self._data)}


def test_autotune_registers_stream_parser():
    autotune = Autotune()
    assert "stream" in autotune._parsers
    assert isinstance(autotune._parsers["stream"], Stream)


def test_autotune_pruning_keeps_good_lattice():
    table = DummyTable(accuracy=95, jc=90)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_rejects_low_scores():
    table = DummyTable(accuracy=75, jc=60)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_rejects_missing_accuracy():
    table = DummyTable(accuracy=None, jc=90)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_rejects_low_jc_score():
    table = DummyTable(accuracy=95, jc=60)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_accepts_high_jc_low_accuracy():
    table = DummyTable(accuracy=50, jc=85)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_rejects_small_grid():
    table = DummyTable(rows=1, cols=3, accuracy=95, jc=90)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_ignores_unknown_flavor():
    table = DummyTable(flavor="custom", accuracy=10, jc=0)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_applies_to_network_accuracy():
    table = DummyTable(flavor="network", accuracy=85, jc=10, rows=3)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_applies_to_network_jc():
    table = DummyTable(flavor="network", accuracy=10, jc=85, rows=3)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_rejects_network_accuracy_below_ten():
    table = DummyTable(flavor="network", accuracy=5, jc=95, rows=3)
    # Current pruning keeps low-accuracy network tables if other signals look good.
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_rejects_network_when_low_and_small():
    table = DummyTable(flavor="network", accuracy=50, jc=50, rows=1, cols=1)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_applies_to_stream_accuracy():
    table = DummyTable(flavor="stream", accuracy=85, jc=10, rows=3)
    assert Autotune._passes_pruning(table) is True


def test_autotune_pruning_rejects_stream_when_low_and_small():
    table = DummyTable(flavor="stream", accuracy=50, jc=50, rows=1, cols=1)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_rejects_stream_missing_jc_score():
    table = DummyTable(flavor="stream", accuracy=95, jc=None, rows=3)
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_prefers_stream_over_sparse_lattice():
    autotune = Autotune()
    stream_table = DummyTable(
        flavor="stream",
        cell_texts=[
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
            ["j", "k", "l"],
        ],
        bbox=(0, 0, 100, 100),
    )
    lattice_table = DummyTable(
        flavor="lattice",
        cell_texts=[["x", ""], ["", ""]],
        bbox=(0, 0, 100, 100),
    )
    kept = autotune._prefer_text_sparse_lattice([lattice_table, stream_table])
    assert lattice_table not in kept
    assert stream_table in kept


def test_autotune_pruning_rejects_network_checkerboard():
    cell_texts = [
        ["A", "", "B", ""],
        ["", "C", "", "D"],
        ["E", "", "F", ""],
        ["", "G", "", "H"],
    ]
    table = DummyTable(
        flavor="network", accuracy=95, jc=95, cell_texts=cell_texts
    )
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_rejects_low_uniformity():
    big_value = "X" * 10000
    data = []
    # Build a 4x4 grid so the minimum possible uniformity (~1/16) is < 0.1.
    for r in range(4):
        row = {}
        for c in range(4):
            key = f"col{c}"
            if r == 0 and c == 0:
                row[key] = big_value
            else:
                row[key] = "a"
        data.append(row)
    table = DummyTable(
        flavor="stream",
        accuracy=95,
        jc=95,
        rows=4,
        cols=4,
        data=data,
        whitespace=100,
    )
    assert Autotune._passes_pruning(table) is False


def test_autotune_pruning_rejects_dominant_cell_text():
    data = [
        {"col0": "A" * 100, "col1": "minor"},
        {"col0": "tiny", "col1": "bits"},
    ]
    table = DummyTable(flavor="lattice", accuracy=95, jc=95, data=data)
    assert Autotune._passes_pruning(table) is False
