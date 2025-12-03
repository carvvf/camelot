from camelot.core import Table


class DummyTextLine:
    def __init__(self, x0, y0, x1, y1, text):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self._text = text

    def get_text(self):
        return self._text


def _build_stream_table():
    cols = [(0.0, 50.0), (50.0, 100.0)]
    rows = [(100.0, 50.0), (50.0, 0.0)]
    table = Table(cols, rows)
    table.flavor = "stream"
    table.page = 1
    table.order = 1
    table.pdf_size = (100.0, 100.0)
    table._bbox = (0.0, 0.0, 100.0, 100.0)
    table.set_border()

    values = [["A1", "B1"], ["A2", "B2"]]
    for r_idx, row in enumerate(values):
        for c_idx, text in enumerate(row):
            cell = table.cells[r_idx][c_idx]
            cell._text = ""
            cell.text = text

    table.textlines = [
        DummyTextLine(5.0, 55.0, 45.0, 95.0, "A1"),
        DummyTextLine(55.0, 55.0, 95.0, 95.0, "B1"),
        DummyTextLine(5.0, 5.0, 45.0, 45.0, "A2"),
        DummyTextLine(55.0, 5.0, 95.0, 45.0, "B2"),
    ]
    return table


def test_stream_json_coords_includes_indicators():
    table = _build_stream_table()
    payload = table.to_json_payload(include_layout=True)
    layout = payload.get("layout")
    assert layout, "expected structured layout in stream json-coords payload"

    logical_cells = layout.get("logical_cells")
    assert logical_cells and len(logical_cells) == 4

    indicators = layout.get("indicators", {})
    assert "jc_accuracy" in indicators
    assert "rectangularity" in indicators
    jc_accuracy = indicators["jc_accuracy"]
    assert jc_accuracy.get("score") is not None
    rectangularity = indicators["rectangularity"]
    assert rectangularity.get("cells_score") is not None
