import json
import os
import statistics

import pandas as pd
from pandas.testing import assert_frame_equal

import camelot
from camelot.parsers.lattice import Lattice

from .data import *


def _assert_contains_table(tables, expected_df):
    """Ensure the extraction produced the expected frame regardless of index."""
    for table in tables:
        try:
            assert_frame_equal(expected_df, table.df)
            return table
        except AssertionError:
            continue
    raise AssertionError("Expected table not found in extracted results.")


def _write_char_split_pdf(path):
    content_lines = [
        "0.5 w",
        "40 250 m 360 250 l S",
        "40 200 m 360 200 l S",
        "40 150 m 360 150 l S",
        "40 100 m 360 100 l S",
        "40 100 m 40 250 l S",
        "170 100 m 170 250 l S",
        "360 100 m 360 250 l S",
        "BT",
        "/F1 12 Tf",
        "55 215 Td",
        "(Ore 07.00) Tj",
        "0 -50 Td",
        "(Ore 07.00) Tj",
        "0 -50 Td",
        "(Ore 08.00/08.45) Tj",
        "140 0 Td",
        "(Ricognizione pista per gara slalom gigante) Tj",
        "ET",
        "BT",
        "/F1 12 Tf",
        "185 215 Td",
        "(Apertura ufficio gare e ritiro pettorali) Tj",
        "0 -50 Td",
        "(Apertura impianti per ricognizione sulle piste) Tj",
        "ET",
    ]
    content = "\n".join(content_lines) + "\n"
    content_bytes = content.encode("latin-1")
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content_bytes)} >>\nstream\n{content}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    header = b"%PDF-1.4\n"
    offsets = []
    parts = []
    current = len(header)

    for index, obj in enumerate(objects, start=1):
        payload = f"{index} 0 obj\n{obj}\nendobj\n".encode("latin-1")
        offsets.append(current)
        parts.append(payload)
        current += len(payload)

    body = b"".join(parts)
    xref_offset = len(header) + len(body)
    xref_entries = ["0000000000 65535 f \n"]
    xref_entries.extend(f"{offset:010d} 00000 n \n" for offset in offsets)
    xref = (
        f"xref\n0 {len(objects) + 1}\n" + "".join(xref_entries)
    ).encode("latin-1")
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(body)
        handle.write(xref)
        handle.write(trailer)


def test_lattice(testdir):
    df = pd.DataFrame(data_lattice)

    filename = os.path.join(
        testdir, "tabula/icdar2013-dataset/competition-dataset-us/us-030.pdf"
    )
    tables = camelot.read_pdf(filename, pages="2")
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_rotated(testdir):
    df = pd.DataFrame(data_lattice_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_1.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)


def test_lattice_two_tables(testdir):
    df1 = pd.DataFrame(data_lattice_two_tables_1)
    df2 = pd.DataFrame(data_lattice_two_tables_2)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename)
    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_lattice_table_regions(testdir):
    df = pd.DataFrame(data_lattice_table_regions)

    filename = os.path.join(testdir, "table_region.pdf")
    tables = camelot.read_pdf(filename, table_regions=["170,370,560,270"])
    assert_frame_equal(df, tables[0].df)


def test_lattice_table_areas(testdir):
    df = pd.DataFrame(data_lattice_table_areas)

    filename = os.path.join(testdir, "twotables_2.pdf")
    tables = camelot.read_pdf(filename, table_areas=["80,693,535,448"])
    assert_frame_equal(df, tables[0].df)


def test_lattice_process_background(testdir):
    df = pd.DataFrame(data_lattice_process_background)

    filename = os.path.join(testdir, "background_lines_1.pdf")
    tables = camelot.read_pdf(filename, process_background=True)
    assert len(tables) >= 1
    _assert_contains_table(tables, df)


def test_lattice_remove_background_artifacts(monkeypatch, testdir):
    calls = {"image_regions": 0, "cleanup": 0, "regions": None}

    def fake_image_regions(self, image_scalers, width, height, **kwargs):
        calls["image_regions"] += 1
        return [
            {
                "index": 0,
                "name": "Im1",
                "rect": (0, 0, width, height),
                "image_bbox": (0, 0, width, height),
                "pdf_bbox": (0.0, 0.0, float(width), float(height)),
            }
        ]

    def fake_cleanup(threshold, line_mask, regions):
        calls["cleanup"] += 1
        calls["regions"] = regions
        return threshold

    monkeypatch.setattr(Lattice, "_image_regions", fake_image_regions)
    monkeypatch.setattr(
        Lattice, "_apply_background_cleanup", staticmethod(fake_cleanup)
    )

    filename = os.path.join(testdir, "background_lines_1.pdf")

    camelot.read_pdf(filename)
    assert calls["image_regions"] == 1
    assert calls["cleanup"] == 1
    assert calls["regions"]
    first_region = calls["regions"][0]
    assert "rect" in first_region
    assert "image_index" in first_region

    calls["image_regions"] = 0
    calls["cleanup"] = 0
    calls["regions"] = None

    camelot.read_pdf(filename, remove_background_artifacts=False)
    assert calls["image_regions"] == 0
    assert calls["cleanup"] == 0
    assert calls["regions"] is None


def test_lattice_remove_text_option(monkeypatch, testdir):
    calls = {"mask": 0, "regions_called": 0}

    def fake_text_regions(self, candidates, scalers, width, height, padding=1, **kwargs):
        calls["regions_called"] += 1
        assert candidates
        return [(0, 0, 10, 10)]

    def fake_apply(threshold, regions):
        calls["mask"] += 1
        assert regions == [(0, 0, 10, 10)]
        return threshold

    monkeypatch.setattr(Lattice, "_text_mask_regions", fake_text_regions)
    monkeypatch.setattr(Lattice, "_apply_text_mask", staticmethod(fake_apply))

    filename = os.path.join(testdir, "background_lines_1.pdf")

    camelot.read_pdf(filename, remove_text=True)
    assert calls["regions_called"] == 1
    assert calls["mask"] == 1

    calls["regions_called"] = 0
    calls["mask"] = 0

    camelot.read_pdf(filename, remove_text=False)
    assert calls["regions_called"] == 0
    assert calls["mask"] == 0


def test_lattice_copy_text(testdir):
    filename = os.path.join(testdir, "row_span_1.pdf")
    tables = camelot.read_pdf(filename, line_scale=60, copy_text="v")
    table = tables[0]
    assert table.df.shape == (40, 4)
    first_row = table.df.iloc[1].tolist()
    assert first_row == [
        "Plan Type \nGMC",
        "County \nSacramento",
        "Plan Name",
        "Totals",
    ]
    assert any(
        isinstance(value, str) and "All Models Total Enrollments" in value
        for value in table.df[0].tolist()
    )


def test_lattice_spanning_cells_column_span(testdir):
    filename = os.path.join(testdir, "column_span_1.pdf")
    table = camelot.read_pdf(filename, line_scale=40)[0]
    header_cells = {
        cell["text"]: cell
        for cell in table.spanning_cells
        if cell["row_start"] == 0
    }
    assert "Accidental Deaths" in header_cells
    assert header_cells["Accidental Deaths"]["col_span"] == 2
    assert "Suicides" in header_cells
    assert header_cells["Suicides"]["col_span"] == 2
    assert "Sl. \nNo." in header_cells
    assert header_cells["Sl. \nNo."]["row_span"] == 2


def test_lattice_spanning_cells_row_span(testdir):
    filename = os.path.join(testdir, "row_span_1.pdf")
    table = camelot.read_pdf(filename, line_scale=60)[0]
    assert table.spanning_cells, "expected spanning cells to be present"
    header_cell = next(
        cell for cell in table.spanning_cells if "Plan Type" in (cell.get("text") or "")
    )
    assert header_cell.get("row_span") == 10


def test_lattice_html_preserves_spans(testdir):
    filename = os.path.join(testdir, "column_span_1.pdf")
    table = camelot.read_pdf(filename, line_scale=40)[0]
    html = table.to_html_string()
    assert 'colspan="2">Accidental Deaths' in html
    assert 'rowspan="2">Sl. <br/>No.' in html


def test_lattice_json_coords_spanning_cells(testdir, tmp_path):
    filename = os.path.join(testdir, "row_span_1.pdf")
    tables = camelot.read_pdf(filename, line_scale=60)
    output = tmp_path / "layout.json"
    tables.export(str(output), f="json-coords")
    payload = json.loads(output.read_text())
    first = payload["tables"][0]
    assert "spanning_cells" not in first
    layout = first.get("layout")
    assert layout
    spans = layout.get("logical_cells")
    assert "confidence" not in layout
    header = next(cell for cell in spans if "Plan Type" in (cell.get("text") or ""))
    assert header.get("row_span") == 10
    assert header.get("source_cells")
    assert "grid" in first
    assert first["grid"]["rows"] >= 1
    indicators = first.get("layout", {}).get("indicators", {})
    assert "jc_accuracy" in indicators
    jc_accuracy = indicators["jc_accuracy"]
    assert jc_accuracy["score"] <= 100
    assert jc_accuracy["score"] >= 0
    assert "rectangularity" in indicators
    rectangularity = indicators["rectangularity"]
    assert 0 <= rectangularity["area_text_score"] <= 100
    assert 0 <= rectangularity["edge_text_score"] <= 100
    assert 0 <= rectangularity["area_full_score"] <= 100
    assert 0 <= rectangularity["edge_full_score"] <= 100
    assert 0 <= rectangularity["cells_score"] <= 100
    assert 0 <= rectangularity["fillers_score"] <= 100
    edge_gaps = rectangularity.get("edge_text_gaps")
    if edge_gaps:
        for side in ("top", "bottom", "left", "right"):
            assert edge_gaps.get(side, 0) >= 0
    edge_full_gaps = rectangularity.get("edge_full_gaps")
    if edge_full_gaps:
        for side in ("top", "bottom", "left", "right"):
            assert edge_full_gaps.get(side, 0) >= 0
    cell_count = layout.get("cell_count")
    if cell_count:
        def _collapse_positions(values, axis_span):
            coords = []
            for value in values:
                try:
                    coords.append(float(value))
                except (TypeError, ValueError):
                    continue
            if len(coords) <= 1:
                return coords
            coords.sort()
            deltas = [
                coords[idx + 1] - coords[idx]
                for idx in range(len(coords) - 1)
                if coords[idx + 1] - coords[idx] > 0
            ]
            typical = statistics.median(deltas) if deltas else None
            thresholds = []
            if axis_span is not None and axis_span > 0:
                thresholds.append(abs(axis_span) * 0.005)
            if typical is not None:
                thresholds.append(typical * 0.3)
            threshold = max(0.5, min(thresholds)) if thresholds else 0.5

            collapsed = [coords[0]]
            for coord in coords[1:]:
                if coord - collapsed[-1] <= threshold:
                    collapsed[-1] = (collapsed[-1] + coord) / 2.0
                else:
                    collapsed.append(coord)
            return collapsed

        table_bbox = layout.get("bbox", {}).get("abs")
        table_width = table_height = None
        if isinstance(table_bbox, list) and len(table_bbox) >= 4:
            table_width = table_bbox[2] - table_bbox[0]
            table_height = table_bbox[3] - table_bbox[1]

        col_positions = layout.get("column_positions") or []
        row_positions = layout.get("row_positions") or []
        collapsed_cols = _collapse_positions(col_positions, table_width)
        collapsed_rows = _collapse_positions(row_positions, table_height)
        effective_grid_total = (
            (len(collapsed_cols) - 1) * (len(collapsed_rows) - 1)
            if len(collapsed_cols) > 1 and len(collapsed_rows) > 1
            else first["grid"]["rows"] * first["grid"]["cols"]
        )
        base_grid_total = first["grid"]["rows"] * first["grid"]["cols"]
        assert base_grid_total == effective_grid_total
        assert cell_count["grid"] == effective_grid_total
        if spans:
            has_merge = any(
                span.get("row_span", 1) > 1
                or span.get("col_span", 1) > 1
                or len(span.get("source_cells", [])) > 1
                for span in spans
            )
            expected_logical = len(spans) if has_merge else effective_grid_total
            assert cell_count.get("logical") == expected_logical


def test_lattice_json_coords_parsing_report_no_duplicates(testdir, tmp_path):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, line_scale=40)
    output = tmp_path / "coords.json"
    tables.export(str(output), f="json-coords")
    payload = json.loads(output.read_text())
    first = payload["tables"][0]
    stats = first.get("parsing_report")
    assert stats
    assert "page" not in stats
    assert "order" not in stats
    assert first["page"] == tables[0].page
    assert first["order"] == tables[0].order


def test_lattice_shift_text(testdir):
    df_lt = pd.DataFrame(data_lattice_shift_text_left_top)
    df_disable = pd.DataFrame(data_lattice_shift_text_disable)
    df_rb = pd.DataFrame(data_lattice_shift_text_right_bottom)

    filename = os.path.join(testdir, "column_span_2.pdf")
    tables = camelot.read_pdf(filename, line_scale=40)
    assert df_lt.equals(tables[0].df)

    tables = camelot.read_pdf(filename, line_scale=40, shift_text=[""])
    assert df_disable.equals(tables[0].df)

    tables = camelot.read_pdf(filename, line_scale=40, shift_text=["r", "b"])
    assert df_rb.equals(tables[0].df)


def test_lattice_arabic(testdir):
    df = pd.DataFrame(data_arabic)

    filename = os.path.join(testdir, "tabula/arabic.pdf")
    tables = camelot.read_pdf(filename)
    assert_frame_equal(df, tables[0].df)


def test_lattice_split_text(testdir):
    df = pd.DataFrame(data_lattice_split_text)

    filename = os.path.join(testdir, "split_text_lattice.pdf")
    tables = camelot.read_pdf(filename, line_scale=60, split_text=True)

    assert_frame_equal(df, tables[0].df)


def test_lattice_character_split(tmp_path):
    pdf_path = tmp_path / "char_split.pdf"
    _write_char_split_pdf(pdf_path)

    tables = camelot.read_pdf(str(pdf_path), flavor="lattice")
    assert len(tables) == 1

    expected = pd.DataFrame(
        [
            ["Ore 07.00", "Apertura ufficio gare e ritiro pettorali"],
            ["Ore 07.00", "Apertura impianti per ricognizione sulle piste"],
            ["Ore 08.00/08.45", "Ricognizione pista per gara slalom gigante"],
        ]
    )

    assert_frame_equal(expected, tables[0].df)
