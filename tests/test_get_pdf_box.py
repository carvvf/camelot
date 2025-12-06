import json

from camelot.core import get_pdf_box


def _make_test_pdf(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    rotated = writer.add_blank_page(width=200, height=100)
    rotated.rotate(90)

    pdf_path = tmp_path / "sample.pdf"
    with pdf_path.open("wb") as fp:
        writer.write(fp)
    return pdf_path


def _write_tables_payload(tmp_path, pdf_path):
    payload = {
        "tables": [
            {
                "page": 1,
                "order": 1,
                "bbox": {"x1": 10, "y1": 20, "x2": 60, "y2": 70},
                "rotation": 0,
                "source": {"file": str(pdf_path)},
            },
            {
                "page": 2,
                "order": 1,
                "bbox": {"norm": [0.1, 0.2, 0.4, 0.6]},
                "rotation": 0,
                "source": {"file": str(pdf_path)},
            },
        ]
    }

    tables_json_path = tmp_path / "tables.json"
    tables_json_path.write_text(json.dumps(payload))
    return tables_json_path


def test_get_pdf_box_matches_overlay_logic(tmp_path):
    pdf_path = _make_test_pdf(tmp_path)
    tables_json_path = _write_tables_payload(tmp_path, pdf_path)

    first_box = get_pdf_box(tables_json_path, "p1-o1", input_pdf=pdf_path)
    assert first_box == (10, 20, 60, 70)

    rotated_box = get_pdf_box(tables_json_path, (2, 1), input_pdf=pdf_path)
    assert rotated_box == (80, 10, 160, 40)


def test_get_pdf_box_infers_pdf_path(tmp_path):
    pdf_path = _make_test_pdf(tmp_path)
    tables_json_path = _write_tables_payload(tmp_path, pdf_path)

    inferred_box = get_pdf_box(tables_json_path, "table-p1-o1")
    assert inferred_box == (10, 20, 60, 70)
