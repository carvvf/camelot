import json
from pathlib import Path

from camelot.core import generate_html_report


def test_generate_html_report_lists_table_flavors(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"")

    tables_payload = {
        "tables": [
            {
                "page": 1,
                "order": 1,
                "flavor": "lattice",
                "data": [{"foo": "bar"}],
                "grid": {"rows": 1, "cols": 1},
                "page_rotation_pdfinfo": 0,
            },
            {
                "page": 2,
                "order": 1,
                "flavor": "stream",
                "data": [{"baz": "qux"}],
                "grid": {"rows": 1, "cols": 1},
                "page_rotation_pdfinfo": 0,
            },
        ]
    }

    tables_json_path = tmp_path / "tables.json"
    tables_json_path.write_text(json.dumps(tables_payload))

    html_output_path = tmp_path / "report.html"

    generate_html_report(
        tables_json=tables_json_path,
        html_output=html_output_path,
        label="flavor-check",
        flavor="autotune",
        pdf_path=pdf_path,
        command_preview="camelot command preview",
        specification={"flavor": "autotune"},
    )

    html_content = html_output_path.read_text()
    assert "flavor: lattice" in html_content
    assert "flavor: stream" in html_content


def test_generate_html_report_includes_uniformity(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"")

    tables_payload = {
        "tables": [
            {
                "page": 1,
                "order": 1,
                "flavor": "lattice",
                "data": [
                    {"c1": "aa", "c2": "bb"},
                    {"c1": "", "c2": ""},
                ],
                "grid": {"rows": 2, "cols": 2},
            }
        ]
    }

    tables_json_path = tmp_path / "tables.json"
    tables_json_path.write_text(json.dumps(tables_payload))

    html_output_path = tmp_path / "report.html"

    generate_html_report(
        tables_json=tables_json_path,
        html_output=html_output_path,
        label="uniformity-check",
        flavor="autotune",
        pdf_path=pdf_path,
        command_preview="camelot command preview",
        specification={"flavor": "autotune"},
    )

    html_content = html_output_path.read_text()
    assert "uniformity: U 0.5" in html_content
    assert "G 0.5" in html_content
