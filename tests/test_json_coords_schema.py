"""Ensure json-coords exports remain compliant with the versioned schema."""

from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema  # type: ignore

import camelot


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "json-coords" / "v1.json"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_json_coords_payload_matches_schema(testdir, tmp_path):
    filename = os.path.join(testdir, "foo.pdf")
    tables = camelot.read_pdf(filename, line_scale=40)
    output = tmp_path / "coords.json"
    tables.export(str(output), f="json-coords")

    payload = json.loads(output.read_text(encoding="utf-8"))
    schema = _load_schema()

    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(payload)

    first_table = payload["tables"][0]
    page_boxes = first_table.get("page_boxes") or {}
    assert "mediabox" in page_boxes, "expected mediabox metadata in json-coords payload"
    mediabox = page_boxes["mediabox"]
    assert mediabox.get("origin") is not None
    assert mediabox.get("size") is not None
    layout = first_table.get("layout", {})
    first_table["isolated_cells"] = [
        {"bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 5.0}, "texts": ["iso"]}
    ]
    layout["isolated_cells"] = [
        {"bbox": {"abs": [0.0, 0.0, 10.0, 5.0]}, "texts": ["iso"]}
    ]
    first_table["layout"] = layout

    validator.validate(payload)
