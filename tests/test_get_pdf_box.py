import json

from camelot.core import get_pdf_box


def _artifact_with_abs_bbox():
    return {
        "tables": [
            {
                "page": 1,
                "order": 1,
                "bbox": {"x1": 10, "y1": 20, "x2": 60, "y2": 70},
                "rotation": 0,
                "layout": {"page_size": {"width": 200, "height": 100}},
                "page_rotation_pdfinfo": 0,
                "page_boxes": {
                    "mediabox": {
                        "origin": [0.0, 0.0],
                        "size": {"width": 200.0, "height": 100.0},
                    }
                },
            }
        ]
    }


def _artifact_with_norm_bbox_and_rotation():
    return {
        "tables": [
            {
                "page": 2,
                "order": 1,
                "bbox": {"norm": [0.1, 0.2, 0.4, 0.6]},
                "rotation": 0,
                "layout": {"page_size": {"width": 200, "height": 100}, "rotation": 0},
                "page_rotation_pdfinfo": 90,
            }
        ]
    }


def test_get_pdf_box_returns_abs_bbox_for_single_table(tmp_path):
    artifact = _artifact_with_abs_bbox()
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact))

    # Path input
    first_box = get_pdf_box(artifact_path)
    assert first_box == (10, 20, 60, 70)

    # Mapping input
    first_box_mapping = get_pdf_box(artifact)
    assert first_box_mapping == (10, 20, 60, 70)


def test_get_pdf_box_handles_normalized_bbox_and_page_rotation(tmp_path):
    artifact = _artifact_with_norm_bbox_and_rotation()
    artifact_path = tmp_path / "artifact_norm.json"
    artifact_path.write_text(json.dumps(artifact))

    rotated_box = get_pdf_box(artifact_path)
    assert rotated_box == (40, 20, 80, 80)


def test_get_pdf_box_requires_single_table(tmp_path):
    payload = {
        "tables": [
            {"page": 1, "order": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            {"page": 2, "order": 1, "bbox": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}},
        ]
    }
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(payload))

    try:
        get_pdf_box(path)
        assert False, "expected ValueError for multi-table artifact"
    except ValueError:
        pass
