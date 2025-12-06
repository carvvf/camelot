"""Contains the core functions to parse tables from PDFs."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import zipfile
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import logging
from collections import OrderedDict
from collections import defaultdict
from html import escape

import cv2
import pandas as pd


if sys.version_info >= (3, 11):
    from typing import TypedDict  # pylint: disable=no-name-in-module
    from typing import Unpack
else:
    from typing_extensions import TypedDict, Unpack

from .backends import ImageConversionBackend
from .utils import build_file_path_in_temp_dir
from .utils import get_index_closest_point
from .utils import get_textline_coords
from .utils import textlines_overlapping_bbox


logger = logging.getLogger("camelot")


# minimum number of vertical textline intersections for a textedge
# to be considered valid
TEXTEDGE_REQUIRED_ELEMENTS = 4
# padding added to table area on the left, right and bottom
TABLE_AREA_PADDING = 10
_SPAN_SKIP = object()

HORIZONTAL_ALIGNMENTS = ["left", "right", "middle"]
VERTICAL_ALIGNMENTS = ["top", "bottom", "center"]
ALL_ALIGNMENTS = HORIZONTAL_ALIGNMENTS + VERTICAL_ALIGNMENTS

JSON_COORDS_SCHEMA_NAME: Final[str] = "camelot.json-coords"
JSON_COORDS_SCHEMA_VERSION: Final[str] = "1.0.0"
JSON_COORDS_SCHEMA_URL: Final[
    str
] = "https://raw.githubusercontent.com/carvvf/camelot/custom/docs/schemas/json-coords/v1.json"
JSON_COORDS_SCHEMA_DESCRIPTOR: Final[Dict[str, str]] = {
    "name": JSON_COORDS_SCHEMA_NAME,
    "version": JSON_COORDS_SCHEMA_VERSION,
    "url": JSON_COORDS_SCHEMA_URL,
}

class TextAlignment:
    """Represents a list of textlines sharing an alignment on a coordinate.

    The alignment can be left/right/middle or top/bottom/center.
    (PDF coordinate space)

    Parameters
    ----------
    coord : float
        coordinate of the initial text edge. Depending on the alignment
        it could be a vertical or horizontal coordinate.
    textline : obj
        the original textline to start the alignment
    align : str
        Name of the alignment (e.g. "left", "top", etc)

    Attributes
    ----------
    coord : float
        The coordinate aligned averaged out across textlines.  It can be along
        the x or y axis.
    textlines : array
        Array of textlines that demonstrate this alignment.
    align : str
        Name of the alignment (e.g. "left", "top", etc)
    """

    def __init__(self, coord, textline, align):
        self.coord = coord
        self.textlines = [textline]
        self.align = align

    def __repr__(self):  # noqa D105
        text_inside = " | ".join(
            map(lambda x: x.get_text(), self.textlines[:2])
        ).replace("\n", "")
        return (
            f"<TextEdge coord={self.coord} tl={len(self.textlines)} "
            f"textlines text='{text_inside}...'>"
        )

    def register_aligned_textline(self, textline, coord):
        """Update new textline to this alignment, adapting its average."""
        # Increase the intersections for this segment, expand it up,
        # and adjust the x based on the new value
        self.coord = (self.coord * len(self.textlines) + coord) / float(
            len(self.textlines) + 1
        )
        self.textlines.append(textline)


class TextEdge(TextAlignment):
    """Defines a text edge coordinates relative to a left-bottom origin.

    (PDF coordinate space)
    An edge is an alignment bounded over a segment.

    Parameters
    ----------
    coord : float
        coordinate of the text edge.  Can be x or y.
    y0 : float
        y-coordinate of bottommost point.
    y1 : float
        y-coordinate of topmost point.
    align : string, optional (default: 'left')
        {'left', 'right', 'middle'}

    Attributes
    ----------
    is_valid: bool
        A text edge is valid if it intersects with at least
        TEXTEDGE_REQUIRED_ELEMENTS horizontal text rows.

    """

    def __init__(self, coord, textline, align):
        super().__init__(coord, textline, align)
        self.y0 = textline.y0
        self.y1 = textline.y1
        self.is_valid = False

    def __repr__(self):  # noqa D105
        x = round(self.coord, 2)
        y0 = round(self.y0, 2)
        y1 = round(self.y1, 2)
        return (
            f"<TextEdge x={x} y0={y0} y1={y1} align={self.align} valid={self.is_valid}>"
        )

    def update_coords(self, x, textline, edge_tol=50):
        """Update text edge coordinates.

        Update the text edge's x and bottom y coordinates and sets
        the is_valid attribute.
        """
        if math.isclose(self.y0, textline.y0, abs_tol=edge_tol):
            self.register_aligned_textline(textline, x)
            self.y0 = textline.y0
            # a textedge is valid only if it extends uninterrupted
            # over a required number of textlines
            if len(self.textlines) > TEXTEDGE_REQUIRED_ELEMENTS:
                self.is_valid = True


class TextAlignments:
    """Defines a dict of text edges across reference alignments."""

    def __init__(self, alignment_names):
        # For each possible alignment, list of tuples coordinate/textlines
        self._text_alignments = {}
        for alignment_name in alignment_names:
            self._text_alignments[alignment_name] = []

    @staticmethod
    def _create_new_text_alignment(coord, textline, align):
        return TextAlignment(coord, textline, align)

    def _update_alignment(self, alignment, coord, textline):
        return NotImplemented

    def _register_textline(self, textline):
        """Update an existing text edge in the current dict."""
        coords = get_textline_coords(textline)
        for alignment_id, alignment_array in self._text_alignments.items():
            coord = coords[alignment_id]

            # Find the index of the closest existing element (or 0 if none)
            idx_closest = get_index_closest_point(
                coord, alignment_array, fn=lambda x: x.coord
            )

            # Check if the edges before/after are close enough
            # that it can be considered aligned
            idx_insert = None
            if idx_closest is None:
                idx_insert = 0
            else:
                coord_closest = alignment_array[idx_closest].coord
                # Note: np.isclose is slow!
                if coord - 0.5 < coord_closest < coord + 0.5:
                    self._update_alignment(
                        alignment_array[idx_closest], coord, textline
                    )
                elif coord_closest < coord:
                    idx_insert = idx_closest + 1
                else:
                    idx_insert = idx_closest
            if idx_insert is not None:
                new_alignment = self._create_new_text_alignment(
                    coord, textline, alignment_id
                )
                alignment_array.insert(idx_insert, new_alignment)


class TextEdges(TextAlignments):
    """Defines a dict text edges on the PDF page.

    The dict contains the left, right and middle text edges found on
    the PDF page. The dict has three keys based on the alignments,
    and each key's value is a list of camelot.core.TextEdge objects.
    """

    def __init__(self, edge_tol=50):
        super().__init__(HORIZONTAL_ALIGNMENTS)
        self.edge_tol = edge_tol

    def _create_new_text_alignment(self, coord, textline, align):
        # In TextEdges, each alignment is a TextEdge
        return TextEdge(coord, textline, align)

    def add(self, coord, textline, align):
        """Add a new text edge to the current dict."""
        te = self._create_new_text_alignment(coord, textline, align)
        self._text_alignments[align].append(te)

    def _update_alignment(self, alignment, coord, textline):
        alignment.update_coords(coord, textline, self.edge_tol)

    def generate(self, textlines):
        """Generates the text edges dict based on horizontal text rows."""
        for tl in textlines:
            if len(tl.get_text().strip()) > 1:  # TODO: hacky
                self._register_textline(tl)

    def get_relevant(self):
        """Return the list of relevant text edges.

        (all share the same alignment)
        based on which list intersects horizontal text rows the most.
        """
        intersections_sum = {
            "left": sum(
                len(te.textlines) for te in self._text_alignments["left"] if te.is_valid
            ),
            "right": sum(
                len(te.textlines)
                for te in self._text_alignments["right"]
                if te.is_valid
            ),
            "middle": sum(
                len(te.textlines)
                for te in self._text_alignments["middle"]
                if te.is_valid
            ),
        }

        # TODO: naive
        # get vertical textedges that intersect maximum number of
        # times with horizontal textlines
        relevant_align = max(intersections_sum.items(), key=itemgetter(1))[0]
        return list(
            filter(lambda te: te.is_valid, self._text_alignments[relevant_align])
        )

    def get_table_areas(self, textlines, relevant_textedges):
        """
        Return a dict of interesting table areas on the PDF page.

        The table areas are calculated using relevant text edges.

        Parameters
        ----------
        textlines : list
            List of text line objects that are relevant for determining table areas.
        relevant_textedges : list
            List of relevant text edge objects used to identify table areas.

        Returns
        -------
        dict
            A dictionary with padded table areas as keys and None as values.
        """
        # Sort relevant text edges in reading order
        relevant_textedges.sort(key=lambda te: (-te.y0, te.coord))

        table_areas = self._initialize_table_areas(relevant_textedges)
        self._extend_table_areas_with_textlines(table_areas, textlines)

        # Add padding to table areas
        average_textline_height = self._calculate_average_textline_height(textlines)
        padded_table_areas = {
            self._pad(area, average_textline_height): None for area in table_areas
        }

        return padded_table_areas

    def _initialize_table_areas(self, relevant_textedges):
        """
        Initialize table areas based on relevant text edges.

        Parameters
        ----------
        relevant_textedges : list
            List of relevant text edge objects used to initialize table areas.

        Returns
        -------
        dict
            A dictionary of table areas initialized from relevant text edges.
        """
        table_areas = {}
        for te in relevant_textedges:
            if not table_areas:
                table_areas[(te.coord, te.y0, te.coord, te.y1)] = None
            else:
                self._update_table_areas(table_areas, te)

        return table_areas

    def _update_table_areas(self, table_areas, te):
        """
        Update table areas by checking for overlaps with new text edges.

        Parameters
        ----------
        table_areas : dict
            Current table areas to be updated.
        te : object
            The new text edge object to check for overlaps.

        Returns
        -------
        None
        """
        found = None
        for area in table_areas:
            # Check for overlap
            if te.y1 >= area[1] and te.y0 <= area[3]:
                found = area
                break

        if found is None:
            table_areas[(te.coord, te.y0, te.coord, te.y1)] = None
        else:
            table_areas.pop(found)
            updated_area = (
                found[0],
                min(te.y0, found[1]),
                max(found[2], te.coord),
                max(found[3], te.y1),
            )
            table_areas[updated_area] = None

    def _extend_table_areas_with_textlines(self, table_areas, textlines):
        """
        Extend table areas based on text lines that overlap vertically.

        Parameters
        ----------
        table_areas : dict
            Current table areas to be extended.
        textlines : list
            List of text line objects relevant for extending table areas.

        Returns
        -------
        None
        """
        for tl in textlines:
            found = None
            for area in table_areas:
                # Check for overlap
                if tl.y0 >= area[1] and tl.y1 <= area[3]:
                    found = area
                    break

            if found is not None:
                table_areas.pop(found)
                updated_area = (
                    min(tl.x0, found[0]),
                    min(tl.y0, found[1]),
                    max(found[2], tl.x1),
                    max(found[3], tl.y1),
                )
                table_areas[updated_area] = None

    def _calculate_average_textline_height(self, textlines):
        """
        Calculate the average height of text lines.

        Parameters
        ----------
        textlines : list
            List of text line objects.

        Returns
        -------
        float
            The average height of the text lines.
        """
        sum_textline_height = sum(tl.y1 - tl.y0 for tl in textlines)
        return sum_textline_height / float(len(textlines)) if textlines else 0

    def _pad(self, area, average_row_height):
        """
        Pad a given area by a constant value.

        Parameters
        ----------
        area : tuple
            The area to be padded defined as (x0, y0, x1, y1).
        average_row_height : float
            The average height of rows to use for padding.

        Returns
        -------
        tuple
            The padded area.
        """
        x0 = area[0] - TABLE_AREA_PADDING
        y0 = area[1] - TABLE_AREA_PADDING
        x1 = area[2] + TABLE_AREA_PADDING
        # Add a constant since table headers can be relatively up
        y1 = area[3] + average_row_height * 5
        return (x0, y0, x1, y1)


class Cell:
    """Defines a cell in a table.

    With coordinates relative to a
    left-bottom origin. (PDF coordinate space)

    Parameters
    ----------
    x1 : float
        x-coordinate of left-bottom point.
    y1 : float
        y-coordinate of left-bottom point.
    x2 : float
        x-coordinate of right-top point.
    y2 : float
        y-coordinate of right-top point.

    Attributes
    ----------
    lb : tuple
        Tuple representing left-bottom coordinates.
    lt : tuple
        Tuple representing left-top coordinates.
    rb : tuple
        Tuple representing right-bottom coordinates.
    rt : tuple
        Tuple representing right-top coordinates.
    left : bool
        Whether or not cell is bounded on the left.
    right : bool
        Whether or not cell is bounded on the right.
    top : bool
        Whether or not cell is bounded on the top.
    bottom : bool
        Whether or not cell is bounded on the bottom.
    text : string
        Text assigned to cell.

    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.lb = (x1, y1)
        self.lt = (x1, y2)
        self.rb = (x2, y1)
        self.rt = (x2, y2)
        self.left = False
        self.right = False
        self.top = False
        self.bottom = False
        self._text = ""

    def __repr__(self):  # noqa D105
        x1 = round(self.x1)
        y1 = round(self.y1)
        x2 = round(self.x2)
        y2 = round(self.y2)
        return f"<Cell x1={x1} y1={y1} x2={x2} y2={y2}>"

    @property
    def text(self):  # noqa D102
        return self._text

    @text.setter
    def text(self, t):  # noqa D105
        self._text = "".join([self._text, t])

    @property
    def hspan(self) -> bool:
        """Whether or not cell spans horizontally."""
        return not self.left or not self.right

    @property
    def vspan(self) -> bool:
        """Whether or not cell spans vertically."""
        return not self.top or not self.bottom

    @property
    def bound(self):
        """The number of sides on which the cell is bounded."""
        return self.top + self.bottom + self.left + self.right


class Table:
    """Defines a table with coordinates relative to a left-bottom origin.

    (PDF coordinate space)

    Parameters
    ----------
    cols : list
        List of tuples representing column x-coordinates in increasing
        order.
    rows : list
        List of tuples representing row y-coordinates in decreasing
        order.

    Attributes
    ----------
    df : :class:`pandas.DataFrame`
    shape : tuple
        Shape of the table.
    accuracy : float
        Accuracy with which text was assigned to the cell.
    whitespace : float
        Percentage of whitespace in the table.
    filename : str
        Path of the original PDF
    order : int
        Table number on PDF page.
    page : int
        PDF page number.

    """

    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.cells = [[Cell(c[0], r[1], c[1], r[0]) for c in cols] for r in rows]
        self.df = pd.DataFrame()
        self.shape = (0, 0)
        self.accuracy = 0
        self.whitespace = 0
        self.filename = None
        self.source = None
        self.pdfinfo_rotation = None
        self.order = None
        self.page = None
        self.flavor = None  # Flavor of the parser that generated the table
        self.pdf_size = None  # Dimensions of the original PDF page
        self.rotation = 0  # Page rotation applied during parsing (degrees)
        self._bbox = None  # Bounding box in original document
        self.parse = None  # Parse information
        self.parse_details = None  # Field holding debug data

        self._image = None
        self._image_path = None  # Temporary file to hold an image of the pdf
        self._spanning_cells = None  # Cached logical cells rebuilt from edges
        self._has_cid_placeholders = False
        self._cid_total_count = 0
        self._cid_unresolved_count = 0
        self.isolated_cells: list[dict[str, Any]] = []

    def __repr__(self):
        """Return a string representation of the class .

        Returns
        -------
        [type]
            [description]
        """
        return f"<{self.__class__.__name__} shape={self.shape}>"

    def __lt__(self, other):
        """Return True if the two pages are less than the current page .

        Parameters
        ----------
        other : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        if self.page == other.page:
            if self.order < other.order:
                return True
        if self.page < other.page:
            return True

    @staticmethod
    def _collapse_anchors(
        values: Iterable[float],
        axis_span: Optional[float],
        min_threshold: float = 0.5,
    ) -> List[float]:
        """Collapse near-duplicate anchor positions (e.g., double borders)."""
        coords: List[float] = []
        for value in values:
            try:
                coords.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(coords) <= 1:
            return coords
        deltas = [
            coords[idx + 1] - coords[idx]
            for idx in range(len(coords) - 1)
            if coords[idx + 1] - coords[idx] > 0
        ]
        typical_gap = statistics.median(deltas) if deltas else None
        thresholds: List[float] = []
        if axis_span is not None and axis_span > 0:
            thresholds.append(abs(axis_span) * 0.005)
        if typical_gap is not None:
            thresholds.append(typical_gap * 0.3)
        threshold = max(float(min_threshold), min(thresholds)) if thresholds else float(min_threshold)

        collapsed: List[float] = [coords[0]]
        for coord in coords[1:]:
            if coord - collapsed[-1] <= threshold:
                collapsed[-1] = (collapsed[-1] + coord) / 2.0
            else:
                collapsed.append(coord)
        return collapsed

    def _effective_grid_dimensions(self) -> Tuple[int, int, List[float], List[float]]:
        """
        Derive grid dimensions while collapsing line artefacts.

        Returns (rows, cols, collapsed_cols, collapsed_rows).
        """
        table_width: Optional[float] = None
        table_height: Optional[float] = None
        if self.bbox is not None and len(self.bbox) >= 4:
            try:
                table_width = float(self.bbox[2]) - float(self.bbox[0])
                table_height = float(self.bbox[3]) - float(self.bbox[1])
            except (TypeError, ValueError):
                table_width = table_height = None

        parse_info = self.parse if isinstance(self.parse, dict) else {}
        col_anchors = parse_info.get("col_anchors")
        row_anchors = parse_info.get("row_anchors")
        if not isinstance(col_anchors, list) and self.cols:
            col_anchors = [self.cols[0][0]] + [c[1] for c in self.cols]
        if not isinstance(row_anchors, list) and self.rows:
            row_anchors = [self.rows[0][0]] + [r[1] for r in self.rows]

        collapsed_cols = self._collapse_anchors(col_anchors or [], table_width)
        collapsed_rows = self._collapse_anchors(row_anchors or [], table_height)

        if len(collapsed_cols) > 1 and len(collapsed_rows) > 1:
            return (
                len(collapsed_rows) - 1,
                len(collapsed_cols) - 1,
                collapsed_cols,
                collapsed_rows,
            )

        base_rows = len(self.cells)
        base_cols = len(self.cells[0]) if self.cells and self.cells[0] else 0
        return base_rows, base_cols, collapsed_cols, collapsed_rows

    @property
    def data(self):
        """Returns two-dimensional list of strings in table."""
        d = []
        for row in self.cells:
            d.append([cell.text.strip() for cell in row])
        return d

    def has_visible_text(self) -> bool:
        """Return ``True`` when at least one cell contains native text."""
        for row in self.cells:
            for cell in row:
                text = cell.text
                if isinstance(text, str):
                    if text.strip():
                        return True
                    continue
                if text is not None and not pd.isna(text):
                    return True

        if self.flavor in (None, "lattice"):
            for span in self.spanning_cells:
                span_text = span.get("text")
                if isinstance(span_text, str):
                    if span_text.strip():
                        return True
                    continue
                if span_text is not None and not pd.isna(span_text):
                    return True
        return False

    @property
    def bbox(self):
        """Return the table bounding box as ``(x1, y1, x2, y2)``."""
        if self._bbox is not None:
            return self._bbox

        if not self.cells:
            return None

        x1 = min(cell.x1 for row in self.cells for cell in row)
        y1 = min(cell.y1 for row in self.cells for cell in row)
        x2 = max(cell.x2 for row in self.cells for cell in row)
        y2 = max(cell.y2 for row in self.cells for cell in row)
        return (x1, y1, x2, y2)

    @property
    def parsing_report(self):
        """Returns a parsing report.

        with % accuracy, % whitespace,
        table number on page and page number.
        """
        # pretty?
        report = {
            "accuracy": round(self.accuracy, 2),
            "whitespace": round(self.whitespace, 2),
            "order": self.order,
            "page": self.page,
        }
        return report

    @staticmethod
    def _normalize_text_value(text: Any) -> Optional[str]:
        """Return a lightly normalized representation of cell text."""
        if text is None:
            return None
        if isinstance(text, str):
            normalized = " ".join(text.split())
            return normalized
        return str(text)

    @staticmethod
    def _textline_within_bbox(textline, bbox, tolerance: float = 1.0) -> bool:
        """Return True if textline bbox lies inside bbox within tolerance."""
        if bbox is None:
            return False
        x0, y0, x1, y1 = bbox
        span_xmin = min(x0, x1) - tolerance
        span_xmax = max(x0, x1) + tolerance
        span_ymin = min(y0, y1) - tolerance
        span_ymax = max(y0, y1) + tolerance
        return (
            textline.x0 >= span_xmin
            and textline.x1 <= span_xmax
            and textline.y0 >= span_ymin
            and textline.y1 <= span_ymax
        )

    def _compute_jc_accuracy_indicator(
        self,
        export_bbox_fn,
        tolerance: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Return coverage indicator for logical cells when data is available."""
        if not self.textlines:
            return None
        logical_cells = self.spanning_cells
        if not logical_cells:
            return None

        textline_cache: Dict[int, Tuple[Optional[str], Optional[Dict[str, Any]]]] = {}

        inspected = 0
        passing = 0
        violations: List[Dict[str, Any]] = []

        for span in logical_cells:
            bbox = span.get("bbox")
            if bbox is None:
                continue

            overlapping = textlines_overlapping_bbox(bbox, self.textlines)
            if not overlapping:
                continue

            meaningful_lines: List[Tuple[Any, str]] = []
            for textline in overlapping:
                cache_entry = textline_cache.get(id(textline))
                if cache_entry is None:
                    normalized_text = self._normalize_text_value(textline.get_text())
                    bbox_payload = (
                        export_bbox_fn(
                            (textline.x0, textline.y0, textline.x1, textline.y1)
                        )
                        if normalized_text
                        else None
                    )
                    cache_entry = (normalized_text, bbox_payload)
                    textline_cache[id(textline)] = cache_entry
                normalized_text, tl_bbox_payload = cache_entry
                if normalized_text:
                    meaningful_lines.append((textline, normalized_text, tl_bbox_payload))
            if not meaningful_lines:
                continue

            inspected += 1
            offenders: List[Tuple[Any, str, Optional[Dict[str, Any]]]] = []
            for textline, normalized_text, tl_bbox_payload in meaningful_lines:
                if not self._textline_within_bbox(textline, bbox, tolerance):
                    offenders.append((textline, normalized_text, tl_bbox_payload))
            if not offenders:
                passing += 1
                continue

            cell_info: Dict[str, Any] = {
                "row_start": span.get("row_start"),
                "row_end": span.get("row_end"),
                "col_start": span.get("col_start"),
                "col_end": span.get("col_end"),
            }
            cell_bbox_payload = export_bbox_fn(bbox)
            if cell_bbox_payload is not None:
                cell_info["bbox"] = cell_bbox_payload

            violation_entry: Dict[str, Any] = {"cell": cell_info, "text_lines": []}
            for textline, normalized_text, tl_bbox_payload in offenders:
                if tl_bbox_payload is None:
                    tl_bbox_payload = export_bbox_fn(
                        (textline.x0, textline.y0, textline.x1, textline.y1)
                    )
                violation_entry["text_lines"].append(
                    {
                        "text": normalized_text,
                        "bbox": tl_bbox_payload,
                    }
                )
            violations.append(violation_entry)

        if inspected == 0:
            return None

        score = (passing / inspected) * 100.0
        indicator: Dict[str, Any] = {
            "score": round(score, 2),
            "inspected": inspected,
        }
        if violations:
            indicator["violations"] = violations
        return indicator

    def _compute_rectangularity_indicator(self) -> Optional[Dict[str, Any]]:
        """Return rectangularity scores derived from content coverage."""
        if not self.cells or not self.cells[0]:
            return None
        logical_cells = self.spanning_cells
        if not logical_cells:
            return None

        logical_total = len(logical_cells)
        if logical_total <= 0:
            return None

        rows_count = len(self.cells)
        cols_count = len(self.cells[0])
        total_slots = rows_count * cols_count
        if total_slots <= 0:
            return None

        rectangular_cells = 0
        filler_slots: Set[Tuple[int, int]] = set()
        logical_cells_seen = 0
        logical_cells_with_content = 0

        top_hits_content = [False] * cols_count
        bottom_hits_content = [False] * cols_count
        left_hits_content = [False] * rows_count
        right_hits_content = [False] * rows_count

        top_hits_full = [False] * cols_count
        bottom_hits_full = [False] * cols_count
        left_hits_full = [False] * rows_count
        right_hits_full = [False] * rows_count

        full_hits_grid = [
            [False for _ in range(cols_count)] for _ in range(rows_count)
        ]
        content_hits_grid = [
            [False for _ in range(cols_count)] for _ in range(rows_count)
        ]
        full_slots = 0
        content_slots = 0

        def clamp(value: int, upper: int) -> int:
            return max(0, min(value, upper))

        for span in logical_cells:
            try:
                row_start = int(span.get("row_start", 0))
                row_end = int(span.get("row_end", row_start))
                col_start = int(span.get("col_start", 0))
                col_end = int(span.get("col_end", col_start))
            except (TypeError, ValueError):
                continue

            row_start = clamp(row_start, rows_count - 1)
            row_end = clamp(row_end, rows_count - 1)
            if row_end < row_start:
                row_start, row_end = row_end, row_start

            col_start = clamp(col_start, cols_count - 1)
            col_end = clamp(col_end, cols_count - 1)
            if col_end < col_start:
                col_start, col_end = col_end, col_start

            normalized_text = self._normalize_text_value(span.get("text"))
            has_content = bool(normalized_text)

            source_coords = span.get("source_cells") or []
            coord_pairs: List[Tuple[int, int]] = []
            has_source_coords = False
            for entry in source_coords:
                coord_row = coord_col = None
                if isinstance(entry, dict):
                    coord_row = entry.get("row")
                    coord_col = entry.get("column")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    coord_row, coord_col = entry[0], entry[1]
                try:
                    coord_row = int(coord_row)
                    coord_col = int(coord_col)
                except (TypeError, ValueError):
                    continue
                if 0 <= coord_row < rows_count and 0 <= coord_col < cols_count:
                    has_source_coords = True
                    coord_pairs.append((coord_row, coord_col))

            if coord_pairs:
                coord_set = set(coord_pairs)
            else:
                coord_set = {
                    (r_idx, c_idx)
                    for r_idx in range(row_start, row_end + 1)
                    for c_idx in range(col_start, col_end + 1)
                }
            if not coord_set:
                continue

            logical_cells_seen += 1
            expected_slots = (row_end - row_start + 1) * (col_end - col_start + 1)
            actual_slots = len(coord_set)
            if has_source_coords:
                if actual_slots == expected_slots:
                    rectangular_cells += 1
                else:
                    filler_slots.update(coord_set)
            else:
                rectangular_cells += 1

            for coord_row, coord_col in coord_set:
                if not full_hits_grid[coord_row][coord_col]:
                    full_hits_grid[coord_row][coord_col] = True
                    full_slots += 1
                    if coord_row == 0:
                        top_hits_full[coord_col] = True
                    if coord_row == rows_count - 1:
                        bottom_hits_full[coord_col] = True
                    if coord_col == 0:
                        left_hits_full[coord_row] = True
                    if coord_col == cols_count - 1:
                        right_hits_full[coord_row] = True

            if has_content:
                logical_cells_with_content += 1
                for coord_row, coord_col in coord_set:
                    if not content_hits_grid[coord_row][coord_col]:
                        content_hits_grid[coord_row][coord_col] = True
                        content_slots += 1
                        if coord_row == 0:
                            top_hits_content[coord_col] = True
                        if coord_row == rows_count - 1:
                            bottom_hits_content[coord_col] = True
                        if coord_col == 0:
                            left_hits_content[coord_row] = True
                        if coord_col == cols_count - 1:
                            right_hits_content[coord_row] = True

        def as_percentage(ratio: float) -> float:
            return round(ratio * 100.0, 2)

        area_score = (
            as_percentage(logical_cells_with_content / logical_total)
            if logical_total
            else 0.0
        )
        area_full_score = (
            as_percentage(logical_cells_seen / logical_total)
            if logical_total
            else 0.0
        )

        perimeter_slots = (2 * cols_count) + (2 * rows_count)

        def edge_metrics(
            top_hits: List[bool],
            bottom_hits: List[bool],
            left_hits: List[bool],
            right_hits: List[bool],
        ) -> Tuple[float, Dict[str, int]]:
            covered_edges = sum(top_hits) + sum(bottom_hits) + sum(left_hits) + sum(right_hits)
            edge_score = (
                as_percentage(covered_edges / perimeter_slots)
                if perimeter_slots
                else 0.0
            )
            gaps = {
                "top": int(cols_count - sum(top_hits)),
                "bottom": int(cols_count - sum(bottom_hits)),
                "left": int(rows_count - sum(left_hits)),
                "right": int(rows_count - sum(right_hits)),
            }
            return edge_score, gaps

        edge_score, edge_gaps = edge_metrics(
            top_hits_content,
            bottom_hits_content,
            left_hits_content,
            right_hits_content,
        )
        edge_full_score, edge_full_gaps = edge_metrics(
            top_hits_full,
            bottom_hits_full,
            left_hits_full,
            right_hits_full,
        )

        indicator: Dict[str, Any] = {
            "area_text_score": area_score,
            "edge_text_score": edge_score,
            "area_full_score": area_full_score,
            "edge_full_score": edge_full_score,
            "edge_text_gaps": edge_gaps,
            "edge_full_gaps": edge_full_gaps,
        }
        if logical_cells_seen > 0:
            cells_score = (rectangular_cells / logical_cells_seen) * 100.0
            indicator["cells_score"] = round(cells_score, 2)
        filler_total = len(filler_slots)
        filler_denominator = (
            total_slots
            if total_slots > 0
            else logical_cells_seen if logical_cells_seen > 0 else logical_total
        )
        fillers_score = (
            (filler_total / filler_denominator) * 100.0 if filler_denominator else 0.0
        )
        indicator["fillers_score"] = round(fillers_score, 2)
        return indicator

    def to_structured_layout(self) -> Dict[str, Any]:
        """Produce a structured layout payload with geometry metadata."""
        rotation_mod = int(self.rotation) % 360

        pdf_width: Optional[float] = None
        pdf_height: Optional[float] = None
        if self.pdf_size is not None:
            try:
                pdf_width = float(self.pdf_size[0])
                pdf_height = float(self.pdf_size[1])
            except (TypeError, ValueError):
                pdf_width = pdf_height = None

        export_pdf_width = pdf_width
        export_pdf_height = pdf_height
        if (
            pdf_width is not None
            and pdf_height is not None
            and rotation_mod in (90, 270)
        ):
            export_pdf_width, export_pdf_height = pdf_height, pdf_width

        def transform_point(x: float, y: float) -> Tuple[float, float]:
            if pdf_width is None or pdf_height is None:
                return float(x), float(y)
            if rotation_mod == 90:
                width_orig = pdf_height
                return float(width_orig - y), float(x)
            if rotation_mod == 270:
                return float(y), float(x)
            if rotation_mod == 180:
                return float(pdf_width - x), float(pdf_height - y)
            return float(x), float(y)

        def export_bbox(
            bbox: Optional[Tuple[float, float, float, float]]
        ) -> Optional[Dict[str, Any]]:
            if bbox is None:
                return None
            x1, y1, x2, y2 = bbox
            x1_t, y1_t = transform_point(x1, y1)
            x2_t, y2_t = transform_point(x2, y2)
            x_low = min(x1_t, x2_t)
            x_high = max(x1_t, x2_t)
            y_low = min(y1_t, y2_t)
            y_high = max(y1_t, y2_t)
            bbox_dict: Dict[str, Any] = {
                "abs": [
                    float(x_low),
                    float(y_low),
                    float(x_high),
                    float(y_high),
                ]
            }
            if (
                export_pdf_width
                and export_pdf_height
                and export_pdf_width != 0
                and export_pdf_height != 0
            ):
                bbox_dict["norm"] = [
                    float(x_low / export_pdf_width),
                    float(y_low / export_pdf_height),
                    float(x_high / export_pdf_width),
                    float(y_high / export_pdf_height),
                ]
            return bbox_dict

        def export_line(
            points: List[Tuple[float, float]],
            orientation: str,
            source: str,
        ) -> Dict[str, Any]:
            abs_points: List[List[float]] = []
            norm_points: List[List[float]] = []
            for px, py in points:
                x_t, y_t = transform_point(px, py)
                abs_points.append([float(x_t), float(y_t)])
                if (
                    export_pdf_width
                    and export_pdf_height
                    and export_pdf_width != 0
                    and export_pdf_height != 0
                ):
                    norm_points.append(
                        [
                            float(x_t / export_pdf_width),
                            float(y_t / export_pdf_height),
                        ]
                    )
            line: Dict[str, Any] = {
                "orientation": orientation,
                "source": source,
                "points": abs_points,
            }
            if norm_points:
                line["norm_points"] = norm_points
            return line

        def dedupe_preserve(values: Iterable[float]) -> List[float]:
            ordered = OrderedDict()
            for value in values:
                try:
                    ordered[float(value)] = None
                except (TypeError, ValueError):
                    continue
            return list(ordered.keys())

        layout: Dict[str, Any] = {}
        if export_pdf_width and export_pdf_height:
            layout["page_size"] = {
                "width": float(export_pdf_width),
                "height": float(export_pdf_height),
            }
        layout["rotation"] = rotation_mod
        table_bbox = export_bbox(self.bbox)
        if table_bbox is not None:
            layout["bbox"] = table_bbox

        base_cells: List[Dict[str, Any]] = []
        base_cell_count = 0
        for row_idx, row in enumerate(self.cells):
            for col_idx, cell in enumerate(row):
                cell_bbox = export_bbox((cell.x1, cell.y1, cell.x2, cell.y2))
                cell_payload: Dict[str, Any] = {
                    "row": row_idx,
                    "column": col_idx,
                    "edges": {
                        "top": bool(cell.top),
                        "bottom": bool(cell.bottom),
                        "left": bool(cell.left),
                        "right": bool(cell.right),
                    },
                    "spanning": {
                        "horizontal": bool(cell.hspan),
                        "vertical": bool(cell.vspan),
                    },
                }
                normalized_text = self._normalize_text_value(cell.text)
                if normalized_text is not None:
                    cell_payload["text"] = normalized_text
                if cell.text is not None:
                    cell_payload["raw_text"] = cell.text
                if cell_bbox is not None:
                    cell_payload["bbox"] = cell_bbox
                base_cells.append(cell_payload)
                base_cell_count += 1
        if base_cells:
            layout["cells"] = base_cells

        logical_cells_payload: List[Dict[str, Any]] = []
        has_real_merges = False
        for span in self.spanning_cells:
            span_bbox = export_bbox(span.get("bbox"))
            span_payload: Dict[str, Any] = {
                "row_start": span["row_start"],
                "row_end": span["row_end"],
                "col_start": span["col_start"],
                "col_end": span["col_end"],
                "row_span": span["row_span"],
                "col_span": span["col_span"],
                "source_cells": [
                    {"row": int(coord[0]), "column": int(coord[1])}
                    for coord in span.get("source_cells", [])
                ],
            }
            normalized_span_text = self._normalize_text_value(span.get("text"))
            if normalized_span_text is not None:
                span_payload["text"] = normalized_span_text
            raw_span_text = span.get("text")
            if raw_span_text is not None:
                span_payload["raw_text"] = raw_span_text
            if span_bbox is not None:
                span_payload["bbox"] = span_bbox
            logical_cells_payload.append(span_payload)
            if span_payload["row_span"] > 1 or span_payload["col_span"] > 1:
                has_real_merges = True
            elif len(span_payload["source_cells"]) > 1:
                has_real_merges = True
        if logical_cells_payload:
            layout["logical_cells"] = logical_cells_payload

        eff_rows, eff_cols, _, _ = self._effective_grid_dimensions()
        effective_grid_total = eff_rows * eff_cols

        if base_cell_count:
            cell_count_entry = {"grid": effective_grid_total}
            if logical_cells_payload:
                cell_count_entry["logical"] = (
                    len(logical_cells_payload)
                    if has_real_merges
                    else effective_grid_total
                )
            layout["cell_count"] = cell_count_entry

        col_positions: List[float] = []
        row_positions: List[float] = []
        parse_info = self.parse if isinstance(self.parse, dict) else {}
        col_anchors = parse_info.get("col_anchors") if parse_info else None
        row_anchors = parse_info.get("row_anchors") if parse_info else None
        if isinstance(col_anchors, list):
            col_positions.extend(col_anchors)
        if isinstance(row_anchors, list):
            row_positions.extend(row_anchors)
        if not col_positions and self.cols:
            for column_bounds in self.cols:
                col_positions.extend(column_bounds)
        if not row_positions and self.rows:
            for row_bounds in self.rows:
                row_positions.extend(row_bounds)
        col_positions = dedupe_preserve(col_positions)
        row_positions = dedupe_preserve(row_positions)

        if col_positions:
            layout["column_positions"] = [float(v) for v in col_positions]
        if row_positions:
            layout["row_positions"] = [float(v) for v in row_positions]

        y_bounds: Optional[Tuple[float, float]] = None
        if row_positions:
            y_bounds = (min(row_positions), max(row_positions))
        elif self.bbox is not None:
            y_bounds = (float(self.bbox[1]), float(self.bbox[3]))

        x_bounds: Optional[Tuple[float, float]] = None
        if col_positions:
            x_bounds = (min(col_positions), max(col_positions))
        elif self.bbox is not None:
            x_bounds = (float(self.bbox[0]), float(self.bbox[2]))

        grid_lines: Dict[str, List[Dict[str, Any]]] = {}
        if y_bounds and col_positions:
            y_min, y_max = y_bounds
            if not math.isclose(y_min, y_max):
                orientation_source = (
                    "parse.col_anchors" if isinstance(col_anchors, list) else "grid.derived"
                )
                vertical_lines = [
                    export_line(
                        [(x, y_min), (x, y_max)],
                        "vertical",
                        orientation_source,
                    )
                    for x in col_positions
                ]
                grid_lines["vertical"] = vertical_lines
        if x_bounds and row_positions:
            x_min, x_max = x_bounds
            if not math.isclose(x_min, x_max):
                orientation_source = (
                    "parse.row_anchors" if isinstance(row_anchors, list) else "grid.derived"
                )
                horizontal_lines = [
                    export_line(
                        [(x_min, y), (x_max, y)],
                        "horizontal",
                        orientation_source,
                    )
                    for y in row_positions
                ]
                grid_lines["horizontal"] = horizontal_lines
        if grid_lines:
            layout["grid_lines"] = grid_lines

        if self.textlines:
            layout["text_line_count"] = len(self.textlines)

        indicators: Dict[str, Any] = {}
        jc_indicator = self._compute_jc_accuracy_indicator(export_bbox)
        if jc_indicator is not None:
            indicators["jc_accuracy"] = jc_indicator
        rectangularity_indicator = self._compute_rectangularity_indicator()
        if rectangularity_indicator is not None:
            indicators["rectangularity"] = rectangularity_indicator
        if indicators:
            layout["indicators"] = indicators

        isolated_cells = getattr(self, "isolated_cells", None)
        if isolated_cells:
            isolated_payload: list[dict[str, Any]] = []
            for cell in isolated_cells:
                entry: dict[str, Any] = {}
                bbox_payload = export_bbox(cell.get("bbox")) if isinstance(cell, dict) else None
                if bbox_payload is not None:
                    entry["bbox"] = bbox_payload
                texts = []
                if isinstance(cell, dict):
                    texts = cell.get("texts") or cell.get("text") or []
                normalized_texts = []
                for text in texts if isinstance(texts, (list, tuple)) else [texts]:
                    normalized = self._normalize_text_value(text)
                    if normalized:
                        normalized_texts.append(normalized)
                if normalized_texts:
                    entry["texts"] = normalized_texts
                if entry:
                    isolated_payload.append(entry)
            if isolated_payload:
                layout["isolated_cells"] = isolated_payload

        return layout

    def to_json_payload(
        self,
        include_dataframe: bool = True,
        include_layout: bool = True,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict describing the table."""
        parsing_report = self.parsing_report
        payload: Dict[str, Any] = {
            "page": self.page,
            "order": self.order,
        }
        if parsing_report:
            parsing_report = {
                k: v for k, v in parsing_report.items() if k not in {"page", "order"}
            }
            if parsing_report:
                payload["parsing_report"] = parsing_report
        if include_dataframe:
            payload["data"] = self.df.to_dict(orient="records")
        if self.flavor is not None:
            payload["flavor"] = self.flavor
        source_file = self.source or self.filename
        if source_file:
            payload["source"] = {"file": source_file}
        if self.pdfinfo_rotation is not None:
            try:
                payload["page_rotation_pdfinfo"] = int(self.pdfinfo_rotation)
            except (TypeError, ValueError):
                payload["page_rotation_pdfinfo"] = self.pdfinfo_rotation
        pdf_size_entry: Optional[Dict[str, float]] = None
        if (
            self.pdf_size is not None
            and self.pdf_size[0] is not None
            and self.pdf_size[1] is not None
        ):
            pdf_size_entry = {
                "width": float(self.pdf_size[0]),
                "height": float(self.pdf_size[1]),
            }
        if self.cells:
            payload["grid"] = {
                "rows": len(self.cells),
                "cols": len(self.cells[0]) if self.cells[0] else 0,
            }
        if getattr(self, "isolated_cells", None):
            isolated_entries: list[dict[str, Any]] = []
            for cell in self.isolated_cells:
                entry: dict[str, Any] = {}
                bbox = cell.get("bbox") if isinstance(cell, dict) else None
                if bbox is not None:
                    entry["bbox"] = {
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                    }
                texts = []
                if isinstance(cell, dict):
                    texts = cell.get("texts") or cell.get("text") or []
                normalized_texts = []
                for text in texts if isinstance(texts, (list, tuple)) else [texts]:
                    normalized = self._normalize_text_value(text)
                    if normalized:
                        normalized_texts.append(normalized)
                if normalized_texts:
                    entry["texts"] = normalized_texts
                if entry:
                    isolated_entries.append(entry)
            if isolated_entries:
                payload["isolated_cells"] = isolated_entries
        layout = self.to_structured_layout() if include_layout else None
        layout_has_data = bool(layout)
        if layout is not None:
            payload["layout"] = layout
        if not layout_has_data:
            payload["rotation"] = self.rotation
            if pdf_size_entry:
                payload["pdf_size"] = pdf_size_entry
        if layout_has_data:
            return {k: v for k, v in payload.items() if v is not None}

        bbox = self.bbox
        if bbox is not None:
            payload["bbox"] = {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
            }
        logical_cells = self.spanning_cells
        if logical_cells:
            spanning_cells = []
            for logical in logical_cells:
                entry: Dict[str, Any] = {
                    "row_start": logical.get("row_start"),
                    "row_end": logical.get("row_end"),
                    "col_start": logical.get("col_start"),
                    "col_end": logical.get("col_end"),
                    "row_span": logical.get("row_span"),
                    "col_span": logical.get("col_span"),
                    "text": logical.get("text"),
                    "source_cells": [
                        {"row": int(coord[0]), "column": int(coord[1])}
                        for coord in logical.get("source_cells") or []
                    ],
                }
                bbox_info = logical.get("bbox")
                if bbox_info is not None:
                    entry["bbox"] = {
                        "x1": float(bbox_info[0]),
                        "y1": float(bbox_info[1]),
                        "x2": float(bbox_info[2]),
                        "y2": float(bbox_info[3]),
                    }
                spanning_cells.append(
                    {k: v for k, v in entry.items() if v is not None}
                )
            if spanning_cells:
                payload["spanning_cells"] = spanning_cells
        return {k: v for k, v in payload.items() if v is not None}

    def get_pdf_image(self):
        """Compute pdf image and cache it."""
        if self._image is None:
            temp_image_path = build_file_path_in_temp_dir(
                os.path.basename(self.filename), ".png"
            )
            self._image_path = temp_image_path
            backend = ImageConversionBackend(use_fallback=True)
            try:
                backend.convert(self.filename, temp_image_path)
                self._image = cv2.imread(temp_image_path)
            finally:
                Path(temp_image_path).unlink(missing_ok=True)
                self._image_path = None
        return self._image

    def set_all_edges(self):
        """Set all table edges to True."""
        for row in self.cells:
            for cell in row:
                cell.left = cell.right = cell.top = cell.bottom = True
        self._spanning_cells = None
        return self

    def set_edges(self, vertical, horizontal, joint_tol=2):
        """Set the edges of the joint.

        Set a cell's edges to True depending on whether the cell's
        coordinates overlap with the line's coordinates within a
        tolerance.

        Parameters
        ----------
        vertical : list
            List of detected vertical lines.
        horizontal : list
            List of detected horizontal lines.
        joint_tol : int, optional
            Tolerance for determining proximity, by default 2
        """
        self._set_vertical_edges(vertical, joint_tol)
        self._set_horizontal_edges(horizontal, joint_tol)
        self._spanning_cells = None
        return self

    def _find_close_point(self, coords, coord, joint_tol):
        for i, t in enumerate(coords):
            if math.isclose(coord, t[0], abs_tol=joint_tol):
                return i
        return None

    def _set_vertical_edges(self, vertical, joint_tol):
        for v in vertical:
            # find closest x coord
            # iterate over y coords and find closest start and end points
            start = self._find_close_point(self.rows, v[3], joint_tol)
            if start is None:
                continue
            end = self._find_close_point(self.rows, v[1], joint_tol)
            if end is None:
                end = len(self.rows)
            i = self._find_close_point(self.cols, v[0], joint_tol)
            self._update_vertical_edges(start, end, i)

    def _update_vertical_edges(self, start, end, index):
        if index is None:  # only right edge
            index = len(self.cols) - 1
            if index >= 0:
                for j in range(start, end):
                    self.cells[j][index].right = True
        elif index == 0:  # only left edge
            for j in range(start, end):
                self.cells[j][0].left = True
        else:  # both left and right edges
            for j in range(start, end):
                self.cells[j][index].left = True
                self.cells[j][index - 1].right = True

    def _set_horizontal_edges(self, horizontal, joint_tol):
        for h in horizontal:
            # find closest y coord
            # iterate over x coords and find closest start and end points
            start = self._find_close_point(self.cols, h[0], joint_tol)
            if start is None:
                continue
            end = self._find_close_point(self.cols, h[2], joint_tol)
            if end is None:
                end = len(self.cols)
            i = self._find_close_point(self.rows, h[1], joint_tol)
            self._update_horizontal_edges(start, end, i)

    def _update_horizontal_edges(self, start, end, index):
        if index is None:  # only bottom edge
            index = len(self.rows) - 1
            if index >= 0:
                for j in range(start, end):
                    self.cells[index][j].bottom = True
        elif index == 0:  # only top edge
            for j in range(start, end):
                self.cells[0][j].top = True
        else:  # both top and bottom edges
            for j in range(start, end):
                self.cells[index][j].top = True
                self.cells[index - 1][j].bottom = True

    def set_border(self):
        """Sets table border edges to True."""
        num_rows = len(self.rows)
        num_cols = len(self.cols)

        # Ensure cells structure is valid
        if num_rows == 0 or num_cols == 0:
            return self  # No rows or columns, nothing to do

        # Check if cells have the expected structure
        if len(self.cells) != num_rows or any(
            len(row) != num_cols for row in self.cells
        ):
            raise ValueError(
                "Inconsistent cells structure: cells should match the dimensions of rows and cols."
            )

        # Set left and right borders for each row
        for row_index in range(num_rows):
            self.cells[row_index][0].left = True  # Set the left border
            self.cells[row_index][num_cols - 1].right = True  # Set the right border

        # Set top and bottom borders for each column
        for col_index in range(num_cols):
            self.cells[0][col_index].top = True  # Set the top border
            self.cells[num_rows - 1][col_index].bottom = True  # Set the bottom border

        self._spanning_cells = None
        return self

    def rebuild_spanning_cells(self):
        """Recompute logical cells by merging adjacent base cells without borders."""
        if not self.cells:
            self._spanning_cells = []
            return self

        num_rows = len(self.cells)
        num_cols = len(self.cells[0]) if num_rows else 0

        if num_rows == 0 or num_cols == 0:
            self._spanning_cells = []
            return self

        if any(len(row) != num_cols for row in self.cells):
            # Grid is inconsistent; bail out to avoid incorrect merges.
            self._spanning_cells = []
            return self

        parent = list(range(num_rows * num_cols))
        rank = [0] * len(parent)

        def idx(row, col):
            return row * num_cols + col

        def find(item):
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            if rank[root_a] < rank[root_b]:
                parent[root_a] = root_b
            elif rank[root_a] > rank[root_b]:
                parent[root_b] = root_a
            else:
                parent[root_b] = root_a
                rank[root_a] += 1

        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                cell = self.cells[r_idx][c_idx]
                cell_text = cell.text.strip()
                if c_idx + 1 < num_cols and not cell.right:
                    neighbor = self.cells[r_idx][c_idx + 1]
                    if not neighbor.left:
                        neighbor_text = neighbor.text.strip()
                        if (
                            not cell_text
                            or not neighbor_text
                            or cell_text == neighbor_text
                        ):
                            union(idx(r_idx, c_idx), idx(r_idx, c_idx + 1))
                if r_idx > 0 and not cell.top:
                    neighbor = self.cells[r_idx - 1][c_idx]
                    if not neighbor.bottom:
                        neighbor_text = neighbor.text.strip()
                        if (
                            not cell_text
                            or not neighbor_text
                            or cell_text == neighbor_text
                        ):
                            union(idx(r_idx, c_idx), idx(r_idx - 1, c_idx))

        grouped = defaultdict(list)
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                grouped[find(idx(r_idx, c_idx))].append((r_idx, c_idx))

        logical_cells = []
        for coords in grouped.values():
            coords.sort()
            row_indices = [r for r, _ in coords]
            col_indices = [c for _, c in coords]
            row_start = min(row_indices)
            row_end = max(row_indices)
            col_start = min(col_indices)
            col_end = max(col_indices)

            merged = [self.cells[r][c] for r, c in coords]
            x1 = min(cell.x1 for cell in merged)
            y1 = min(cell.y1 for cell in merged)
            x2 = max(cell.x2 for cell in merged)
            y2 = max(cell.y2 for cell in merged)

            values = OrderedDict()
            for r_idx, c_idx in coords:
                value = self.cells[r_idx][c_idx].text.strip()
                if value:
                    values.setdefault(value, None)

            logical_cells.append(
                {
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "row_span": row_end - row_start + 1,
                    "col_span": col_end - col_start + 1,
                    "bbox": (x1, y1, x2, y2),
                    "text": "\n".join(values.keys()),
                    "source_cells": coords,
                }
            )

        logical_cells.sort(key=lambda item: (item["row_start"], item["col_start"]))
        self._spanning_cells = logical_cells
        return self

    @property
    def spanning_cells(self):
        """Return merged logical cells, computing them from edges if needed."""
        if self.flavor not in (None, "lattice", "network", "stream", "hybrid", "autotune"):
            return []
        if self._spanning_cells is None:
            self.rebuild_spanning_cells()
        return self._spanning_cells

    def _render_html(self, preserve_spans=None, **kwargs):
        """Render the table as an HTML string, optionally preserving spans."""
        render_kwargs = dict(kwargs)

        logical_cells = None
        preserve = False
        if preserve_spans is None:
            if self.flavor == "lattice":
                logical_cells = self.spanning_cells
                preserve = bool(logical_cells)
        elif preserve_spans:
            logical_cells = self.spanning_cells
            preserve = True

        if not preserve:
            return self.df.to_html(**render_kwargs)

        if logical_cells is None:
            logical_cells = self.spanning_cells

        classes = render_kwargs.pop("classes", None)
        table_id = render_kwargs.pop("table_id", None)
        border = render_kwargs.pop("border", None)
        table_attributes = render_kwargs.pop("table_attributes", None)
        escape_text = render_kwargs.pop("escape", True)
        na_rep = render_kwargs.pop("na_rep", "")
        render_kwargs.pop("index", None)
        render_kwargs.pop("header", None)

        if render_kwargs:
            # Unsupported keyword arguments, fallback to pandas implementation.
            return self.df.to_html(
                classes=classes,
                table_id=table_id,
                border=border,
                table_attributes=table_attributes,
                escape=escape_text,
                na_rep=na_rep,
                **render_kwargs,
            )

        base_rows = len(self.cells)
        base_cols = len(self.cells[0]) if base_rows else 0
        grid = [[None for _ in range(base_cols)] for _ in range(base_rows)]

        isolated_texts: set[str] = set()
        for iso in getattr(self, "isolated_cells", []) or []:
            texts = iso.get("texts") or iso.get("text") or []
            if not isinstance(texts, (list, tuple)):
                texts = [texts]
            for text in texts:
                if not text:
                    continue
                normalized = self._normalize_text_value(text)
                if normalized:
                    isolated_texts.add(normalized)

        def _cell_has_isolated(text_value: str) -> bool:
            if not isolated_texts:
                return False
            normalized_cell = self._normalize_text_value(text_value) or ""
            for token in isolated_texts:
                if token and token in normalized_cell:
                    return True
            return False

        for info in logical_cells:
            row_start = info["row_start"]
            col_start = info["col_start"]
            row_span = info["row_span"]
            col_span = info["col_span"]
            top_left = (row_start, col_start)
            for r_idx in range(row_start, row_start + row_span):
                for c_idx in range(col_start, col_start + col_span):
                    if (r_idx, c_idx) == top_left:
                        grid[r_idx][c_idx] = info
                    else:
                        grid[r_idx][c_idx] = _SPAN_SKIP

        attrs = []
        if classes:
            if isinstance(classes, (list, tuple, set)):
                class_attr = " ".join(str(cls) for cls in classes)
            else:
                class_attr = str(classes)
            if class_attr:
                attrs.append(f'class="{escape(class_attr, quote=True)}"')
        if table_id:
            attrs.append(f'id="{escape(str(table_id), quote=True)}"')
        if border is not None:
            attrs.append(f'border="{int(border)}"')

        table_open = "<table"
        if table_attributes:
            table_open += f" {table_attributes}"
        if attrs:
            table_open += " " + " ".join(attrs)
        table_open += ">"
        table_close = "</table>"

        rendered_rows: list[list[dict[str, object]]] = []
        for r_idx in range(base_rows):
            row_cells: list[dict[str, object]] = []
            for c_idx in range(base_cols):
                entry = grid[r_idx][c_idx]
                if entry is _SPAN_SKIP:
                    continue
                if entry is None:
                    cell = self.cells[r_idx][c_idx]
                    entry = {
                        "row_span": 1,
                        "col_span": 1,
                        "text": cell.text.strip(),
                    }

                col_span = int(entry.get("col_span", 1) or 1)
                row_span = int(entry.get("row_span", 1) or 1)
                text_value = entry.get("text", "")
                if not text_value:
                    text_value = na_rep
                display = text_value
                if escape_text:
                    display = escape(display, quote=True)
                display = display.replace("\n", "<br/>")

                row_cells.append(
                    {
                        "col_span": col_span,
                        "row_span": row_span,
                        "html": display,
                        "isolated": _cell_has_isolated(text_value),
                    }
                )
            if row_cells:
                rendered_rows.append(row_cells)

        if not rendered_rows:
            return table_open + "<tbody></tbody>" + table_close

        top_span_cells = [
            cell for cell in logical_cells if cell.get("row_start", 0) == 0
        ]
        if top_span_cells:
            header_depth = max(cell["row_end"] for cell in top_span_cells) + 1
        else:
            header_depth = 1
        header_depth = max(1, min(header_depth, len(rendered_rows)))

        def _format_cell(cell: dict[str, object], tag: str) -> str:
            attr_parts: list[str] = []
            col_span = int(cell.get("col_span", 1) or 1)
            row_span = int(cell.get("row_span", 1) or 1)
            classes: list[str] = []
            if cell.get("isolated"):
                classes.append("isolated-cell")
            if classes:
                attr_parts.append(f'class="{" ".join(classes)}"')
            if col_span > 1:
                attr_parts.append(f'colspan="{col_span}"')
            if row_span > 1:
                attr_parts.append(f'rowspan="{row_span}"')
            attrs = " " + " ".join(attr_parts) if attr_parts else ""
            return f"<{tag}{attrs}>{cell.get('html', '&nbsp;')}</{tag}>"

        lines = [table_open]

        if header_depth > 0:
            lines.append("<thead>")
            for row in rendered_rows[:header_depth]:
                lines.append("<tr>")
                for cell in row:
                    lines.append(_format_cell(cell, "th"))
                lines.append("</tr>")
            lines.append("</thead>")

        if header_depth < len(rendered_rows):
            lines.append("<tbody>")
            for row in rendered_rows[header_depth:]:
                lines.append("<tr>")
                for cell in row:
                    lines.append(_format_cell(cell, "td"))
                lines.append("</tr>")
            lines.append("</tbody>")
        else:
            lines.append("<tbody></tbody>")

        lines.append(table_close)
        html_output = "".join(lines)

        isolated_cells = getattr(self, "isolated_cells", None)
        if isolated_cells:
            report_lines: list[str] = []
            report_lines.append('<div class="camelot-isolated-cells">')
            report_lines.append(
                f"<strong>Isolated cells detected:</strong> {len(isolated_cells)}"
            )
            report_lines.append("<ul>")
            for idx, cell in enumerate(isolated_cells, start=1):
                texts = []
                if isinstance(cell, dict):
                    texts = cell.get("texts") or cell.get("text") or []
                normalized_texts = []
                for text in texts if isinstance(texts, (list, tuple)) else [texts]:
                    if not text:
                        continue
                    normalized_texts.append(escape(str(text), quote=True))
                bbox = cell.get("bbox") if isinstance(cell, dict) else None
                bbox_str = ""
                if bbox is not None:
                    bbox_str = (
                        f" (bbox: x1={round(bbox[0], 2)}, y1={round(bbox[1], 2)}, "
                        f"x2={round(bbox[2], 2)}, y2={round(bbox[3], 2)})"
                    )
                text_label = ", ".join(normalized_texts) if normalized_texts else "—"
                report_lines.append(f"<li>#{idx}: {text_label}{bbox_str}</li>")
            report_lines.append("</ul></div>")
            html_output = html_output + "".join(report_lines)

        return html_output

    def to_html_string(self, preserve_spans=None, **kwargs):
        """Return the table rendered as HTML without writing to disk."""
        return self._render_html(preserve_spans=preserve_spans, **kwargs)

    def copy_spanning_text(self, copy_text=None):
        """
        Copies over text in empty spanning cells.

        Parameters
        ----------
        copy_text : list of str, optional (default: None)
            Select one or more of the following strings: {'h', 'v'} to specify
            the direction in which text should be copied over when a cell spans
            multiple rows or columns.

        Returns
        -------
        camelot.core.Table
            The updated table with copied text in spanning cells.
        """
        if copy_text is None:
            return self

        for direction in copy_text:
            if direction == "h":
                self._copy_horizontal_text()
            elif direction == "v":
                self._copy_vertical_text()

        self._spanning_cells = None
        return self

    def _copy_horizontal_text(self):
        """
        Copies text horizontally in empty spanning cells.

        This method iterates through the cells and fills empty cells that span
        horizontally with the text from the left adjacent cell.

        Returns
        -------
        None
        """
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if (
                    self.cells[i][j].text.strip() == ""
                    and self.cells[i][j].hspan
                    and not self.cells[i][j].left
                ):
                    self.cells[i][j].text = self.cells[i][j - 1].text

    def _copy_vertical_text(self):
        """
        Copies text vertically in empty spanning cells.

        This method iterates through the cells and fills empty cells that span
        vertically with the text from the top adjacent cell.

        Returns
        -------
        None
        """
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if (
                    self.cells[i][j].text.strip() == ""
                    and self.cells[i][j].vspan
                    and not self.cells[i][j].top
                ):
                    self.cells[i][j].text = self.cells[i - 1][j].text

    def to_csv(self, path, **kwargs):
        """Write Table(s) to a comma-separated values (csv) file.

        For kwargs, check :meth:`pandas.DataFrame.to_csv`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8", "index": False, "header": False, "quoting": 1}
        kw.update(kwargs)
        self.df.to_csv(path, **kw)

    def to_json(self, path, **kwargs):
        """Write Table(s) to a JSON file.

        For kwargs, check :meth:`pandas.DataFrame.to_json`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"orient": "records"}
        kw.update(kwargs)
        json_string = self.df.to_json(**kw)
        with open(path, "w") as f:
            f.write(json_string)

    def to_excel(self, path, **kwargs):
        """Write Table(s) to an Excel file.

        For kwargs, check :meth:`pandas.DataFrame.to_excel`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"encoding": "utf-8"}
        sheet_name = f"page-{self.page}-table-{self.order}"
        kw.update(kwargs)
        writer = pd.ExcelWriter(path)
        self.df.to_excel(writer, sheet_name=sheet_name, **kw)

    def to_html(self, path, **kwargs):
        """Write Table(s) to an HTML file.

        For kwargs, check :meth:`pandas.DataFrame.to_html`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        html_string = self._render_html(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_string)

    def to_markdown(self, path, **kwargs):
        """Write Table(s) to a Markdown file.

        For kwargs, check :meth:`pandas.DataFrame.to_markdown`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        md_string = self.df.to_markdown(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_string)

    def to_sqlite(self, path, **kwargs):
        """Write Table(s) to sqlite database.

        For kwargs, check :meth:`pandas.DataFrame.to_sql`.

        Parameters
        ----------
        path : str
            Output filepath.

        """
        kw = {"if_exists": "replace", "index": False}
        kw.update(kwargs)
        conn = sqlite3.connect(path)
        table_name = f"page-{self.page}-table-{self.order}"
        self.df.to_sql(table_name, conn, **kw)
        conn.commit()
        conn.close()


class _Kw(TypedDict):
    """Helper class to define file related arguments."""

    path: os.PathLike[Any] | str
    dirname: str
    root: str
    ext: str


class TableList:
    """Defines a list of camelot.core.Table objects.

    Each table can be accessed using its index.

    Attributes
    ----------
    n : int
        Number of tables in the list.

    """

    def __init__(
        self,
        tables: Iterable[Table],
        cleanup_callbacks: Iterable[Callable[[], None]] | None = None,
    ) -> None:  # noqa D105
        self._tables: Iterable[Table] = tables
        self._cleanup_callbacks: list[Callable[[], None]] = list(
            cleanup_callbacks or []
        )

    def __repr__(self):  # noqa D105
        return f"<{self.__class__.__name__} n={self.n}>"

    def __len__(self):  # noqa D105
        return len(self._tables)

    def __getitem__(self, idx):  # noqa D105
        return self._tables[idx]

    def __iter__(self) -> Iterator[Table]:  # noqa D105
        return iter(self._tables)

    def __next__(self) -> Table:  # noqa D105
        return next(self)

    def close(self) -> None:
        """Release any temporary resources associated with the tables."""
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        for cb in callbacks:
            try:
                cb()
            except Exception:
                # Best-effort cleanup; avoid raising during shutdown.
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _format_func(table, f):
        return getattr(table, f"to_{f}")

    @property
    def n(self) -> int:
        """The number of tables in the list."""
        return len(self)

    def _write_file(self, f=None, **kwargs: Unpack[_Kw]) -> None:
        dirname = kwargs["dirname"]
        root = kwargs["root"]
        ext = kwargs["ext"]
        for table in self._tables:
            filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
            filepath = os.path.join(dirname, filename)
            to_format = self._format_func(table, f)
            to_format(filepath)

    def _compress_dir(self, **kwargs: Unpack[_Kw]) -> None:
        path = kwargs["path"]
        dirname = kwargs["dirname"]
        root = kwargs["root"]
        ext = kwargs["ext"]
        zipname = os.path.join(os.path.dirname(path), root) + ".zip"
        with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
            for table in self._tables:
                filename = f"{root}-page-{table.page}-table-{table.order}{ext}"
                filepath = os.path.join(dirname, filename)
                z.write(filepath, os.path.basename(filepath))

    def to_structured_data(
        self,
        skip_empty: bool = True,
        include_dataframe: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return a structured payload for each table in the list."""
        payloads: List[Dict[str, Any]] = []
        for table in self._tables:
            if skip_empty and not table.has_visible_text():
                table_label = f"table-p{table.page}-o{table.order}"
                logger.info(
                    "Skipping %s due to lack of native text.",
                    table_label,
                )
                continue
            payloads.append(
                table.to_json_payload(
                    include_dataframe=include_dataframe,
                    include_layout=True,
                )
            )
        return payloads

    def _write_json_with_coordinates(self, filepath: str) -> None:
        payload = {
            "schema": dict(JSON_COORDS_SCHEMA_DESCRIPTOR),
            "tables": self.to_structured_data(),
        }
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    def export(self, path: str, f="csv", compress=False):
        """Export the list of tables to specified file format.

        Parameters
        ----------
        path : str
            Output filepath.
        f : str
            File format. Can be csv, excel, html, json, markdown or sqlite.
        compress : bool
            Whether or not to add files to a ZIP archive.

        """
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        root, ext = os.path.splitext(basename)
        temp_dir: Optional[str] = None
        if compress:
            temp_dir = tempfile.mkdtemp()
            dirname = temp_dir

        kwargs: _Kw = {"path": path, "dirname": dirname, "root": root, "ext": ext}

        try:
            if f in ["csv", "html", "json", "markdown"]:
                self._write_file(f=f, **kwargs)
                if compress:
                    self._compress_dir(**kwargs)
            elif f == "json-coords":
                filepath = os.path.join(dirname, basename)
                self._write_json_with_coordinates(filepath)
                if compress:
                    zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                    with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                        z.write(filepath, os.path.basename(filepath))
            elif f == "excel":
                filepath = os.path.join(dirname, basename)
                writer = pd.ExcelWriter(filepath)
                for table in self._tables:
                    sheet_name = f"page-{table.page}-table-{table.order}"
                    table.df.to_excel(writer, sheet_name=sheet_name)
                writer.close()
                if compress:
                    zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                    with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                        z.write(filepath, os.path.basename(filepath))
            elif f == "sqlite":
                filepath = os.path.join(dirname, basename)
                for table in self._tables:
                    table.to_sqlite(filepath)
                if compress:
                    zipname = os.path.join(os.path.dirname(path), root) + ".zip"
                    with zipfile.ZipFile(zipname, "w", allowZip64=True) as z:
                        z.write(filepath, os.path.basename(filepath))
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


def _overlay_boxes(
    tables_json: os.PathLike[str] | str,
    input_pdf: os.PathLike[str] | str,
    output_pdf: os.PathLike[str] | str,
    style: Dict[str, Any],
    *,
    hatch_diagonals: bool,
) -> None:
    from pypdf import PdfReader
    from pypdf import PdfWriter
    from pypdf.annotations import Line
    from pypdf.annotations import Rectangle
    from pypdf.generic import ArrayObject
    from pypdf.generic import FloatObject
    from pypdf.generic import NameObject
    from pypdf.generic import TextStringObject

    tables_json_path = Path(tables_json).expanduser()
    input_pdf_path = Path(input_pdf).expanduser()
    output_pdf_path = Path(output_pdf).expanduser()

    if not tables_json_path.is_file():
        raise FileNotFoundError(f"JSON file '{tables_json_path}' not found.")
    if not input_pdf_path.is_file():
        raise FileNotFoundError(f"PDF file '{input_pdf_path}' not found.")

    if output_pdf_path.parent:
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    tables = _load_tables_payload(tables_json_path)

    reader = PdfReader(str(input_pdf_path))
    writer = PdfWriter()

    annotations_by_page: Dict[int, List[Dict[str, Any]]] = {}

    for table in tables:
        if not isinstance(table, dict):
            continue
        page_number = table.get("page")
        layout_info = table.get("layout") or {}
        bbox = table.get("bbox") or layout_info.get("bbox")
        if not isinstance(page_number, int) or bbox is None:
            continue
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(reader.pages):
            continue
        rotation_value = table.get("rotation")
        if rotation_value is None:
            rotation_value = layout_info.get("rotation")
        annotations_by_page.setdefault(page_index, []).append(
            {
                "bbox": bbox,
                "rotation": int(rotation_value or 0),
            }
        )

    def diagonal_lines(
        bbox: tuple[float, float, float, float], spacing: float = 16.0
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        x1, y1, x2, y2 = bbox
        c_min = x1 - y2
        c_max = x2 - y1
        lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
        c = c_min
        while c <= c_max + spacing:
            intersections: list[tuple[float, float]] = []

            y_at_x1 = x1 - c
            if y1 <= y_at_x1 <= y2:
                intersections.append((x1, y_at_x1))

            y_at_x2 = x2 - c
            if y1 <= y_at_x2 <= y2:
                intersections.append((x2, y_at_x2))

            x_at_y1 = c + y1
            if x1 <= x_at_y1 <= x2:
                intersections.append((x_at_y1, y1))

            x_at_y2 = c + y2
            if x1 <= x_at_y2 <= x2:
                intersections.append((x_at_y2, y2))

            unique_points: list[tuple[float, float]] = []
            for pt in intersections:
                if pt not in unique_points:
                    unique_points.append(pt)

            if len(unique_points) >= 2:
                unique_points.sort(key=lambda p: (p[0], p[1]))
                start = unique_points[0]
                end = unique_points[-1]
                lines.append((start, end))

            c += spacing

        return lines

    for page_index, page in enumerate(reader.pages):
        writer.add_page(page)
        target_index = len(writer.pages) - 1

        page_rotation = int(page.get("/Rotate", 0) or 0)
        page_items = annotations_by_page.get(page_index, [])

        mediabox = page.mediabox
        media_width = float(mediabox.width)
        media_height = float(mediabox.height)
        media_x0 = float(mediabox.left)
        media_y0 = float(mediabox.bottom)
        if page_rotation in (0, 180):
            width = media_width
            height = media_height
        else:
            width = media_height
            height = media_width

        def build_annotation(
            rect: tuple[float, float, float, float],
            label: str | None,
            style_dict: Dict[str, Any],
        ) -> None:
            stroke_color = style_dict["stroke"]
            fill_color = style_dict["fill"]
            border_width = float(style_dict.get("border", 3.0))
            diagonal_border_width = float(
                style_dict.get("diagonal_border", max(border_width / 2.0, 1.0))
            )
            opacity = float(style_dict.get("opacity", 0.25))
            annotation = Rectangle(rect)
            annotation[NameObject("/C")] = ArrayObject(
                [
                    FloatObject(stroke_color[0]),
                    FloatObject(stroke_color[1]),
                    FloatObject(stroke_color[2]),
                ]
            )
            annotation[NameObject("/IC")] = ArrayObject(
                [
                    FloatObject(fill_color[0]),
                    FloatObject(fill_color[1]),
                    FloatObject(fill_color[2]),
                ]
            )
            annotation[NameObject("/CA")] = FloatObject(opacity)
            annotation[NameObject("/ca")] = FloatObject(opacity)
            annotation[NameObject("/Border")] = ArrayObject(
                [FloatObject(0), FloatObject(0), FloatObject(border_width)]
            )
            if label:
                text_label = TextStringObject(label)
                annotation[NameObject("/Contents")] = text_label
                annotation[NameObject("/T")] = text_label
                annotation[NameObject("/Subj")] = text_label
            writer.add_annotation(target_index, annotation)

            if hatch_diagonals:
                for start, end in diagonal_lines(rect):
                    line_rect = (
                        min(start[0], end[0]),
                        min(start[1], end[1]),
                        max(start[0], end[0]),
                        max(start[1], end[1]),
                    )
                    line_annotation = Line(start, end, line_rect)
                    line_annotation[NameObject("/C")] = ArrayObject(
                        [
                            FloatObject(stroke_color[0]),
                            FloatObject(stroke_color[1]),
                            FloatObject(stroke_color[2]),
                        ]
                    )
                    line_annotation[NameObject("/Border")] = ArrayObject(
                        [
                            FloatObject(0),
                            FloatObject(0),
                            FloatObject(diagonal_border_width),
                        ]
                    )
                    line_annotation[NameObject("/CA")] = FloatObject(opacity)
                    line_annotation[NameObject("/ca")] = FloatObject(opacity)
                    if label:
                        text_label = TextStringObject(label)
                        line_annotation[NameObject("/Contents")] = text_label
                    writer.add_annotation(target_index, line_annotation)

        for item in page_items:
            rect = _resolve_pdf_bbox(
                item.get("bbox"),
                rotation=item.get("rotation"),
                page_rotation=page_rotation,
                page_width=width,
                page_height=height,
            )
            if rect is None:
                continue
            build_annotation(
                rect,
                label=None,
                style_dict=style,
            )

    with output_pdf_path.open("wb") as fp:
        writer.write(fp)


_DEFAULT_DRAW_STYLE: Dict[str, Any] = {
    "stroke": (0.62, 0.36, 0.75),
    "fill": (0.93, 0.84, 0.97),
    "border": 6.0,
    "diagonal_border": 2.0,
    "opacity": 0.55,
}


def draw_boxes(
    tables_json: os.PathLike[str] | str,
    input_pdf: os.PathLike[str] | str,
    output_pdf: os.PathLike[str] | str,
) -> None:
    """Overlay extracted table bounding boxes onto a PDF with hatched fills."""

    _overlay_boxes(
        tables_json,
        input_pdf,
        output_pdf,
        style=dict(_DEFAULT_DRAW_STYLE),
        hatch_diagonals=True,
    )


def crop_boxes(
    tables_json: os.PathLike[str] | str,
    input_pdf: os.PathLike[str] | str,
    output_pdf: os.PathLike[str] | str,
) -> None:
    """Cover table bounding boxes with opaque patches matching the page background."""

    style = {
        "stroke": (1.0, 1.0, 1.0),
        "fill": (1.0, 1.0, 1.0),
        "border": 0.0,
        "diagonal_border": 0.0,
        "opacity": 1.0,
    }
    _overlay_boxes(
        tables_json,
        input_pdf,
        output_pdf,
        style=style,
        hatch_diagonals=False,
    )


def get_pdf_box(
    tables_json: os.PathLike[str] | str,
    table_id: object,
    input_pdf: os.PathLike[str] | str | None = None,
) -> tuple[float, float, float, float]:
    """
    Return the PDF coordinates of a table's bounding box from a json-coords export.

    This mirrors the coordinate normalization used by :func:`draw_boxes` and
    :func:`crop_boxes`, accounting for page rotations and normalized bbox values.
    The PDF path can be passed explicitly or inferred from the table's ``source.file``.
    """

    tables_json_path = Path(tables_json).expanduser()
    if not tables_json_path.is_file():
        raise FileNotFoundError(f"JSON file '{tables_json_path}' not found.")

    page, order = _normalize_table_identifier(table_id)
    tables = _load_tables_payload(tables_json_path)

    target_table: dict[str, Any] | None = None
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_page = _coerce_positive_int(table.get("page"))
        table_order = _coerce_nonnegative_int(table.get("order"))
        if table_page == page and table_order == order:
            target_table = table
            break

    if target_table is None:
        raise ValueError(
            f"Table with page={page} and order={order} not found in {tables_json_path}"
        )

    layout_info = target_table.get("layout") or {}
    raw_bbox = target_table.get("bbox") or layout_info.get("bbox")
    if raw_bbox is None:
        raise ValueError(
            f"Table with page={page} and order={order} is missing bbox data."
        )

    rotation_value = target_table.get("rotation")
    if rotation_value is None and isinstance(layout_info, dict):
        rotation_value = layout_info.get("rotation")

    try:
        page_index = int(page) - 1
    except (TypeError, ValueError):
        page_index = -1

    inferred_pdf = None
    if input_pdf is None:
        inferred_pdf = _resolve_source_pdf_path(target_table, tables_json_path)
    pdf_path = Path(input_pdf).expanduser() if input_pdf else inferred_pdf

    page_rotation = _extract_page_rotation_pdfinfo(target_table) or 0
    page_width: float | None = None
    page_height: float | None = None

    if pdf_path is not None and pdf_path.is_file():
        from pypdf import PdfReader  # Imported lazily to avoid heavy dependencies at import time.

        reader = PdfReader(str(pdf_path))
        if page_index < 0 or page_index >= len(reader.pages):
            raise ValueError(
                f"PDF page {page} not available in '{pdf_path}' (tables_json={tables_json_path})"
            )
        pdf_page = reader.pages[page_index]
        try:
            page_rotation = int(pdf_page.get("/Rotate", 0) or 0)
        except Exception:
            page_rotation = 0
        mediabox = pdf_page.mediabox
        media_width = float(mediabox.width)
        media_height = float(mediabox.height)
        if page_rotation in (0, 180):
            page_width = media_width
            page_height = media_height
        else:
            page_width = media_height
            page_height = media_width
    else:
        page_width, page_height = _extract_page_size(target_table)

    if page_width is None or page_height is None:
        raise ValueError(
            "Unable to determine page dimensions. Provide input_pdf or ensure page_size is present."
        )

    rect = _resolve_pdf_bbox(
        raw_bbox,
        rotation=rotation_value,
        page_rotation=int(page_rotation or 0),
        page_width=page_width,
        page_height=page_height,
    )
    if rect is None:
        raise ValueError(
            f"Could not resolve bounding box for table page={page} order={order}."
        )
    return rect


_PDFINFO_PAGE_ROTATION = re.compile(r"^Page\s+(\d+)\s+rot:\s+(-?\d+)")
_PDFINFO_SINGLE_PAGE_ROTATION = re.compile(r"^Page\s+rot:\s+(-?\d+)")

_DEFAULT_SPECIFICATION_SUMMARY: Dict[str, Any] = {
    "pages": "all",
    "format": "json-coords",
    "zip": False,
    "quiet": False,
    "parallel": False,
    "password": None,
    "split_text": False,
    "flag_size": False,
    "strip_text": "",
    "margins": None,
    "global_args": (),
    "args": (),
    "table_regions": (),
    "table_areas": (),
    "columns": (),
    "column_repeat": 8,
    "column_pad": 0,
    "remove_background_artifacts": True,
    "remove_text": True,
    "draw_boxes": True,
    "html_report": True,
    "extra_formats": (),
}

_FLAVOR_PARAMETER_SCHEMAS: Dict[str, List[Dict[str, Any]]] = {
    "lattice": [
        {
            "key": "process_background",
            "default": False,
            "kind": "flag",
            "flags": ["--process_background", "-back"],
        },
        {
            "key": "remove_background_artifacts",
            "default": True,
            "kind": "bool",
            "flags": ["--remove_background_artifacts"],
            "negative_flags": ["--no-remove_background_artifacts"],
        },
        {
            "key": "remove_text",
            "default": False,
            "kind": "bool",
            "flags": ["--remove_text"],
            "negative_flags": ["--no-remove_text"],
        },
        {
            "key": "line_scale",
            "default": 40,
            "kind": "value",
            "value_type": "int",
            "flags": ["--line_scale", "-scale"],
        },
        {
            "key": "line_tol",
            "default": 2,
            "kind": "value",
            "value_type": "int",
            "flags": ["--line_tol", "-l"],
        },
        {
            "key": "joint_tol",
            "default": 2,
            "kind": "value",
            "value_type": "int",
            "flags": ["--joint_tol", "-j"],
        },
        {
            "key": "threshold_blocksize",
            "default": 15,
            "kind": "value",
            "value_type": "int",
            "flags": ["--threshold_blocksize", "-block"],
        },
        {
            "key": "threshold_constant",
            "default": -2,
            "kind": "value",
            "value_type": "int",
            "flags": ["--threshold_constant", "-const"],
        },
        {
            "key": "iterations",
            "default": 0,
            "kind": "value",
            "value_type": "int",
            "flags": ["--iterations", "-I"],
        },
        {
            "key": "resolution",
            "default": 300,
            "kind": "value",
            "value_type": "int",
            "flags": ["--resolution", "-res"],
        },
        {
            "key": "copy_text",
            "default": [],
            "kind": "multi",
            "value_type": "str",
            "flags": ["--copy_text", "-copy"],
        },
        {
            "key": "shift_text",
            "default": ["l", "t"],
            "kind": "multi",
            "value_type": "str",
            "flags": ["--shift_text", "-shift"],
        },
        {
            "key": "plot_type",
            "default": None,
            "kind": "value",
            "value_type": "str",
            "flags": ["--plot_type", "-plot"],
        },
    ],
    "stream": [
        {
            "key": "remove_background_artifacts",
            "default": True,
            "kind": "bool",
            "flags": ["--remove_background_artifacts"],
            "negative_flags": ["--no-remove_background_artifacts"],
        },
        {
            "key": "edge_tol",
            "default": 50,
            "kind": "value",
            "value_type": "int",
            "flags": ["--edge_tol", "-e"],
        },
        {
            "key": "row_tol",
            "default": 2,
            "kind": "value",
            "value_type": "int",
            "flags": ["--row_tol", "-r"],
        },
        {
            "key": "column_tol",
            "default": 0,
            "kind": "value",
            "value_type": "int",
            "flags": ["--column_tol", "-c"],
        },
        {
            "key": "plot_type",
            "default": None,
            "kind": "value",
            "value_type": "str",
            "flags": ["--plot_type", "-plot"],
        },
    ],
    "hybrid": [
        {
            "key": "remove_background_artifacts",
            "default": True,
            "kind": "bool",
            "flags": ["--remove_background_artifacts"],
            "negative_flags": ["--no-remove_background_artifacts"],
        },
        {
            "key": "remove_text",
            "default": False,
            "kind": "bool",
            "flags": ["--remove_text"],
            "negative_flags": ["--no-remove_text"],
        },
        {
            "key": "edge_tol",
            "default": 50,
            "kind": "value",
            "value_type": "int",
            "flags": ["--edge_tol", "-e"],
        },
        {
            "key": "row_tol",
            "default": 2,
            "kind": "value",
            "value_type": "int",
            "flags": ["--row_tol", "-r"],
        },
        {
            "key": "column_tol",
            "default": 0,
            "kind": "value",
            "value_type": "int",
            "flags": ["--column_tol", "-c"],
        },
        {
            "key": "plot_type",
            "default": None,
            "kind": "value",
            "value_type": "str",
            "flags": ["--plot_type", "-plot"],
        },
    ],
}


def _interpret_bool(value: object, default: bool = False) -> bool:
    """Best-effort coercion of configuration values to booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
        if lowered in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        return default
    return bool(value)


def _coerce_positive_int(value: object) -> Optional[int]:
    """Convert arbitrary values to positive integers when possible."""
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


def _coerce_nonnegative_int(value: object) -> Optional[int]:
    """Convert arbitrary values to non-negative integers when possible."""
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    return candidate if candidate >= 0 else None


def _collect_pdfinfo_page_rotations(
    pdf_path: Path,
    pages: Sequence[int],
) -> Dict[int, int]:
    """
    Return a mapping of page numbers to their rotation as reported by pdfinfo.

    Failures (missing pdfinfo binary, command errors, etc.) are swallowed so the
    HTML report generation continues even if rotation data is unavailable.
    """
    unique_pages = sorted({page for page in pages if isinstance(page, int) and page > 0})
    if not unique_pages:
        return {}

    pdfinfo_bin = shutil.which("pdfinfo")
    if not pdfinfo_bin:
        return {}

    min_page = min(unique_pages)
    max_page = max(unique_pages)

    try:
        completed = subprocess.run(
            [pdfinfo_bin, "-f", str(min_page), "-l", str(max_page), str(pdf_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}

    if completed.returncode != 0:
        return {}

    page_rotations: Dict[int, int] = {}
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        match = _PDFINFO_PAGE_ROTATION.match(line)
        if match:
            page_rotations[int(match.group(1))] = int(match.group(2))

    if not page_rotations and min_page == max_page:
        match = _PDFINFO_SINGLE_PAGE_ROTATION.search(completed.stdout)
        if match:
            page_rotations[min_page] = int(match.group(1))

    return {page: page_rotations[page] for page in unique_pages if page in page_rotations}


def _normalize_bbox_dict(candidate: object) -> Optional[Dict[str, float]]:
    if not isinstance(candidate, dict):
        return None
    coords = ("x1", "y1", "x2", "y2")
    if not all(coord in candidate for coord in coords):
        return None
    try:
        return {coord: float(candidate[coord]) for coord in coords}
    except (TypeError, ValueError):
        return None


def _extract_table_bbox(table: dict) -> Dict[str, float]:
    bbox = _normalize_bbox_dict(table.get("bbox"))
    if bbox:
        return bbox
    layout_info = table.get("layout") or {}
    layout_bbox = layout_info.get("bbox")
    bbox = _normalize_bbox_dict(layout_bbox)
    if bbox:
        return bbox
    if isinstance(layout_bbox, dict):
        abs_bbox = layout_bbox.get("abs")
        if isinstance(abs_bbox, (list, tuple)) and len(abs_bbox) == 4:
            try:
                x1, y1, x2, y2 = (float(value) for value in abs_bbox)
            except (TypeError, ValueError):
                return {}
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {}


def _extract_table_rotation(table: dict):
    rotation = table.get("rotation")
    if rotation is not None:
        return rotation
    layout_info = table.get("layout") or {}
    return layout_info.get("rotation")


def _extract_page_rotation_pdfinfo(table: dict):
    rotation = table.get("page_rotation_pdfinfo")
    if rotation is None:
        return None
    try:
        return int(rotation)
    except (TypeError, ValueError):
        return None


def _extract_spanning_cells(table: dict) -> List:
    spans = table.get("spanning_cells")
    if isinstance(spans, list) and spans:
        return spans
    layout_info = table.get("layout") or {}
    spans = layout_info.get("logical_cells")
    return spans if isinstance(spans, list) else []


def _normalize_table_identifier(table_id: object) -> tuple[int, int]:
    """Convert flexible table identifiers to (page, order)."""

    def _normalize_pair(page_value: object, order_value: object) -> tuple[int, int]:
        page = _coerce_positive_int(page_value)
        order = _coerce_nonnegative_int(order_value)
        if page is None or order is None:
            raise ValueError(
                "Table identifier must include a valid page (>0) and order (>=0)."
            )
        return page, order

    if isinstance(table_id, (list, tuple)) and len(table_id) >= 2:
        return _normalize_pair(table_id[0], table_id[1])

    if isinstance(table_id, dict):
        return _normalize_pair(table_id.get("page"), table_id.get("order"))

    if isinstance(table_id, str):
        normalized = table_id.strip()
        patterns = [
            r"table-p(?P<page>\d+)-o(?P<order>\d+)",
            r"p(?P<page>\d+)-o(?P<order>\d+)",
            r"(?P<page>\d+)[,.:/_-]+(?P<order>\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return _normalize_pair(match.group("page"), match.group("order"))
        segments = re.split(r"[\s,;:]+", normalized)
        if len(segments) == 2:
            return _normalize_pair(segments[0], segments[1])

    raise ValueError(
        "Unsupported table identifier. Use (page, order) or a string like 'p1-o2'."
    )


def _extract_page_size(table: Mapping[str, object]) -> tuple[Optional[float], Optional[float]]:
    """Return page dimensions from the payload when available."""
    layout_info = table.get("layout") if isinstance(table, Mapping) else None
    page_size = layout_info.get("page_size") if isinstance(layout_info, Mapping) else None
    if not page_size:
        page_size = table.get("pdf_size") if isinstance(table, Mapping) else None
    if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
        try:
            return float(page_size[0]), float(page_size[1])
        except (TypeError, ValueError):
            return (None, None)
    if not isinstance(page_size, Mapping):
        return (None, None)
    try:
        width = float(page_size.get("width"))
        height = float(page_size.get("height"))
    except (TypeError, ValueError):
        return (None, None)
    return (width, height)


def _load_tables_payload(tables_json_path: Path) -> List[dict[str, Any]]:
    """Read and validate a json-coords payload."""
    with tables_json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object containing a 'tables' array")

    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("JSON payload missing 'tables' array")
    return tables


def _resolve_source_pdf_path(table_entry: Mapping[str, object], tables_json_path: Path) -> Path | None:
    """Infer the PDF path from the table entry."""
    source_info = table_entry.get("source") if isinstance(table_entry, Mapping) else None
    pdf_path_value = source_info.get("file") if isinstance(source_info, Mapping) else None
    if not pdf_path_value:
        return None
    pdf_path = Path(str(pdf_path_value)).expanduser()
    if not pdf_path.is_absolute():
        candidate = (tables_json_path.parent / pdf_path).resolve()
        pdf_path = candidate if candidate.exists() else pdf_path
    return pdf_path if pdf_path.exists() else None


def _resolve_pdf_bbox(
    raw_bbox: Any,
    *,
    rotation: object,
    page_rotation: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    """Convert a Camelot bbox entry into PDF coordinates."""

    if page_width is None or page_height is None:
        raise ValueError("Page dimensions are required to resolve bounding boxes.")

    def _expand(rect_coords: tuple[float, float, float, float]):
        x1, y1, x2, y2 = rect_coords
        return (
            math.floor(min(x1, x2)),
            math.floor(min(y1, y2)),
            math.ceil(max(x1, x2)),
            math.ceil(max(y1, y2)),
        )

    def _as_tuple(values: Any) -> tuple[float, float, float, float] | None:
        try:
            x1, y1, x2, y2 = values
        except (TypeError, ValueError):
            return None
        try:
            return (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )
        except (TypeError, ValueError):
            return None

    rect: tuple[float, float, float, float] | None = None
    if isinstance(raw_bbox, dict):
        if {"x1", "y1", "x2", "y2"} <= raw_bbox.keys():
            rect = _as_tuple(
                (
                    raw_bbox.get("x1"),
                    raw_bbox.get("y1"),
                    raw_bbox.get("x2"),
                    raw_bbox.get("y2"),
                )
            )
        if rect is None:
            abs_bbox = raw_bbox.get("abs")
            rect = _as_tuple(abs_bbox) if abs_bbox is not None else None
        if rect is None:
            norm_bbox = raw_bbox.get("norm")
            if isinstance(norm_bbox, (list, tuple)) and len(norm_bbox) == 4:
                normalized = _as_tuple(norm_bbox)
                if normalized is not None:
                    nx1, ny1, nx2, ny2 = normalized
                    rect = (
                        nx1 * page_width,
                        ny1 * page_height,
                        nx2 * page_width,
                        ny2 * page_height,
                    )
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        rect = _as_tuple(raw_bbox)
    else:
        rect = _as_tuple(raw_bbox)
    if rect is None:
        return None

    try:
        rotation_value = int(rotation or 0)
    except (TypeError, ValueError):
        rotation_value = 0

    x1, y1, x2, y2 = rect
    normalized = (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )
    if page_rotation == 180:
        swapped_x = (page_width - normalized[0], page_width - normalized[2])
        swapped_y = (page_height - normalized[1], page_height - normalized[3])
        normalized = (
            min(swapped_x),
            min(swapped_y),
            max(swapped_x),
            max(swapped_y),
        )
    if page_rotation == 90:
        swapped_x = (page_height - normalized[1], page_height - normalized[3])
        swapped_y = (normalized[0], normalized[2])
        normalized = (
            min(swapped_x),
            min(swapped_y),
            max(swapped_x),
            max(swapped_y),
        )
    if page_rotation == 270:
        swapped_x = (normalized[1], normalized[3])
        swapped_y = (page_width - normalized[0], page_width - normalized[2])
        normalized = (
            min(swapped_x),
            min(swapped_y),
            max(swapped_x),
            max(swapped_y),
        )
    rotation_mod = int(rotation_value) % 360
    if (page_rotation in (0, 180)) and rotation_mod == 270:
        swapped_x = (normalized[0], normalized[2])
        swapped_y = (page_height - normalized[1], page_height - normalized[3])
        return _expand(
            (
                min(swapped_x),
                min(swapped_y),
                max(swapped_x),
                max(swapped_y),
            )
        )
    if (page_rotation in (90, 270)) and rotation_mod == 270:
        swapped_x = (page_height - normalized[0], page_height - normalized[2])
        swapped_y = (normalized[1], normalized[3])
        return _expand(
            (
                min(swapped_x),
                min(swapped_y),
                max(swapped_x),
                max(swapped_y),
            )
        )
    return _expand(normalized)


def _cell_coord_key(entry: object) -> Optional[Tuple[int, int, int, int]]:
    """Return a tuple identifying a cell based on grid coordinates."""
    if not isinstance(entry, dict):
        return None
    try:
        row_start = int(entry.get("row_start"))
        col_start = int(entry.get("col_start"))
    except (TypeError, ValueError):
        return None

    def _resolve_end(end_value: object, span_value: object, start_value: int) -> Optional[int]:
        try:
            return int(end_value)
        except (TypeError, ValueError):
            try:
                span_int = int(span_value)
            except (TypeError, ValueError):
                return None
            if span_int <= 0:
                return None
            return start_value + span_int - 1

    row_end = _resolve_end(entry.get("row_end"), entry.get("row_span"), row_start)
    col_end = _resolve_end(entry.get("col_end"), entry.get("col_span"), col_start)
    if row_end is None or col_end is None:
        return None
    if row_end < row_start or col_end < col_start:
        return None
    return (row_start, row_end, col_start, col_end)


def _find_rectangularity_issues(spans: List[dict]) -> Tuple[List[dict], Dict[int, dict]]:
    """Return non-rectangular spans and a lookup keyed by object id."""
    issues: List[dict] = []
    info_by_id: Dict[int, dict] = {}
    for entry in spans or []:
        if not isinstance(entry, dict):
            continue
        try:
            row_span = max(int(entry.get("row_span", 1)), 1)
            col_span = max(int(entry.get("col_span", 1)), 1)
        except (TypeError, ValueError):
            continue
        expected = row_span * col_span
        src_coords = []
        for coord in entry.get("source_cells") or []:
            if not isinstance(coord, dict):
                continue
            try:
                r = int(coord.get("row"))
                c = int(coord.get("column"))
            except (TypeError, ValueError):
                continue
            src_coords.append((r, c))
        if src_coords and len(src_coords) != expected:
            issue = {
                "span": entry,
                "expected": expected,
                "source_count": len(src_coords),
                "source_cells": src_coords,
            }
            issues.append(issue)
            info_by_id[id(entry)] = issue
    return issues, info_by_id


def _render_table_from_rows(rows: object) -> Tuple[List[str], Optional[str]]:
    if not isinstance(rows, list) or not rows:
        return [], "This table contains no cell data."

    columns: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            for key in row.keys():
                if key not in columns:
                    columns.append(key)

    if not columns:
        return [], "Table data present but column headers could not be determined."

    lines = ["  <table>", "    <tbody>"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append("      <tr>")
        for col in columns:
            value = row.get(col)
            cell = "" if value is None else str(value)
            cell_display = escape(cell) if cell else "&nbsp;"
            lines.append(f"        <td>{cell_display}</td>")
        lines.append("      </tr>")
    lines.append("    </tbody>")
    lines.append("  </table>")
    return lines, None


def _render_table_with_spans(
    spanning_cells: object,
    grid_rows: object,
    grid_cols: object,
    *,
    cell_flags: Optional[Dict[int, Dict[str, object]]] = None,
) -> Tuple[List[str], Optional[str]]:
    try:
        rows_count = int(grid_rows)
    except (TypeError, ValueError):
        rows_count = 0
    try:
        cols_count = int(grid_cols)
    except (TypeError, ValueError):
        cols_count = 0

    if not isinstance(spanning_cells, list) or not spanning_cells:
        return [], None
    if rows_count <= 0 or cols_count <= 0:
        return [], "Missing geometry to rebuild the table layout."

    skip_marker: object = object()
    grid: List[List[object | None]] = [
        [None for _ in range(cols_count)] for _ in range(rows_count)
    ]

    for entry in spanning_cells:
        if not isinstance(entry, dict):
            continue
        try:
            row_start = int(entry.get("row_start", 0))
            col_start = int(entry.get("col_start", 0))
            row_span = max(int(entry.get("row_span", 1)), 1)
            col_span = max(int(entry.get("col_span", 1)), 1)
        except (TypeError, ValueError):
            continue

        if not (0 <= row_start < rows_count and 0 <= col_start < cols_count):
            continue

        row_end = min(rows_count, row_start + row_span)
        col_end = min(cols_count, col_start + col_span)

        for r_idx in range(row_start, row_end):
            for c_idx in range(col_start, col_end):
                grid[r_idx][c_idx] = skip_marker
        grid[row_start][col_start] = entry

    rendered_rows: List[List[Dict[str, object]]] = []
    for r_idx in range(rows_count):
        row_cells: List[Dict[str, object]] = []
        row_has_coverage = False
        for c_idx in range(cols_count):
            entry = grid[r_idx][c_idx]
            if entry is skip_marker:
                row_has_coverage = True
                continue
            if not isinstance(entry, dict):
                entry = {"row_span": 1, "col_span": 1, "text": ""}
            else:
                row_has_coverage = True

            try:
                col_span = max(int(entry.get("col_span", 1)), 1)
            except (TypeError, ValueError):
                col_span = 1

            try:
                row_span = max(int(entry.get("row_span", 1)), 1)
            except (TypeError, ValueError):
                row_span = 1

            raw_text = entry.get("text", "")
            text_str = str(raw_text)
            if text_str.strip():
                cell_html = escape(text_str, quote=True).replace("\n", "<br/>")
            else:
                cell_html = "&nbsp;"

            css_classes: List[str] = []
            extras: List[str] = []
            flags = cell_flags.get(id(entry)) if cell_flags else None
            rect_issue = None
            jc_flags: List[dict] = []
            isolated_flag = False
            if isinstance(flags, dict):
                rect_issue = flags.get("rect_issue")
                isolated_flag = bool(flags.get("isolated"))
                try:
                    jc_flags = list(flags.get("jc_violations") or [])
                except Exception:
                    jc_flags = []
            if rect_issue:
                css_classes.append("non-rect-cell")
                actual = rect_issue.get("source_count")
                expected = rect_issue.get("expected")
                if actual is not None and expected:
                    extras.append(
                        f'<div class="cell-note">source cells {actual}/{expected}</div>'
                    )
                    try:
                        fill_pct = max(
                            0.0, min(float(actual) / float(expected), 1.0)
                        )
                    except Exception:
                        fill_pct = None
                    if fill_pct is not None:
                        width = round(fill_pct * 100, 1)
                        extras.append(
                            '<div class="filler-meter"><span style="width: '
                            f'{width}%"></span></div>'
                        )
            if jc_flags:
                css_classes.append("jc-violation")
            if isolated_flag:
                css_classes.append("isolated-cell")

            class_attr = (
                f' class="{" ".join(css_classes)}"' if css_classes else ""
            )
            row_cells.append(
                {
                    "row_span": row_span,
                    "col_span": col_span,
                    "html": cell_html,
                    "class_attr": class_attr,
                    "extras": "".join(extras),
                }
            )
        if row_cells or row_has_coverage:
            rendered_rows.append(row_cells)

    if not rendered_rows:
        return [], "This table contains no cell data."

    def _format_cell(cell: Dict[str, object]) -> str:
        attr_parts: List[str] = []
        col_span = cell.get("col_span", 1)
        row_span = cell.get("row_span", 1)
        try:
            col_span_int = int(col_span)
        except (TypeError, ValueError):
            col_span_int = 1
        try:
            row_span_int = int(row_span)
        except (TypeError, ValueError):
            row_span_int = 1
        if col_span_int > 1:
            attr_parts.append(f'colspan="{col_span_int}"')
        if row_span_int > 1:
            attr_parts.append(f'rowspan="{row_span_int}"')
        class_attr = str(cell.get("class_attr", ""))
        attrs = (
            " " + " ".join(attr_parts)
            if attr_parts
            else ""
        )
        html = str(cell.get("html", "&nbsp;"))
        extras = str(cell.get("extras", ""))
        return f"        <td{class_attr}{attrs}>{html}{extras}</td>"

    lines = ["  <table>", "    <tbody>"]
    for row in rendered_rows:
        lines.append("      <tr>")
        for cell in row:
            lines.append(_format_cell(cell))
        lines.append("      </tr>")
    lines.append("    </tbody>")
    lines.append("  </table>")
    return lines, None


def _normalized_sequence(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _clone_option_default(value: Any) -> Any:
    if isinstance(value, (list, dict, set)):
        return copy.deepcopy(value)
    return value


def _convert_option_value(value: object, value_type: str) -> Any:
    if value_type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if value_type == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if value is None:
        return None
    return str(value)


def _summarize_flavor_options(
    flavor: object,
    flavor_args: Sequence[object] | None,
    flavor_schemas: Mapping[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    flavor_key = str(flavor).lower() if isinstance(flavor, str) else flavor
    schema_lookup = flavor_schemas or _FLAVOR_PARAMETER_SCHEMAS
    schema = schema_lookup.get(flavor_key, [])
    if not schema:
        return {}

    values: Dict[str, Any] = {
        entry["key"]: _clone_option_default(entry.get("default")) for entry in schema
    }
    overridden_multi: Set[str] = set()
    tokens = [str(arg) for arg in flavor_args] if flavor_args else []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        entry: Dict[str, Any] | None = None
        positive = True
        for candidate in schema:
            if token in candidate.get("flags", []):
                entry = candidate
                positive = True
                break
            if token in candidate.get("negative_flags", []):
                entry = candidate
                positive = False
                break

        if entry is None:
            idx += 1
            continue

        kind = entry.get("kind", "value")
        key = entry["key"]

        if kind == "value":
            if positive and idx + 1 < len(tokens):
                value = _convert_option_value(
                    tokens[idx + 1], entry.get("value_type", "str")
                )
                values[key] = value
                idx += 2
            else:
                idx += 1
            continue

        if kind == "multi":
            if positive and idx + 1 < len(tokens):
                if key not in overridden_multi:
                    values[key] = []
                    overridden_multi.add(key)
                value = _convert_option_value(
                    tokens[idx + 1], entry.get("value_type", "str")
                )
                values[key].append(value)
                idx += 2
            else:
                idx += 1
            continue

        if kind == "flag":
            if positive:
                values[key] = True
            idx += 1
            continue

        if kind == "bool":
            values[key] = bool(positive)
            idx += 1
            continue

        idx += 1

    return values


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, set):
        return [_normalize_json_value(v) for v in sorted(value)]
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    return value


def _summarize_specification(
    spec: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
    flavor_args: Sequence[object] | None = None,
    command_parts: Sequence[object] | None = None,
    flavor_schemas: Mapping[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    spec = spec or {}
    defaults = defaults or _DEFAULT_SPECIFICATION_SUMMARY

    def _get(key: str, fallback: Any) -> Any:
        if key in spec:
            return spec[key]
        return defaults.get(key, fallback)

    remove_background_default = defaults.get("remove_background_artifacts", False)
    remove_text_default = defaults.get("remove_text", False)

    summary: Dict[str, Any] = {
        "label": spec.get("label"),
        "flavor": spec.get("flavor", "lattice"),
        "pages": _get("pages", None),
        "format": _get("format", None),
        "zip": bool(_get("zip", False)),
        "quiet": bool(_get("quiet", False)),
        "parallel": bool(_get("parallel", False)),
        "password": _get("password", None),
        "split_text": bool(_get("split_text", False)),
        "flag_size": bool(_get("flag_size", False)),
        "strip_text": _get("strip_text", ""),
        "margins": _get("margins", None),
        "global_args": _normalized_sequence(_get("global_args", ())),
        "args": _normalized_sequence(_get("args", ())),
        "table_regions": _get("table_regions", ()),
        "table_areas": _get("table_areas", ()),
        "columns": _get("columns", ()),
        "column_repeat": int(_get("column_repeat", 0)),
        "column_pad": int(_get("column_pad", 0)),
        "remove_background_artifacts": _interpret_bool(
            _get("remove_background_artifacts", remove_background_default),
            remove_background_default,
        ),
        "remove_text": _interpret_bool(
            _get("remove_text", remove_text_default),
            remove_text_default,
        ),
        "draw_boxes": bool(_get("draw_boxes", True)),
        "html_report": bool(_get("html_report", True)),
        "extra_formats": list(_normalized_sequence(_get("extra_formats", ()))),  # type: ignore[arg-type]
    }

    summary["effective_args"] = (
        [str(value) for value in flavor_args] if flavor_args is not None else []
    )
    summary["command_parts"] = (
        [str(value) for value in command_parts] if command_parts is not None else []
    )
    summary["flavor_parameters"] = _summarize_flavor_options(
        summary.get("flavor"), flavor_args, flavor_schemas
    )

    return {key: _normalize_json_value(value) for key, value in summary.items()}


def format_command(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def generate_html_report(
    *,
    tables_json: os.PathLike[str] | str | Path,
    html_output: os.PathLike[str] | str | Path,
    label: str,
    flavor: str,
    pdf_path: os.PathLike[str] | str | Path,
    command_preview: str,
    specification: Mapping[str, object] | None,
    command_parts: Sequence[str] | None = None,
    flavor_args: Sequence[str] | None = None,
    specification_defaults: Mapping[str, object] | None = None,
    flavor_parameter_schemas: Mapping[str, List[Dict[str, Any]]] | None = None,
) -> None:
    """Build an HTML overview from a json-coords export."""

    tables_json_path = Path(tables_json)
    html_output_path = Path(html_output)
    source_pdf_path = Path(pdf_path)

    if not tables_json_path.exists():
        raise FileNotFoundError(f"JSON results not found: {tables_json_path}")

    with tables_json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise ValueError("Invalid JSON payload: missing 'tables' list")

    def _format_metric(value: object) -> str:
        if isinstance(value, (int, float)):
            formatted = f"{value:.3f}".rstrip("0").rstrip(".")
            return formatted
        if value is None:
            return ""
        return str(value)

    def _normalize_iso_text(value: object) -> str:
        if value is None:
            return ""
        try:
            return " ".join(str(value).split())
        except Exception:
            return ""

    def _collect_isolated_texts(entries: object) -> set[str]:
        texts: set[str] = set()
        if not isinstance(entries, list):
            return texts
        for iso in entries:
            if not isinstance(iso, dict):
                continue
            iso_texts = iso.get("texts") or iso.get("text") or []
            if not isinstance(iso_texts, (list, tuple)):
                iso_texts = [iso_texts]
            for text in iso_texts:
                normalized = _normalize_iso_text(text)
                if normalized:
                    texts.add(normalized)
        return texts

    def _span_is_isolated(span_entry: object, iso_texts: set[str]) -> bool:
        if not iso_texts or not isinstance(span_entry, dict):
            return False
        span_text = span_entry.get("text") or span_entry.get("raw_text")
        normalized = _normalize_iso_text(span_text)
        if not normalized:
            return False
        return any(token in normalized for token in iso_texts)

    def _compute_uniformity_from_data(
        table_entry: Mapping[str, object] | object,
    ) -> Dict[str, object] | None:
        data = table_entry.get("data") if isinstance(table_entry, dict) else None
        if not isinstance(data, list):
            return None

        lengths: List[int] = []
        nonempty = 0
        for row in data:
            if isinstance(row, dict):
                values = row.values()
            elif isinstance(row, (list, tuple)):
                values = row
            else:
                continue
            for value in values:
                if value is None:
                    lengths.append(0)
                    continue
                try:
                    if pd.isna(value):
                        lengths.append(0)
                        continue
                except Exception:
                    pass
                try:
                    normalized = " ".join(str(value).split())
                except Exception:
                    normalized = None
                length = len(normalized) if normalized else 0
                lengths.append(length)
                if length > 0:
                    nonempty += 1

        if not lengths:
            return None

        total_chars = sum(lengths)
        if total_chars <= 0:
            return None

        sorted_vals = sorted(lengths)
        n = len(sorted_vals)
        cumulative = 0.0
        for idx, val in enumerate(sorted_vals, start=1):
            cumulative += idx * val
        gini = (2 * cumulative) / (n * total_chars) - (n + 1) / n
        uniformity = 1.0 - gini
        return {
            "uniformity": round(uniformity, 4),
            "gini": round(gini, 4),
            "cells": len(lengths),
            "nonempty_cells": nonempty,
            "total_chars": total_chars,
            "max_chars": max(lengths),
        }

    def _compute_network_uniformity_from_data(
        table_entry: Mapping[str, object] | object,
    ) -> Dict[str, object] | None:
        """Compute uniformity of non-empty cell text lengths."""
        data = table_entry.get("data") if isinstance(table_entry, dict) else None
        if not isinstance(data, list):
            return None

        lengths: List[int] = []
        for row in data:
            if isinstance(row, dict):
                values = row.values()
            elif isinstance(row, (list, tuple)):
                values = row
            else:
                continue
            for value in values:
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                try:
                    normalized = " ".join(str(value).split())
                except Exception:
                    normalized = None
                length = len(normalized) if normalized else 0
                if length > 0:
                    lengths.append(length)

        if not lengths:
            return None

        total_chars = sum(lengths)
        if total_chars <= 0:
            return None
        avg_len = total_chars / len(lengths) if lengths else 0.0

        sorted_vals = sorted(lengths)
        n = len(sorted_vals)
        cumulative = 0.0
        for idx, val in enumerate(sorted_vals, start=1):
            cumulative += idx * val
        gini = (2 * cumulative) / (n * total_chars) - (n + 1) / n
        uniformity = 1.0 - gini
        return {
            "uniformity": round(uniformity, 4),
            "gini": round(gini, 4),
            "cells": len(lengths),
            "total_chars": total_chars,
            "max_chars": max(lengths),
            "avg_chars": round(avg_len, 4),
        }

    page_numbers: List[int] = []
    missing_rotation_pages: Set[int] = set()
    for table in tables:
        if not isinstance(table, dict):
            continue
        page_value = _coerce_positive_int(table.get("page"))
        if page_value is not None:
            page_numbers.append(page_value)
            if _extract_page_rotation_pdfinfo(table) is None:
                missing_rotation_pages.add(page_value)

    page_rotations: Dict[int, int] = {}
    if missing_rotation_pages:
        page_rotations = _collect_pdfinfo_page_rotations(
            source_pdf_path, sorted(missing_rotation_pages)
        )
    html_output_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"{source_pdf_path.stem} · {label} · {flavor}"
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        f"  <title>{escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; color: #1f1f24; }",
        "    header { margin-bottom: 2.5rem; }",
        "    header h1 { margin: 0 0 0.5rem; font-size: 1.75rem; }",
        "    header .meta { font-size: 0.9rem; color: #555; }",
        "    code { background: #ececec; padding: 0.15rem 0.35rem; border-radius: 4px; }",
        "    pre.command { background: #272822; color: #f8f8f2; padding: 1rem; overflow-x: auto; border-radius: 6px; }",
        "    section.table-block { background: #eaf1ff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 2rem; padding: 1.5rem; }",
        "    section.table-block h2 { margin-top: 0; font-size: 1.25rem; padding: 0.5rem 0; color: #1f2a44; }",
        "    section.table-block .stats { font-size: 0.85rem; color: #1f2a44; margin-bottom: 0.75rem; line-height: 1.4; }",
        "    section.table-block .stats .stat-line { display: block; margin-bottom: 0.15rem; background: rgba(255,255,255,0.55); padding: 0.25rem 0.6rem; border-radius: 4px; }",
        "    table { border-collapse: collapse; width: 100%; font-size: 0.95rem; background: #fff; border: 2px solid #9cb5ff; box-shadow: 0 0 0 1px #dbe3ff inset; }",
        "    thead { background: #373a9c; color: #f8f8ff; }",
        "    th, td { border: 1.5px solid #9cb5ff; padding: 0.45rem 0.6rem; vertical-align: top; }",
        "    tbody tr:nth-child(even) { background: #fafafa; }",
        "    tbody tr:hover { background: #f1f4ff; }",
        "    .empty-table { font-style: italic; color: #666; margin: 1rem 0; }",
        "    details.jc-violations { margin: 0.5rem 0 1rem; }",
        "    details.jc-violations summary, details.rect-issues summary, details.isolated-issues summary { cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; list-style: none; }",
        "    details.jc-violations summary::-webkit-details-marker, details.rect-issues summary::-webkit-details-marker, details.isolated-issues summary::-webkit-details-marker { display: none; }",
        "    details.jc-violations summary::marker, details.rect-issues summary::marker, details.isolated-issues summary::marker { content: \"\"; }",
        "    details.jc-violations ul { margin: 0.5rem 0 0 1rem; padding-left: 1.25rem; }",
        "    details.jc-violations li { margin-bottom: 0.35rem; }",
        "    details.rect-issues { margin: 0.5rem 0 1rem; }",
        "    details.rect-issues ul { margin: 0.5rem 0 0 1rem; padding-left: 1.25rem; }",
        "    details.rect-issues li { margin-bottom: 0.35rem; }",
        "    details.isolated-issues { margin: 0.5rem 0 1rem; }",
        "    details.isolated-issues ul { margin: 0.5rem 0 0 1rem; padding-left: 1.25rem; }",
        "    details.isolated-issues li { margin-bottom: 0.35rem; }",
        "    .summary-swatch { width: 0.85rem; height: 0.85rem; border-radius: 3px; display: inline-block; box-shadow: 0 0 0 1px rgba(0,0,0,0.08); flex-shrink: 0; }",
        "    .summary-swatch.rect { background: #ffe8e8; border: 1px solid #e58a8a; }",
        "    .summary-swatch.jc { background: #fff5c2; border: 1px solid #d6b000; }",
        "    .summary-swatch.iso { background-color: #f3ebff; border: 1px solid #8d65d9; background-image: linear-gradient(90deg, rgba(128, 90, 213, 0.35) 1px, transparent 1px), linear-gradient(0deg, rgba(128, 90, 213, 0.35) 1px, transparent 1px); background-size: 6px 6px, 6px 6px; }",
        "    .summary-marker { width: 0; height: 0; border-top: 0.35rem solid transparent; border-bottom: 0.35rem solid transparent; border-left: 0.55rem solid #1f1f24; transition: transform 0.2s ease; transform-origin: 35% 50%; flex-shrink: 0; }",
        "    details[open] > summary .summary-marker { transform: rotate(90deg); }",
        "    td.non-rect-cell { background: #ffe8e8; border-color: #e58a8a; position: relative; }",
        "    td.jc-violation { background: #fff5c2; border-color: #d6b000; position: relative; }",
        "    td.non-rect-cell.jc-violation { background: linear-gradient(135deg, #ffe8e8 50%, #fff5c2 50%); border-color: #d19a5a; }",
        "    td .cell-note { display: block; margin-top: 0.35rem; font-size: 0.8em; color: #8c1a1a; }",
        "    td .filler-meter { margin-top: 0.2rem; width: 100%; height: 4px; background: #f6cfd1; }",
        "    td .filler-meter span { display: block; height: 100%; background: #c62828; }",
        "    td.isolated-cell, th.isolated-cell { background-color: #f8f2ff; background-image: linear-gradient(90deg, rgba(128, 90, 213, 0.22) 1px, transparent 1px), linear-gradient(0deg, rgba(128, 90, 213, 0.22) 1px, transparent 1px); background-size: 12px 12px, 12px 12px; border-color: #b592e6; }",
        "  </style>",
        "</head>",
        "<body>",
        "<header>",
        f"  <h1>{escape(title)}</h1>",
        f"  <div class=\"meta\"><strong>Source PDF:</strong> {escape(str(source_pdf_path))}</div>",
        f"  <div class=\"meta\"><strong>Results JSON:</strong> {escape(str(tables_json_path))}</div>",
        f"  <div class=\"meta\"><strong>Command:</strong></div>",
        f"  <pre class=\"command\">{escape(command_preview)}</pre>",
        "</header>",
    ]

    if not tables:
        html_lines.append("<p class=\"empty-table\">No tables detected.</p>")
    else:
        for index, table in enumerate(tables, start=1):
            if not isinstance(table, dict):
                continue
            page = table.get("page")
            normalized_page = _coerce_positive_int(page)
            page_display = normalized_page if normalized_page is not None else page
            order = table.get("order")
            stats = table.get("parsing_report") or {}
            bbox = _extract_table_bbox(table)
            rotation = _extract_table_rotation(table)
            layout_info = table.get("layout") or {}
            grid_info = table.get("grid") or {}
            indicators = layout_info.get("indicators")
            jc_indicator = None
            rect_indicator = None
            uniformity_indicator = _compute_uniformity_from_data(table)
            network_uniformity_indicator = _compute_network_uniformity_from_data(table)
            if isinstance(indicators, dict):
                jc_indicator = indicators.get("jc_accuracy")
                rect_indicator = indicators.get("rectangularity")
                if uniformity_indicator is None:
                    uniformity_indicator = indicators.get("uniformity")
                if isinstance(jc_indicator, dict):
                    jc_violations = jc_indicator.get("violations") or []
                else:
                    jc_violations = []
            else:
                jc_violations = []
            layout_cell_count = (
                layout_info.get("cell_count") if isinstance(layout_info, dict) else None
            )
            page_rotation = _extract_page_rotation_pdfinfo(table)
            if page_rotation is None and normalized_page is not None:
                page_rotation = page_rotations.get(normalized_page)

            section_title = f"Table {index} — page {page_display} (order {order})"
            html_lines.append("<section class=\"table-block\">")
            html_lines.append(f"  <h2>{escape(section_title)}</h2>")

            meta_bits: List[str] = []
            table_flavor = table.get("flavor")
            if not table_flavor and isinstance(layout_info, dict):
                table_flavor = layout_info.get("flavor")
            if not table_flavor:
                table_flavor = flavor
            try:
                flavor_label = str(table_flavor).strip() if table_flavor is not None else ""
            except Exception:
                flavor_label = ""
            if flavor_label:
                meta_bits.append("flavor: " + flavor_label)
            if isinstance(grid_info, dict):
                grid_rows = grid_info.get("rows")
                grid_cols = grid_info.get("cols")
                grid_desc: List[str] = []
                if grid_rows is not None and grid_cols is not None:
                    grid_desc.append(f"{grid_rows}x{grid_cols}")
                    try:
                        total_cells = int(grid_rows) * int(grid_cols)
                    except (TypeError, ValueError):
                        total_cells = None
                    if total_cells is not None:
                        grid_desc.append(f"total {total_cells}")
                else:
                    if grid_rows is not None:
                        grid_desc.append(f"rows {grid_rows}")
                    if grid_cols is not None:
                        grid_desc.append(f"cols {grid_cols}")
                if grid_desc:
                    meta_bits.append("grid: " + ", ".join(grid_desc))
            if isinstance(layout_cell_count, dict):
                grid_cells = layout_cell_count.get("grid")
                logical_cells = layout_cell_count.get("logical")
                count_bits: List[str] = []
                if grid_cells is not None:
                    count_bits.append(f"grid {grid_cells}")
                if logical_cells is not None:
                    count_bits.append(f"logical {logical_cells}")
                if count_bits:
                    meta_bits.append("layout cell_count: " + ", ".join(count_bits))
            if stats:
                accuracy = stats.get("accuracy")
                whitespace = stats.get("whitespace")
                order_stat = stats.get("order")
                meta_stats: List[str] = []
                if accuracy is not None:
                    meta_stats.append(f"accuracy {accuracy}")
                if whitespace is not None:
                    meta_stats.append(f"whitespace {whitespace}")
                if order_stat is not None:
                    meta_stats.append(f"order {order_stat}")
                if meta_stats:
                    meta_bits.append("report: " + ", ".join(meta_stats))
            if jc_indicator:
                jc_score = jc_indicator.get("score")
                jc_inspected = jc_indicator.get("inspected")
                summary_bits = []
                if jc_score is not None:
                    summary_bits.append(f"score {jc_score}")
                if jc_inspected is not None:
                    summary_bits.append(f"spans {jc_inspected}")
                summary_bits.append(f"violations {len(jc_violations)}")
                meta_bits.append("jc_accuracy: " + ", ".join(summary_bits))
            if uniformity_indicator:
                uniformity_value = uniformity_indicator.get("uniformity")
                gini_value = uniformity_indicator.get("gini")
                total_cells_uniformity = uniformity_indicator.get("cells")
                nonempty_cells_uniformity = uniformity_indicator.get("nonempty_cells")
                summary_bits = []
                if uniformity_value is not None:
                    summary_bits.append(f"U {_format_metric(uniformity_value)}")
                if gini_value is not None:
                    summary_bits.append(f"G {_format_metric(gini_value)}")
                if (
                    total_cells_uniformity is not None
                    and nonempty_cells_uniformity is not None
                ):
                    summary_bits.append(
                        f"nonempty {nonempty_cells_uniformity}/{total_cells_uniformity}"
                    )
                if summary_bits:
                    meta_bits.append("uniformity: " + ", ".join(summary_bits))
            if network_uniformity_indicator:
                nu_value = network_uniformity_indicator.get("uniformity")
                nu_gini = network_uniformity_indicator.get("gini")
                nu_cells = network_uniformity_indicator.get("cells")
                nu_chars = network_uniformity_indicator.get("total_chars")
                nu_max = network_uniformity_indicator.get("max_chars")
                nu_avg = network_uniformity_indicator.get("avg_chars")
                summary_bits = []
                if nu_value is not None:
                    summary_bits.append(f"U {_format_metric(nu_value)}")
                if nu_gini is not None:
                    summary_bits.append(f"G {_format_metric(nu_gini)}")
                if nu_cells is not None:
                    summary_bits.append(f"cells {nu_cells}")
                if nu_chars is not None:
                    summary_bits.append(f"chars {nu_chars}")
                if nu_avg is not None:
                    summary_bits.append(f"avg { _format_metric(nu_avg)}")
                if nu_max is not None:
                    summary_bits.append(f"max {nu_max}")
                if summary_bits:
                    meta_bits.append("text uniformity (nonempty): " + ", ".join(summary_bits))
            if rect_indicator:
                area_score = rect_indicator.get("area_text_score")
                edge_score = rect_indicator.get("edge_text_score")
                edge_gaps = rect_indicator.get("edge_text_gaps") or {}
                area_full = rect_indicator.get("area_full_score")
                edge_full = rect_indicator.get("edge_full_score")
                edge_full_gaps = rect_indicator.get("edge_full_gaps") or {}
                summary_bits = []
                if area_score is not None:
                    summary_bits.append(f"area(text) {area_score}")
                if area_full is not None:
                    summary_bits.append(f"area(all) {area_full}")
                if edge_score is not None:
                    summary_bits.append(f"edges(text) {edge_score}")
                if edge_full is not None:
                    summary_bits.append(f"edges(all) {edge_full}")
                cells_score = rect_indicator.get("cells_score")
                fillers_score = rect_indicator.get("fillers_score")
                if cells_score is not None:
                    summary_bits.append(f"cells {cells_score}")
                if fillers_score is not None:
                    summary_bits.append(f"fillers {fillers_score}")
                gap_bits: List[str] = []
                for edge_name in ("top", "bottom", "left", "right"):
                    gap_value = edge_gaps.get(edge_name)
                    if isinstance(gap_value, int) and gap_value > 0:
                        gap_bits.append(f"{edge_name} {gap_value}")
                if gap_bits:
                    summary_bits.append("gaps(text) " + "/".join(gap_bits))
                full_gap_bits: List[str] = []
                for edge_name in ("top", "bottom", "left", "right"):
                    gap_value = edge_full_gaps.get(edge_name)
                    if isinstance(gap_value, int) and gap_value > 0:
                        full_gap_bits.append(f"{edge_name} {gap_value}")
                if full_gap_bits:
                    summary_bits.append("gaps(all) " + "/".join(full_gap_bits))
                if summary_bits:
                    meta_bits.append("rectangularity: " + ", ".join(summary_bits))
            if bbox:
                meta_bits.append(
                    "bbox (x1,y1,x2,y2)="
                    + ", ".join(
                        str(bbox.get(coord)) for coord in ("x1", "y1", "x2", "y2")
                    )
                )
            if rotation is not None:
                meta_bits.append(f"table rotation {rotation} deg")
            if page_rotation is not None:
                meta_bits.append(f"pdfinfo page rotation {page_rotation} deg")
            if meta_bits:
                html_lines.append("  <div class=\"stats\">")
                for entry in meta_bits:
                    html_lines.append(
                        f"    <span class=\"stat-line\">{escape(entry)}</span>"
                    )
                html_lines.append("  </div>")

            isolated_cells = None
            isolated_texts: Set[str] = set()
            layout_isolated = layout_info.get("isolated_cells") if isinstance(layout_info, dict) else None
            if isinstance(layout_isolated, list) and layout_isolated:
                isolated_cells = layout_isolated
            elif isinstance(table.get("isolated_cells"), list) and table.get("isolated_cells"):
                isolated_cells = table.get("isolated_cells")
            if isolated_cells:
                isolated_texts = _collect_isolated_texts(isolated_cells)

            table_lines: List[str] = []
            message: Optional[str] = None

            spans = _extract_spanning_cells(table)
            rect_issues: List[dict] = []
            rect_issue_lookup: Dict[int, dict] = {}
            span_lookup: Dict[Tuple[int, int, int, int], dict] = {}
            cell_flags: Dict[int, Dict[str, object]] = {}

            for span in spans or []:
                key = _cell_coord_key(span)
                if key:
                    span_lookup[key] = span

            if spans:
                rect_issues, rect_issue_lookup = _find_rectangularity_issues(spans)
                for entry_id, issue in rect_issue_lookup.items():
                    cell_flags.setdefault(entry_id, {})["rect_issue"] = issue

            if jc_violations and span_lookup:
                for violation in jc_violations:
                    cell_info = violation.get("cell") or {}
                    key = _cell_coord_key(cell_info)
                    if key is None:
                        continue
                    span = span_lookup.get(key)
                    if span is None:
                        continue
                    cell_flags.setdefault(id(span), {}).setdefault(
                        "jc_violations", []
                    ).append(violation)
            if isolated_texts and spans:
                for span in spans:
                    if _span_is_isolated(span, isolated_texts):
                        cell_flags.setdefault(id(span), {})["isolated"] = True

            if spans:
                rows_count = grid_info.get("rows") if isinstance(grid_info, dict) else None
                cols_count = grid_info.get("cols") if isinstance(grid_info, dict) else None
                table_lines, message = _render_table_with_spans(
                    spans,
                    rows_count,
                    cols_count,
                    cell_flags=cell_flags,
                )

            if not table_lines:
                fallback_lines, fallback_message = _render_table_from_rows(table.get("data"))
                table_lines = fallback_lines
                if fallback_message:
                    message = fallback_message

            if table_lines:
                html_lines.extend(table_lines)
            else:
                failure_reason = message or "This table contains no cell data."
                html_lines.append(f"  <p class=\"empty-table\">{escape(failure_reason)}</p>")

            if rect_issues:
                meta_bits.append(f"rectangularity issues: {len(rect_issues)} non-rectangular cells")

            if jc_indicator:
                if jc_violations:
                    html_lines.append("  <details class=\"jc-violations\">")
                    html_lines.append(
                        "    <summary><span class=\"summary-swatch jc\"></span><span class=\"summary-marker\"></span>JC accuracy violations</summary>"
                    )
                    html_lines.append("    <ul>")
                    max_entries = 10
                    for violation in jc_violations[:max_entries]:
                        cell_info = violation.get("cell") or {}
                        row_start = cell_info.get("row_start")
                        row_end = cell_info.get("row_end")
                        col_start = cell_info.get("col_start")
                        col_end = cell_info.get("col_end")
                        cell_desc_parts = []
                        if row_start is not None and row_end is not None:
                            cell_desc_parts.append(f"rows {row_start}-{row_end}")
                        if col_start is not None and col_end is not None:
                            cell_desc_parts.append(f"cols {col_start}-{col_end}")
                        cell_desc = ", ".join(cell_desc_parts) or "cell"
                        text_lines = violation.get("text_lines") or []
                        text_snippets = [
                            (tl.get("text") or "").strip() for tl in text_lines if tl
                        ]
                        snippet_display = "; ".join(filter(None, text_snippets)) or "<empty>"
                        html_lines.append(
                            f"      <li><strong>{escape(cell_desc)}:</strong> {escape(snippet_display)}</li>"
                        )
                    remaining = len(jc_violations) - max_entries
                    if remaining > 0:
                        html_lines.append(
                            f"      <li>… and {remaining} more violations.</li>"
                        )
                    html_lines.append("    </ul>")
                    html_lines.append("  </details>")

            if rect_issues:
                html_lines.append("  <details class=\"rect-issues\">")
                html_lines.append(
                    "    <summary><span class=\"summary-swatch rect\"></span><span class=\"summary-marker\"></span>Rectangularity issues</summary>"
                )
                html_lines.append("    <ul>")
                max_entries = 10
                for issue in rect_issues[:max_entries]:
                    span = issue.get("span", {})
                    try:
                        rs = int(span.get("row_start", 0))
                        cs = int(span.get("col_start", 0))
                        re = int(span.get("row_end", rs))
                        ce = int(span.get("col_end", cs))
                    except Exception:
                        rs = cs = re = ce = None
                    actual = issue.get("source_count")
                    expected = issue.get("expected")
                    desc_parts = []
                    if rs is not None and re is not None:
                        desc_parts.append(f"rows {rs}-{re}")
                    if cs is not None and ce is not None:
                        desc_parts.append(f"cols {cs}-{ce}")
                    span_label = ", ".join(desc_parts) or "cell"
                    html_lines.append(
                        "      <li><strong>"
                        f"{escape(span_label)}"
                        "</strong>: "
                        f"{escape(str(actual))}/{escape(str(expected))} source cells"
                        "</li>"
                    )
                remaining = len(rect_issues) - max_entries
                if remaining > 0:
                    html_lines.append(f"      <li>… and {remaining} more issues.</li>")
                html_lines.append("    </ul>")
                html_lines.append("  </details>")

            if isolated_cells:
                html_lines.append("  <details class=\"isolated-issues\">")
                html_lines.append(
                    "    <summary><span class=\"summary-swatch iso\"></span><span class=\"summary-marker\"></span>Isolated cells</summary>"
                )
                html_lines.append("    <ul>")
                max_iso = 12
                for cell in isolated_cells[:max_iso]:
                    texts = cell.get("texts") or cell.get("text") or []
                    if not isinstance(texts, (list, tuple)):
                        texts = [texts]
                    label = ", ".join(str(t) for t in texts if t) or "—"
                    bbox = cell.get("bbox") or {}
                    if isinstance(bbox, dict) and {"x1", "y1", "x2", "y2"} <= set(bbox):
                        coords = f"{bbox['x1']}, {bbox['y1']}, {bbox['x2']}, {bbox['y2']}"
                    elif isinstance(bbox, dict):
                        abs_bbox = bbox.get("abs")
                        coords = ", ".join(map(str, abs_bbox)) if isinstance(abs_bbox, (list, tuple)) and len(abs_bbox) == 4 else ""
                    else:
                        coords = ""
                    if coords:
                        desc = f"\"{label}\" (bbox: {coords})"
                    else:
                        desc = f"\"{label}\""
                    html_lines.append(f"      <li>{escape(desc)}</li>")
                remaining_iso = len(isolated_cells) - max_iso
                if remaining_iso > 0:
                    html_lines.append(f"      <li>… and {remaining_iso} more isolated cells.</li>")
                html_lines.append("    </ul>")
                html_lines.append("  </details>")

            html_lines.append("</section>")

    spec_summary = _summarize_specification(
        specification,
        flavor_args=flavor_args,
        command_parts=command_parts,
        defaults=specification_defaults,
        flavor_schemas=flavor_parameter_schemas,
    )
    if spec_summary:
        try:
            spec_repr = json.dumps(spec_summary, indent=2, default=str)
        except TypeError:
            spec_repr = str(spec_summary)
        html_lines.extend(
            [
                "<section class=\"table-block\">",
                "  <h2>Parameter Set Configuration</h2>",
                "  <pre class=\"command\">" + escape(spec_repr) + "</pre>",
                "</section>",
            ]
        )

    html_lines.append("</body>")
    html_lines.append("</html>")

    html_output_path.write_text("\n".join(html_lines), encoding="utf-8")
