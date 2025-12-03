"""Implementation of the Lattice table parser."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from pdfminer.layout import LTAnno
from pdfminer.layout import LTChar

import numpy as np
import statistics
from ..backends import ImageConversionBackend
from ..core import Table
from ..image_processing import adaptive_threshold
from ..image_processing import apply_region_mask
from ..image_processing import find_contours
from ..image_processing import find_joints
from ..image_processing import find_lines
from ..utils import build_file_path_in_temp_dir
from ..utils import flag_font_size
from ..utils import get_table_index
from ..utils import merge_close_lines
from ..utils import scale_image
from ..utils import scale_pdf
from ..utils import segments_in_bbox
from ..utils import text_in_bbox_per_axis
from ..utils import text_strip
from .base import BaseParser, logger


class Lattice(BaseParser):
    """Lattice method looks for lines between text to parse the table.

    Parameters
    ----------
    table_regions : list, optional (default: None)
        List of page regions that may contain tables of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    table_areas : list, optional (default: None)
        List of table area strings of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in PDF coordinate space.
    process_background : bool, optional (default: False)
        Process background lines.
    remove_background_artifacts : bool, optional (default: True)
        Attempt to remove background images or watermarks that overlap
        a detected table area before extracting line structure.
    remove_text : bool, optional (default: False)
        Zero out text glyphs inside candidate table areas before detecting
        lines. This helps preserve solid borders on scans where text pixels
        break the table contours. Text content is still recovered from the
        original PDF objects, so only the raster analysis is affected.
    line_scale : int, optional (default: 15)
        Line size scaling factor. The larger the value the smaller
        the detected lines. Making it very large will lead to text
        being detected as lines.
    copy_text : list, optional (default: None)
        {'h', 'v'}
        Direction in which text in a spanning cell will be copied
        over.
    shift_text : list, optional (default: ['l', 't'])
        {'l', 'r', 't', 'b'}
        Direction in which text in a spanning cell will flow.
    split_text : bool, optional (default: False)
        Split text that spans across multiple cells.
    flag_size : bool, optional (default: False)
        Flag text based on font size. Useful to detect
        super/subscripts. Adds <s></s> around flagged text.
    strip_text : str, optional (default: '')
        Characters that should be stripped from a string before
        assigning it to a cell.
    line_tol : int, optional (default: 2)
        Tolerance parameter used to merge close vertical and horizontal
        lines.
    joint_tol : int, optional (default: 2)
        Tolerance parameter used to decide whether the detected lines
        and points lie close to each other.
    threshold_blocksize : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    threshold_constant : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    iterations : int, optional (default: 0)
        Number of times for erosion/dilation is applied.

        For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.
    backend* : str, optional by default "pdfium"
        The backend to use for converting the PDF to an image so it can be processed by OpenCV.
    use_fallback* : bool, optional
        Fallback to another backend if unavailable, by default True
    resolution : int, optional (default: 300)
        Resolution used for PDF to PNG conversion.

    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        process_background=False,
        remove_background_artifacts=True,
        remove_text=False,
        line_scale=15,
        copy_text=None,
        shift_text=None,
        split_text=False,
        flag_size=False,
        strip_text="",
        line_tol=2,
        joint_tol=2,
        threshold_blocksize=15,
        threshold_constant=-2,
        iterations=0,
        resolution=300,
        use_fallback=True,
        backend="pdfium",
        **kwargs,
    ):
        super().__init__("lattice")
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.process_background = process_background
        self.remove_background_artifacts = remove_background_artifacts
        self.remove_text = remove_text
        self.line_scale = line_scale
        self.copy_text = copy_text
        self.shift_text = shift_text or ["l", "t"]
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.threshold_blocksize = threshold_blocksize
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution
        self.use_fallback = use_fallback
        self.icb = ImageConversionBackend(use_fallback=use_fallback, backend=backend)
        self.image_path = None
        self.pdf_image = None
        self._cleanup_regions: list[dict[str, Any]] | None = None
        self._cleanup_pdf_bboxes: list[tuple[float, float, float, float]] | None = None

    @staticmethod
    def _shift_index(
        table: Any, r_idx: int, c_idx: int, direction: str
    ) -> tuple[int, int]:
        """
        Shift the index based on the specified direction.

        Parameters
        ----------
        table : camelot.core.Table
            The table structure containing rows and columns.
        r_idx : int
            Row index of the cell.
        c_idx : int
            Column index of the cell.
        direction : str
            Direction in which to shift the index ('l', 'r', 't', 'b').

        Returns
        -------
        tuple
            New row and column indices after the shift.
        """
        if direction == "l":
            while c_idx > 0 and not table.cells[r_idx][c_idx].left:
                c_idx -= 1
        elif direction == "r":
            while (
                c_idx < len(table.cells[r_idx]) - 1
                and not table.cells[r_idx][c_idx].right
            ):
                c_idx += 1
        elif direction == "t":
            while r_idx > 0 and not table.cells[r_idx][c_idx].top:
                r_idx -= 1
        elif direction == "b":
            while r_idx < len(table.cells) - 1 and not table.cells[r_idx][c_idx].bottom:
                r_idx += 1

        return r_idx, c_idx

    @staticmethod
    def _reduce_index(
        table: Any, idx: list[tuple[int, int, str]], shift_text: list[str]
    ) -> list[tuple[int, int, str]]:
        """
        Reduces the index of a text object if it lies within a spanning cell.

        Parameters
        ----------
        table : camelot.core.Table
            The table structure containing rows and columns.
        idx : list of tuples
            List of tuples of the form (r_idx, c_idx, text) where r_idx
            is the row index, c_idx is the column index, and text is the
            associated text for that index.
        shift_text : list of str
            A list containing one or more of the following strings:
            {'l', 'r', 't', 'b'} to specify the direction in which the
            text in a spanning cell should flow. 'l' for left, 'r' for right,
            't' for top, 'b' for bottom.

        Returns
        -------
        list of tuples
            List of tuples of the form (r_idx, c_idx, text) where r_idx
            and c_idx are the new row and column indices for the text after
            adjustment.
        """
        indices = []

        for r_idx, c_idx, text in idx:
            # Adjust the index based on specified shift directions
            for direction in shift_text:
                r_idx, c_idx = Lattice._shift_index(table, r_idx, c_idx, direction)

            indices.append((r_idx, c_idx, text))

        return indices

    @staticmethod
    def _rect_from_bbox(
        bbox: tuple[float, float, float, float], width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Convert a bbox expressed as (x1, y1, x2, y2) into a clipped rectangle."""
        x1, y1, x2, y2 = bbox
        xs = sorted((int(round(x1)), int(round(x2))))
        ys = sorted((int(round(y1)), int(round(y2))))

        x_min = max(0, xs[0])
        x_max = min(width, xs[1])
        y_min = max(0, ys[0])
        y_max = min(height, ys[1])

        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            return None
        return (x_min, y_min, w, h)

    @staticmethod
    def _intersect_rectangles(
        lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """Return the intersection between two rectangles, if any."""
        x1 = max(lhs[0], rhs[0])
        y1 = max(lhs[1], rhs[1])
        x2 = min(lhs[0] + lhs[2], rhs[0] + rhs[2])
        y2 = min(lhs[1] + lhs[3], rhs[1] + rhs[3])

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _image_regions(
        self, image_scalers: tuple[float, float, float], width: int, height: int
    ) -> list[dict[str, Any]]:
        """Return image metadata and bounding boxes expressed in image coordinates."""
        regions: list[dict[str, Any]] = []
        for index, image in enumerate(self.images or []):
            bbox = getattr(image, "bbox", None)
            if bbox is None:
                continue
            scaled_bbox = scale_pdf(bbox, image_scalers)
            rect = self._rect_from_bbox(scaled_bbox, width, height)
            if rect is None:
                continue
            regions.append(
                {
                    "index": index,
                    "name": getattr(image, "name", None),
                    "rect": rect,
                    "image_bbox": scaled_bbox,
                    "pdf_bbox": bbox,
                }
            )
        return regions

    @staticmethod
    def _expand_rect(
        rect: tuple[int, int, int, int],
        padding: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        """Expand a rectangle by padding pixels, clamped to image bounds."""
        if padding <= 0:
            return rect
        x, y, w, h = rect
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        if x2 <= x1 or y2 <= y1:
            return rect
        return (x1, y1, x2 - x1, y2 - y1)

    def _text_mask_regions(
        self,
        candidate_regions: list[tuple[int, int, int, int]],
        image_scalers: tuple[float, float, float],
        width: int,
        height: int,
        padding: int = 1,
    ) -> list[tuple[int, int, int, int]]:
        """Collect rectangles that cover text glyphs inside candidate areas."""
        if not candidate_regions:
            return []

        textlines = []
        if self.horizontal_text:
            textlines.extend(self.horizontal_text)
        if self.vertical_text:
            textlines.extend(self.vertical_text)
        if not textlines:
            return []

        mask_regions: list[tuple[int, int, int, int]] = []
        for textline in textlines:
            bbox = getattr(textline, "bbox", None)
            if bbox is None:
                continue
            scaled_bbox = scale_pdf(bbox, image_scalers)
            rect = self._rect_from_bbox(scaled_bbox, width, height)
            if rect is None:
                continue
            expanded = self._expand_rect(rect, padding, width, height)
            for candidate in candidate_regions:
                intersection = self._intersect_rectangles(expanded, candidate)
                if intersection is not None:
                    mask_regions.append(intersection)
        return mask_regions

    @staticmethod
    def _apply_text_mask(
        threshold: np.ndarray, mask_regions: list[tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Zero out pixels inside each text rectangle."""
        if not mask_regions:
            return threshold
        cleaned = threshold.copy()
        height, width = cleaned.shape
        for mask in mask_regions:
            x, y, w, h = mask
            if w <= 0 or h <= 0:
                continue
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            if x2 <= x or y2 <= y:
                continue
            cleaned[y:y2, x:x2] = 0
        return cleaned

    @staticmethod
    def _apply_background_cleanup(
        threshold: np.ndarray,
        line_mask: np.ndarray,
        cleanup_regions: list[dict[str, Any]],
    ) -> np.ndarray:
        """Remove background pixels while preserving structural lines."""
        cleaned = threshold.copy()
        height, width = cleaned.shape
        for region in cleanup_regions:
            x, y, w, h = region["rect"]
            if w <= 0 or h <= 0:
                continue
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            if x2 <= x or y2 <= y:
                continue
            cleaned[y:y2, x:x2] = 0
        return cleaned

    def extract_tables(self):
        tables = super().extract_tables()
        if (
            not tables
            or not self.remove_background_artifacts
            or not self._cleanup_pdf_bboxes
        ):
            return tables

        filtered: list[Any] = []
        for table in tables:
            if self._should_drop_background_table(table):
                continue
            filtered.append(table)
        return filtered

    def _should_drop_background_table(self, table) -> bool:
        bbox = getattr(table, "_bbox", None)
        if bbox is None:
            return False
        if not self._bbox_overlaps_cleanup(bbox):
            return False

        if not getattr(table, "cells", None):
            return True

        total_cells = sum(len(row) for row in table.cells)
        filled_cells = sum(1 for row in table.cells for cell in row if cell.text.strip())
        if filled_cells == 0:
            return True

        if total_cells == 0:
            return True

        # If almost all cells are empty (noise), drop it.
        fill_ratio = filled_cells / total_cells
        if filled_cells <= 2 and fill_ratio < 0.25:
            return True

        distinct_rows = sum(1 for row in table.cells if any(cell.text.strip() for cell in row))
        if distinct_rows <= 1 and fill_ratio < 0.4:
            return True

        return False

    def _bbox_overlaps_cleanup(self, bbox: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = bbox
        for region_bbox in self._cleanup_pdf_bboxes or ():
            rx0, ry0, rx1, ry1 = region_bbox
            if not (x1 <= rx0 or x0 >= rx1 or y1 <= ry0 or y0 >= ry1):
                return True
        return False

    def record_parse_metadata(self, table):
        """Record data about the origin of the table."""
        super().record_parse_metadata(table)
        table.isolated_cells = self._detect_isolated_cells(table)
        # for plotting
        table._image = self.pdf_image  # Reuse the image used for calc
        table._segments = (self.vertical_segments, self.horizontal_segments)

    def _detect_isolated_cells(self, table):
        """Detect rectangles fully disconnected from the parsed grid."""
        bbox = getattr(table, "_bbox", None) or table.bbox
        if bbox is None:
            return []

        table_width = bbox[2] - bbox[0]
        table_height = bbox[3] - bbox[1]
        min_grid_v = table_height * 0.6
        min_grid_h = table_width * 0.6

        grid_x = set()
        grid_y = set()

        tol = max(self.joint_tol, 2)

        def near_grid(coord, anchors):
            return any(abs(coord - anchor) <= tol for anchor in anchors)

        v_s, h_s = segments_in_bbox(
            bbox, self.vertical_segments, self.horizontal_segments
        )

        for vx1, vy1, vx2, vy2 in v_s:
            length = abs(vy2 - vy1)
            if length >= min_grid_v:
                grid_x.add(float(vx1))
        for hx1, hy, hx2, _ in h_s:
            length = abs(hx2 - hx1)
            if length >= min_grid_h:
                grid_y.add(float(hy))

        grid_x.update([bbox[0], bbox[2]])
        grid_y.update([bbox[1], bbox[3]])

        verticals_map: dict[float, tuple[float, float, float, float]] = {}
        for vx1, vy1, vx2, vy2 in v_s:
            x = float(vx1)
            if near_grid(x, grid_x):
                continue
            y_bottom = min(vy1, vy2)
            y_top = max(vy1, vy2)
            key = round(x / tol) * tol
            existing = verticals_map.get(key)
            if existing is None or (existing[3] - existing[1]) < (y_top - y_bottom):
                verticals_map[key] = (x, y_bottom, x, y_top)

        horizontals_map: dict[float, tuple[float, float, float, float]] = {}
        for hx1, hy, hx2, _ in h_s:
            y = float(hy)
            if near_grid(y, grid_y):
                continue
            x_left = min(hx1, hx2)
            x_right = max(hx1, hx2)
            key = round(y / tol) * tol
            existing = horizontals_map.get(key)
            if existing is None or (existing[2] - existing[0]) < (x_right - x_left):
                horizontals_map[key] = (x_left, y, x_right, y)

        verticals = sorted(verticals_map.values(), key=lambda line: line[0])
        horizontals = sorted(horizontals_map.values(), key=lambda line: line[1])

        if not horizontals:
            return []

        min_size = tol * 2
        table_width = bbox[2] - bbox[0]
        rectangles: list[tuple[float, float, float, float]] = []

        # Heuristic 1: closed boxes defined by off-grid vertical + horizontal lines.
        if verticals:
            for idx, left in enumerate(verticals[:-1]):
                lx, ly0, _, ly1 = left
                right = verticals[idx + 1]
                rx, ry0, _, ry1 = right
                if rx - lx <= min_size:
                    continue
                y_min = max(ly0, ry0)
                y_max = min(ly1, ry1)
                if y_max - y_min <= min_size:
                    continue
                candidates = [
                    h
                    for h in horizontals
                    if h[0] <= lx + tol
                    and h[2] >= rx - tol
                    and y_min - tol <= h[1] <= y_max + tol
                ]
                if len(candidates) < 2:
                    continue
                candidates.sort(key=lambda h: h[1])
                for h_idx in range(len(candidates) - 1):
                    bottom = candidates[h_idx]
                    top = candidates[h_idx + 1]
                    if top[1] <= bottom[1] + min_size:
                        continue
                    y_bottom = bottom[1]
                    y_top = top[1]
                    if y_top - y_bottom <= min_size:
                        continue
                    if not (ly0 <= y_bottom + tol and ly1 >= y_top - tol):
                        continue
                    if not (ry0 <= y_bottom + tol and ry1 >= y_top - tol):
                        continue
                    rectangles.append((lx, y_bottom, rx, y_top))

        # Heuristic 2: matched horizontal spans not tied to the grid.
        span_groups: dict[tuple[float, float], list[tuple[float, float, float, float]]] = defaultdict(list)
        for hx1, hy, hx2, _ in horizontals:
            span_width = hx2 - hx1
            if span_width <= min_size:
                continue
            if table_width and span_width >= 0.9 * table_width:
                continue
            if near_grid(hx1, grid_x) or near_grid(hx2, grid_x):
                continue
            key = (round(hx1 / tol) * tol, round(hx2 / tol) * tol)
            span_groups[key].append((hx1, hy, hx2, hy))

        for (gx1, gx2), lines in span_groups.items():
            if len(lines) < 2:
                continue
            lines.sort(key=lambda line: line[1])
            for idx in range(len(lines) - 1):
                y_bottom = lines[idx][1]
                y_top = lines[idx + 1][1]
                if y_top - y_bottom <= min_size:
                    continue
                rectangles.append((gx1, y_bottom, gx2, y_top))

        deduped: list[tuple[float, float, float, float]] = []
        for rect in rectangles:
            if any(
                abs(rect[0] - other[0]) <= tol
                and abs(rect[1] - other[1]) <= tol
                and abs(rect[2] - other[2]) <= tol
                and abs(rect[3] - other[3]) <= tol
                for other in deduped
            ):
                continue
            deduped.append(rect)

        # Remove rectangles that are fully contained in larger ones.
        deduped.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
        filtered: list[tuple[float, float, float, float]] = []
        for rect in deduped:
            if any(
                other[0] - tol <= rect[0] <= other[2] + tol
                and other[1] - tol <= rect[1] <= other[3] + tol
                and other[0] - tol <= rect[2] <= other[2] + tol
                and other[1] - tol <= rect[3] <= other[3] + tol
                for other in filtered
            ):
                continue
            filtered.append(rect)
        deduped = filtered

        results: list[dict[str, Any]] = []
        textlines = getattr(table, "textlines", []) or []
        for rect in deduped:
            x1, y1, x2, y2 = rect
            texts = []
            for tl in textlines:
                cx = (tl.x0 + tl.x1) / 2.0
                cy = (tl.y0 + tl.y1) / 2.0
                if x1 - tol <= cx <= x2 + tol and y1 - tol <= cy <= y2 + tol:
                    text = tl.get_text().strip()
                    if text:
                        texts.append(text)
            results.append({"bbox": rect, "texts": texts})

        return results

    def _generate_table_bbox(self):
        def scale_areas(area_strings):
            scaled_areas = []
            for area in area_strings:
                x1, y1, x2, y2 = (float(coord) for coord in area.split(","))
                scaled_bbox = scale_pdf((x1, y1, x2, y2), image_scalers)
                rect = self._rect_from_bbox(scaled_bbox, image_width, image_height)
                if rect is not None:
                    scaled_areas.append(rect)
            return scaled_areas

        self.image_path = build_file_path_in_temp_dir(
            os.path.basename(self.filename), ".png"
        )
        try:
            self.icb.convert(self.filename, self.image_path)

            self.pdf_image, self.threshold = adaptive_threshold(
                self.image_path,
                process_background=self.process_background,
                blocksize=self.threshold_blocksize,
                c=self.threshold_constant,
            )
        finally:
            Path(self.image_path).unlink(missing_ok=True)
            self.image_path = None

        image_width = self.pdf_image.shape[1]
        image_height = self.pdf_image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        pdf_width_scaler = self.pdf_width / float(image_width)
        pdf_height_scaler = self.pdf_height / float(image_height)
        image_scalers = (image_width_scaler, image_height_scaler, self.pdf_height)
        pdf_scalers = (pdf_width_scaler, pdf_height_scaler, image_height)

        scaled_table_regions = (
            scale_areas(self.table_regions) if self.table_regions is not None else None
        )
        scaled_table_areas = (
            scale_areas(self.table_areas) if self.table_areas is not None else None
        )

        if self.remove_text:
            text_mask_candidates: list[tuple[int, int, int, int]] | None = None
            if scaled_table_areas:
                text_mask_candidates = list(scaled_table_areas)
            elif scaled_table_regions:
                text_mask_candidates = list(scaled_table_regions)
            else:
                text_mask_candidates = [(0, 0, image_width, image_height)]

            text_regions = self._text_mask_regions(
                text_mask_candidates,
                image_scalers,
                image_width,
                image_height,
            )
            if text_regions:
                page_number = getattr(self, "page", None)
                logger.info(
                    "Masking %s text region(s) before lattice parsing (page=%s)",
                    len(text_regions),
                    page_number,
                )
                self.threshold = self._apply_text_mask(self.threshold, text_regions)

        def detect_tables(threshold_img):
            regions_arg = None if scaled_table_areas is not None else scaled_table_regions
            precomputed_mask = (
                apply_region_mask(threshold_img, regions_arg)
                if regions_arg is not None
                else None
            )
            vertical_mask_local, vertical_segments_local = find_lines(
                threshold_img,
                regions=regions_arg,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
                precomputed_mask=precomputed_mask,
            )
            horizontal_mask_local, horizontal_segments_local = find_lines(
                threshold_img,
                regions=regions_arg,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
                precomputed_mask=precomputed_mask,
            )
            if scaled_table_areas is not None:
                table_bbox_local = find_joints(
                    scaled_table_areas, vertical_mask_local, horizontal_mask_local
                )
            else:
                contours = find_contours(vertical_mask_local, horizontal_mask_local)
                table_bbox_local = find_joints(
                    contours, vertical_mask_local, horizontal_mask_local
                )
            return (
                vertical_mask_local,
                vertical_segments_local,
                horizontal_mask_local,
                horizontal_segments_local,
                table_bbox_local,
            )

        (
            vertical_mask,
            vertical_segments,
            horizontal_mask,
            horizontal_segments,
            table_bbox,
        ) = detect_tables(self.threshold)

        cleaned_threshold = self.threshold
        if self.remove_background_artifacts:
            image_regions = self._image_regions(image_scalers, image_width, image_height)
            candidate_regions: list[tuple[int, int, int, int]] = []
            if scaled_table_areas:
                candidate_regions.extend(scaled_table_areas)
            elif table_bbox:
                for bbox in table_bbox:
                    rect = self._rect_from_bbox(bbox, image_width, image_height)
                    if rect is not None:
                        candidate_regions.append(rect)
            elif scaled_table_regions:
                candidate_regions.extend(scaled_table_regions)
            else:
                candidate_regions.append((0, 0, image_width, image_height))

            cleanup_regions: list[dict[str, Any]] = []
            if image_regions and candidate_regions:
                for table_rect in candidate_regions:
                    for image_entry in image_regions:
                        image_rect = image_entry["rect"]
                        intersection = self._intersect_rectangles(
                            table_rect, image_rect
                        )
                        if intersection is not None:
                            cleanup_regions.append(
                                {
                                    "rect": intersection,
                                    "table_rect": table_rect,
                                    "image_index": image_entry["index"],
                                    "image_name": image_entry["name"],
                                    "image_rect": image_rect,
                                    "image_bbox": image_entry["image_bbox"],
                                    "image_pdf_bbox": image_entry["pdf_bbox"],
                                }
                            )

            if cleanup_regions:
                self._cleanup_regions = cleanup_regions
                self._cleanup_pdf_bboxes = [
                    region["image_pdf_bbox"]
                    for region in cleanup_regions
                    if region.get("image_pdf_bbox") is not None
                ]
                page_number = getattr(self, "page", None)
                for region in cleanup_regions:
                    logger.info(
                        "table-p%s-o%s - Masking background image name=%s, intersection=%s",
                        page_number,
                        region["image_index"]+1,
                        region["image_name"] or "<unknown>",
                        region["rect"],
                    )
                line_mask = np.maximum(vertical_mask, horizontal_mask)
                cleaned_threshold = self._apply_background_cleanup(
                    cleaned_threshold, line_mask, cleanup_regions
                )
                (
                    vertical_mask,
                    vertical_segments,
                    horizontal_mask,
                    horizontal_segments,
                    table_bbox,
                ) = detect_tables(cleaned_threshold)
            else:
                self._cleanup_regions = []
                self._cleanup_pdf_bboxes = []

        self.threshold = cleaned_threshold

        [self.table_bbox_parses, self.vertical_segments, self.horizontal_segments] = (
            scale_image(table_bbox, vertical_segments, horizontal_segments, pdf_scalers)
        )

        for bbox, parse in self.table_bbox_parses.items():
            joints = parse["joints"]

            # Merge x coordinates that are close together
            line_tol = self.line_tol
            # Sort the joints, make them a list of lists (instead of sets)
            joints_normalized = list(
                map(lambda x: list(x), sorted(joints, key=lambda j: -j[0]))
            )
            for idx in range(1, len(joints_normalized)):
                x_left, x_right = (
                    joints_normalized[idx - 1][0],
                    joints_normalized[idx][0],
                )
                if x_left - line_tol <= x_right <= x_left + line_tol:
                    joints_normalized[idx][0] = x_left

            # Merge y coordinates that are close together
            joints_normalized = sorted(joints_normalized, key=lambda j: -j[1])
            for idx in range(1, len(joints_normalized)):
                y_bottom, y_top = (
                    joints_normalized[idx - 1][1],
                    joints_normalized[idx][1],
                )
                if y_bottom - line_tol <= y_top <= y_bottom + line_tol:
                    joints_normalized[idx][1] = y_bottom

            # TODO: check this is useful, otherwise get rid of the code
            # above
            parse["joints_normalized"] = joints_normalized

            cols = list(map(lambda coords: coords[0], joints))
            cols.extend([bbox[0], bbox[2]])
            rows = list(map(lambda coords: coords[1], joints))
            rows.extend([bbox[1], bbox[3]])

            # sort horizontal and vertical segments
            cols = merge_close_lines(sorted(cols), line_tol=self.line_tol)
            rows = merge_close_lines(sorted(rows, reverse=True), line_tol=self.line_tol)
            parse["col_anchors"] = cols
            parse["row_anchors"] = rows

    def _generate_columns_and_rows(self, bbox, user_cols):
        def _collapse_anchors(values: list[float], axis_span: float | None) -> list[float]:
            if not values:
                return values
            coords = []
            for val in values:
                try:
                    coords.append(float(val))
                except (TypeError, ValueError):
                    continue
            if len(coords) <= 1:
                return coords
            deltas = [
                abs(coords[idx + 1] - coords[idx])
                for idx in range(len(coords) - 1)
                if coords[idx + 1] != coords[idx]
            ]
            typical_gap = statistics.median(deltas) if deltas else None
            thresholds: list[float] = []
            if axis_span is not None and axis_span > 0:
                thresholds.append(abs(axis_span) * 0.005)
            if typical_gap is not None:
                thresholds.append(typical_gap * 0.3)
            threshold = max(self.line_tol, min(thresholds)) if thresholds else self.line_tol

            collapsed = [coords[0]]
            for coord in coords[1:]:
                if abs(coord - collapsed[-1]) <= threshold:
                    collapsed[-1] = (collapsed[-1] + coord) / 2.0
                else:
                    collapsed.append(coord)
            return collapsed

        # select elements which lie within table_bbox
        v_s, h_s = segments_in_bbox(
            bbox, self.vertical_segments, self.horizontal_segments
        )
        self.t_bbox = text_in_bbox_per_axis(
            bbox, self.horizontal_text, self.vertical_text
        )
        parse = self.table_bbox_parses[bbox]

        x_span = bbox[2] - bbox[0] if bbox and len(bbox) >= 4 else None
        y_span = bbox[3] - bbox[1] if bbox and len(bbox) >= 4 else None
        collapsed_cols = _collapse_anchors(parse["col_anchors"], x_span)
        collapsed_rows = _collapse_anchors(parse["row_anchors"], y_span)
        parse["col_anchors"] = collapsed_cols
        parse["row_anchors"] = collapsed_rows

        cols = [
            (collapsed_cols[i], collapsed_cols[i + 1])
            for i in range(0, len(collapsed_cols) - 1)
        ]
        rows = [
            (collapsed_rows[i], collapsed_rows[i + 1])
            for i in range(0, len(collapsed_rows) - 1)
        ]
        return cols, rows, v_s, h_s

    def _generate_table(self, table_idx, bbox, cols, rows, **kwargs):
        v_s = kwargs.get("v_s")
        h_s = kwargs.get("h_s")
        if v_s is None or h_s is None:
            raise ValueError(f"No segments found on {self.rootname}")

        table = self._initialize_new_table(table_idx, bbox, cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(v_s, h_s, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()

        self.record_parse_metadata(table)
        return table

    def compute_parse_errors(self, table):
        """Compute parse errors while rebuilding cell text from native glyphs."""
        pos_errors = []
        if not self.t_bbox:
            return pos_errors

        for direction in ["vertical", "horizontal"]:
            for textline in self.t_bbox.get(direction, []):
                indices, error = get_table_index(
                    table,
                    textline,
                    direction,
                    split_text=self.split_text,
                    flag_size=self.flag_size,
                    strip_text=self.strip_text,
                )
                if indices and indices[0][:2] != (-1, -1):
                    pos_errors.append(error)

        self._assign_text_to_cells(table)
        return pos_errors

    def _assign_text_to_cells(self, table):
        """Assign text to cells using geometric overlap of characters."""
        if not table.cells:
            return

        for row in table.cells:
            for cell in row:
                cell._text = ""

        if not self.t_bbox:
            return

        cell_segments = defaultdict(list)
        shift_directives = self.shift_text or []

        if self.shift_text == [""]:
            self._assign_text_without_split(table, shift_directives)
            return

        for direction in ("horizontal", "vertical"):
            for textline in self.t_bbox.get(direction, []):
                for (
                    r_idx,
                    c_idx,
                    raw_text,
                    objs,
                    line_break,
                ) in self._split_textline_into_cells(table, textline):
                    if r_idx is None or c_idx is None:
                        continue
                    reduced = type(self)._reduce_index(
                        table, [(r_idx, c_idx, "")], shift_text=shift_directives
                    )
                    if reduced:
                        r_idx, c_idx, _ = reduced[0]
                    filtered_objs = [
                        obj for obj in objs if isinstance(obj, (LTChar, LTAnno))
                    ]
                    cell_segments[(r_idx, c_idx)].append(
                        (direction, raw_text, filtered_objs, line_break)
                    )

        for (r_idx, c_idx), segments in cell_segments.items():
            if not segments:
                continue
            text_chunks: list[tuple[str, bool]] = []
            for direction, raw_text, objs, line_break in segments:
                if self.flag_size and objs:
                    rendered = flag_font_size(
                        objs, direction, strip_text=self.strip_text
                    )
                else:
                    rendered = text_strip(raw_text, self.strip_text)
                segment_text = rendered if rendered is not None else ""
                text_chunks.append((segment_text, line_break))

            combined_parts: list[str] = []
            for chunk, line_break in text_chunks:
                combined_parts.append(chunk)
                if line_break and (not chunk.endswith("\n")):
                    combined_parts.append("\n")
            combined = "".join(combined_parts)
            if combined and combined.strip():
                table.cells[r_idx][c_idx].text = combined

    def _split_textline_into_cells(self, table, textline):
        """Split a textline into table cells based on glyph overlap."""
        fragments: list[tuple[int, int, str, list[LTChar | LTAnno], bool]] = []
        current_cell = None
        buffer_chars: list[str] = []
        buffer_objs: list[LTChar | LTAnno] = []

        for obj in getattr(textline, "_objs", []):
            text = getattr(obj, "get_text", lambda: "")()

            if isinstance(obj, LTAnno):
                if text in ("\n", "\r"):
                    if buffer_chars and current_cell is not None:
                        fragments.append(
                            (
                                current_cell[0],
                                current_cell[1],
                                "".join(buffer_chars),
                                buffer_objs[:],
                                True,
                            )
                        )
                        buffer_chars = []
                        buffer_objs = []
                        current_cell = None
                    elif fragments:
                        # mark previous fragment as ending with a newline
                        prev_r, prev_c, prev_text, prev_objs, _ = fragments[-1]
                        fragments[-1] = (
                            prev_r,
                            prev_c,
                            prev_text,
                            prev_objs,
                            True,
                        )
                    continue
                target_cell = current_cell
            else:
                bbox = getattr(obj, "bbox", None)
                if bbox is None and hasattr(obj, "x0"):
                    bbox = (obj.x0, obj.y0, obj.x1, obj.y1)
                target_cell = self._locate_bbox_cell(table, bbox) if bbox else current_cell

            if target_cell is None:
                continue

            if target_cell != current_cell:
                if buffer_chars and current_cell is not None:
                    fragments.append(
                        (
                            current_cell[0],
                            current_cell[1],
                            "".join(buffer_chars),
                            buffer_objs[:],
                            False,
                        )
                    )
                current_cell = target_cell
                buffer_chars = []
                buffer_objs = []

            if text:
                buffer_chars.append(text)

            if isinstance(obj, (LTChar, LTAnno)):
                buffer_objs.append(obj)

        if buffer_chars and current_cell is not None:
            ends_line = textline.get_text().endswith("\n")
            fragments.append(
                (
                    current_cell[0],
                    current_cell[1],
                    "".join(buffer_chars),
                    buffer_objs[:],
                    ends_line,
                )
            )
        elif fragments and textline.get_text().endswith("\n"):
            prev_r, prev_c, prev_text, prev_objs, _ = fragments[-1]
            fragments[-1] = (
                prev_r,
                prev_c,
                prev_text,
                prev_objs,
                True,
            )
        return fragments

    def _assign_text_without_split(self, table, shift_directives):
        """Fallback assignment replicating legacy behavior without splitting."""
        for direction in ["vertical", "horizontal"]:
            for textline in self.t_bbox.get(direction, []):
                indices, _ = get_table_index(
                    table,
                    textline,
                    direction,
                    split_text=self.split_text,
                    flag_size=self.flag_size,
                    strip_text=self.strip_text,
                )
                if not indices or indices[0][:2] == (-1, -1):
                    continue
                indices = type(self)._reduce_index(
                    table, indices, shift_text=shift_directives
                )
                for r_idx, c_idx, text in indices:
                    if r_idx == -1 or c_idx == -1 or text is None:
                        continue
                    table.cells[r_idx][c_idx].text = text

    def _locate_bbox_cell(self, table, bbox):
        """Return the cell indices whose geometry best matches the bbox."""
        if not bbox or not table.rows or not table.cols:
            return None

        x0, y0, x1, y1 = bbox
        left, right = sorted((x0, x1))
        bottom, top = sorted((y0, y1))
        width = right - left
        height = top - bottom
        area = max(abs(width * height), 1e-6)
        best_cell = None
        best_overlap = 0.0

        for r_idx, (row_top, row_bottom) in enumerate(table.rows):
            y_overlap = min(row_top, top) - max(row_bottom, bottom)
            if y_overlap <= 0:
                continue
            for c_idx, (col_left, col_right) in enumerate(table.cols):
                x_overlap = min(col_right, right) - max(col_left, left)
                if x_overlap <= 0:
                    continue
                overlap_area = x_overlap * y_overlap
                overlap_ratio = overlap_area / area
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cell = (r_idx, c_idx)

        if best_cell is not None and best_overlap > 0:
            return best_cell

        center_x = (left + right) / 2.0
        center_y = (bottom + top) / 2.0
        row_idx = self._row_index_for_y(table, center_y)
        col_idx = self._col_index_for_x(table, center_x)
        if row_idx is None or col_idx is None:
            return None
        return row_idx, col_idx

    @staticmethod
    def _row_index_for_y(table, y_coord):
        if not table.rows:
            return None
        best_idx = None
        best_distance = float("inf")
        for idx, (row_top, row_bottom) in enumerate(table.rows):
            if row_top >= y_coord >= row_bottom:
                return idx
            distance = min(abs(y_coord - row_top), abs(y_coord - row_bottom))
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        return best_idx

    @staticmethod
    def _col_index_for_x(table, x_coord):
        if not table.cols:
            return None
        best_idx = None
        best_distance = float("inf")
        for idx, (col_left, col_right) in enumerate(table.cols):
            if col_left <= x_coord <= col_right:
                return idx
            distance = min(abs(x_coord - col_left), abs(x_coord - col_right))
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        return best_idx
