"""Autotune parser: orchestrates lattice, stream and network runs for QA-driven exports."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from typing import Any, Iterable

from .lattice import Lattice
from .network import Network
from .stream import Stream
from ..utils import boundaries_to_split_lines


LINE_BREAK_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z]+")

logger = logging.getLogger("camelot")


class Autotune:
    """Run lattice, stream and network parsers in parallel and merge results."""

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        columns=None,
        flag_size=False,
        split_text=False,
        strip_text="",
        copy_text=None,
        shift_text=None,
        line_tol=2,
        joint_tol=2,
        threshold_blocksize=15,
        threshold_constant=-2,
        iterations=0,
        resolution=400,
        line_scale=60,
        remove_background_artifacts=True,
        remove_text=False,
        edge_tol=None,
        row_tol=2,
        column_tol=0,
        use_fallback=True,
        backend="pdfium",
        debug=False,
        **kwargs,
    ):
        lattice_args: dict[str, Any] = {
            "table_regions": table_regions,
            "table_areas": table_areas,
            "flag_size": flag_size,
            "split_text": split_text,
            "strip_text": strip_text,
            "copy_text": copy_text,
            "shift_text": shift_text,
            "line_tol": line_tol,
            "joint_tol": joint_tol,
            "threshold_blocksize": threshold_blocksize,
            "threshold_constant": threshold_constant,
            "iterations": iterations,
            "resolution": resolution,
            "line_scale": line_scale,
            "remove_background_artifacts": remove_background_artifacts,
            "remove_text": remove_text,
            "use_fallback": use_fallback,
            "backend": backend,
        }
        network_args: dict[str, Any] = {
            "table_regions": table_regions,
            "table_areas": table_areas,
            "columns": columns,
            "flag_size": flag_size,
            "split_text": split_text,
            "strip_text": strip_text,
            "edge_tol": edge_tol,
            "row_tol": row_tol,
            "column_tol": column_tol,
        }
        stream_args: dict[str, Any] = {
            "table_regions": table_regions,
            "table_areas": table_areas,
            "columns": columns,
            "flag_size": flag_size,
            "split_text": split_text,
            "strip_text": strip_text,
            "edge_tol": 50 if edge_tol is None else edge_tol,
            "row_tol": row_tol,
            "column_tol": column_tol,
            "remove_background_artifacts": remove_background_artifacts,
        }
        self.id = "autotune"
        self._parsers = {
            "standard": Lattice(
                process_background=False,
                debug=debug,
                **lattice_args,
            ),
            "process_background": Lattice(
                process_background=True,
                debug=debug,
                **lattice_args,
            ),
            "network": Network(
                debug=debug,
                **network_args,
            ),
            "stream": Stream(
                debug=debug,
                **stream_args,
            ),
        }

    def prepare_page_parse(
        self,
        filename,
        layout,
        dimensions,
        page_idx,
        images,
        horizontal_text,
        vertical_text,
        *,
        rotation: int = 0,
        layout_kwargs,
        source_filepath=None,
        source_page_rotation=None,
        page_boxes=None,
    ):
        """Prepare all autotune parsers for parsing."""
        for parser in self._parsers.values():
            parser.prepare_page_parse(
                filename,
                layout,
                dimensions,
                page_idx,
                images,
                horizontal_text,
                vertical_text,
                rotation=rotation,
                layout_kwargs=layout_kwargs,
                source_filepath=source_filepath,
                source_page_rotation=source_page_rotation,
                page_boxes=page_boxes,
            )

    def _table_sort_key(self, table):
        bbox = getattr(table, "_bbox", None) or getattr(table, "bbox", None)
        y_top = -(bbox[1]) if bbox else 0
        x_left = bbox[0] if bbox else 0
        variant = getattr(table, "autotune_variant", "")
        variant_rank = {
            "standard": 0,
            "process_background": 1,
            "network": 2,
            "stream": 3,
        }.get(variant, 99)
        return (table.page, y_top, x_left, variant_rank)

    @staticmethod
    def _format_table_label(table: Any) -> str:
        page = getattr(table, "page", None)
        order = getattr(table, "order", None)
        variant = getattr(table, "autotune_variant", None) or getattr(
            table, "flavor", None
        )
        label_parts = ["table"]
        if page is not None:
            label_parts.append(f"p{page}")
        if order is not None:
            label_parts.append(f"o{order}")
        if variant:
            label_parts.append(str(variant))
        return "-".join(label_parts)

    @staticmethod
    def _format_metric(value: float | int | None) -> str:
        if value is None:
            return "None"
        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_bbox(bbox: tuple[float, float, float, float] | None) -> str:
        if bbox is None:
            return "None"
        return (
            f"[{Autotune._format_metric(bbox[0])}, {Autotune._format_metric(bbox[1])}, "
            f"{Autotune._format_metric(bbox[2])}, {Autotune._format_metric(bbox[3])}]"
        )

    @staticmethod
    def _log_overlap_skip(
        dropped: Any,
        kept: Any,
        order: int,
        reason: str,
    ) -> None:
        logger.info(
            "Skipping %s due to autotune overlap comparison #%s (%s). kept=%s",
            Autotune._format_table_label(dropped),
            order,
            reason,
            Autotune._format_table_label(kept),
        )

    @staticmethod
    def _log_overlap_merge(
        target: Any,
        source: Any,
        order: int,
        reason: str,
        new_bbox: tuple[float, float, float, float] | None,
    ) -> None:
        logger.info(
            "Applied autotune hybrid merge #%s (%s). target=%s source=%s new_bbox=%s",
            order,
            reason,
            Autotune._format_table_label(target),
            Autotune._format_table_label(source),
            Autotune._format_bbox(new_bbox),
        )

    @staticmethod
    def _log_overlap_drop(
        dropped: Any,
        kept: Any,
        order: int,
        reason: str,
    ) -> None:
        logger.info(
            "Dropping %s after autotune hybrid merge #%s (%s). kept=%s",
            Autotune._format_table_label(dropped),
            order,
            reason,
            Autotune._format_table_label(kept),
        )

    @staticmethod
    def _grid_shape(table: Any) -> tuple[int, int]:
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        return rows, cols

    @staticmethod
    def _nonempty_grid_shape(table: Any) -> tuple[int, int]:
        """Return count of rows/cols that contain at least one nonempty cell."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0, 0

        def _is_nonempty(cell: Any) -> bool:
            text = getattr(cell, "text", "")
            try:
                normalized = " ".join(str(text).split()) if text is not None else ""
            except Exception:
                normalized = ""
            return bool(normalized)

        nonempty_rows = 0
        nonempty_cols: list[bool] = [False] * cols
        for r in range(rows):
            row_has_text = False
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    continue
                if _is_nonempty(cell):
                    row_has_text = True
                    nonempty_cols[c] = True
            if row_has_text:
                nonempty_rows += 1

        nonempty_cols_count = sum(1 for val in nonempty_cols if val)
        return nonempty_rows, nonempty_cols_count

    @staticmethod
    def _distinct_grid_counts(table: Any) -> tuple[int, int]:
        """Return grid size after collapsing duplicate rows/columns by text."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0, 0

        row_signatures: list[tuple[str, ...]] = []

        def _norm(text: Any) -> str:
            try:
                return " ".join(str(text).split()) if text is not None else ""
            except Exception:
                return ""

        for row in cells:
            signature: list[str] = []
            for cell in row:
                text = getattr(cell, "text", "")
                signature.append(_norm(text))
            row_signatures.append(tuple(signature))

        # Preserve order while removing duplicates.
        unique_rows = list(dict.fromkeys(row_signatures))

        col_signatures: list[tuple[str, ...]] = []
        for c_idx in range(cols):
            column_values: list[str] = []
            for row_sig in row_signatures:
                value = row_sig[c_idx] if c_idx < len(row_sig) else ""
                column_values.append(value)
            col_signatures.append(tuple(column_values))
        unique_cols = list(dict.fromkeys(col_signatures))

        return len(unique_rows), len(unique_cols)

    @staticmethod
    def _max_nonempty_per_column(table: Any) -> int:
        """Return max count of nonempty cells across columns."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0

        col_counts: list[int] = [0] * cols
        for r in range(rows):
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    continue
                text = getattr(cell, "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                if normalized:
                    col_counts[c] += 1
        return max(col_counts) if col_counts else 0

    @staticmethod
    def _max_line_breaks_per_column(table: Any) -> int:
        """Return max count of spaced line breaks across columns."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0

        col_breaks: list[int] = [0] * cols
        for r in range(rows):
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    continue
                text = getattr(cell, "text", "")
                try:
                    text_str = str(text) if text is not None else ""
                except Exception:
                    text_str = ""
                breaks = Autotune._spaced_line_break_count(text_str)
                if breaks > 0:
                    col_breaks[c] += breaks
        return max(col_breaks) if col_breaks else 0

    @staticmethod
    def _spaced_line_break_count(
        text: str, *, min_tokens_between: int = 4
    ) -> int:
        """Return line break count only when separated by enough words/numbers."""
        if not text:
            return 0
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in normalized:
            return 0

        segments = normalized.split("\n")
        breaks = 0
        for segment in segments[:-1]:
            tokens = LINE_BREAK_TOKEN_PATTERN.findall(segment)
            if len(tokens) >= min_tokens_between:
                breaks += 1
        return breaks

    @staticmethod
    def _max_nonempty_per_row(table: Any) -> int:
        """Return max count of nonempty cells across rows."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0

        row_counts: list[int] = [0] * rows
        for r in range(rows):
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    continue
                text = getattr(cell, "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                if normalized:
                    row_counts[r] += 1
        return max(row_counts) if row_counts else 0

    @staticmethod
    def _normalized_cell_texts(table: Any) -> list[str]:
        """Return normalized nonempty cell texts for a table."""
        cells = getattr(table, "cells", None) or []
        texts: list[str] = []
        for row in cells:
            for cell in row or []:
                text = getattr(cell, "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                if normalized:
                    texts.append(normalized.lower())
        return texts

    @staticmethod
    def _text_matches_any(text: str, others: list[str]) -> bool:
        """Return True if `text` appears inside any string in `others`."""
        if not text or not others:
            return False
        for other in others:
            if text in other:
                return True
        return False

    @staticmethod
    def _matched_nonempty_counts(
        table: Any, other_texts: list[str]
    ) -> tuple[int, int, int]:
        """Return (rows, cols, max_col_nonempty) counting only cells matching other_texts."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return 0, 0, 0

        row_has: list[bool] = [False] * rows
        col_counts: list[int] = [0] * cols
        for r in range(rows):
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    continue
                text = getattr(cell, "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                if not normalized:
                    continue
                normalized = normalized.lower()
                if not Autotune._text_matches_any(normalized, other_texts):
                    continue
                row_has[r] = True
                col_counts[c] += 1

        matched_rows = sum(1 for flag in row_has if flag)
        matched_cols = sum(1 for count in col_counts if count > 0)
        max_col_nonempty = max(col_counts) if col_counts else 0
        return matched_rows, matched_cols, max_col_nonempty

    @staticmethod
    def _log_pruning_skip(
        table: Any,
        reason: str,
        *,
        rows: int | None,
        cols: int | None,
        accuracy: float | None,
        jc_score: float | None,
    ) -> None:
        logger.info(
            "Skipping %s due to autotune pruning (%s). rows=%s cols=%s accuracy=%s jc_accuracy=%s",
            Autotune._format_table_label(table),
            reason,
            rows if rows is not None else "?",
            cols if cols is not None else "?",
            Autotune._format_metric(accuracy),
            Autotune._format_metric(jc_score),
        )

    @staticmethod
    def _dominant_cell_share(table: Any) -> tuple[float | None, int, int]:
        """Return dominant cell share, total chars and max cell chars from JSON data."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return None, 0, 0
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return None, 0, 0
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None, 0, 0

        lengths: list[int] = []
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
                    text_len = len(str(value))
                except Exception:
                    continue
                if text_len > 0:
                    lengths.append(text_len)

        if not lengths:
            return None, 0, 0

        total_chars = sum(lengths)
        max_chars = max(lengths)
        if total_chars <= 0:
            return None, total_chars, max_chars

        return max_chars / total_chars, total_chars, max_chars

    @staticmethod
    def _compute_uniformity_from_table(table: Any) -> float | None:
        """Compute U = 1 - Gini for character counts per cell."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return None
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None

        lengths: list[int] = []
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
                    text = str(value)
                except Exception:
                    text = None
                length = len(text) if text else 0
                lengths.append(length)

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
        return 1.0 - gini

    @staticmethod
    def _compute_network_text_uniformity(
        table: Any,
    ) -> tuple[float, float, int, int, int] | None:
        """Return (uniformity, avg_len, count, total_chars, max_len) over non-empty cells."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return None
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None

        lengths: list[int] = []
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
        avg_len = total_chars / len(lengths)

        sorted_vals = sorted(lengths)
        n = len(sorted_vals)
        cumulative = 0.0
        for idx, val in enumerate(sorted_vals, start=1):
            cumulative += idx * val
        gini = (2 * cumulative) / (n * total_chars) - (n + 1) / n
        uniformity = 1.0 - gini
        return uniformity, avg_len, len(lengths), total_chars, max(lengths)

    @staticmethod
    def _compute_avg_text_length(
        table: Any,
    ) -> tuple[float, int, int, int] | None:
        """Return (avg_len, count, total_chars, max_len) over non-empty cells."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return None
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None

        lengths: list[int] = []
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
                    normalized = " ".join(str(value).split())
                except Exception:
                    normalized = None
                length = len(normalized) if normalized else 0
                if length > 0:
                    lengths.append(length)

        if not lengths:
            return None

        total_chars = sum(lengths)
        avg_len = total_chars / len(lengths)
        return avg_len, len(lengths), total_chars, max(lengths)

    @staticmethod
    def _detect_list_like_table(table: Any) -> tuple[bool, dict[str, float]]:
        """Detect list-like tables based on left column bullet/number markers."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return False, {}
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return False, {}
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return False, {}

        rows_as_lists: list[list[Any]] = []
        first_row = None
        for row in data:
            if isinstance(row, dict):
                values = [value for _, value in sorted(row.items(), key=lambda item: item[0])]
            elif isinstance(row, (list, tuple)):
                values = list(row)
            else:
                continue
            if first_row is None:
                first_row = values
            rows_as_lists.append(values)

        col_count = len(first_row) if first_row is not None else 0
        if col_count not in {2, 4}:
            return False, {}

        bullet_chars = {"•", "·", "∙", "◦", "●", "○", "■", "□", "▪", "◾", "◼", "-", "–", "—"}
        bullet_regex = re.compile(
            r"""
            ^[\(\[]?                                   # optional opening bracket
            \s*
            (?:                                        # numeric markers require trailing punctuation
                (?P<num>\d{1,3}(?:\.\d{1,3})*)\s*[\.\)\]-]
                |
                (?P<alpha>[ivxlcdmIVXLCDM]+|[a-zA-Z])[\.\)\]-]?  # alpha/roman markers allow optional punctuation
            )
            \s*$
            """,
            re.VERBOSE,
        )

        def _detect_pair(offset: int, label: str) -> tuple[bool, dict[str, float]]:
            total_rows = 0
            bullet_rows = 0
            short_rows = 0
            numbers: list[int] = []
            for values in rows_as_lists:
                if len(values) <= offset:
                    continue
                # Skip rows where both the marker and content cells are empty/whitespace
                try:
                    marker_text_raw = values[offset]
                    content_text_raw = values[offset + 1] if len(values) > offset + 1 else ""
                except Exception:
                    marker_text_raw = ""
                    content_text_raw = ""
                marker_text = ""
                content_text = ""
                try:
                    marker_text = " ".join(str(marker_text_raw).split()) if marker_text_raw is not None else ""
                except Exception:
                    marker_text = ""
                try:
                    content_text = " ".join(str(content_text_raw).split()) if content_text_raw is not None else ""
                except Exception:
                    content_text = ""
                if not marker_text and not content_text:
                    continue
                total_rows += 1
                first_text = marker_text
                first_len = len(first_text)
                if first_len <= 3:
                    short_rows += 1
                is_bullet = False
                if first_text in bullet_chars:
                    is_bullet = True
                else:
                    match = bullet_regex.match(first_text)
                    if match:
                        is_bullet = True
                        num_group = match.group("num") if match.groupdict() else None
                        if num_group is not None:
                            try:
                                num_val = int(num_group)
                                numbers.append(num_val)
                            except Exception:
                                pass
                if not is_bullet and 1 <= first_len <= 3:
                    try:
                        if first_text and not any(ch.isalnum() for ch in first_text):
                            is_bullet = True
                    except Exception:
                        pass
                if is_bullet:
                    bullet_rows += 1

            if total_rows < 3:
                return False, {}

            bullet_ratio = bullet_rows / total_rows if total_rows else 0.0
            short_ratio = short_rows / total_rows if total_rows else 0.0

            seq_ratio = 0.0
            if len(numbers) >= 2:
                seq_hits = 0
                for idx in range(len(numbers) - 1):
                    if numbers[idx + 1] - numbers[idx] in {0, 1}:
                        seq_hits += 1
                seq_ratio = seq_hits / (len(numbers) - 1) if len(numbers) > 1 else 0.0

            detected = (
                bullet_ratio >= 0.4
                and short_ratio >= 0.5
            ) or (bullet_ratio >= 0.4 and seq_ratio >= 0.5)

            return detected, {
                "pair": label,
                "bullet_ratio": bullet_ratio,
                "short_ratio": short_ratio,
                "seq_ratio": seq_ratio,
                "rows": float(total_rows),
            }

        pairs = []
        if col_count == 2:
            pairs.append(_detect_pair(0, "left"))
        elif col_count == 4:
            pairs.append(_detect_pair(0, "left"))
            pairs.append(_detect_pair(2, "right"))

        for detected, metrics in pairs:
            if detected:
                return True, metrics
        return False, {}

    @staticmethod
    def _has_sparse_rows_or_cols(table: Any) -> tuple[bool, dict[str, int]]:
        """Check if every row or every column has text in only one (or none) cell."""
        cells = getattr(table, "cells", None) or []
        if not cells:
            return False, {}
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows == 0 or cols == 0:
            return False, {}

        def _nonempty_count(items):
            count = 0
            for cell in items:
                text = getattr(cell, "text", "")
                if text is None:
                    continue
                if isinstance(text, str):
                    if text.strip():
                        count += 1
                else:
                    try:
                        if str(text).strip():
                            count += 1
                    except Exception:
                        continue
            return count

        row_counts = [_nonempty_count(row) for row in cells]
        col_counts: list[int] = [0] * cols
        for r in range(rows):
            for c in range(cols):
                text = getattr(cells[r][c], "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                if normalized:
                    col_counts[c] += 1

        # ratio of rows/cols with <=1 nonempty cell
        rows_le1 = sum(1 for count in row_counts if count <= 1)
        cols_le1 = sum(1 for count in col_counts if count <= 1)
        row_ratio = rows_le1 / rows if rows else 0.0
        col_ratio = cols_le1 / cols if cols else 0.0

        rows_sparse = max(row_counts) <= 1 or row_ratio >= 0.6
        cols_sparse = max(col_counts) <= 1 or col_ratio >= 0.6

        return rows_sparse or cols_sparse, {
            "rows": rows,
            "cols": cols,
            "rows_nonempty_max": max(row_counts) if row_counts else 0,
            "cols_nonempty_max": max(col_counts) if col_counts else 0,
            "row_ratio_le1": row_ratio,
            "col_ratio_le1": col_ratio,
        }

    @staticmethod
    def _detect_checkerboard_pattern(table: Any) -> tuple[bool, dict[str, float]]:
        """Detect alternating empty/nonempty cells in a checkerboard layout."""
        cells = getattr(table, "cells", None) or []
        rows = len(cells)
        cols = len(cells[0]) if rows and cells and cells[0] else 0
        if rows < 3 or cols < 3:
            return False, {}

        grid: list[list[bool]] = []
        for r in range(rows):
            row_flags: list[bool] = []
            for c in range(cols):
                try:
                    cell = cells[r][c]
                except Exception:
                    row_flags.append(False)
                    continue
                text = getattr(cell, "text", "")
                try:
                    normalized = " ".join(str(text).split()) if text is not None else ""
                except Exception:
                    normalized = ""
                row_flags.append(bool(normalized))
            grid.append(row_flags)

        total_cells = rows * cols
        filled_counts = [sum(row) for row in grid]
        filled_total = sum(filled_counts)
        if total_cells == 0 or filled_total == 0:
            return False, {}

        filled_ratio = filled_total / total_cells
        row_mixed_ratio = sum(1 for count in filled_counts if 0 < count < cols) / rows
        col_counts = [sum(grid[r][c] for r in range(rows)) for c in range(cols)]
        col_mixed_ratio = sum(1 for count in col_counts if 0 < count < rows) / cols

        patterns: list[float] = []
        for offset in (0, 1):
            matches = 0
            for r in range(rows):
                for c in range(cols):
                    expected_filled = ((r + c + offset) % 2) == 0
                    actual_filled = grid[r][c]
                    if actual_filled == expected_filled:
                        matches += 1
            patterns.append(matches / total_cells if total_cells else 0.0)
        best_match = max(patterns) if patterns else 0.0

        detected = (
            0.35 <= filled_ratio <= 0.65
            and row_mixed_ratio >= 0.75
            and col_mixed_ratio >= 0.75
            and best_match >= 0.9
        )
        return detected, {
            "match_ratio": best_match,
            "filled_ratio": filled_ratio,
            "row_mixed_ratio": row_mixed_ratio,
            "col_mixed_ratio": col_mixed_ratio,
        }

    @staticmethod
    def _collect_text_stats(table: Any) -> tuple[Counter[str], int]:
        """Return Counter of normalized cell texts and total characters."""
        to_json = getattr(table, "to_json_payload", None)
        if not callable(to_json):
            return Counter(), 0
        try:
            payload = to_json(include_dataframe=True, include_layout=False)
        except Exception:
            return Counter(), 0
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return Counter(), 0

        counts: Counter[str] = Counter()
        total = 0
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
                    normalized = " ".join(str(value).split())
                except Exception:
                    continue
                if not normalized:
                    continue
                length = len(normalized)
                counts[normalized] += 1
                total += length
        return counts, total

    @staticmethod
    def _count_nonempty_logical_cells(table: Any) -> int:
        """Return count of logical cells with nonempty text (lattice)."""
        try:
            logical_cells = getattr(table, "spanning_cells", None)
        except Exception:
            logical_cells = None
        if not logical_cells:
            return 0

        nonempty = 0
        for cell in logical_cells:
            if not isinstance(cell, dict):
                continue
            text = cell.get("text")
            try:
                normalized = " ".join(str(text).split()) if text is not None else ""
            except Exception:
                normalized = ""
            if normalized:
                nonempty += 1
        return nonempty

    @staticmethod
    def _shared_text_share(
        table_a: Any, table_b: Any
    ) -> tuple[float | None, float | None, float]:
        """Return (share_a, share_b, shared_chars) based on common cell texts."""
        counts_a, total_a = Autotune._collect_text_stats(table_a)
        counts_b, total_b = Autotune._collect_text_stats(table_b)
        if total_a <= 0 or total_b <= 0:
            return None, None, 0.0

        shared_chars = 0
        for text, count_a in counts_a.items():
            count_b = counts_b.get(text)
            if count_b:
                shared_chars += min(count_a, count_b) * len(text)

        share_a = shared_chars / total_a if total_a > 0 else None
        share_b = shared_chars / total_b if total_b > 0 else None
        return share_a, share_b, float(shared_chars)

    @staticmethod
    def _uniformity_margin(flavor_a: str | None, flavor_b: str | None) -> float:
        """Return bonus margin for lattice vs text-based (network/stream) comparisons."""
        flavor_a = (flavor_a or "").lower()
        flavor_b = (flavor_b or "").lower()
        text_flavors = {"network", "stream"}
        if flavor_a == "lattice" and flavor_b in text_flavors:
            return 0.26
        return 0.0

    @staticmethod
    def _augment_boundaries_with_splits(
        boundaries: list[list[float]], splits: list[float] | None, tolerance: float
    ) -> list[list[float]]:
        """Augment existing boundaries using provided hard splits (hybrid-style)."""
        if not splits:
            return boundaries
        idx_boundaries = len(boundaries) - 1
        idx_splits = len(splits) - 1
        previous_boundary = None
        while True:
            if idx_splits < 0:
                break
            split = splits[idx_splits]

            if idx_boundaries < 0:
                new_boundary = [split, boundaries[0][0]]
                boundaries.insert(0, new_boundary)
                idx_splits = idx_splits - 1
            else:
                boundary = boundaries[idx_boundaries]
                if boundary[1] < split + tolerance:
                    boundary[1] = split
                    if previous_boundary is not None:
                        previous_boundary[0] = split
                    idx_splits = idx_splits - 1
                elif boundary[0] > split - tolerance:
                    idx_boundaries = idx_boundaries - 1
                    previous_boundary = boundary
                    if idx_boundaries < 0:
                        boundary[0] = split
                        idx_splits = idx_splits - 1
                else:
                    new_boundary = [split, boundary[1]]
                    boundaries.insert(idx_boundaries + 1, new_boundary)
                    boundary[1] = split
                    previous_boundary = new_boundary
                    idx_splits = idx_splits - 1
        return boundaries

    @staticmethod
    def _extract_bbox(table: Any) -> tuple[float, float, float, float] | None:
        bbox = getattr(table, "_bbox", None) or getattr(table, "bbox", None)
        if bbox is None or len(bbox) != 4:
            return None
        try:
            return tuple(float(coord) for coord in bbox)  # type: ignore[return-value]
        except Exception:
            return None

    @staticmethod
    def _bboxes_close(
        bbox_a: tuple[float, float, float, float] | None,
        bbox_b: tuple[float, float, float, float] | None,
        threshold: float = 0.8,
    ) -> bool:
        if bbox_a is None or bbox_b is None:
            return False
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b
        width_a = max(0.0, x2_a - x1_a)
        height_a = max(0.0, y2_a - y1_a)
        width_b = max(0.0, x2_b - x1_b)
        height_b = max(0.0, y2_b - y1_b)
        area_a = width_a * height_a
        area_b = width_b * height_b
        if area_a <= 0 or area_b <= 0:
            return False

        inter_width = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
        inter_height = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
        inter_area = inter_width * inter_height
        if inter_area <= 0:
            return False

        overlap_a = inter_area / area_a
        overlap_b = inter_area / area_b
        return overlap_a >= threshold and overlap_b >= threshold

    @staticmethod
    def _bboxes_overlap(
        bbox_a: tuple[float, float, float, float] | None,
        bbox_b: tuple[float, float, float, float] | None,
    ) -> bool:
        if bbox_a is None or bbox_b is None:
            return False
        return not (
            bbox_a[2] <= bbox_b[0]
            or bbox_b[2] <= bbox_a[0]
            or bbox_a[3] <= bbox_b[1]
            or bbox_b[3] <= bbox_a[1]
        )

    @staticmethod
    def _overlap_ratios(
        bbox_a: tuple[float, float, float, float] | None,
        bbox_b: tuple[float, float, float, float] | None,
    ) -> tuple[float | None, float | None]:
        """Return overlap ratios (A over B, B over A) for two bboxes."""
        if bbox_a is None or bbox_b is None:
            return None, None
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b
        width_a = max(0.0, x2_a - x1_a)
        height_a = max(0.0, y2_a - y1_a)
        width_b = max(0.0, x2_b - x1_b)
        height_b = max(0.0, y2_b - y1_b)
        area_a = width_a * height_a
        area_b = width_b * height_b
        if area_a <= 0 or area_b <= 0:
            return None, None

        inter_width = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
        inter_height = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
        inter_area = inter_width * inter_height
        if inter_area <= 0:
            return 0.0, 0.0

        return inter_area / area_a, inter_area / area_b

    @staticmethod
    def _compare_contained_tables(
        tables: list[Any],
        *,
        threshold: float = 0.6,
        tolerance: float = 2.0,
        order: int = 200,
    ) -> list[Any]:
        """Drop tables fully contained in a larger one.

        A small tolerance (default 2.0) lets nearly identical boxes count as contained.
        The ``threshold`` argument is retained for compatibility but is ignored.
        """
        if len(tables) <= 1:
            return tables

        def _area(bbox: tuple[float, float, float, float] | None) -> float:
            if bbox is None:
                return 0.0
            x1, y1, x2, y2 = bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        by_page: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, table in enumerate(tables):
            by_page[getattr(table, "page", 0)].append((idx, table))

        to_drop: set[int] = set()
        for page_tables in by_page.values():
            for i in range(len(page_tables)):
                idx_i, table_i = page_tables[i]
                if idx_i in to_drop:
                    continue
                bbox_i = Autotune._extract_bbox(table_i)
                area_i = _area(bbox_i)
                if area_i <= 0:
                    continue
                for j in range(len(page_tables)):
                    if i == j:
                        continue
                    idx_j, table_j = page_tables[j]
                    if idx_j in to_drop:
                        continue
                    bbox_j = Autotune._extract_bbox(table_j)
                    area_j = _area(bbox_j)
                    if area_j <= 0:
                        continue
                    # Is j contained within i?
                    if (
                        bbox_j is not None
                        and bbox_i is not None
                        and bbox_j[0] >= bbox_i[0] - tolerance
                        and bbox_j[1] >= bbox_i[1] - tolerance
                        and bbox_j[2] <= bbox_i[2] + tolerance
                        and bbox_j[3] <= bbox_i[3] + tolerance
                    ):
                        to_drop.add(idx_j)
                        area_share = (
                            (area_j / area_i) * 100 if area_i > 0 else None
                        )
                        area_info = (
                            f" (area share {Autotune._format_metric(area_share)}%)"
                            if area_share is not None
                            else ""
                        )
                        reason = (
                            f"contained within {Autotune._format_table_label(table_i)}"
                            f"{area_info}"
                        )
                        Autotune._log_overlap_skip(table_j, table_i, order, reason)

        return [table for idx, table in enumerate(tables) if idx not in to_drop]

    def _prefer_text_sparse_lattice(
        self,
        tables: list[Any],
        *,
        overlap_threshold: float = 0.7,
        row_ratio_threshold: float = 0.4,
        col_ratio_threshold: float = 0.5,
        order: int = 50,
    ) -> list[Any]:
        """Drop lattice tables overlapping text parsers with much smaller grid."""
        if len(tables) <= 1:
            return tables

        to_drop: set[int] = set()
        by_page: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, table in enumerate(tables):
            by_page[getattr(table, "page", 0)].append((idx, table))

        text_flavors = {"network", "stream"}
        for page_tables in by_page.values():
            lattice_entries = [
                (idx, table)
                for idx, table in page_tables
                if (getattr(table, "flavor", "") or "").lower() == "lattice"
            ]
            text_entries = [
                (idx, table)
                for idx, table in page_tables
                if (getattr(table, "flavor", "") or "").lower() in text_flavors
            ]
            if not lattice_entries or not text_entries:
                continue

            for idx_text, text_table in text_entries:
                text_flavor = (getattr(text_table, "flavor", "") or "").lower()
                text_bbox = Autotune._extract_bbox(text_table)
                if text_bbox is None:
                    continue
                text_rows = Autotune._max_nonempty_per_column(text_table)
                text_cols = Autotune._max_nonempty_per_row(text_table)
                for idx_lat, lattice_table in lattice_entries:
                    if idx_lat in to_drop:
                        continue
                    lat_bbox = Autotune._extract_bbox(lattice_table)
                    lat_rows_base = Autotune._max_nonempty_per_column(lattice_table)
                    lat_breaks = Autotune._max_line_breaks_per_column(lattice_table)
                    lat_rows = lat_rows_base + lat_breaks
                    lat_cols = Autotune._max_nonempty_per_row(lattice_table)
                    if text_rows <= 0 or text_cols <= 0:
                        continue
                    overlap_lat, overlap_text = Autotune._overlap_ratios(
                        lat_bbox, text_bbox
                    )
                    if (
                        overlap_lat is None
                        or overlap_text is None
                        or overlap_lat < overlap_threshold
                        or overlap_text < overlap_threshold
                    ):
                        continue

                    rows_condition = text_rows > 0 and lat_rows <= (text_rows * row_ratio_threshold)
                    cols_condition = text_cols > 0 and lat_cols <= (text_cols * col_ratio_threshold)
                    if not (rows_condition or cols_condition):
                        continue

                    to_drop.add(idx_lat)
                    overlap_info = (
                        f"overlap L {Autotune._format_metric(overlap_lat * 100)}% / "
                        f"{text_flavor} {Autotune._format_metric(overlap_text * 100)}%"
                    )
                    grid_info = (
                        f"grid lattice {lat_rows}(base {lat_rows_base}+breaks {lat_breaks})x{lat_cols} "
                        f"vs {text_flavor} {text_rows}x{text_cols}"
                    )
                    deficit_parts = []
                    if rows_condition:
                        deficit_parts.append("rows<=40%")
                    if cols_condition:
                        deficit_parts.append("cols<=50%")
                    deficit_info = " & ".join(deficit_parts) if deficit_parts else "grid smaller"
                    reason = (
                        f"{overlap_info}; {deficit_info} ({grid_info})"
                    )
                    Autotune._log_overlap_skip(
                        lattice_table, text_table, order, reason
                    )

        return [table for idx, table in enumerate(tables) if idx not in to_drop]

    def _compare_text_overlaps_uniformity(
        self, tables: list[Any], *, text_overlap_threshold: float = 0.8, order: int = 150
    ) -> list[Any]:
        """Drop tables with highly overlapping text using uniformity tie-breaker."""
        if len(tables) <= 1:
            return tables

        uniformity_cache: dict[Any, float | None] = {}

        def _uniformity(table: Any) -> float | None:
            if table not in uniformity_cache:
                uniformity_cache[table] = Autotune._compute_uniformity_from_table(table)
            return uniformity_cache[table]

        by_page: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, table in enumerate(tables):
            by_page[getattr(table, "page", 0)].append((idx, table))

        to_drop: set[int] = set()
        for page_tables in by_page.values():
            for i in range(len(page_tables)):
                idx_i, table_i = page_tables[i]
                if idx_i in to_drop:
                    continue
                for j in range(i + 1, len(page_tables)):
                    idx_j, table_j = page_tables[j]
                    if idx_j in to_drop:
                        continue
                    share_i, share_j, _ = Autotune._shared_text_share(table_i, table_j)
                    if (
                        share_i is None
                        or share_j is None
                        or share_i < text_overlap_threshold
                        or share_j < text_overlap_threshold
                    ):
                        continue
                    ui = _uniformity(table_i)
                    uj = _uniformity(table_j)
                    if ui is None and uj is None:
                        continue
                    flavor_i = getattr(table_i, "flavor", None)
                    flavor_j = getattr(table_j, "flavor", None)
                    margin_i = self._uniformity_margin(flavor_i, flavor_j)
                    margin_j = self._uniformity_margin(flavor_j, flavor_i)
                    adj_ui = ui + margin_i if ui is not None else None
                    adj_uj = uj + margin_j if uj is not None else None

                    def _fmt_adj(value: float | None, margin: float) -> str:
                        if value is None:
                            return "None"
                        base = self._format_metric(value)
                        if margin:
                            return f"{base}+{self._format_metric(margin)}={self._format_metric(value + margin)}"
                        return base

                    keep_first = adj_uj is None or (adj_ui is not None and adj_ui >= (adj_uj - 1e-6))
                    if keep_first:
                        to_drop.add(idx_j)
                        reason = (
                            f"text overlap >= {int(text_overlap_threshold * 100)}% "
                            f"uniformity {_fmt_adj(uj, margin_j)} < {_fmt_adj(ui, margin_i)}"
                        )
                        self._log_overlap_skip(table_j, table_i, order, reason)
                    else:
                        to_drop.add(idx_i)
                        reason = (
                            f"text overlap >= {int(text_overlap_threshold * 100)}% "
                            f"uniformity {_fmt_adj(ui, margin_i)} < {_fmt_adj(uj, margin_j)}"
                        )
                        self._log_overlap_skip(table_i, table_j, order, reason)
                        break

        return [table for idx, table in enumerate(tables) if idx not in to_drop]

    def _compare_overlaps_uniformity(
        self, tables: list[Any], *, overlap_threshold: float = 0.8, order: int = 100
    ) -> list[Any]:
        """Drop overlapping tables with lower uniformity."""
        if len(tables) <= 1:
            return tables

        uniformity_cache: dict[Any, float | None] = {}

        def _uniformity(table: Any) -> float | None:
            if table not in uniformity_cache:
                uniformity_cache[table] = Autotune._compute_uniformity_from_table(table)
            return uniformity_cache[table]

        by_page: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, table in enumerate(tables):
            by_page[getattr(table, "page", 0)].append((idx, table))

        to_drop: set[int] = set()
        for page_tables in by_page.values():
            for i in range(len(page_tables)):
                idx_i, table_i = page_tables[i]
                if idx_i in to_drop:
                    continue
                bbox_i = self._extract_bbox(table_i)
                for j in range(i + 1, len(page_tables)):
                    idx_j, table_j = page_tables[j]
                    if idx_j in to_drop:
                        continue
                    bbox_j = self._extract_bbox(table_j)
                    if not self._bboxes_close(bbox_i, bbox_j, threshold=overlap_threshold):
                        continue
                    ui = _uniformity(table_i)
                    uj = _uniformity(table_j)
                    if ui is None and uj is None:
                        continue
                    flavor_i = getattr(table_i, "flavor", None)
                    flavor_j = getattr(table_j, "flavor", None)
                    margin_i = self._uniformity_margin(flavor_i, flavor_j)
                    margin_j = self._uniformity_margin(flavor_j, flavor_i)
                    adj_ui = ui + margin_i if ui is not None else None
                    adj_uj = uj + margin_j if uj is not None else None

                    def _fmt_adj(value: float | None, margin: float) -> str:
                        if value is None:
                            return "None"
                        base = self._format_metric(value)
                        if margin:
                            return f"{base}+{self._format_metric(margin)}={self._format_metric(value + margin)}"
                        return base

                    keep_first = adj_uj is None or (adj_ui is not None and adj_ui >= (adj_uj - 1e-6))
                    if keep_first:
                        to_drop.add(idx_j)
                        reason = f"uniformity {_fmt_adj(uj, margin_j)} < {_fmt_adj(ui, margin_i)}"
                        self._log_overlap_skip(table_j, table_i, order, reason)
                    else:
                        to_drop.add(idx_i)
                        reason = f"uniformity {_fmt_adj(ui, margin_i)} < {_fmt_adj(uj, margin_j)}"
                        self._log_overlap_skip(table_i, table_j, order, reason)
                        break

        return [table for idx, table in enumerate(tables) if idx not in to_drop]

    def _merge_lattice_into_network(
        self,
        lattice_table: Any,
        network_table: Any,
        *,
        joint_tol: float,
        order: int,
    ) -> bool:
        lattice_bbox = Autotune._extract_bbox(lattice_table)
        network_bbox = Autotune._extract_bbox(network_table)
        if lattice_bbox is None or network_bbox is None:
            return False

        lattice_parse = lattice_table.parse if isinstance(lattice_table.parse, dict) else {}
        network_parse = (
            dict(network_table.parse) if isinstance(network_table.parse, dict) else {}
        )

        lattice_cols = lattice_parse.get("col_anchors")
        network_cols_boundaries = network_parse.get("cols_boundaries")

        augmented_boundaries: list[list[float]] | None = None
        if isinstance(network_cols_boundaries, list) and network_cols_boundaries:
            boundaries_copy: list[list[float]] = []
            for segment in network_cols_boundaries:
                if (
                    isinstance(segment, (list, tuple))
                    and len(segment) == 2
                    and all(isinstance(coord, (int, float)) for coord in segment)
                ):
                    boundaries_copy.append([float(segment[0]), float(segment[1])])
            augmented_boundaries = Autotune._augment_boundaries_with_splits(
                boundaries_copy, lattice_cols if isinstance(lattice_cols, list) else None, joint_tol
            )
            network_parse["cols_boundaries"] = augmented_boundaries
            network_parse["cols_anchors"] = boundaries_to_split_lines(augmented_boundaries)

        if augmented_boundaries:
            new_left = augmented_boundaries[0][0]
            new_right = augmented_boundaries[-1][1]
        else:
            new_left = network_bbox[0]
            new_right = network_bbox[2]

        new_bbox = (
            new_left,
            min(lattice_bbox[1], network_bbox[1]),
            new_right,
            max(lattice_bbox[3], network_bbox[3]),
        )
        network_table._bbox = new_bbox
        network_parse["bbox_full"] = new_bbox
        network_table.parse = network_parse

        Autotune._log_overlap_merge(
            network_table,
            lattice_table,
            order,
            "expanded network bbox with lattice splits",
            new_bbox,
        )
        return True

    def _apply_hybrid_merge(
        self, tables: list[Any], *, order: int = 300
    ) -> list[Any]:
        """Hybrid-style fusion: expand network bbox using overlapping lattice splits."""
        if len(tables) <= 1:
            return tables

        joint_tol = getattr(self._parsers.get("standard"), "joint_tol", 0)
        to_drop: set[int] = set()

        by_page: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for idx, table in enumerate(tables):
            by_page[getattr(table, "page", 0)].append((idx, table))

        for page_tables in by_page.values():
            lattice_entries = [
                (idx, table)
                for idx, table in page_tables
                if (getattr(table, "flavor", "") or "").lower() == "lattice"
            ]
            if not lattice_entries:
                continue

            for idx_network, network_table in page_tables:
                if (getattr(network_table, "flavor", "") or "").lower() != "network":
                    continue
                if idx_network in to_drop:
                    continue
                network_bbox = Autotune._extract_bbox(network_table)
                if network_bbox is None:
                    continue
                net_rows_before, net_cols_before = Autotune._grid_shape(network_table)

                for idx_lattice, lattice_table in lattice_entries:
                    if idx_lattice in to_drop:
                        continue
                    lattice_bbox = Autotune._extract_bbox(lattice_table)
                    if not Autotune._bboxes_overlap(network_bbox, lattice_bbox):
                        continue
                    lat_rows, lat_cols = Autotune._grid_shape(lattice_table)
                    merged = self._merge_lattice_into_network(
                        lattice_table,
                        network_table,
                        joint_tol=joint_tol,
                        order=order,
                    )
                    if merged:
                        net_rows_after, net_cols_after = Autotune._grid_shape(
                            network_table
                        )
                        target_rows = max(net_rows_before, lat_rows)
                        target_cols = max(net_cols_before, lat_cols)
                        improved = (
                            net_rows_after > target_rows or net_cols_after > target_cols
                        )
                        if improved:
                            to_drop.add(idx_lattice)
                            Autotune._log_overlap_drop(
                                lattice_table,
                                network_table,
                                order,
                                (
                                    "removed source lattice after merge; "
                                    f"grid improved from ({net_rows_before},{net_cols_before})/"
                                    f"({lat_rows},{lat_cols}) to ({net_rows_after},{net_cols_after})"
                                ),
                            )
                        else:
                            to_drop.add(idx_network)
                            Autotune._log_overlap_drop(
                                network_table,
                                lattice_table,
                                order,
                                (
                                    "grid not expanded by merge; "
                                    f"network ({net_rows_after},{net_cols_after}) vs "
                                    f"lattice ({lat_rows},{lat_cols})"
                                ),
                            )
                            break
                        network_bbox = Autotune._extract_bbox(network_table)

        return [table for idx, table in enumerate(tables) if idx not in to_drop]

    def _run_overlap_comparisons(self, tables: list[Any]) -> list[Any]:
        """Run ordered overlap comparisons for near-duplicate tables."""
        comparisons = [
            (50, self._prefer_text_sparse_lattice),
            (100, self._compare_overlaps_uniformity),
            (150, self._compare_text_overlaps_uniformity),
            (200, self._compare_contained_tables),
            (300, self._apply_hybrid_merge),
        ]
        for _, comparison in sorted(comparisons, key=lambda entry: entry[0]):
            tables = comparison(tables)
        return tables

    @staticmethod
    def _dedupe_tables(tables: Iterable[Any], tol: float = 0.5):
        def _variant_rank(table: Any) -> int:
            variant = getattr(table, "autotune_variant", "")
            return {
                "standard": 0,
                "process_background": 1,
                "network": 2,
                "stream": 3,
            }.get(variant, 99)

        seen: dict[tuple[int, int, int, int, int], Any] = {}
        unique: list[Any] = []
        for table in tables:
            bbox = getattr(table, "_bbox", None) or getattr(table, "bbox", None)
            if bbox is not None:
                key = (
                    getattr(table, "page", None),
                    round(float(bbox[0]) / tol),
                    round(float(bbox[1]) / tol),
                    round(float(bbox[2]) / tol),
                    round(float(bbox[3]) / tol),
                )
                existing = seen.get(key)
                if existing is None or _variant_rank(table) < _variant_rank(existing):
                    seen[key] = table
                continue
            unique.append(table)
        unique.extend(seen.values())
        return unique

    @staticmethod
    def _passes_pruning(table: Any) -> bool:
        """Apply quality gates for autotune outputs."""
        flavor_value = getattr(table, "flavor", None)
        flavor_lower = flavor_value.lower() if isinstance(flavor_value, str) else ""
        is_lattice = flavor_lower == "lattice"
        is_text_parser = flavor_lower in {"network", "stream"}
        if not (is_lattice or is_text_parser):
            return True
        flavor_label = flavor_lower or "table"

        bbox = Autotune._extract_bbox(table)
        pdf_size = getattr(table, "pdf_size", None)
        if (
            isinstance(pdf_size, (list, tuple))
            and len(pdf_size) == 2
            and bbox is not None
        ):
            width, height = pdf_size
            if (
                isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and width > 0
                and height > 0
            ):
                tol = 2.0
                x0, y0, x1, y1 = bbox
                if (
                    abs(x0) <= tol
                    and abs(y0) <= tol
                    and abs(x1 - width) <= tol
                    and abs(y1 - height) <= tol
                ):
                    Autotune._log_pruning_skip(
                        table,
                        "bbox covers entire page",
                        rows=None,
                        cols=None,
                        accuracy=None,
                        jc_score=None,
                    )
                    return False

        accuracy_raw = getattr(table, "accuracy", None)
        try:
            accuracy = float(accuracy_raw)
        except (TypeError, ValueError):
            accuracy = None
        if isinstance(accuracy, float) and accuracy != accuracy:  # NaN guard
            accuracy = None

        cells = getattr(table, "cells", None) or []
        rows, cols = Autotune._distinct_grid_counts(table)
        min_rows_required = 3 if is_text_parser else 2
        if rows < min_rows_required or cols < 2:
            Autotune._log_pruning_skip(
                table,
                f"requires at least a {min_rows_required}x2 grid",
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=None,
            )
            return False

        if is_lattice:
            nonempty_cells = Autotune._count_nonempty_logical_cells(table)
            if nonempty_cells <= 0:
                text_counts, _ = Autotune._collect_text_stats(table)
                nonempty_cells = sum(text_counts.values())
        else:
            text_counts, _ = Autotune._collect_text_stats(table)
            nonempty_cells = sum(text_counts.values())
        if nonempty_cells < 4:
            Autotune._log_pruning_skip(
                table,
                f"fewer than 4 nonempty cells (count={nonempty_cells})",
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=None,
            )
            return False

        uniformity_thresholds = {
            "lattice": 0.12,
            "network": 0.34,
            "stream": 0.45,
        }
        whitespace_thresholds = {
            "lattice": 75.0,
            "network": 60.0,
            "stream": 45.0,
        }
        accuracy_thresholds = {
            "lattice": 80.0,
            "network": 80.0,
            "stream": 80.0,
        }
        jc_accuracy_thresholds = {
            "lattice": 80.0,
            "network": 80.0,
            "stream": 80.0,
        }
        uniformity_all = Autotune._compute_uniformity_from_table(table)
        threshold_uniformity = uniformity_thresholds.get(flavor_lower)
        try:
            whitespace = float(getattr(table, "whitespace", 0))
        except (TypeError, ValueError):
            whitespace = None
        whitespace_threshold = whitespace_thresholds.get(flavor_lower)
        accuracy_threshold = accuracy_thresholds.get(flavor_lower)
        jc_threshold = jc_accuracy_thresholds.get(flavor_lower)
        try:
            layout_payload = table.to_structured_layout()
        except Exception:
            layout_payload = None
        indicators = layout_payload.get("indicators") if isinstance(layout_payload, dict) else None
        jc_indicator = indicators.get("jc_accuracy") if isinstance(indicators, dict) else None
        jc_score = jc_indicator.get("score") if isinstance(jc_indicator, dict) else None
        try:
            jc_score_value = float(jc_score)
        except (TypeError, ValueError):
            jc_score_value = None

        whitespace_high = (
            whitespace is not None
            and whitespace_threshold is not None
            and whitespace > whitespace_threshold
        )
        accuracy_low = (
            accuracy_threshold is not None
            and accuracy_threshold > 0
            and accuracy is not None
            and accuracy < accuracy_threshold
        )
        jc_low = (
            jc_threshold is not None
            and jc_threshold > 0
            and jc_score_value is not None
            and jc_score_value < jc_threshold
        )

        if (
            threshold_uniformity is not None
            and uniformity_all is not None
            and uniformity_all < threshold_uniformity
            and (whitespace_high or accuracy_low or jc_low)
        ):
            Autotune._log_pruning_skip(
                table,
                (
                    "uniformity "
                    f"{Autotune._format_metric(uniformity_all)} < "
                    f"{Autotune._format_metric(threshold_uniformity)} "
                    f"and (whitespace {Autotune._format_metric(whitespace)}% > "
                    f"{Autotune._format_metric(whitespace_threshold)}% "
                    f"or accuracy {Autotune._format_metric(accuracy)}% < "
                    f"{Autotune._format_metric(accuracy_threshold)} "
                    f"or jc_accuracy {Autotune._format_metric(jc_score_value)}% < "
                    f"{Autotune._format_metric(jc_threshold)})"
                ),
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=jc_score_value,
            )
            return False

        dense_text_stats = Autotune._compute_network_text_uniformity(table)
        if dense_text_stats is not None:
            uniformity, avg_len, count, total_chars, max_len = dense_text_stats
            if avg_len >= 22 and uniformity >= 0.55:
                Autotune._log_pruning_skip(
                    table,
                    (
                        f"{flavor_label} text appears dense (avg chars "
                        f"{Autotune._format_metric(avg_len)}, U={Autotune._format_metric(uniformity)}); "
                        "likely paragraph"
                    ),
                    rows=rows,
                    cols=cols,
                    accuracy=accuracy,
                    jc_score=None,
                )
                return False

        if is_text_parser:
            list_detected, list_metrics = Autotune._detect_list_like_table(table)
            if list_detected:
                Autotune._log_pruning_skip(
                    table,
                    (
                        f"{list_metrics.get('pair','left')} column looks like list markers "
                        f"(bullet_ratio={Autotune._format_metric(list_metrics.get('bullet_ratio'))}, "
                        f"short_ratio={Autotune._format_metric(list_metrics.get('short_ratio'))}, "
                        f"seq_ratio={Autotune._format_metric(list_metrics.get('seq_ratio'))})"
                    ),
                    rows=rows,
                    cols=cols,
                    accuracy=accuracy,
                    jc_score=None,
                )
                return False
            sparse_detected, sparse_metrics = Autotune._has_sparse_rows_or_cols(table)
            if sparse_detected:
                Autotune._log_pruning_skip(
                    table,
                    (
                        f"{flavor_label} table has rows/cols with <=1 nonempty cell "
                        f"(rows_nonempty_max={sparse_metrics.get('rows_nonempty_max')}, "
                        f"cols_nonempty_max={sparse_metrics.get('cols_nonempty_max')}, "
                        f"row_ratio_le1={Autotune._format_metric(sparse_metrics.get('row_ratio_le1'))}, "
                        f"col_ratio_le1={Autotune._format_metric(sparse_metrics.get('col_ratio_le1'))})"
                    ),
                    rows=rows,
                    cols=cols,
                    accuracy=accuracy,
                    jc_score=None,
                )
                return False

            checkerboard_detected, checkerboard_metrics = Autotune._detect_checkerboard_pattern(
                table
            )
            if checkerboard_detected:
                Autotune._log_pruning_skip(
                    table,
                    (
                        f"{flavor_label} table alternates empty/full cells like a checkerboard "
                        f"(match={Autotune._format_metric(checkerboard_metrics.get('match_ratio'))}, "
                        f"filled_ratio={Autotune._format_metric(checkerboard_metrics.get('filled_ratio'))})"
                    ),
                    rows=rows,
                    cols=cols,
                    accuracy=accuracy,
                    jc_score=None,
                )
                return False

        avg_text_stats = Autotune._compute_avg_text_length(table)
        if avg_text_stats is not None:
            avg_len, count_cells, total_chars_avg, max_len_avg = avg_text_stats
            if avg_len >= 200:
                Autotune._log_pruning_skip(
                    table,
                    (
                        "average cell text length "
                        f"{Autotune._format_metric(avg_len)} >= 200 chars "
                        "(likely paragraph)"
                    ),
                    rows=rows,
                    cols=cols,
                    accuracy=accuracy,
                    jc_score=None,
                )
                return False

        dominant_share, total_chars, max_chars = Autotune._dominant_cell_share(table)
        if dominant_share is not None and dominant_share >= 0.85:
            percent = Autotune._format_metric(dominant_share * 100)
            Autotune._log_pruning_skip(
                table,
                f"cell text dominates {percent}% of total (max_cell={max_chars}, total={total_chars})",
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=None,
            )
            return False

        layout = layout_payload
        if layout is None:
            try:
                layout = table.to_structured_layout()
            except Exception:
                return False
        if jc_score_value is None:
            indicators = layout.get("indicators") if isinstance(layout, dict) else None
            jc_indicator = indicators.get("jc_accuracy") if isinstance(indicators, dict) else None
            jc_score = jc_indicator.get("score") if isinstance(jc_indicator, dict) else None
            try:
                jc_score_value = float(jc_score)
            except (TypeError, ValueError):
                jc_score_value = None
        if isinstance(jc_score_value, float) and jc_score_value != jc_score_value:
            jc_score_value = None

        accuracy_under_min = accuracy is None or accuracy < 1
        jc_under_min = jc_score_value is None or jc_score_value < 1
        if accuracy_under_min or jc_under_min:
            Autotune._log_pruning_skip(
                table,
                "accuracy or jc_accuracy missing or < 1",
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=jc_score_value,
            )
            return False

        has_accuracy_90 = accuracy is not None and accuracy >= 90
        has_jc_score_90 = jc_score_value is not None and jc_score_value >= 90
        if has_accuracy_90 or has_jc_score_90:
            return True

        has_accuracy_80 = accuracy is not None and accuracy >= 80
        has_jc_score_80 = jc_score_value is not None and jc_score_value >= 80
        if has_accuracy_80 or has_jc_score_80:
            uniformity_info = Autotune._compute_network_text_uniformity(table)
            uniformity_value = uniformity_info[0] if uniformity_info else None
            if uniformity_value is not None and uniformity_value >= 0.45:
                return True
            Autotune._log_pruning_skip(
                table,
                (
                    "accuracy or jc_accuracy >= 80 but uniformity "
                    f"{Autotune._format_metric(uniformity_value)} < 0.45"
                ),
                rows=rows,
                cols=cols,
                accuracy=accuracy,
                jc_score=jc_score_value,
            )
            return False

        Autotune._log_pruning_skip(
            table,
            "accuracy and jc_accuracy below thresholds (80/90 gates)",
            rows=rows,
            cols=cols,
            accuracy=accuracy,
            jc_score=jc_score_value,
        )
        return False

    @staticmethod
    def _annotate_table(table, variant: str, flavor: str):
        table.autotune_variant = variant
        parse_info = table.parse if isinstance(table.parse, dict) else {}
        if isinstance(parse_info, dict):
            updated_parse = dict(parse_info)
            updated_parse["autotune_variant"] = variant
            table.parse = updated_parse

    def extract_tables(self):
        """Run all autotune parsers and merge the outputs."""
        results: list[Any] = []
        with ThreadPoolExecutor(max_workers=len(self._parsers)) as executor:
            futures = {
                executor.submit(parser.extract_tables): name
                for name, parser in self._parsers.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                tables = future.result()
                for table in tables:
                    self._annotate_table(table, name, self.id)
                results.extend(tables)

        merged = [
            table
            for table in self._dedupe_tables(results)
            if self._passes_pruning(table)
        ]
        merged = self._run_overlap_comparisons(merged)
        # Temporarily disable pruning and overlap comparisons.
        #merged = list(results)

        by_page: dict[int, list[Any]] = defaultdict(list)
        for table in merged:
            by_page[getattr(table, "page", 0)].append(table)
        for page_tables in by_page.values():
            page_tables.sort(key=self._table_sort_key)
            for idx, table in enumerate(page_tables, start=1):
                table.order = idx

        return merged
