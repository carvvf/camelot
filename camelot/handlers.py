"""Functions to handle all operations on the PDF's."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable

from pdfminer.layout import LTChar
from pdfminer.layout import LTImage
from pdfminer.layout import LTTextLineHorizontal
from pdfminer.layout import LTTextLineVertical
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf._utils import StrByteType

from .core import TableList
from .parsers import Autotune
from .parsers import Hybrid
from .parsers import Lattice
from .parsers import Network
from .parsers import Stream
from .utils import TemporaryDirectory
from .utils import download_url
from .utils import get_image_char_and_text_objects
from .utils import get_page_layout
from .utils import get_rotation
from .utils import is_url


logger = logging.getLogger("camelot")


PARSERS = {
    "lattice": Lattice,
    "stream": Stream,
    "network": Network,
    "hybrid": Hybrid,
    "autotune": Autotune,
}


def _parse_page_in_worker(
    handler,
    parser_key,
    page_num,
    tempdir,
    suppress_stdout,
    layout_kwargs,
    parser_kwargs,
):
    """Top-level worker entrypoint to keep multiprocessing pickle-friendly."""
    parser_cls = PARSERS[parser_key]
    parser = parser_cls(debug=handler.debug, **parser_kwargs)
    return handler._parse_page(page_num, tempdir, parser, suppress_stdout, layout_kwargs)


class PDFHandler:
    """Handles all operations on the PDF's.

    Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    debug : bool, optional (default: False)
        Whether the parser should store debug information during parsing.
    """

    def __init__(
        self,
        filepath: StrByteType | Path | str,
        pages="1",
        password=None,
        debug=False,
    ):
        self.debug = debug
        self._downloaded_file: Path | None = None
        if is_url(filepath):
            filepath = download_url(str(filepath))
            self._downloaded_file = Path(filepath)
        self.filepath: StrByteType | Path | str = filepath
        self._reader: PdfReader | None = None

        if isinstance(filepath, str) and not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""  # noqa: S105
        else:
            self.password = password
        # Derive the page list using a lightweight probe, defer holding a PdfReader
        # until parse time to keep the handler picklable for multiprocessing.
        probe_reader = PdfReader(self.filepath, strict=False)
        try:
            if probe_reader.is_encrypted:
                probe_reader.decrypt(self.password)
            self._num_pages = len(probe_reader.pages)
        finally:
            # PdfReader does not expose an explicit close, rely on GC.
            probe_reader = None
        self.pages = self._get_pages(pages)

    def _get_pages(self, pages):
        """Convert pages string to list of integers.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        P : list
            List of int page numbers.

        """
        page_numbers = []

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            infile = PdfReader(self.filepath, strict=False)

            if infile.is_encrypted:
                infile.decrypt(self.password)

            if pages == "all":
                page_numbers.append({"start": 1, "end": len(infile.pages)})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = len(infile.pages)
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})

        result = []
        for p in page_numbers:
            result.extend(range(p["start"], p["end"] + 1))
        return sorted(set(result))

    def _save_page(
        self, filepath: StrByteType | Path, page: int, temp: str, **layout_kwargs
    ) -> tuple[
            Any,
            tuple[float, float],
            list[LTImage],
            list[LTChar],
            list[LTTextLineHorizontal],
            list[LTTextLineVertical],
            int,
            dict[str, dict[str, object]] | None,
        ]:
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        temp : str
            Tmp directory.


        Returns
        -------
        layout : object

        dimensions : tuple
            The dimensions of the pdf page

        filepath : str
            The path of the single page PDF - either the original, or a
            normalized version.

        """
        if self._reader is None:
            self._reader = PdfReader(filepath, strict=False)
            if self._reader.is_encrypted:
                self._reader.decrypt(self.password)
        fpath = os.path.join(temp, f"page-{page}.pdf")
        froot, fext = os.path.splitext(fpath)
        p = self._reader.pages[page - 1]
        rotation_total = 0
        outfile = PdfWriter()

        page_rotation = p.get("/Rotate", 0) or 0
        try:
            page_rotation = int(page_rotation)
        except Exception:  # pragma: no cover - safeguard
            page_rotation = 0
        pdfinfo_page_rotation = page_rotation

        if page_rotation % 360 != 0:
            correction = (-page_rotation) % 360
            if correction > 180:
                correction -= 360
            if correction % 90 != 0:
                raise ValueError(
                    f"Unsupported page rotation angle: {page_rotation} degrees."
                )
            if correction != 0:
                p.rotate(correction)
                rotation_total += correction

        outfile.add_page(p)
        with open(fpath, "wb") as f:
            outfile.write(f)
        layout, dimensions = get_page_layout(fpath, **layout_kwargs)
        # fix rotated PDF
        images, chars, horizontal_text, vertical_text = get_image_char_and_text_objects(
            layout
        )
        rotation = get_rotation(chars, horizontal_text, vertical_text)
        if rotation != "":
            fpath_new = "".join([froot.replace("page", "p"), "_rotated", fext])
            os.rename(fpath, fpath_new)
            instream = open(fpath_new, "rb")
            infile = PdfReader(instream, strict=False)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            outfile = PdfWriter()
            p = infile.pages[0]
            if rotation == "anticlockwise":
                delta = 90
            elif rotation == "clockwise":
                delta = -90
            else:  # pragma: no cover
                delta = 0
            if delta != 0:
                p.rotate(delta)
                rotation_total += delta
            outfile.add_page(p)
            with open(fpath, "wb") as f:
                outfile.write(f)
            # Only recompute layout and dimension after rotating the pdf
            layout, dimensions = get_page_layout(fpath, **layout_kwargs)
            images, chars, horizontal_text, vertical_text = (
                get_image_char_and_text_objects(layout)
            )
            instream.close()
        rotation_ccw = rotation_total % 360
        if rotation_ccw not in (0, 90, 180, 270):
            raise ValueError(
                f"Unsupported normalized rotation angle: {rotation_ccw} degrees."
            )

        def _extract_box_metadata(box_obj):
            try:
                llx, lly = box_obj.lower_left
                urx, ury = box_obj.upper_right
            except Exception:
                return None
            try:
                x0 = float(llx)
                y0 = float(lly)
                x1 = float(urx)
                y1 = float(ury)
            except Exception:
                return None
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            return {
                "origin": (x_min, y_min),
                "size": {"width": float(x_max - x_min), "height": float(y_max - y_min)},
                "bounds": (x_min, y_min, x_max, y_max),
            }

        page_boxes: dict[str, dict[str, object]] | None = None
        media_meta = _extract_box_metadata(getattr(p, "mediabox", None))
        crop_meta = _extract_box_metadata(getattr(p, "cropbox", None))
        if media_meta or crop_meta:
            page_boxes = {}
            if media_meta:
                page_boxes["mediabox"] = media_meta
            if crop_meta:
                page_boxes["cropbox"] = crop_meta

        def _box_size(box_obj):
            try:
                ll = box_obj.lower_left
                ur = box_obj.upper_right
                return float(ur[0] - ll[0]), float(ur[1] - ll[1])
            except Exception:
                return None

        def _effective_dimensions(pdf_file, fallback_dim, text_objs):
            try:
                page_obj = PdfReader(pdf_file, strict=False).pages[0]
            except Exception:
                return fallback_dim

            crop_size = _box_size(getattr(page_obj, "cropbox", None))
            media_size = _box_size(getattr(page_obj, "mediabox", None))
            chosen = crop_size if crop_size and all(v > 0 for v in crop_size) else media_size
            if not chosen or not all(v > 0 for v in chosen):
                return fallback_dim

            try:
                rotation_val = int(page_obj.get("/Rotate", 0) or 0) % 360
            except Exception:
                rotation_val = 0
            width, height = chosen
            if rotation_val in (90, 270):
                width, height = height, width

            if text_objs:
                max_x = max(getattr(t, "x1", 0) for t in text_objs)
                max_y = max(getattr(t, "y1", 0) for t in text_objs)
                tol = 0.05
                if max_x > width * (1 + tol) or max_y > height * (1 + tol):
                    return fallback_dim

            return (width, height)

        text_objs = list(horizontal_text) + list(vertical_text)
        dimensions = _effective_dimensions(fpath, dimensions, text_objs)

        return (
            layout,
            dimensions,
            images,
            chars,
            horizontal_text,
            vertical_text,
            rotation_ccw,
            pdfinfo_page_rotation,
            page_boxes,
        )

    def parse(
        self,
        flavor: str = "lattice",
        suppress_stdout: bool = False,
        parallel: bool = False,
        layout_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Extract tables by calling parser.get_tables on all single page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use.
            Lattice is used by default.
        suppress_stdout : bool (default: False)
            Suppress logs and warnings.
        parallel : bool (default: False)
            Process pages in parallel using all available cpu cores.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        if layout_kwargs is None:
            layout_kwargs = {}

        tables = []
        parser_cls = PARSERS[flavor]
        tempdir_ctx = TemporaryDirectory()
        tempdir = tempdir_ctx.__enter__()
        cleanup_callbacks: list[Callable[[], None]] = [tempdir_ctx.cleanup]
        if self._downloaded_file is not None:
            download_path = Path(self._downloaded_file)
            cleanup_callbacks.append(lambda p=download_path: p.unlink(missing_ok=True))
            # Ensure the handler does not attempt to clean it up again.
            self._downloaded_file = None

        try:
            cpu_count = mp.cpu_count()
            # Using multiprocessing only when cpu_count > 1 to prevent a stallness issue
            # when cpu_count is 1
            if parallel and len(self.pages) > 1 and cpu_count > 1:
                try:
                    with mp.get_context("spawn").Pool(processes=cpu_count) as pool:
                        jobs = []
                        for p in self.pages:
                            j = pool.apply_async(
                                _parse_page_in_worker,
                                (
                                    self,
                                    flavor,
                                    p,
                                    tempdir,
                                    suppress_stdout,
                                    layout_kwargs,
                                    kwargs,
                                ),
                            )
                            jobs.append(j)

                        for j in jobs:
                            t = j.get()
                            tables.extend(t)
                except (OSError, PermissionError) as exc:
                    logger.warning(
                        "Parallel parsing unavailable (%s), falling back to sequential.",
                        exc,
                    )
                    for p in self.pages:
                        t = self._parse_page(
                            p,
                            tempdir,
                            parser_cls(debug=self.debug, **kwargs),
                            suppress_stdout,
                            layout_kwargs,
                        )
                        tables.extend(t)
            else:
                for p in self.pages:
                    t = self._parse_page(
                        p,
                        tempdir,
                        parser_cls(debug=self.debug, **kwargs),
                        suppress_stdout,
                        layout_kwargs,
                    )
                    tables.extend(t)

            return TableList(sorted(tables), cleanup_callbacks=cleanup_callbacks)
        except Exception:
            for cb in cleanup_callbacks:
                try:
                    cb()
                except Exception:
                    pass
            raise

    def _parse_page(
        self, page: int, tempdir: str, parser, suppress_stdout: bool, layout_kwargs
    ):
        """Extract tables by calling parser.get_tables on a single page PDF.

        Parameters
        ----------
        page : int
            Page number to parse
        parser : Lattice, Stream, Network or Hybrid
            The parser to use.
        suppress_stdout : bool
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        (
            layout,
            dimensions,
            images,
            chars,
            horizontal_text,
            vertical_text,
            rotation_angle,
            pdfinfo_page_rotation,
            page_boxes,
        ) = (self._save_page(self.filepath, page, tempdir, **layout_kwargs))
        page_path = os.path.join(tempdir, f"page-{page}.pdf")
        try:
            source_filepath = os.path.abspath(os.fsdecode(self.filepath))
        except (AttributeError, TypeError, ValueError):
            source_filepath = None
        parser.prepare_page_parse(
            page_path,
            layout,
            dimensions,
            page,
            images,
            horizontal_text,
            vertical_text,
            rotation=rotation_angle,
            layout_kwargs=layout_kwargs,
            source_filepath=source_filepath,
            source_page_rotation=pdfinfo_page_rotation,
            page_boxes=page_boxes,
        )
        tables = parser.extract_tables()
        return tables

    def close(self) -> None:
        """Release resources owned by the handler."""
        if self._downloaded_file is not None:
            Path(self._downloaded_file).unlink(missing_ok=True)
            self._downloaded_file = None
        self._reader = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid raising during interpreter shutdown.
            pass
