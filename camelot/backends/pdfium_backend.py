"""Creates a pdfium backend class to convert a pdf to a png file."""

from camelot.backends.base import ConversionBackend


PDFIUM_EXC = None

try:
    import pypdfium2 as pdfium

except ModuleNotFoundError as e:
    PDFIUM_EXC = e


class PdfiumBackend(ConversionBackend):
    """Classmethod to create PdfiumBackend."""

    def installed(self) -> bool:  # noqa D102
        if not PDFIUM_EXC:
            return True
        return False

    def convert(self, pdf_path: str, png_path: str, resolution: int = 300) -> None:
        """Convert PDF to png.

        Parameters
        ----------
        pdf_path : str
            Path where to read the pdf file.
        png_path : str
            Path where to save png file.

        Raises
        ------
        OSError
            Raise an error if pdfium is not installed
        """
        if not self.installed():
            raise OSError(f"pypdfium2 is not available: {PDFIUM_EXC!r}")
        doc = pdfium.PdfDocument(pdf_path)
        page = None
        image = None
        try:
            doc.init_forms()
            page = doc[0]
            image = page.render(scale=resolution / 72).to_pil()
            image.save(png_path)
        finally:
            if image is not None:
                image.close()
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
            try:
                doc.close()
            except Exception:
                pass
