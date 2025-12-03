"""Implementation of the command line interface."""

import logging

import click


try:
    import matplotlib.pyplot as plt
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True

import importlib.metadata
from typing import Optional

from . import plot
from . import read_pdf


logger = logging.getLogger("camelot")
logger.setLevel(logging.INFO)


def get_version() -> Optional[str]:
    """Get the version information."""
    try:
        return importlib.metadata.version("camelot-py")
    except importlib.metadata.PackageNotFoundError:
        # Package is not installed in editable mode or in the current environment
        return None


__version__ = get_version() or "0.0.0+unknown"


class Config:
    """Class method for creating a new class."""

    def __init__(self):
        """Initialize the configuration."""
        self.config = {}

    def set_config(self, key, value):
        """Set a configuration value for a given key."""
        self.config[key] = value


pass_config = click.make_pass_decorator(Config)


@click.group(name="camelot")
@click.version_option(version=__version__)
@click.option(
    "-q", "--quiet", is_flag=False, default=False, help="Suppress logs and warnings."
)
@click.option(
    "-p",
    "--pages",
    default="1",
    help="Comma-separated page numbers." " Example: 1,3,4 or 1,4-end or all.",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Read pdf pages in parallel using all CPU cores.",
)
@click.option("-pw", "--password", help="Password for decryption.")
@click.option("-o", "--output", help="Output file path.")
@click.option(
    "-f",
    "--format",
    type=click.Choice([
        "csv",
        "excel",
        "html",
        "json",
        "json-coords",
        "markdown",
        "sqlite",
    ]),
    help="Output file format. Use 'json-coords' for bounding boxes metadata.",
)
@click.option("-z", "--zip", is_flag=True, help="Create ZIP archive.")
@click.option(
    "-split",
    "--split_text",
    is_flag=True,
    help="Split text that spans across multiple cells.",
)
@click.option(
    "--json-coords",
    "json_coords",
    is_flag=True,
    default=False,
    help="Shortcut to force --format json-coords.",
)
@click.option(
    "-flag",
    "--flag_size",
    is_flag=True,
    help="Flag text based on" " font size. Useful to detect super/subscripts.",
)
@click.option(
    "-strip",
    "--strip_text",
    help="Characters that should be stripped from a string before"
    " assigning it to a cell.",
)
@click.option(
    "-M",
    "--margins",
    nargs=3,
    default=(1.0, 0.5, 0.1),
    help="PDFMiner char_margin, line_margin and word_margin.",
)
@click.pass_context
def cli(ctx, *args, **kwargs):
    """Camelot: PDF Table Extraction for Humans.

    ### Custom build extras: ##############################################
    
      • `--format json-coords` exports bounding boxes metadata, and
      flavor details for each detected table.

      • Supplementary QA scripts prefixed `jc-` under `tests/` help visualise
      extraction results when working with the custom distribution.

    See jc-config.debug.env for more options.

    #######################################################################
    """
    ctx.obj = Config()
    for key, value in kwargs.items():
        ctx.obj.set_config(key, value)


@cli.command("lattice")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-back", "--process_background", is_flag=True, help="Process background lines."
)
@click.option(
    "--remove_background_artifacts/--no-remove_background_artifacts",
    default=True,
    help="Remove background images that overlap detected table areas (enabled by default).",
)
@click.option(
    "--remove_text/--no-remove_text",
    default=False,
    help="Mask text glyphs inside candidate table areas before detecting grid lines.",
)
@click.option(
    "-scale",
    "--line_scale",
    default=40,
    help="Line size scaling factor. The larger the value,"
    " the smaller the detected lines.",
)
@click.option(
    "-copy",
    "--copy_text",
    default=[],
    type=click.Choice(["h", "v"]),
    multiple=True,
    help="Direction in which text in a spanning cell" " will be copied over.",
)
@click.option(
    "-shift",
    "--shift_text",
    default=["l", "t"],
    type=click.Choice(["", "l", "r", "t", "b"]),
    multiple=True,
    help="Direction in which text in a spanning cell will flow.",
)
@click.option(
    "-l",
    "--line_tol",
    default=2,
    help="Tolerance parameter used to merge close vertical" " and horizontal lines.",
)
@click.option(
    "-j",
    "--joint_tol",
    default=2,
    help="Tolerance parameter used to decide whether"
    " the detected lines and points lie close to each other.",
)
@click.option(
    "-block",
    "--threshold_blocksize",
    default=15,
    help="For adaptive thresholding, size of a pixel"
    " neighborhood that is used to calculate a threshold value for"
    " the pixel. Example: 3, 5, 7, and so on.",
)
@click.option(
    "-const",
    "--threshold_constant",
    default=-2,
    help="For adaptive thresholding, constant subtracted"
    " from the mean or weighted mean. Normally, it is positive but"
    " may be zero or negative as well.",
)
@click.option(
    "-I",
    "--iterations",
    default=0,
    help="Number of times for erosion/dilation will be applied.",
)
@click.option(
    "-res",
    "--resolution",
    default=300,
    help="Resolution used for PDF to PNG conversion.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "joint", "line"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def lattice(c, *args, **kwargs):
    """Use lines between text to parse the table."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    json_coords = conf.pop("json_coords", False)
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    copy_text = list(kwargs["copy_text"])
    kwargs["copy_text"] = None if not copy_text else copy_text
    kwargs["shift_text"] = list(kwargs["shift_text"])

    if json_coords:
        if f not in (None, "json-coords"):
            raise click.UsageError(
                "--json-coords cannot be combined with a different --format"
            )
        f = "json-coords"

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath, pages=pages, flavor="lattice", suppress_stdout=quiet, **kwargs
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)


@cli.command("autotune")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "--remove_background_artifacts/--no-remove_background_artifacts",
    default=True,
    help="Remove background images that overlap detected table areas (enabled by default).",
)
@click.option(
    "--remove_text/--no-remove_text",
    default=False,
    help="Mask text glyphs inside candidate table areas before detecting grid lines.",
)
@click.option(
    "-scale",
    "--line_scale",
    default=60,
    help="Line size scaling factor. The larger the value,"
    " the smaller the detected lines.",
)
@click.option(
    "-copy",
    "--copy_text",
    default=[],
    type=click.Choice(["h", "v"]),
    multiple=True,
    help="Direction in which text in a spanning cell" " will be copied over.",
)
@click.option(
    "-shift",
    "--shift_text",
    default=["l", "t"],
    type=click.Choice(["", "l", "r", "t", "b"]),
    multiple=True,
    help="Direction in which text in a spanning cell will flow.",
)
@click.option(
    "-l",
    "--line_tol",
    default=2,
    help="Tolerance parameter used to merge close vertical" " and horizontal lines.",
)
@click.option(
    "-j",
    "--joint_tol",
    default=2,
    help="Tolerance parameter used to decide whether"
    " the detected lines and points lie close to each other.",
)
@click.option(
    "-block",
    "--threshold_blocksize",
    default=15,
    help="For adaptive thresholding, size of a pixel"
    " neighborhood that is used to calculate a threshold value for"
    " the pixel. Example: 3, 5, 7, and so on.",
)
@click.option(
    "-const",
    "--threshold_constant",
    default=-2,
    help="For adaptive thresholding, constant subtracted"
    " from the mean or weighted mean. Normally, it is positive but"
    " may be zero or negative as well.",
)
@click.option(
    "-I",
    "--iterations",
    default=0,
    help="Number of times for erosion/dilation will be applied.",
)
@click.option(
    "-res",
    "--resolution",
    default=400,
    help="Resolution used for PDF to PNG conversion.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "joint", "line"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def autotune(c, *args, **kwargs):
    """Autotune: run lattice (with/without background), stream and network, then merge results."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    json_coords = conf.pop("json_coords", False)
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    copy_text = list(kwargs["copy_text"])
    kwargs["copy_text"] = None if not copy_text else copy_text
    kwargs["shift_text"] = list(kwargs["shift_text"])

    if json_coords:
        if f not in (None, "json-coords"):
            raise click.UsageError(
                "--json-coords cannot be combined with a different --format"
            )
        f = "json-coords"

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath,
        pages=pages,
        flavor="autotune",
        suppress_stdout=quiet,
        **kwargs,
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)


@cli.command("stream")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-C",
    "--columns",
    default=[],
    multiple=True,
    help="X coordinates of column separators.",
)
@click.option(
    "-e",
    "--edge_tol",
    default=50,
    help="Tolerance parameter" " for extending textedges vertically.",
)
@click.option(
    "-r",
    "--row_tol",
    default=2,
    help="Tolerance parameter" " used to combine text vertically, to generate rows.",
)
@click.option(
    "-c",
    "--column_tol",
    default=0,
    help="Tolerance parameter"
    " used to combine text horizontally, to generate columns.",
)
@click.option(
    "--remove_background_artifacts/--no-remove_background_artifacts",
    default=True,
    help="Remove background images that overlap detected table areas (enabled by default).",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "textedge"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def stream(c, *args, **kwargs):
    """Use spaces between text to parse the table."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    json_coords = conf.pop("json_coords", False)
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    columns = list(kwargs["columns"])
    kwargs["columns"] = None if not columns else columns

    margins = conf.pop("margins")

    if json_coords:
        if f not in (None, "json-coords"):
            raise click.UsageError(
                "--json-coords cannot be combined with a different --format"
            )
        f = "json-coords"

    if margins is None:
        layout_kwargs = {}
    else:
        layout_kwargs = {
            "char_margin": margins[0],
            "line_margin": margins[1],
            "word_margin": margins[2],
        }

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath,
        pages=pages,
        flavor="stream",
        suppress_stdout=quiet,
        layout_kwargs=layout_kwargs,
        **kwargs,
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)


@cli.command("hybrid")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-C",
    "--columns",
    default=[],
    multiple=True,
    help="X coordinates of column separators.",
)
@click.option(
    "--remove_background_artifacts/--no-remove_background_artifacts",
    default=True,
    help="Remove background images that overlap detected table areas (enabled by default).",
)
@click.option(
    "--remove_text/--no-remove_text",
    default=False,
    help="Mask text glyphs inside candidate table areas before detecting grid lines.",
)
@click.option(
    "-e",
    "--edge_tol",
    default=50,
    help="Tolerance parameter" " for extending textedges vertically.",
)
@click.option(
    "-r",
    "--row_tol",
    default=2,
    help="Tolerance parameter" " used to combine text vertically, to generate rows.",
)
@click.option(
    "-c",
    "--column_tol",
    default=0,
    help="Tolerance parameter"
    " used to combine text horizontally, to generate columns.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "textedge"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def hybrid(c, *args, **kwargs):
    """Combines the strengths of both the Network and the Lattice parser."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    json_coords = conf.pop("json_coords", False)
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    columns = list(kwargs["columns"])
    kwargs["columns"] = None if not columns else columns

    if json_coords:
        if f not in (None, "json-coords"):
            raise click.UsageError(
                "--json-coords cannot be combined with a different --format"
            )
        f = "json-coords"

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath, pages=pages, flavor="hybrid", suppress_stdout=quiet, **kwargs
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)


@cli.command("network")
@click.option(
    "-R",
    "--table_regions",
    default=[],
    multiple=True,
    help="Page regions to analyze. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-T",
    "--table_areas",
    default=[],
    multiple=True,
    help="Table areas to process. Example: x1,y1,x2,y2"
    " where x1, y1 -> left-top and x2, y2 -> right-bottom.",
)
@click.option(
    "-C",
    "--columns",
    default=[],
    multiple=True,
    help="X coordinates of column separators.",
)
@click.option(
    "-e",
    "--edge_tol",
    default=50,
    help="Tolerance parameter" " for extending textedges vertically.",
)
@click.option(
    "-r",
    "--row_tol",
    default=2,
    help="Tolerance parameter" " used to combine text vertically, to generate rows.",
)
@click.option(
    "-c",
    "--column_tol",
    default=0,
    help="Tolerance parameter"
    " used to combine text horizontally, to generate columns.",
)
@click.option(
    "-plot",
    "--plot_type",
    type=click.Choice(["text", "grid", "contour", "textedge"]),
    help="Plot elements found on PDF page for visual debugging.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def network(c, *args, **kwargs):
    """Use text alignments to parse the table."""
    conf = c.config
    pages = conf.pop("pages")
    output = conf.pop("output")
    f = conf.pop("format")
    json_coords = conf.pop("json_coords", False)
    compress = conf.pop("zip")
    quiet = conf.pop("quiet")
    plot_type = kwargs.pop("plot_type")
    filepath = kwargs.pop("filepath")
    kwargs.update(conf)

    table_regions = list(kwargs["table_regions"])
    kwargs["table_regions"] = None if not table_regions else table_regions
    table_areas = list(kwargs["table_areas"])
    kwargs["table_areas"] = None if not table_areas else table_areas
    columns = list(kwargs["columns"])
    kwargs["columns"] = None if not columns else columns

    if json_coords:
        if f not in (None, "json-coords"):
            raise click.UsageError(
                "--json-coords cannot be combined with a different --format"
            )
        f = "json-coords"

    if plot_type is not None:
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")
    else:
        if output is None:
            raise click.UsageError("Please specify output file path using --output")
        if f is None:
            raise click.UsageError("Please specify output file format using --format")

    tables = read_pdf(
        filepath, pages=pages, flavor="network", suppress_stdout=quiet, **kwargs
    )
    click.echo(f"Found {tables.n} tables")
    if plot_type is not None:
        for table in tables:
            plot(table, kind=plot_type)
            plt.show()
    else:
        tables.export(output, f=f, compress=compress)
