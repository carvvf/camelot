import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import camelot
from camelot.parsers.base import TextBaseParser
from camelot.parsers.stream import Stream

from .data import *


def test_stream(testdir):
    df = pd.DataFrame(data_stream)

    filename = os.path.join(testdir, "health.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_remove_background_preserves_tables(testdir):
    filename = os.path.join(testdir, "health.pdf")
    tables_default = camelot.read_pdf(filename, flavor="stream")
    assert len(tables_default) == 1

    tables_disabled = camelot.read_pdf(
        filename, flavor="stream", remove_background_artifacts=False
    )
    assert len(tables_disabled) == 1


def test_stream_table_rotated(testdir):
    df = pd.DataFrame(data_stream_table_rotated)

    filename = os.path.join(testdir, "clockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)

    filename = os.path.join(testdir, "anticlockwise_table_2.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_two_tables(testdir):
    df1 = pd.DataFrame(data_stream_two_tables_1)
    df2 = pd.DataFrame(data_stream_two_tables_2)

    filename = os.path.join(testdir, "tabula/12s0324.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")

    assert len(tables) == 2
    assert df1.equals(tables[0].df)
    assert df2.equals(tables[1].df)


def test_stream_table_regions(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_regions=["320,460,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_table_areas(testdir):
    df = pd.DataFrame(data_stream_table_areas)

    filename = os.path.join(testdir, "tabula/us-007.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", table_areas=["320,500,573,335"]
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_columns(testdir):
    df = pd.DataFrame(data_stream_columns)

    filename = os.path.join(testdir, "mexican_towns.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", columns=["67,180,230,425,475"], row_tol=10
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_split_text(testdir):
    df = pd.DataFrame(data_stream_split_text)

    filename = os.path.join(testdir, "tabula/m27.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
        columns=["72,95,209,327,442,529,566,606,683"],
        split_text=True,
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_flag_size(testdir):
    df = pd.DataFrame(data_stream_flag_size)

    filename = os.path.join(testdir, "superscript.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", flag_size=True)
    assert_frame_equal(df, tables[0].df)


def test_stream_strip_text(testdir):
    df = pd.DataFrame(data_stream_strip_text)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", strip_text=" ,\n")
    assert_frame_equal(df, tables[0].df)


def test_stream_edge_tol(testdir):
    df = pd.DataFrame(data_stream_edge_tol)

    filename = os.path.join(testdir, "edge_tol.pdf")
    tables = camelot.read_pdf(filename, flavor="stream", edge_tol=500)
    assert_frame_equal(df, tables[0].df)


def test_stream_layout_kwargs(testdir):
    df = pd.DataFrame(data_stream_layout_kwargs)

    filename = os.path.join(testdir, "detect_vertical_false.pdf")
    tables = camelot.read_pdf(
        filename, flavor="stream", layout_kwargs={"detect_vertical": False}
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_duplicated_text(testdir):
    df = pd.DataFrame(data_stream_duplicated_text)

    filename = os.path.join(testdir, "birdisland.pdf")
    tables = camelot.read_pdf(filename, flavor="stream")
    assert_frame_equal(df, tables[0].df)


def test_stream_inner_outer_columns(testdir):
    df = pd.DataFrame(data_stream_inner_outer_columns)

    filename = os.path.join(testdir, "stream_inner_outer_columns.pdf")
    tables = camelot.read_pdf(
        filename,
        flavor="stream",
    )
    assert_frame_equal(df, tables[0].df)


def test_stream_handles_empty_row_groups():
    rows = TextBaseParser._group_rows([], row_tol=2)
    assert rows == []
    joined = TextBaseParser._join_rows(rows, text_y_max=100, text_y_min=0)
    assert joined == []


def test_stream_handles_empty_textlines_bbox():
    parser = Stream()
    parser.horizontal_text = []
    parser.vertical_text = []

    bbox = (0, 0, 10, 10)
    with pytest.warns(UserWarning, match="No tables found in table area"):
        cols, rows, v_s, h_s = parser._generate_columns_and_rows(bbox, None)

    assert cols == []
    assert rows == []
    assert v_s is None
    assert h_s is None
