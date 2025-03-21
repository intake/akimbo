import datetime

import awkward as ak
import pyarrow as pa
import pytest

import akimbo.cudf
import akimbo.io

cudf = pytest.importorskip("cudf")


def test_operator_overload():
    s = pa.array([[1, 2, 3], [], [4, 5]], type=pa.list_(pa.int32()))
    series = cudf.Series(s)
    # assert ak.backend(series.ak.array) == "cuda"
    s2 = series.ak + 1
    # assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [[2, 3, 4], [], [5, 6]]


def test_inner_slicing():
    s = pa.array([[1, 2, 3], [0], [4, 5]], type=pa.list_(pa.int32()))
    series = cudf.Series(s)
    assert ak.backend(series.ak.array) == "cuda"
    s2 = series.ak[:, 0]
    assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [1, 0, 4]
    s2 = series.ak[:, :2]
    assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [[1, 2], [0], [4, 5]]
    s2 = series.ak[:, ::2]
    assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [[1, 3], [0], [4]]


def test_string_methods():
    s = pa.array(
        [{"s": ["hey", "Ho"], "i": [0]}, {"s": ["Gar", "go"], "i": [2]}],
        type=pa.struct([("s", pa.list_(pa.string())), ("i", pa.list_(pa.int32()))]),
    )
    series = cudf.Series(s)
    s2 = series.ak.str.upper()
    assert s2.ak.to_list() == [
        {"s": ["HEY", "HO"], "i": [0]},
        {"s": ["GAR", "GO"], "i": [2]},
    ]

    assert series.ak.str.upper.__doc__
    # kwargs
    s2 = series.ak.str.replace(pat="h", repl="B")
    assert s2.ak.to_list() == [
        {"s": ["Bey", "Ho"], "i": [0]},
        {"s": ["Gar", "go"], "i": [2]},
    ]

    # positional args
    s2 = series.ak.str.replace("h", "B")
    assert s2.ak.to_list() == [
        {"s": ["Bey", "Ho"], "i": [0]},
        {"s": ["Gar", "go"], "i": [2]},
    ]

    # non-str output
    s2 = series.ak.str.len()
    assert s2.ak.to_list() == [{"s": [3, 2], "i": [0]}, {"s": [3, 2], "i": [2]}]


def test_cast():
    s = cudf.Series([0, 1, 2])
    # shows that cast to timestamp needs to be two-step in cudf
    s2 = s.ak.cast("m8[s]").ak.cast("M8[s]")
    out = s2.ak.to_list()
    assert out == [
        datetime.datetime(1970, 1, 1, 0, 0),
        datetime.datetime(1970, 1, 1, 0, 0, 1),
        datetime.datetime(1970, 1, 1, 0, 0, 2),
    ]


def test_times():
    data = [
        datetime.datetime(1970, 1, 1, 0, 0),
        datetime.datetime(1970, 1, 1, 0, 0, 1),
        None,
        datetime.datetime(1970, 1, 1, 0, 0, 2),
    ]
    arr = ak.Array([[data], [], [data]])
    s = akimbo.io.ak_to_series(arr, "cudf")
    s2 = s.ak.dt.second()
    assert s2.ak.to_list() == [[[0, 1, None, 2]], [], [[0, 1, None, 2]]]


def test_dataframe():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = df.ak.array
    assert ak.backend(arr) == "cuda"
    assert arr.fields == ["a", "b"]
