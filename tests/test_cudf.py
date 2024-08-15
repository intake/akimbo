import pytest

import pyarrow as pa
import awkward as ak

pytest.importorskip("akimbo.cudf")

import cudf


def test_operator_overload():
    s = pa.array([[1, 2, 3], [], [4, 5]], type=pa.list_(pa.int32()))
    series = cudf.Series(s)
    assert ak.backend(series.ak.array) == "cuda"
    s2 = series.ak + 1
    assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [[2, 3, 4], [], [5, 6]]


def test_string_methods():
    s = pa.array([{"s": ["hey", "Ho"], "i": [0]}, {"s": ["Gar", "go"], "i": [2]}],
                 type=pa.struct([("s", pa.list_(pa.string())), ("i", pa.list_(pa.int32()))]))
    series = cudf.Series(s)
    s2 = series.ak.str.upper()
    assert s2.ak.to_list() == [{"s": ["HEY", "HO"], "i": [0]}, {"s": ["GAR", "GO"], "i": [2]}]
