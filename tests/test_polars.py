import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("akimbo.polars")


def test_simple():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak[:, -1:]
    assert s2.to_list() == [[3], [], [5]]


def test_apply():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak.apply(np.negative)
    assert s2.to_list() == [[-1, -2, -3], [], [-4, -5]]


def test_apply_where():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.Series(data)
    s2 = s.ak.apply(np.negative, where="a")
    assert s2[0] == {"a": [-1, -2, -3], "b": [1, 2, 3]}


def test_pack_unpack():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.Series(data)
    df = s.ak.unpack()
    assert df["a"].to_list() == [[1, 2, 3]] * 3
    s2 = df.ak.pack()
    assert s.to_list() == s2.to_list()


def test_operator():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak + 1
    assert s2.to_list() == [[2, 3, 4], [], [5, 6]]


def test_ufunc():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = np.negative(s.ak)
    assert s2.to_list() == [[-1, -2, -3], [], [-4, -5]]

    s2 = np.add(s.ak, 1)
    assert s2.to_list() == [[2, 3, 4], [], [5, 6]]

    df = pl.DataFrame({"a": s})
    df2 = df.ak + 1
    assert df2["a"].to_list() == [[2, 3, 4], [], [5, 6]]
