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
