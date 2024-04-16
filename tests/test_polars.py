import numpy as np
import pytest

import awkward_pandas.polars  # noqa: F401

pl = pytest.importorskip("polars")


def test_simple():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak[:, -1:]
    assert s2.to_list() == [[3], [], [5]]


def test_apply():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak.apply(np.negative)
    assert s2.to_list() == [[-1, -2, -3], [], [-4, -5]]


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
