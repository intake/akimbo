import pytest

import awkward as ak

pytest.importorskip("awkward_pandas.cudf")


def test_operator_overload():
    import cudf
    s = [[1, 2, 3], [], [4, 5]]
    series = cudf.Series(s)
    assert ak.backend(series.ak.array) == "cuda"
    s2 = series.ak + 1
    assert ak.backend(s2.ak.array) == "cuda"
    assert isinstance(s2, cudf.Series)
    assert s2.ak.to_list() == [[2, 3, 4], [], [5, 6]]

