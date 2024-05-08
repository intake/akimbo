import datetime

import pytest

import awkward_pandas.pandas  # noqa

pd = pytest.importorskip("pandas")


def test_cast():
    s = pd.Series([[0, 1], [1, 0], [2]])
    out = s.ak.dt.cast("timestamp[s]")
    assert str(out.dtype) == "list<item: timestamp[s]>[pyarrow]"
    assert out.to_list() == [
        [datetime.datetime(1970, 1, 1, 0, 0), datetime.datetime(1970, 1, 1, 0, 0, 1)],
        [datetime.datetime(1970, 1, 1, 0, 0, 1), datetime.datetime(1970, 1, 1, 0, 0)],
        [datetime.datetime(1970, 1, 1, 0, 0, 2)],
    ]


def test_unary_unit():
    s = pd.Series([[0, 1], [1, 0], [2]])
    ts = s.ak.dt.cast("timestamp[s]")
    s2 = ts.ak.dt.second()
    assert s.to_list() == s2.to_list()


def test_bad_type():
    # consider more specific exception rather than hitting arrow's one
    s = pd.Series([[0, 1], [1, 0], [2]])
    with pytest.raises(NotImplementedError):
        s.ak.dt.second()
