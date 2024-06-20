import datetime

import pytest

import akimbo.pandas  # noqa

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


def test_binary():
    s = pd.Series([[0, 1], [1, 0], [2]])
    s2 = s.ak + 1
    ts1 = s.ak.dt.cast("timestamp[s]")
    ts2 = s2.ak.dt.cast("timestamp[s]")

    out = ts1.ak.dt.nanoseconds_between(ts2)
    assert out.tolist() == [
        [1000000000, 1000000000],
        [1000000000, 1000000000],
        [1000000000],
    ]
    assert str(out.dtype) == "list<item: int64>[pyarrow]"


def test_binary_with_kwargs():
    s = pd.Series([[0, 1], [1, 0], [2]])
    s2 = s.ak + int(24 * 3600 * 7 * 2.5)
    ts1 = s.ak.dt.cast("timestamp[s]")
    ts2 = s2.ak.dt.cast("timestamp[s]")

    out = ts1.ak.dt.weeks_between(ts2, count_from_zero=False, week_start=2)
    assert out.tolist() == [[2, 2], [2, 2], [2]]
    out = ts1.ak.dt.weeks_between(ts2, count_from_zero=False, week_start=5)
    assert out.tolist() == [[3, 3], [3, 3], [3]]
