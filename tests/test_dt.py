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
