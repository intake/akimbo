import pandas as pd
import pytest
import awkward_pandas


def test_len():
    s = pd.Series(awkward_pandas.AwkwardArray([[6, 2, 3], [4, 5]]))
    assert s.ak.count() == 5
    s2 = s.ak.count(axis=1)
    assert s2.tolist() == [3, 2]


def test_no_access():
    s = pd.Series([1, 2])
    with pytest.raises(Exception):
        s.ak.count()
