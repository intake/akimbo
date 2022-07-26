import numpy as np
import pandas as pd

import awkward_pandas


def test_ufunc(int_series, int_raw_array):
    s = int_series
    s2 = np.add(s, 1)
    assert s2.dtype == "awkward"
    assert s2.tolist() == (int_raw_array + 1).tolist()


def test_select():
    s = pd.Series(awkward_pandas.AwkwardArray([[6, 2, 3], [4, 5]]))
    s2 = s[0]
    assert s2.dtype == "awkward"
    assert isinstance(s2, awkward_pandas.AwkwardArray)
    assert s2.tolist() == [6, 2, 3]

    s2 = s[0:1]
    assert s2.dtype == "awkward"
    assert isinstance(s2.values, awkward_pandas.AwkwardArray)
    assert s2.tolist() == [[6, 2, 3]]


def test_astype_to_ak():
    s = pd.Series([[6, 2, 3], [4, 5]], dtype=object)
    s2 = s.astype("awkward")
    assert s2.dtype == "awkward"
    assert s2.tolist() == [[6, 2, 3], [4, 5]]
    # assert (s2 == s).all()
