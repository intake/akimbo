import numpy as np
import pandas as pd

import awkward_pandas


def test_ufunc():
    s = pd.Series(awkward_pandas.AwkwardArray([[6, 2, 3], [4, 5]]))
    s2 = np.add(s, 1)
    assert s2.tolist() == [[7, 3, 4], [5, 6]]


def test_dunder():
    s = pd.Series(awkward_pandas.AwkwardArray([[6, 2, 3], [4, 5]]))
    s2 = s + 1
    assert s2.tolist() == [[7, 3, 4], [5, 6]]
