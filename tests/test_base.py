import numpy as np
import pandas as pd

import awkward_pandas


def test_ufunc():
    s = pd.Series(awkward_pandas.AwkwardArray([[1, 2, 3], [4]]))
    s2 = np.add(s, [1])
    assert s2.tolist() == [[2, 3, 4], [5]]
