import pandas as pd

import awkward_pandas


def test1():
    s = pd.Series(["hello", "world"], dtype='awkward')
    out = s.ak.upper()
    assert out.tolist() == ["HELLO", "WORLD"]
