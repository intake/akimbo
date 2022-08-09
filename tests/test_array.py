import pandas as pd

import awkward._v2 as ak

from awkward_pandas import merge


def test_merge_no_ak():
    df = pd.DataFrame({'a': [1, 2, 3], "b": ["hay", "ho", "hi"],
                       "c": pd.Series(["hay", "ho", "hi"], dtype="string[pyarrow]"),
                       "d": [[1, 2, 3], None, []]})
    s = merge(df, name='test')
    assert s.name == "test"
    assert s.dtype == "awkward"
    assert len(s) == 3
    arr = s.values._data
    assert arr.fields == ["a", "b", "c", "d"]
    assert arr["a"].tolist() == [1, 2, 3]
    assert arr["b"].tolist() == ["hay", "ho", "hi"]
    assert arr["c"].tolist() == ["hay", "ho", "hi"]
    assert arr["d"].tolist() == [[1, 2, 3], None, []]

# def test_merge_one_ak:

# def test_merge_all_ak:
