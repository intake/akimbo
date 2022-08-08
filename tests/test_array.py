import pandas as pd

# import awkward._v2 as ak

from awkward_pandas import merge


def test_merge_no_ak():
    df = pd.DataFrame({'a': [1, 2, 3], "b": ["hay", "ho", "hi"],
                       "c": pd.Series(["hay", "ho", "hi"], dtype="string[pyarrow]")})
    s = merge(df)
    assert s.dtype == "awkward"
    assert len(s) == 3
    arr = s.values._data
    assert arr.fields == ["a", "b", "c"]
    assert arr["a"].tolist() == [1, 2, 3]

# def test_merge_one_ak:

# def test_merge_all_ak:
