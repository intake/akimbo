import awkward._v2 as ak
import pandas as pd

import awkward_pandas


def test_merge_columns():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    )

    new_df = awkward_pandas.merge_columns(df, ["a", "b"], name="d")
    assert (
        ak.zip({"a": df["a"], "b": df["b"]}).tolist() == new_df.d.values._data.tolist()
    )
    cols = list(new_df.columns)
    assert "a" not in cols
    assert "b" not in cols
    assert "c" in cols

    new_df = awkward_pandas.merge_columns(df, ["c", "a"])
    assert (
        ak.zip({"c": df["c"], "a": df["a"]}).tolist() == new_df.ca.values._data.tolist()
    )
    cols = list(new_df.columns)
    assert "a" not in cols
    assert "b" in cols
    assert "c" not in cols
