import numpy as np
import pandas as pd

import akimbo.pandas


def test_join():
    left = pd.DataFrame({"key": [1, 2, 3, 4, 5, 6, 7, 8], "value": ["a"] * 8})
    right = pd.DataFrame(
        {"key": np.random.randint(1, 9, size=30), "value": ["b"] * 30}
    ).sort_values("key", ignore_index=True)
    out = akimbo.join(left.ak.array, right.ak.array, "key", "match")
    assert len(out[0, "match"]) == (right.key == 1).sum()
    assert all(out[0, "match", "key"] == out[0, "key"])
    assert all(out[0, "match", "value"] == "b")


def test_join_acc():
    left = pd.DataFrame({"key": [1, 2, 3, 4, 5, 6, 7, 8], "value": ["a"] * 8})
    right = pd.DataFrame(
        {"key": np.random.randint(1, 9, size=30), "value": ["b"] * 30}
    ).sort_values("key", ignore_index=True)
    out = left.ak.join(right, "key", "match")
    assert len(out.ak[0, "match"]) == (right.key == 1).sum()
    assert all(out.ak[0, "match", "key"] == out.ak[0, "key"])
    assert all(out.ak[0, "match", "value"] == "b")

    right = pd.DataFrame({"key": np.random.randint(1, 9, size=30), "value": ["b"] * 30})
    out = left.ak.join(right, "key", "match", sort=True)
    assert len(out.ak[0, "match"]) == (right.key == 1).sum()
    assert all(out.ak[0, "match", "key"] == out.ak[0, "key"])
    assert all(out.ak[0, "match", "value"] == "b")
