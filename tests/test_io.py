import os

import numpy as np
import pandas as pd
import pytest
from fsspec.conftest import m  # noqa

import akimbo.pandas


@pytest.mark.parametrize("numba", [True, False])
def test_join(numba):
    pytest.importorskip("numba")
    left = pd.DataFrame({"key": [1, 2, 3, 4, 5, 6, 7, 8], "value": ["a"] * 8})
    right = pd.DataFrame(
        {"key": np.random.randint(1, 9, size=30), "value": ["b"] * 30}
    ).sort_values("key", ignore_index=True)
    out = akimbo.join(left.ak.array, right.ak.array, "key", "match", numba=numba)
    assert len(out[0, "match"]) == (right.key == 1).sum()
    assert all(out[0, "match", "key"] == out[0, "key"])
    assert all(out[0, "match", "value"] == "b")


def test_join_acc():
    pytest.importorskip("numba")
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


def test_read_parquet(m):  # noqa (m is a fixture)
    fn = "memory://file.parquet"
    data = [[1, 2, 3], [], [4, 5]]
    df = pd.DataFrame({"a": pd.Series(data)})
    df.to_parquet(fn)

    out = akimbo.read_parquet(fn)
    meta = akimbo.get_parquet_schema(fn)
    assert meta["columns"] == ["a.list.element"]  # parquet column naming convention
    assert out.columns == ["a"]
    assert out.a.to_list() == data


def test_read_json(m):  # noqa (m is a fixture)
    import json

    fn = "memory://file.json"
    data = [
        {"a": [[1, 2, 3], [], [4, 5]], "b": list("abc")},
        {"a": [[1, 2, 3], [], [4, 5]], "b": list("abc")},
    ]
    with m.open(fn, "wt") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")

    meta = akimbo.io.get_json_schema(fn)
    assert list(meta["properties"]) == ["a", "b"]

    out = akimbo.read_json(fn)
    assert list(out.columns) == ["a", "b"]
    assert out.b.tolist() == [["a", "b", "c"], ["a", "b", "c"]]

    meta["properties"].pop("a")
    out = akimbo.read_json(fn, schema=meta)
    assert list(out.columns) == ["b"]
    assert out.b.tolist() == [["a", "b", "c"], ["a", "b", "c"]]


def test_read_avro():
    fn = os.path.join(os.path.dirname(__file__), "weather.avro")
    data = akimbo.read_avro(fn)
    assert len(data) == 4
    assert list(data.columns) == ["station", "time", "temp"]

    meta = akimbo.io.get_avro_schema(fn)
    assert meta.fields == ["station", "time", "temp"]
