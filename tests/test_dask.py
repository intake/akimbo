import awkward as ak
import pandas as pd
import pytest

import awkward_pandas  # noqa

dd = pytest.importorskip("dask.dataframe")


def test_simple_map():
    data = [[0], [0, 1]] * 2
    s = pd.Series(data, dtype="awkward")
    df = pd.DataFrame({"s": s})
    ddf = dd.from_pandas(df, 2)
    out = ddf.s.map(ak.count)
    assert out.dtype == "int64"
    assert out.compute().tolist() == [1, 2, 1, 2]

    out = ddf + 1
    assert out.s.dtype == "awkward"
    assert out.compute().s.tolist() == [[1], [1, 2]] * 2


def test_accessor():
    data = [[0], [0, 1]] * 2
    s = pd.Series(data, dtype="awkward")
    df = pd.DataFrame({"s": s})
    ddf = dd.from_pandas(df, 2)
    out = ddf.s.ak.count()
    assert out.compute().tolist() == [3, 3]

    out = ddf.s.ak.count(axis=1).compute()
    assert out.tolist() == [1, 2, 1, 2]


def test_distributed():
    distributed = pytest.importorskip("distributed")
    with distributed.Client(n_workers=1, threads_per_worker=1):
        data = [[0], [0, 1]] * 2
        s = pd.Series(data, dtype="awkward")
        df = pd.DataFrame({"s": s})
        ddf = dd.from_pandas(df, 2)
        out = ddf.s.ak.count()
        assert out.compute().tolist() == [3, 3]
        out = ddf.s.ak.count(axis=1).compute()
        assert out.tolist() == [1, 2, 1, 2]
