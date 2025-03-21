import pandas as pd
import pyarrow as pa
import pytest

dd = pytest.importorskip("dask.dataframe")

import akimbo.dask  # noqa


def test_simple_map():
    data = pd.arrays.ArrowExtensionArray(pa.array([[0], [0, 1]] * 2))
    s = pd.Series(data)
    df = pd.DataFrame({"s": s})
    ddf = dd.from_pandas(df, 2)
    out = ddf.s.ak.count(axis=0)
    assert "int64" in str(out.dtype)
    result = out.compute(scheduler="sync")
    assert set(result) == {1, 2}

    out = ddf.s.ak + 1
    assert out.compute(scheduler="sync").ak.to_list() == [[1], [1, 2]] * 2


def test_accessor():
    data = pd.arrays.ArrowExtensionArray(pa.array([[0], [0, 1]] * 2))
    s = pd.Series(data)
    df = pd.DataFrame({"s": s})
    ddf = dd.from_pandas(df, 2)
    out = ddf.s.ak.count()  # causes dask warning, as each partition reduces to scalar
    assert out.compute().tolist() == [3, 3]

    out = ddf.s.ak.count(axis=1).compute()
    assert out.tolist() == [1, 2, 1, 2]


def test_select():
    data = pd.arrays.ArrowExtensionArray(pa.array([[0], [0, 1]] * 2))
    s = pd.Series(data)
    df = pd.DataFrame({"s": s})
    ddf = dd.from_pandas(df, 2)
    s = ddf.ak["s"]
    s2 = ddf["s"]
    assert s.compute().tolist() == s2.compute().tolist()


def test_to_dak():
    dak = pytest.importorskip("dask_awkward")
    data = pd.arrays.ArrowExtensionArray(pa.array([[0], [0, 1]] * 2))
    s = dd.from_pandas(pd.Series(data), 1)
    da = s.ak.to_dask_awkward()
    assert isinstance(da, dak.Array)
    out = da.compute().to_list()
    assert out == [[0], [0, 1]] * 2


def test_distributed():
    distributed = pytest.importorskip("distributed")

    with distributed.Client(n_workers=1, threads_per_worker=1):
        data = pd.arrays.ArrowExtensionArray(pa.array([[0], [0, 1]] * 2))
        s = pd.Series(data)
        df = pd.DataFrame({"s": s})
        ddf = dd.from_pandas(df, 2)
        out = ddf.s.ak.count()
        assert out.compute().tolist() == [3, 3]
        out = ddf.s.ak.count(axis=1).compute()
        assert out.tolist() == [1, 2, 1, 2]
