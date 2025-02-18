import os
import sys

import awkward as ak
import numpy as np
import pytest

WIN = sys.platform.startswith("win")
if WIN and sys.version_info >= (3, 12):
    # TODO: investigate this condition
    pytest.skip("newer pyspark fails on windows", allow_module_level=True)
pd = pytest.importorskip("pandas")
pyspark = pytest.importorskip("pyspark")

pytest.importorskip("akimbo.pandas")
pytest.importorskip("akimbo.spark")

x = pd.Series([[[1, 2, 3], [], [3, 4, None]], [None]] * 100).ak.to_output()
y = pd.Series([["hey", None], ["hi", "ho"]] * 100).ak.to_output()


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    return (
        SparkSession.builder
        # .config("spark.sql.execution.arrow.enabled", "true")  # this was spark<3.0.0
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
        .appName("test")
        .getOrCreate()
    )


@pytest.fixture()
def df(spark, tmpdir):
    pd.DataFrame({"x": x, "y": y}).to_parquet(f"{tmpdir}/a.parquet")
    return spark.read.parquet(f"{tmpdir}/a.parquet")


def test_unary(df):
    out = df.ak.is_none()
    result = out.ak.to_output()
    expected = x.ak.is_none()
    assert result.x.tolist() == expected.tolist()

    out = df.ak.is_none(axis=1)
    result = out.ak.to_output()
    expected = x.ak.is_none(axis=1)
    assert result.x.tolist() == expected.tolist()

    out = df.ak.str.upper()
    result = out.ak.to_output()
    expected = y.ak.str.upper()
    assert result.y.tolist() == expected.tolist()


@pytest.mark.skipif(WIN, reason="may not have locale on windows")
def test_dt(spark):
    data = (
        pd.DataFrame(
            {
                "_ak_series_": pd.Series(
                    pd.date_range(start="2024-01-01", end="2024-01-02", freq="h")
                )
            }
        )
        .ak.to_output()
        .ak.unpack()
    )
    df = spark.createDataFrame(data)
    out = df.ak.dt.strftime()
    result = out.ak.to_output()

    assert result[0] == "2024-01-01T00:00:00.000000"

    out = df.ak.dt.hour()
    result = out.ak.to_output()
    assert set(result) == set(range(24))


def test_select(df):
    out = df.ak["x"]
    result = out.ak.to_output()
    assert isinstance(result, pd.Series)
    assert result.tolist() == x.tolist()
    out = df.ak[:, ::2]
    result = out.ak.to_output()
    expected = x.ak[:, ::2]
    assert result.x.tolist() == expected.tolist()


def test_binary(df):
    out = df.ak["x"].ak.isclose(1)
    result = out.ak.to_output()
    expected = x.ak.isclose(1)
    assert result.tolist() == expected.tolist()

    out = df.ak["x"].ak.isclose(df.ak["x"])
    result = out.ak.to_output()
    expected = x.ak.isclose(x)
    assert result.tolist() == expected.tolist()


def test_ufunc(df):
    out = np.negative(df.ak["x"].ak)
    result = out.ak.to_output()
    expected = np.negative(x.ak)
    assert result.tolist() == expected.tolist()

    out = np.add(df.ak["x"].ak, df.ak["x"])
    result = out.ak.to_output()
    expected = x.ak * 2
    assert result.tolist() == expected.tolist()


def test_ufunc_where(df):
    out = np.add(df.ak, df, where="x")
    result = out.ak.to_output()
    expected = x.ak * 2
    assert result.x.tolist() == expected.tolist()


def test_overload(spark):
    x = pd.Series([1, 2, 3])
    df = spark.createDataFrame(pd.DataFrame(x, columns=["_ak_series_"]))

    out = df.ak + 1  # scalar
    result = out.ak.to_output()
    assert result.tolist() == [2, 3, 4]

    out = df.ak + [1]  # array-like, broadcast
    result = out.ak.to_output()
    assert result.tolist() == [2, 3, 4]

    out = df.ak == df.ak  # matching layout
    result = out.ak.to_output()
    assert result.tolist() == [True, True, True]


def test_dir(df):
    assert "flatten" in dir(df.ak)
    assert "upper" in dir(df.ak.str)


def test_apply_numba(df):
    numba = pytest.importorskip("numba")

    def f(data: ak.Array, builder: ak.ArrayBuilder) -> None:
        for i, item in enumerate(data.x):
            if item[0] is None:
                builder.append(None)
            else:
                builder.append(item[0][2] + item[2][0])  # always 6

    def f2(data):
        if len(data):
            builder = ak.ArrayBuilder()
            numba.njit(f)(data, builder)
            return builder.snapshot()
        else:
            # default output for zero-length schema guesser
            return ak.Array([None, 6])

    out = df.ak.apply(f2, where="x")
    result = out.ak.to_output()
    assert result.ak.tolist() == [6, None] * 100
