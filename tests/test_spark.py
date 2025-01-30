import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pyspark = pytest.importorskip("pyspark")

pytest.importorskip("akimbo.pandas")
pytest.importorskip("akimbo.spark")

x = pd.Series([[[1, 2, 3], [], [3, 4, None]], [None]] * 100).ak.to_output()
y = pd.Series([["hey", None], ["hi", "ho"]] * 100).ak.to_output()


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

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


def test_dt(spark):
    data = pd.DataFrame(
        {"a": pd.Series(pd.date_range(start="2024-01-01", end="2024-01-02", freq="h"))}
    ).ak.to_output()
    df = spark.createDataFrame(data)
    out = df.ak.dt.strftime()
    result = out.ak.to_output()

    assert result.a.tolist() == data.a.tolist()


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
