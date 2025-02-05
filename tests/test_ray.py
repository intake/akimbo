import awkward as ak
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
ray = pytest.importorskip("ray")

pytest.importorskip("akimbo.pandas")
pytest.importorskip("akimbo.ray")

x = pd.Series([[[1, 2, 3], [], [3, 4, None]], [None]] * 100).ak.to_output()
y = pd.Series([["hey", None], ["hi", "ho"]] * 100).ak.to_output()


@pytest.fixture(scope="module")
def rayc():
    context = ray.init()
    yield context
    context.disconnect()


@pytest.fixture()
def df(rayc, tmpdir):
    import ray.data

    pd.DataFrame({"x": x, "y": y}).to_parquet(f"{tmpdir}/a.parquet")
    return ray.data.read_parquet(f"{tmpdir}/a.parquet", override_num_blocks=2)


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


def test_dt(rayc):
    data = pd.DataFrame(
        {
            "_ak_series_": pd.Series(
                pd.date_range(start="2024-01-01", end="2024-01-02", freq="h")
            )
        }
    )
    df = ray.data.from_arrow(data.ak.arrow)
    out = df.ak.dt.strftime()
    result = out.ak.to_output()

    assert result[0] == "2024-01-01T00:00:00.000000000"  # defaults to ns

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


def test_overload(rayc):
    x = pd.Series([1, 2, 3])
    df = ray.data.from_arrow(pd.DataFrame(x, columns=["_ak_series_"]).ak.arrow)

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

    @numba.njit()
    def f(data: ak.Array, builder: ak.ArrayBuilder) -> None:
        for i, item in enumerate(data.x):
            if item[0] is None:
                builder.append(None)
            else:
                builder.append(item[0][2] + item[2][0])  # always 6

    def f2(data):
        builder = ak.ArrayBuilder()
        f(data, builder)
        return builder.snapshot()

    out = df.ak.apply(f2, where="x")
    result = out.ak.to_output()
    assert result.ak.tolist() == [6, None] * 100
