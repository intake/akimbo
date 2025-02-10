import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
import akimbo.duck  # noqa


@pytest.fixture()
def db():
    conn = duckdb.connect(":default:")
    yield conn
    conn.close()


x = pd.Series([[[1, 2, 3], [], [3, 4, None]], [None]] * 100).ak.to_output()
y = pd.Series([["hey", None], ["hi", "ho"]] * 100).ak.to_output()


@pytest.fixture()
def df(db, tmpdir):
    pd.DataFrame({"x": x, "y": y}).to_parquet(f"{tmpdir}/a.parquet")
    return db.from_parquet(f"{tmpdir}/a.parquet")


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


def test_select(df):
    out = df.ak["x"]
    result = out.ak.to_output()
    assert isinstance(result, pd.Series)
    assert result.tolist() == x.tolist()
    out = df.ak[:, ::2]
    result = out.ak.to_output()
    expected = x.ak[:, ::2]
    assert result.x.tolist() == expected.tolist()
