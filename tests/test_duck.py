import awkward as ak
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")


@pytest.fixture()
def db():
    conn = duckdb.connect()
    yield
    conn.close()


x = pd.Series([[[1, 2, 3], [], [3, 4, None]], [None]] * 100).ak.to_output()
y = pd.Series([["hey", None], ["hi", "ho"]] * 100).ak.to_output()


@pytest.fixture()
def df(db, tmpdir):
    pd.DataFrame({"x": x, "y": y}).to_parquet(f"{tmpdir}/a.parquet")
    return db.from_parquet(f"{tmpdir}/a.parquet")
