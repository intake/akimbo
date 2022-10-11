import awkward as ak
import pandas as pd
import pytest

import awkward_pandas


def test_len():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([[6, 2, 3], [4, 5]]))
    assert s.ak.count() == 5
    s2 = s.ak.count(axis=1)
    assert s2.tolist() == [3, 2]


def test_no_access():
    s = pd.Series([1, 2])
    with pytest.raises(Exception):
        s.ak.count()


def test_getitem():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([[6, 2, 3], [4, 5]]))
    s2 = s.ak[:, :1]
    assert isinstance(s2, pd.Series)
    assert s2.dtype == "awkward"
    assert s2.tolist() == [[6], [4]]


def test_to_column():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([6, 2, 3]), name="test")
    s2 = s.ak.to_column()
    assert s2.dtype == "int64"

    s = pd.Series(awkward_pandas.AwkwardExtensionArray(["6", "2", "3"]), name="test")
    s2 = s.ak.to_column()
    assert s2.dtype == "string[pyarrow]"

    s = pd.Series(awkward_pandas.AwkwardExtensionArray([["6", "2", "3"]]), name="test")
    with pytest.raises(ValueError):
        s.ak.to_column()


def test_to_columns():
    s = pd.Series(
        awkward_pandas.AwkwardExtensionArray(
            {"num": [6, 2, 3], "deep": [[0], [], None], "text": ["hi", "ho", "hum"]}
        ),
        name="test",
    )
    df = s.ak.to_columns()
    assert df.columns.tolist() == ["num", "text", "test"]
    assert df.num.tolist() == [6, 2, 3]
    assert df.test.tolist() == [{"deep": [0]}, {"deep": []}, {"deep": None}]
    assert df.text.tolist() == ["hi", "ho", "hum"]

    df = s.ak.to_columns(cull=False)
    assert df.columns.tolist() == ["num", "text", "test"]
    assert df.num.tolist() == [6, 2, 3]
    assert df.test[0].tolist() == {"num": 6, "deep": [0], "text": "hi"}
    assert df.text.tolist() == ["hi", "ho", "hum"]


def test_apply():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([[6, 2, 3], [4]]), name="test")
    applied = s.ak.apply(lambda x: ak.num(x))
    assert applied.values._data.tolist() == ak.num(s.values._data).tolist()


def test_dir():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([6, 2, 3]), name="test")
    assert "sum" in dir(s.ak)
    assert "Array" not in dir(s.ak)
    assert "ak_num" not in dir(s.ak)
    assert "_util" not in dir(s.ak)
