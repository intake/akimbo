import awkward as ak
import pandas as pd
import pytest

pytest.importorskip("akimbo.pandas")


def test_len():
    s = pd.Series([[6, 2, 3], [4, 5]])
    assert s.ak.count() == 5
    s2 = s.ak.count(axis=1)
    assert s2.tolist() == [3, 2]


def test_getitem():
    s = pd.Series([[6, 2, 3], [4, 5]])
    s2 = s.ak[:, :1]
    assert isinstance(s2, pd.Series)
    assert s2.tolist() == [[6], [4]]


def test_apply():
    s = pd.Series([[6, 2, 3], [4]], name="test")
    applied = s.ak.apply(lambda x: ak.num(x))
    assert applied.tolist() == ak.num(s.ak.array).tolist()


def test_dir():
    s = pd.Series([6, 2, 3], name="test")
    assert "sum" in dir(s.ak)
    assert "Array" not in dir(s.ak)
    assert "ak_num" not in dir(s.ak)
    assert "_util" not in dir(s.ak)


def test_array_property():
    a = [[1, 2, 3], [4, 5], [6]]
    s = pd.Series(a)
    # ensure that the array associated with the accessor is the same as the original
    assert isinstance(s.ak.array, ak.Array)
    assert a == s.ak.array.tolist()


def test_ufunc():
    a = [[1, 2, 3], [4, 5], [6]]
    s = pd.Series(a)
    assert (s.ak + 1).tolist() == [[2, 3, 4], [5, 6], [7]]

    assert (s.ak + s.ak).tolist() == [[2, 4, 6], [8, 10], [12]]
    assert (s.ak + s).tolist() == [[2, 4, 6], [8, 10], [12]]

    s = pd.DataFrame({"a": s})
    assert (s.ak + 1).a.tolist() == [[2, 3, 4], [5, 6], [7]]

    assert (s.ak + s.ak).a.tolist() == [[2, 4, 6], [8, 10], [12]]
    assert (s.ak + s).a.tolist() == [[2, 4, 6], [8, 10], [12]]


def test_manual_ufunc():
    from akimbo.apply_tree import numeric

    df = pd.DataFrame(
        {"a": [["hey", "hi", "ho"], [None], ["blah"]], "b": [[1, 2, 3], [4, 5], [6]]}
    )
    df2 = df.ak.transform(
        lambda x: x + 1, match=numeric, inmode="numpy", outtype=ak.contents.NumpyArray
    )
    expected = [
        {"a": ["hey", "hi", "ho"], "b": [2, 3, 4]},
        {"a": [None], "b": [5, 6]},
        {"a": ["blah"], "b": [7]},
    ]
    assert df2.tolist() == expected


def test_mixed_ufunc():
    # ufuncs are numeric only by default, doesn't touch strings
    df = pd.DataFrame(
        {"a": [["hey", "hi", "ho"], [None], ["blah"]], "b": [[1, 2, 3], [4, 5], [6]]}
    )
    df2 = df.ak + 1
    expected = [
        {"a": ["hey", "hi", "ho"], "b": [2, 3, 4]},
        {"a": [None], "b": [5, 6]},
        {"a": ["blah"], "b": [7]},
    ]
    assert df2.ak.tolist() == expected

    df2 = df.ak * 2
    expected = [
        {"a": ["hey", "hi", "ho"], "b": [2, 4, 6]},
        {"a": [None], "b": [8, 10]},
        {"a": ["blah"], "b": [12]},
    ]
    assert df2.ak.tolist() == expected

    df2 = df.ak == df.ak
    expected = [[True, True, True], [True, True], [True]]
    assert df2["b"].tolist() == expected


def test_to_autoarrow():
    a = [[1, 2, 3], [4, 5], [6]]
    s = pd.Series(a)
    s2 = s.ak.to_output()
    assert s2.tolist() == a
    assert "pyarrow" in str(s2.dtype)


def test_rename():
    a = [{"a": [{"b": {"c": 0}}] * 2}] * 3
    s = pd.Series(a)

    s2 = s.ak.rename(("a", "b"), "d")
    assert s2.tolist() == [{"a": [{"d": {"c": 0}}] * 2}] * 3

    s2 = s.ak.rename("a", "d")
    assert s2.tolist() == [{"d": [{"b": {"c": 0}}] * 2}] * 3

    s2 = s.ak.rename(("a", "b", "c"), "d")
    assert s2.tolist() == [{"a": [{"b": {"d": 0}}] * 2}] * 3
