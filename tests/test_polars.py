import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("akimbo.polars")


### EAGER


def test_simple():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak[:, -1:]
    assert s2.to_list() == [[3], [], [5]]


def test_apply():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak.apply(np.negative)
    assert s2.to_list() == [[-1, -2, -3], [], [-4, -5]]


def test_apply_where():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.Series(data)
    s2 = s.ak.apply(np.negative, where="a")
    assert s2[0] == {"a": [-1, -2, -3], "b": [1, 2, 3]}


def test_pack_unpack():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.Series(data)
    df = s.ak.unpack()
    assert df["a"].to_list() == [[1, 2, 3]] * 3
    s2 = df.ak.pack()
    assert s.to_list() == s2.to_list()


def test_operator():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = s.ak + 1
    assert s2.to_list() == [[2, 3, 4], [], [5, 6]]


def test_ufunc():
    s = pl.Series([[1, 2, 3], [], [4, 5]])
    s2 = np.negative(s.ak)
    assert s2.to_list() == [[-1, -2, -3], [], [-4, -5]]

    s2 = np.add(s.ak, 1)
    assert s2.to_list() == [[2, 3, 4], [], [5, 6]]

    df = pl.DataFrame({"a": s})
    df2 = df.ak + 1
    assert df2["a"].to_list() == [[2, 3, 4], [], [5, 6]]


def test_binary():
    s = pl.Series([[1, 2, 3], [], [4, 5]])

    df = pl.DataFrame({"a": s})
    df2 = df.ak == df
    assert df2["a"].to_list() == [[True, True, True], [], [True, True]]

    s2 = np.negative(s.ak)
    s3 = s.ak + s2
    assert s3.to_list() == [[0, 0, 0], [], [0, 0]]


def test_str_sugar():
    s = pl.Series([["hay", "hi", "hola"], [], ["bye", "yo"]])
    out = s.ak.str.repeat(3)
    out2 = s.ak.str * 3
    expected = [["hayhayhay", "hihihi", "holaholahola"], [], ["byebyebye", "yoyoyo"]]
    assert out.ak.tolist() == out2.ak.tolist() == expected

    out = s.ak.str.join_el(s)
    out2 = s.ak.str + s
    expected = [["hayhay", "hihi", "holahola"], [], ["byebye", "yoyo"]]
    assert out.ak.tolist() == out2.ak.tolist() == expected


def test_unexplode():
    df = pl.DataFrame(
        {
            "x": [1, 1, 1, 2, 1, 3, 3, 1],
            "y": [1, 1, 1, 2, 1, 3, 3, 1],
            "z": [1, 1, 1, 2, 1, 3, 3, 2],
        }
    )
    out = df.ak.unexplode("x")
    compact = out["grouped"].to_list()
    expected = [
        [
            {"y": 1, "z": 1},
            {"y": 1, "z": 1},
            {"y": 1, "z": 1},
            {"y": 1, "z": 1},
            {"y": 1, "z": 2},
        ],
        [{"y": 2, "z": 2}],
        [{"y": 3, "z": 3}, {"y": 3, "z": 3}],
    ]
    assert compact == expected

    out = df.ak.unexplode("x", "y")
    compact = out["grouped"].to_list()
    expected = [
        [{"z": 1}, {"z": 1}, {"z": 1}, {"z": 1}, {"z": 2}],
        [{"z": 2}],
        [{"z": 3}, {"z": 3}],
    ]
    assert compact == expected

    with pytest.raises(ValueError):
        df.ak.unexplode("x", "y", "z")

    with pytest.raises(ValueError):
        df.ak.unexplode("unknown")


### LAZY


def test_simple_lazy():
    s = pl.DataFrame({"a": [[1, 2, 3], [], [4, 5]]}).lazy()
    s2 = s.ak[:, -1:]
    assert s2.collect()["a"].to_list() == [[3], [], [5]]


def test_apply_lazy():
    s = pl.DataFrame({"_ak_series_": pl.Series([[1, 2, 3], [], [4, 5]])}).lazy()
    s2 = s.ak.apply(np.negative)
    assert s2.ak.to_output().to_list() == [[-1, -2, -3], [], [-4, -5]]


def test_apply_where_lazy():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.DataFrame(data).lazy()
    s2 = s.ak.apply(np.negative, where="a")
    assert s2.collect()[0].to_dicts() == [{"a": [-1, -2, -3], "b": [1, 2, 3]}]


def test_pack_unpack_lazy():
    data = [
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2, 3], "b": [1, 2, 3]},
    ]
    s = pl.DataFrame({"_ak_series_": pl.Series(data)}).lazy()
    df = s.ak.unpack()
    assert df.collect()["a"].to_list() == [[1, 2, 3]] * 3
    s2 = df.ak.pack().ak.to_output()
    assert s2.to_list() == data


def test_operator_lazy():
    s = pl.DataFrame({"_ak_series_": pl.Series([[1, 2, 3], [], [4, 5]])}).lazy()
    s2 = s.ak + 1
    assert s2.ak.to_output().to_list() == [[2, 3, 4], [], [5, 6]]


def test_ufunc_lazy():
    s0 = pl.DataFrame({"_ak_series_": pl.Series([[1, 2, 3], [], [4, 5]])})
    s = s0.lazy()
    s2 = np.negative(s.ak)
    assert s2.ak.to_output().to_list() == [[-1, -2, -3], [], [-4, -5]]

    s2 = np.add(s.ak, 1)
    assert s2.ak.to_output().to_list() == [[2, 3, 4], [], [5, 6]]

    df = pl.DataFrame({"a": s0}).lazy()
    df2 = df.ak + 1
    assert df2.ak.to_output()["a"].to_list() == [[2, 3, 4], [], [5, 6]]


def test_binary_lazy():
    s = pl.Series([[1, 2, 3], [], [4, 5]])

    df = pl.DataFrame({"a": s})
    df2 = df.ak == df
    assert df2["a"].to_list() == [[True, True, True], [], [True, True]]

    s2 = np.negative(s.ak)
    s3 = s.ak + s2
    assert s3.to_list() == [[0, 0, 0], [], [0, 0]]
