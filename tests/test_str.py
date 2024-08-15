import pandas as pd
import pytest


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("method", ["upper", "capitalize", "isalpha"])
def test_unary_methods(method, binary):
    s = pd.Series(["hello world", "oi"])
    if binary:
        s = s.ak.str.encode()
    out = getattr(s.ak.str, method)()
    expected = [getattr(_, method)() for _ in s.tolist()]
    assert out.tolist() == expected


def test_with_argument():
    s = pd.Series(["hello world", "oi"])
    out1 = s.ak.str.starts_with("hello")
    out2 = s.ak.str.startswith("hello")
    expected = [_.startswith("hello") for _ in s.tolist()]
    assert out1.tolist() == expected
    assert out2.tolist() == expected


def test_encode_decode():
    s = pd.Series(["hello world", "oi"])
    s2 = s.ak.str.encode()
    assert s2.tolist() == [_.encode() for _ in s.tolist()]
    s3 = s2.ak.str.decode()
    assert (s == s3).all()


def test_split():
    s = pd.Series(["hello world", "oio", pd.NA, ""])
    s2 = s.ak.str.split_whitespace()
    assert s2.tolist() == [["hello", "world"], ["oio"], pd.NA, [""]]
    s2 = s.ak.str.split_pattern("i")
    assert s2.tolist() == [["hello world"], ["o", "o"], pd.NA, [""]]

    s = pd.Series([b"hello world", b"oio", b""])
    s2 = s.ak.str.split_whitespace()
    assert s2.tolist() == [[b"hello", b"world"], [b"oio"], [b""]]
