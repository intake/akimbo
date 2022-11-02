from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("method", ["upper", "capitalize", "isalpha"])
def test_unary_methods(method, binary):
    s = pd.Series(["hello world", "oi"], dtype="awkward")
    if binary:
        s = s.ak.encode()
    out = getattr(s.ak, method)()
    expected = [getattr(_, method)() for _ in s.tolist()]
    assert out.tolist() == expected


def test_with_argument():
    s = pd.Series(["hello world", "oi"], dtype="awkward")
    out = s.ak.startswith("hello")
    expected = [_.startswith("hello") for _ in s.tolist()]
    assert out.tolist() == expected


def test_encode_decode():
    s = pd.Series(["hello world", "oi"], dtype="awkward")
    s2 = s.ak.encode()
    assert s2.tolist() == [_.encode() for _ in s.tolist()]
    s3 = s2.ak.decode()
    assert (s == s3).all()


def test_split():
    s = pd.Series(["hello world", "oio", ""], dtype="awkward")
    s2 = s.ak.split()
    assert s2.tolist() == [["hello", "world"], ["oio"], [""]]
    s2 = s.ak.split("i")
    assert s2.tolist() == [["hello world"], ["o", "o"], [""]]

    s = pd.Series([b"hello world", b"oio", b""], dtype="awkward")
    s2 = s.ak.split()
    assert s2.tolist() == [[b"hello", b"world"], [b"oio"], [b""]]
