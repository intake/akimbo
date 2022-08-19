import pytest

import pandas as pd

import awkward_pandas


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize(
    "method", ["upper", "capitalize", "isalpha"]
)
def test_unary_methods(method, binary):
    s = pd.Series(["hello world", "oi"], dtype='awkward')
    if binary:
        s = s.ak.encode()
    out = getattr(s.ak, method)()
    expected = [getattr(_, method)() for _ in s.tolist()]
    assert out.tolist() == expected


def test_with_argument():
    s = pd.Series(["hello world", "oi"], dtype='awkward')
    out = s.ak.startswith("hello")
    expected = [_.startswith("hello") for _ in s.tolist()]
    assert out.tolist() == expected


def test_encode_decode():
    s = pd.Series(["hello world", "oi"], dtype='awkward')
    s2 = s.ak.encode()
    assert s2.tolist() == [_.encode() for _ in s.tolist()]
    s3 = s2.ak.decode()
    assert (s == s3).all()
