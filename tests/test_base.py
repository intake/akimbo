import awkward as ak
import pandas as pd
import pytest

import awkward_pandas


def test_select():
    s = pd.Series(awkward_pandas.AwkwardExtensionArray([[6, 2, 3], [4, 5]]))
    s2 = s[0]
    assert isinstance(s2, ak.Array)
    assert s2.tolist() == [6, 2, 3]

    s2 = s[0:1]
    assert s2.dtype == "awkward"
    assert isinstance(s2.values, awkward_pandas.AwkwardExtensionArray)
    assert s2.tolist() == [[6, 2, 3]]


@pytest.mark.xfail(reason='numpy dtype("O") comparison giving issues')
def test_astype_to_ak():
    s = pd.Series([[6, 2, 3], [4, 5]], dtype=object)
    s2 = s.astype("awkward")
    assert s2.dtype == "awkward"
    assert s2.tolist() == [[6, 2, 3], [4, 5]]
    assert (s2 == s).tolist() == [[True, True, True], [True, True]]
    assert (s2 == s).all()
