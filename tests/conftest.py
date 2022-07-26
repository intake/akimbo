import awkward._v2 as ak
import pandas as pd
import pytest

from awkward_pandas import AwkwardArray


@pytest.fixture
def int_raw_array():
    return ak.Array([[1, 2, 3], [4, 5], [6]])


@pytest.fixture
def int_array(int_raw_array: ak.Array) -> AwkwardArray:
    return AwkwardArray(int_raw_array)


@pytest.fixture
def int_series(int_array: AwkwardArray) -> pd.Series:
    return pd.Series(int_array)
