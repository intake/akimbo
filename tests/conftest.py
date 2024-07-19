from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.conftest import *  # noqa

import akimbo.pandas  # noqa


@pytest.fixture
def data(dtype):
    """Fixture overriding function in pandas/tests/extension/conftest.py"""
    return pd.array([[1, 2, 3], [4, 5]] * 50, dtype=dtype)


@pytest.fixture
def data_missing(data):
    """Fixture overriding function in pandas/tests/extension/conftest.py"""
    return type(data)._from_sequence([None, data[0]])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.
    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def na_value():
    return np.nan
