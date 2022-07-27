import numpy as np
import pandas as pd
import pytest

from awkward_pandas import AwkwardDtype


@pytest.fixture
def dtype():
    """Fixture overriding function in pandas/tests/extension/conftest.py"""
    return AwkwardDtype()


@pytest.fixture
def data(dtype):
    """Fixture overriding function in pandas/tests/extension/conftest.py"""
    return pd.array([[1, 2, 3], [4, 5]] * 50, dtype=dtype)


@pytest.fixture
def data_missing(data):
    """Fixture overriding function in pandas/tests/extension/conftest.py"""
    return type(data)._from_sequence([None, data[0]])


@pytest.fixture
def na_value():
    return np.nan
