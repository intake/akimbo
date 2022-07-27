from __future__ import annotations

from pandas.tests.extension.base import BaseConstructorsTests, BaseDtypeTests

import awkward_pandas


def test_version():
    assert awkward_pandas.__version__


class TestAwkwardDtype(BaseDtypeTests):
    pass


class TestAwkwardConstructors(BaseConstructorsTests):
    pass
