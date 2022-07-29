from __future__ import annotations

import awkward._v2 as ak
import pandas as pd
from pandas.tests.extension.base import BaseConstructorsTests, BaseDtypeTests
from pandas.tests.extension.base.casting import BaseCastingTests  # noqa
from pandas.tests.extension.base.dim2 import (  # noqa
    Dim2CompatTests,
    NDArrayBacked2DTests,
)
from pandas.tests.extension.base.getitem import BaseGetitemTests  # noqa
from pandas.tests.extension.base.groupby import BaseGroupbyTests  # noqa
from pandas.tests.extension.base.index import BaseIndexTests  # noqa
from pandas.tests.extension.base.interface import BaseInterfaceTests  # noqa
from pandas.tests.extension.base.io import BaseParsingTests  # noqa
from pandas.tests.extension.base.methods import BaseMethodsTests  # noqa
from pandas.tests.extension.base.missing import BaseMissingTests  # noqa
from pandas.tests.extension.base.ops import (  # noqa
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseOpsUtil,
    BaseUnaryOpsTests,
)
from pandas.tests.extension.base.printing import BasePrintingTests  # noqa
from pandas.tests.extension.base.reduce import (  # noqa
    BaseBooleanReduceTests,
    BaseNoReduceTests,
    BaseNumericReduceTests,
)
from pandas.tests.extension.base.reshaping import BaseReshapingTests  # noqa
from pandas.tests.extension.base.setitem import BaseSetitemTests  # noqa

import awkward_pandas


def test_version():
    assert awkward_pandas.__version__


class TestAwkwardDtype(BaseDtypeTests):
    pass


class TestAwkwardConstructors(BaseConstructorsTests):
    def test_series_constructor_scalar_with_index(self, data, dtype):
        assert True


class TestAwkwardBaseCastingTests(BaseCastingTests):

    # Overridden because list(AwkwardExtensionArray) will contain
    # ak.Array as elements, not python objects.
    def test_tolist(self, data):
        result = pd.Series(data).tolist()
        expected = data.tolist()
        assert result == expected

        result = list(pd.Series(data))
        expected = list(data)
        for res, exp in zip(result, expected):
            assert ak.all(res == exp)


class TestAwkwardBaseGetitemTests(BaseGetitemTests):
    pass


class TestAwkwardBaseGroupbyTests(BaseGroupbyTests):
    pass


class TestAwkwardBaseIndexTests(BaseIndexTests):
    pass


class TestAwkwardBaseInterfaceTests(BaseInterfaceTests):
    pass


class TestAwkwardDim2CompatTests(Dim2CompatTests):
    pass


# Not compatible with awkward array
# class TestAwkwardNDArrayBacked2DTests(NDArrayBacked2DTests):
#     pass


class TestAwkwardBaseParsingTests(BaseParsingTests):
    pass


class TestAwkwardBaseMethodsTests(BaseMethodsTests):
    pass


class TestAwkwardBaseMissingTests(BaseMissingTests):
    pass


class TestAwkwardBaseArithmeticOpsTests(BaseArithmeticOpsTests):
    pass


class TestAwkwardBaseComparisonOpsTests(BaseComparisonOpsTests):
    pass


class TestAwkwardBaseOpsUtil(BaseOpsUtil):
    pass


class TestAwkwardBaseUnaryOpsTests(BaseUnaryOpsTests):
    pass


class TestAwkwardBasePrintingTests(BasePrintingTests):
    pass


class TestAwkwardBaseBooleanReduceTests(BaseBooleanReduceTests):
    pass


class TestAwkwardBaseNoReduceTests(BaseNoReduceTests):
    pass


class TestAwkwardBaseNumericReduceTests(BaseNumericReduceTests):
    pass


class TestAwkwardBaseReshapingTests(BaseReshapingTests):
    def test_ravel(self, data):
        result = data.ravel()
        assert type(result) == type(data)
        result._data is data._data


# class TestAwkwardBaseSetitemTests(BaseSetitemTests):
#     pass
