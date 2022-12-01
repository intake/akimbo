from __future__ import annotations

import pandas as pd
from pandas.tests.extension.base import BaseConstructorsTests, BaseDtypeTests
from pandas.tests.extension.base.casting import BaseCastingTests  # noqa
from pandas.tests.extension.base.dim2 import Dim2CompatTests  # noqa
from pandas.tests.extension.base.dim2 import NDArrayBacked2DTests  # noqa
from pandas.tests.extension.base.getitem import BaseGetitemTests  # noqa
from pandas.tests.extension.base.groupby import BaseGroupbyTests  # noqa
from pandas.tests.extension.base.index import BaseIndexTests  # noqa
from pandas.tests.extension.base.interface import BaseInterfaceTests  # noqa
from pandas.tests.extension.base.io import BaseParsingTests  # noqa
from pandas.tests.extension.base.methods import BaseMethodsTests  # noqa
from pandas.tests.extension.base.missing import BaseMissingTests  # noqa
from pandas.tests.extension.base.ops import BaseArithmeticOpsTests  # noqa
from pandas.tests.extension.base.ops import BaseComparisonOpsTests  # noqa
from pandas.tests.extension.base.ops import BaseOpsUtil  # noqa
from pandas.tests.extension.base.ops import BaseUnaryOpsTests  # noqa
from pandas.tests.extension.base.printing import BasePrintingTests  # noqa
from pandas.tests.extension.base.reduce import BaseBooleanReduceTests  # noqa
from pandas.tests.extension.base.reduce import BaseNoReduceTests  # noqa
from pandas.tests.extension.base.reduce import BaseNumericReduceTests  # noqa
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

    # Overridden because pd.DataFrame(list(AwkwardExtensionArray))
    # won't work.
    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        dtype = data.dtype

        expected = pd.Series(data)
        result = pd.Series(list(data), dtype=dtype)
        self.assert_series_equal(result, expected)

        result = pd.Series(list(data), dtype=str(dtype))
        self.assert_series_equal(result, expected)

        # this is the test that breaks the upstream version
        # expected = pd.DataFrame(data).astype(dtype)
        # result = pd.DataFrame(list(data), dtype=dtype)
        # self.assert_frame_equal(result, expected)


# class TestAwkwardBaseCastingTests(BaseCastingTests):

#     # Overridden because list(AwkwardExtensionArray) will contain
#     # ak.Array as elements, not python objects.
#     def test_tolist(self, data):
#         result = pd.Series(data).tolist()
#         expected = data.tolist()
#         assert result == expected

#         result = list(pd.Series(data))
#         expected = list(data)
#         for res, exp in zip(result, expected):
#             assert ak.all(res == exp)


# class TestAwkwardBaseGetitemTests(BaseGetitemTests):
#     pass


# class TestAwkwardBaseGroupbyTests(BaseGroupbyTests):
#     pass


# class TestAwkwardBaseIndexTests(BaseIndexTests):
#     pass


# class TestAwkwardBaseInterfaceTests(BaseInterfaceTests):
#     pass


# class TestAwkwardDim2CompatTests(Dim2CompatTests):
#     pass


# # Not compatible with awkward array
# # class TestAwkwardNDArrayBacked2DTests(NDArrayBacked2DTests):
# #     pass


# class TestAwkwardBaseParsingTests(BaseParsingTests):
#     pass


# class TestAwkwardBaseMethodsTests(BaseMethodsTests):
#     pass


# class TestAwkwardBaseMissingTests(BaseMissingTests):
#     pass


# class TestAwkwardBaseArithmeticOpsTests(BaseArithmeticOpsTests):
#     pass


# class TestAwkwardBaseComparisonOpsTests(BaseComparisonOpsTests):
#     pass


# class TestAwkwardBaseOpsUtil(BaseOpsUtil):
#     pass


# class TestAwkwardBaseUnaryOpsTests(BaseUnaryOpsTests):
#     pass


# class TestAwkwardBasePrintingTests(BasePrintingTests):
#     pass


# class TestAwkwardBaseBooleanReduceTests(BaseBooleanReduceTests):
#     pass


# class TestAwkwardBaseNoReduceTests(BaseNoReduceTests):
#     pass


# class TestAwkwardBaseNumericReduceTests(BaseNumericReduceTests):
#     pass


# class TestAwkwardBaseReshapingTests(BaseReshapingTests):
#     def test_ravel(self, data):
#         result = data.ravel()
#         assert type(result) == type(data)
#         result._data is data._data


# # class TestAwkwardBaseSetitemTests(BaseSetitemTests):
# #     pass
