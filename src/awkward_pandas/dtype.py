from __future__ import annotations

from typing import TYPE_CHECKING

import awkward._v2 as ak
import pandas as pd
from pandas.api.extensions import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING:
    from awkward_pandas.series import AwkwardSeries


class AwkwardDtypeRep(type):
    def __repr__(self):
        return "AwkwardDtype"


@register_extention_dtype
class AwkwardDtype(ExtensionDtype, metaclass=AwkwardDtypeRep):
    name = "awkward"
    type: type[ak.Array] = ak.Array
    kind: str = "O"
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls) -> type[AwkwardSeries]:
        from awkward_pandas.series import AwkwardSeries

        return AwkwardSeries
