from __future__ import annotations

from typing import TYPE_CHECKING

import awkward._v2 as ak
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING:
    from awkward_pandas.array import AwkwardExtensionArray


@register_extension_dtype
class AwkwardDtype(ExtensionDtype):
    name = "awkward"
    kind: str = "O"
    na_value = pd.NA

    @property
    def type(self) -> type[ak.Array]:
        return ak.Array

    @classmethod
    def construct_array_type(cls) -> type[AwkwardExtensionArray]:
        from awkward_pandas.array import AwkwardExtensionArray

        return AwkwardExtensionArray
