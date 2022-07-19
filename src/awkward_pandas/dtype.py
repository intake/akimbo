from __future__ import annotations

from typing import TYPE_CHECKING

import awkward._v2 as ak
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING:
    from awkward_pandas.array import AwkwardExtensionArray


@register_extension_dtype
class AwkwardDtype(ExtensionDtype):
    _name: str = "awkward"
    _kind: str = "O"
    _na_value = pd.NA

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> type[ak.Array]:
        return ak.Array

    @property
    def na_value(self):
        return self._na_value

    @classmethod
    def construct_from_string(cls, string: str) -> AwkwardDtype:
        return cls()

    @classmethod
    def construct_array_type(cls) -> type[AwkwardExtensionArray]:
        from awkward_pandas.array import AwkwardExtensionArray

        return AwkwardExtensionArray

    def __from_arrow__(self, data) -> AwkwardExtensionArray:
        from awkward_pandas.array import AwkwardExtensionArray

        return AwkwardExtensionArray(ak.from_arrow(data))

    def __repr__(self) -> str:
        return "<AwkwardDtype>"
