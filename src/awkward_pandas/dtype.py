from __future__ import annotations

from typing import TYPE_CHECKING, Any

import awkward._v2 as ak
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING:
    from awkward_pandas.array import AwkwardArray


@register_extension_dtype
class AwkwardDtype(ExtensionDtype):
    @property
    def name(self) -> str:
        return "awkward"

    @property
    def type(self) -> type[ak.Array]:
        return ak.Array

    @property
    def kind(self) -> str:
        return "O"

    @property
    def na_value(self) -> object:
        return pd.NA

    @property
    def _is_numeric(self) -> bool:
        return True

    @property
    def _is_boolean(self) -> bool:
        return True

    @classmethod
    def construct_from_string(cls, string: str) -> AwkwardDtype:
        if string == cls().name:
            return cls()
        else:
            raise TypeError(f"Cannot construct a {cls} from '{string}'")

    @classmethod
    def construct_array_type(cls) -> type[AwkwardArray]:
        from awkward_pandas.array import AwkwardArray

        return AwkwardArray

    def __from_arrow__(self, data: Any) -> AwkwardArray:
        from awkward_pandas.array import AwkwardArray

        return AwkwardArray(ak.from_arrow(data))

    def __repr__(self) -> str:
        return "<AwkwardDtype>"
