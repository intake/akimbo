from __future__ import annotations

from typing import TYPE_CHECKING, Any

import awkward._v2 as ak
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING:
    from awkward_pandas.array import AwkwardExtensionArray


@register_extension_dtype
class AwkwardDtype(ExtensionDtype):
    def __init__(self, ak_type=None):
        self.ak_type = ak_type

    @property
    def name(self) -> str:
        return f"awkward[{self.ak_type}]"

    @property
    def type(self) -> type[ak.Array]:
        return ak.Array

    @property
    def kind(self) -> str:
        return "O"

    @property
    def na_value(self) -> object:
        return np.nan

    @property
    def _is_numeric(self) -> bool:
        return True

    @property
    def _is_boolean(self) -> bool:
        return True

    @classmethod
    def construct_from_string(cls, string: str) -> AwkwardDtype:
        """Construct an instance from a string.

        Parameters
        ----------
        string : str
            Should be "awkward".

        Returns
        -------
        AwkwardDtype
            Instance of the dtype.

        """

        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string == cls().name:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls) -> type[AwkwardExtensionArray]:  # type: ignore[valid-type]
        from awkward_pandas.array import AwkwardExtensionArray

        return AwkwardExtensionArray

    def __from_arrow__(self, data: Any) -> AwkwardExtensionArray:
        from awkward_pandas.array import AwkwardExtensionArray

        return AwkwardExtensionArray(ak.from_arrow(data))

    def __repr__(self) -> str:
        return f"<AwkwardDtype({self.ak_type})>"
