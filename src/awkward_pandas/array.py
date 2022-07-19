from __future__ import annotations

from typing import Literal

from pandas.core.arrays.base import ExtensionArray

from awkward_pandas.dtype import AwkwardDtype


class AwkwardExtensionArray(ExtensionArray):
    _dtype: AwkwardDtype

    @property
    def ndim(self) -> Literal[1]:
        return 1
