from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pandas.api.extensions import ExtensionArray

from awkward_pandas.dtype import AwkwardDtype


class AwkwardSeries(ExtensionArray):
    _dtype: AwkwardDtype

    @property
    def ndim(self) -> Literal[1]:
        return 1
