from __future__ import annotations

from typing import Literal

import awkward._v2 as ak
from pandas.core.arrays.base import ExtensionArray

from awkward_pandas.dtype import AwkwardDtype


class AwkwardExtensionArray(ExtensionArray):
    _data: ak.Array
    _dtype: AwkwardDtype

    @property
    def data(self) -> ak.Array:
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._data.layout.nbytes

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self):
        return (len(self._data),)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        raise NotImplementedError
