from __future__ import annotations

from typing import Any, Literal

import awkward._v2 as ak
import numpy as np
from pandas._typing import Dtype
from pandas.core.arrays import ExtensionScalarOpsMixin
from pandas.core.arrays.base import ExtensionArray

from awkward_pandas.dtype import AwkwardDtype


class AwkwardExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    _dtype: AwkwardDtype
    _data: ak.Array

    def __init__(self, data: Any) -> None:
        self._dtype = AwkwardDtype()
        if isinstance(data, ak.Array):
            self._data = data

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        raise NotImplementedError

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, AwkwardExtensionArray):
            return type(self)(self.data == other.data)
        return self == type(self)(other)

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._data.layout.nbytes

    def isna(self):
        return np.array(ak.is_none(self.data))

    def take(self, indices, *, allow_fill=False, fill_value=None):
        raise NotImplementedError

    def copy(self):
        type(self)(ak.copy(self.data))

    @classmethod
    def _concat_same_type(cls, to_concat):
        raise NotImplementedError

    @property
    def data(self) -> ak.Array:
        return self._data

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self):
        return (len(self._data),)

    def __arrow_array__(self):
        import pyarrow as pa

        return pa.chunked_array(ak.to_arrow(self.data))

    def __repr__(self) -> str:
        return f"pandas: {self.data.__repr__()}"


AwkwardExtensionArray._add_arithmetic_ops()
AwkwardExtensionArray._add_comparison_ops()
