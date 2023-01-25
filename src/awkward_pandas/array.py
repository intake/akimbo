from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import awkward as ak
import numpy as np
import pandas as pd
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
    set_function_name,
)
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries

from awkward_pandas.dtype import AwkwardDtype

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class AwkwardExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    _dtype: AwkwardDtype
    _data: ak.Array

    def __init__(self, data: Any) -> None:
        """

        Parameters
        ----------
        data : awkward array, dict, JSON string, iterable
            Construct extension array from this data. If an iterable or dict,
            pass to awkward to generate the internal array. If a JSON string,
            parse it using awkward.
        """

        self._dtype = AwkwardDtype()
        if isinstance(data, type(self)):
            self._data = data._data
        elif isinstance(data, ak.Array):
            self._data = data
        elif isinstance(data, dict):
            self._data = ak.Array(data)
        elif isinstance(data, str):
            self._data = ak.from_json(data)
        elif isinstance(data, Iterable):
            self._data = ak.from_iter(None if a is pd.NA else a for a in data)
        elif data is None:
            self._data = ak.Array([])
        else:
            raise ValueError

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _empty(cls, shape, dtype):
        if isinstance(shape, tuple) and len(shape) != 1:
            raise ValueError
        if isinstance(shape, tuple):
            return cls([None] * shape[0])
        return cls([None] * shape)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def __getitem__(self, item):
        if isinstance(item, int):
            return operator.getitem(self._data, item)
        elif isinstance(item, (slice, np.ndarray, ak.Array)):
            new = operator.getitem(self._data, item)
            return type(self)(new)
        else:
            raise ValueError(f"bad item passed to getitem: {type(item)}")

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self._data[i]

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        def _binop(self, other):
            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            lvalues = self
            if isinstance(other, list) or (
                isinstance(other, pd.Series) and other.dtype == "O"
            ):
                rvalues = cls(other)
                return cls(op(lvalues._data, rvalues._data))
            else:
                return cls(op(lvalues._data, other))

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    def _reduce(self, name: str, *, skipna: bool = True, axis=None, **kwargs):
        return getattr(ak, name)(self._data, **kwargs)

    @property
    def dtype(self) -> AwkwardDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.layout.nbytes

    def isna(self):
        return np.array(ak.is_none(self._data))

    def take(self, indices, *, allow_fill=False, fill_value=None):
        return self[indices]

    def copy(self):
        return type(self)(ak.copy(self._data))

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(ak.concatenate(to_concat))

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return (len(self._data),)

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray:
        dtype = np.dtype(object) if dtype is None else np.dtype(dtype)

        if dtype == np.dtype("O"):
            return np.asarray(self._data.tolist(), dtype=dtype)

        return np.asarray(self._data, dtype=dtype)

    def __arrow_array__(self, type=None):
        import pyarrow as pa

        return pa.chunked_array(ak.to_arrow(self._data), type=type)

    def tolist(self) -> list:
        return self._data.tolist()

    def __array_ufunc__(self, *inputs, **kwargs):
        return type(self)(self._data.__array_ufunc__(*inputs, **kwargs))

    def max(self, **kwargs):
        print("max?")
        return ak.max(self._data, **kwargs)

    def min(self, **kwargs):
        return ak.min(self._data, **kwargs)

    def mean(self, **kwargs):
        return ak.mean(self._data, **kwargs)

    def std(self, **kwargs):
        return ak.std(self._data, **kwargs)

    def sum(self, axis=None, **kwargs):
        return ak.sum(self._data, axis=axis, **kwargs)


AwkwardExtensionArray._add_arithmetic_ops()
AwkwardExtensionArray._add_comparison_ops()
