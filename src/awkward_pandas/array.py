from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any, Literal

import awkward._v2 as ak
import numpy as np
import pandas as pd
from pandas.core.arrays.base import ExtensionArray, ExtensionScalarOpsMixin, set_function_name
from pandas.core.dtypes.generic import ABCSeries, ABCIndex, ABCDataFrame

from awkward_pandas.dtype import AwkwardDtype


class AwkwardArray(ExtensionArray, ExtensionScalarOpsMixin):
    _dtype: AwkwardDtype
    _data: ak.Array

    def __init__(self, data: Any) -> None:
        self._dtype = AwkwardDtype()

        if isinstance(data, type(self)):
            self._data = data._data
        elif isinstance(data, ak.Array):
            self._data = data
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
    def _from_factorized(cls, values, original):
        return cls(values)

    def __getitem__(self, item):
        new = operator.getitem(self._data, item)

        return type(self)(new)

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
            rvalues = cls(other)

            return cls(op(lvalues._data, rvalues._data))

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._data.layout.nbytes

    def isna(self):
        return np.array(ak.is_none(self._data))

    def take(self, indices, *, allow_fill=False, fill_value=None):
        return self[indices]

    def copy(self):
        return type(self)(ak.copy(self._data))

    @classmethod
    def _concat_same_type(cls, to_concat):
        raise NotImplementedError

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self):
        return (len(self._data),)

    def __array__(self):
        return np.asarray(self._data.tolist(), dtype=object)

    def __arrow_array__(self):
        import pyarrow as pa

        return pa.chunked_array(ak.to_arrow(self._data))

    def __repr__(self) -> str:
        return f"pandas: {self._data.__repr__()}"

    def __str__(self) -> str:
        return str(self._data)

    def tolist(self) -> list:
        return self._data.tolist()

    # @classmethod
    # def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
    #     def f(self, *args, **kwargs):
    #         at = getattr(ak.Array, op_name)
    #         return cls(at(self, *args, **kwargs))
    #
    #     op_name = f"__{op.__name__}__"
    #     return getattr(ak.Array, op_name)
    #
    def __array_ufunc__(self, *inputs, **kwargs):
        return type(self)(self._data.__array_ufunc__(*inputs, **kwargs))


AwkwardArray._add_arithmetic_ops()
AwkwardArray._add_comparison_ops()
