import functools
import inspect

import awkward._v2 as ak
import pandas as pd

from awkward_pandas.array import AwkwardArray
from awkward_pandas.dtype import AwkwardDtype

funcs = [n for n in dir(ak) if inspect.isfunction(getattr(ak, n))]


@pd.api.extensions.register_series_accessor("ak")
class AwkwardAccessor:
    def __init__(self, pandas_obj):
        if not self._validate(pandas_obj):
            raise AttributeError("ak accessor called on incompatible data")
        self._obj = pandas_obj
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            if isinstance(self._obj, AwkwardArray):
                self._arr = self._obj
            elif isinstance(self._obj.dtype, AwkwardDtype) and isinstance(
                self._obj, pd.Series
            ):
                # this is a pandas Series that contains an Awkward
                self._arr = self._obj.values
            elif isinstance(self._obj.dtype, AwkwardDtype):
                # a dask series - figure out what to do here
                raise NotImplementedError
            else:
                # this recreates series, possibly by iteration
                self._arr = AwkwardArray(self._obj)
        return self._arr

    def __getitem__(self, *items):
        return pd.Series(AwkwardArray(self.arr._data.__getitem__(*items)))

    @staticmethod
    def _validate(obj):
        return isinstance(obj, (AwkwardArray, ak.Array, ak.Record)) or isinstance(obj.values, AwkwardArray)

    def to_arrow(self):
        return self.arr._data.to_arrow()

    def cartesian(self, other, **kwargs):
        if isinstance(other, AwkwardArray):
            other = other._data
        return AwkwardArray(ak.cartesian([self.arr._data, other], **kwargs))

    def __getattr__(self, item):
        # replace with concrete implementations of all top-level ak functions
        if item not in funcs:
            raise AttributeError
        func = getattr(ak, item)

        @functools.wraps(func)
        def f(*others, **kwargs):
            others = [
                other._data
                if isinstance(getattr(other, "_data", None), ak.Array)
                else other
                for other in others
            ]
            ak_arr = func(self.arr._data, *others, **kwargs)
            # TODO: special case to carry over index and name information where output
            #  is similar to input, e.g., has same length
            if isinstance(ak_arr, ak.Array):
                # TODO: perhaps special case here if the output can be represented
                #  as a regular num/cupy array
                return AwkwardArray(ak_arr)
            return ak_arr

        return f
