import functools
from typing import Callable

import awkward as ak
import cudf
from cudf import DataFrame, Series
from cudf.core.column.string import StringMethods

from akimbo.ak_from_cudf import cudf_to_awkward as from_cudf
from akimbo.mixin import Accessor
from akimbo.strings import StringAccessor
from akimbo.apply_tree import dec


def match_string(arr):
    return arr.parameters.get("__array__", "") == "string"


class CudfStringAccessor(StringAccessor):
    def decode(self, encoding: str = "utf-8"):
        raise NotImplementedError("cudf does not support bytearray type, so we can't automatically identify them")

    def encode(self, encoding: str = "utf-8"):
        raise NotImplementedError("cudf does not support bytearray type")


for meth in dir(StringMethods):
    if meth.startswith("_"):
        continue

    def f(lay, *args, method=meth, **kwargs):
        if not match_string(lay):
            return

        col = getattr(StringMethods(cudf.Series(lay._to_cudf(cudf, None, len(lay)))), method)(*args, **kwargs)
        return from_cudf(col).layout

    setattr(CudfStringAccessor, meth, dec(func=f, match=match_string, inmode="ak"))


class CudfAwkwardAccessor(Accessor):
    series_type = Series
    dataframe_type = DataFrame

    @classmethod
    def _to_output(cls, arr):
        if isinstance(arr, ak.Array):
            return ak.to_cudf(arr)
        return arr

    @classmethod
    def to_array(cls, data) -> ak.Array:
        return from_cudf(data)

    @property
    def array(self) -> ak.Array:
        return self.to_array(self._obj)

    @property
    def str(self):
        """Nested string operations"""
        # need to find string ops within cudf
        return CudfStringAccessor(self)

    @property
    def dt(self):
        """Nested datetime operations"""
        # need to find datetime ops within cudf
        raise NotImplementedError

    def apply(self, fn: Callable, *args, **kwargs):
        if "CPUDispatcher" in str(fn):
            # auto wrap original function for GPU
            raise NotImplementedError
        super().apply(fn, *args, **kwargs)


@property  # type:ignore
def ak_property(self):
    return CudfAwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
