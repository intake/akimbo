import functools
from typing import Callable

import awkward as ak
import numpy as np

from akimbo.utils import NoAttributes

try:
    import cudf
    from cudf import DataFrame, Series
    from cudf import _lib as libcudf
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.string import StringMethods
except ImportError:
    StringMethods = NoAttributes()
    DatetimeColumn = NoAttributes()
    libcudf = NoAttributes()
    DataFrame = Series = NoAttributes()


from akimbo.ak_from_cudf import cudf_to_awkward as from_cudf
from akimbo.apply_tree import dec, leaf
from akimbo.datetimes import DatetimeAccessor
from akimbo.datetimes import match as match_t
from akimbo.mixin import Accessor
from akimbo.strings import StringAccessor


def match_string(arr):
    return arr.parameters.get("__array__", "") == "string"


class CudfStringAccessor(StringAccessor):
    """String operations on nested/var-length data"""

    def decode(self, encoding: str = "utf-8"):
        raise NotImplementedError(
            "cudf does not support bytearray type, so we can't automatically identify them"
        )

    def encode(self, encoding: str = "utf-8"):
        raise NotImplementedError("cudf does not support bytearray type")


def dec_cu(op, match=match_string):
    @functools.wraps(op)
    def f(lay, **kwargs):
        # op(column, ...)->column
        col = op(lay._to_cudf(cudf, None, len(lay)), **kwargs)
        if hasattr(cudf.Series, "_from_column"):
            return from_cudf(cudf.Series._from_column(col)).layout
        return from_cudf(cudf.Series(col)).layout

    return dec(func=f, match=match, inmode="ak")


for meth in dir(StringMethods):
    if meth.startswith("_"):
        continue

    @functools.wraps(getattr(StringMethods, meth))
    def f(lay, method=meth, **kwargs):
        # this is different from dec_cu, because we need to instantiate StringMethods
        # before getting the method from it
        if hasattr(cudf.Series, "_from_column"):
            ser = cudf.Series._from_column(lay._to_cudf(cudf, None, len(lay)))
        else:
            ser = cudf.Series(lay._to_cudf(cudf, None, len(lay)))
        col = getattr(StringMethods(ser), method)(**kwargs)
        return from_cudf(col).layout

    setattr(CudfStringAccessor, meth, dec(func=f, match=match_string, inmode="ak"))


class CudfDatetimeAccessor(DatetimeAccessor):
    ...


for meth in dir(DatetimeColumn):
    if meth.startswith("_") or meth == "strptime":
        # strptime belongs in .str, not here!
        continue

    @functools.wraps(getattr(DatetimeColumn, meth))
    def f(lay, method=meth, **kwargs):
        # this is different from dec_cu, because we need to instantiate StringMethods
        # before getting the method from it
        m = getattr(lay._to_cudf(cudf, None, len(lay)), method)
        if callable(m):
            col = m(**kwargs)
        else:
            # attributes giving components
            col = m
        if hasattr(cudf.Series, "_from_column"):
            return from_cudf(cudf.Series._from_column(col)).layout
        return from_cudf(cudf.Series(col)).layout

    if isinstance(getattr(DatetimeColumn, meth), property):
        setattr(
            CudfDatetimeAccessor,
            meth,
            property(dec(func=f, match=match_t, inmode="ak")),
        )
    else:
        setattr(CudfDatetimeAccessor, meth, dec(func=f, match=match_t, inmode="ak"))


class CudfAwkwardAccessor(Accessor):
    """Operations on cuDF dataframes on the GPU.

    Data are kept in GPU memory and use views rather than copies where
    possible.
    """

    series_type = Series
    dataframe_type = DataFrame

    @classmethod
    def _arrow_to_series(cls, data):
        # this implies CPU->GPU copy
        return Series(data)

    @classmethod
    def _to_output(cls, arr):
        if isinstance(arr, ak.Array):
            return ak.to_cudf(arr)
        elif isinstance(arr, ak.contents.Content):
            return arr._to_cudf(cudf, None, len(arr))
        return arr

    @classmethod
    def to_array(cls, data) -> ak.Array:
        if isinstance(data, cls.series_type):
            return from_cudf(data)
        out = {}
        for col in data.columns:
            out[col] = from_cudf(data[col])
        return ak.Array(out)

    @property
    def array(self) -> ak.Array:
        return self.to_array(self._obj)

    @property
    def str(self):
        """Nested string operations"""
        # need to find string ops within cudf
        return CudfStringAccessor(self)

    try:
        cast = dec_cu(libcudf.unary.cast, match=leaf)
    except AttributeError:

        def cast_inner(col, dtype):
            return cudf.core.column.ColumnBase(
                col.data,
                size=len(col),
                dtype=np.dtype(dtype),
                mask=None,
                offset=0,
                children=(),
            )

        cast = dec_cu(cast_inner, match=leaf)

    @property
    def dt(self):
        """Nested datetime operations"""
        # need to find datetime ops within cudf
        return CudfDatetimeAccessor(self)

    def apply(self, fn: Callable, *args, **kwargs):
        if "CPUDispatcher" in str(fn):
            # auto wrap original function for GPU
            raise NotImplementedError
        super().apply(fn, *args, **kwargs)


@property  # type:ignore
def ak_property(self):
    return CudfAwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
DataFrame.ak = ak_property  # no official register function?
