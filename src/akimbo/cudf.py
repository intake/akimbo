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
from akimbo.datetimes import match as match_dt
from akimbo.datetimes import methods as dt_methods
from akimbo.mixin import EagerAccessor, numeric
from akimbo.ops import rename
from akimbo.strings import _SA_METHODMAPPING, StringAccessor, match_string, strptime


def dec_cu(op, match=match_string):
    @functools.wraps(op)
    def f(lay, **kwargs):
        # op(column, ...)->column
        col = op(lay._to_cudf(cudf, None, len(lay)), **kwargs)
        if hasattr(cudf.Series, "_from_column"):
            return from_cudf(cudf.Series._from_column(col)).layout
        return from_cudf(cudf.Series(col)).layout

    return dec(func=f, match=match, inmode="ak")


class CudfStringAccessor(StringAccessor):
    """String operations on nested/var-length data"""

    def decode(self, arr, encoding: str = "utf-8"):
        raise NotImplementedError(
            "cudf does not support bytearray type, so we can't automatically identify them"
        )

    def encode(self, arr, encoding: str = "utf-8"):
        raise NotImplementedError("cudf does not support bytearray type")

    @staticmethod
    def method_name(attr: str) -> str:
        return _SA_METHODMAPPING.get(attr, attr)

    def __getattr__(self, attr: str) -> callable:
        attr = self.method_name(attr)

        @functools.wraps(getattr(StringMethods, attr))
        def f(lay, **kwargs):
            col = lay._to_cudf(cudf, None, len(lay))
            if hasattr(cudf.Series, "_from_column"):
                ser = cudf.Series._from_column(col)
            else:
                ser = cudf.Series(col)
            col = getattr(StringMethods(ser), attr)(**kwargs)
            return from_cudf(col).layout

        return dec(f, match=match_string, inmode="ak")

    strptime = staticmethod(dec_cu(strptime, match=match_string))


class CudfDatetimeAccessor(DatetimeAccessor):
    def __getattr__(self, attr):
        @functools.wraps(getattr(DatetimeColumn, attr))
        def f(lay, **kwargs):
            col = lay._to_cudf(cudf, None, len(lay))
            col = getattr(col, attr)
            if callable(col):
                # as opposed to an attribute
                col = col(**kwargs)
            if hasattr(cudf.Series, "_from_column"):
                ser = cudf.Series._from_column(col)
            else:
                ser = cudf.Series(col)
            return from_cudf(ser).layout

        return dec(f, match=match_dt, inmode="ak")

    def __dir__(self):
        return dt_methods


def _cast_inner(col, dtype):
    try:
        return libcudf.unary.cast(col, dtype)
    except AttributeError:
        return cudf.core.column.ColumnBase(
            col.data,
            size=len(col),
            dtype=np.dtype(dtype),
            mask=None,
            offset=0,
            children=(),
        )


class CudfAwkwardAccessor(EagerAccessor):
    """Operations on cuDF dataframes on the GPU.

    Data are kept in GPU memory and use views rather than copies where
    possible.
    """

    funcs = {"cast": dec_cu(_cast_inner, match=leaf), "rename": rename}

    series_type = Series
    dataframe_type = DataFrame
    subaccessors = {"str": CudfStringAccessor, "dt": CudfDatetimeAccessor}

    def to_output(self, data=None):
        arr = data if data is not None else self._obj
        if isinstance(arr, ak.Array):
            return ak.to_cudf(arr)
        elif isinstance(arr, ak.contents.Content):
            return arr._to_cudf(cudf, None, len(arr))
        return arr

    @property
    def array(self):
        data = self._obj
        if isinstance(data, self.series_type):
            return from_cudf(data)
        out = {}
        for col in data.columns:
            out[col] = from_cudf(data[col])
        return ak.Array(out)

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            args = [
                ak.Array([_], backend="cuda")
                if isinstance(_, (str, int, float, np.number))
                else _
                for _ in args
            ]
            return self.transform(op, *args, match=numeric, **kwargs)

        return run

    def apply(self, fn: Callable, *args, **kwargs):
        if "CPUDispatcher" in str(fn):
            # auto wrap original function for GPU numba func
            raise NotImplementedError
        super().apply(fn, *args, **kwargs)


@property  # type:ignore
def ak_property(self):
    return CudfAwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
DataFrame.ak = ak_property  # no official register function?
