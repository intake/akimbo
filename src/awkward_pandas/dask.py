import functools

import awkward as ak
import dask.dataframe as dd
from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from awkward_pandas.mixin import Accessor as AkAccessor
from awkward_pandas.pandas import PandasAwkwardAccessor


class DaskAwkwardAccessor(AkAccessor):
    series_type = dd.Series
    dataframe_type = dd.DataFrame
    aggregations = False  # you need dask-awkward for that

    @staticmethod
    def _to_tt(data):
        # self._obj._meta.convert_dtypes(dtype_backend="pyarrow")
        data = data._meta if hasattr(data, "_meta") else data
        arr = PandasAwkwardAccessor.to_arrow(data)
        return ak.to_backend(ak.from_arrow(arr), "typetracer")

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            try:
                tt = self._to_tt(self._obj)
                ar = (
                    ak.to_backend(ar) if isinstance(ar, (ak.Array, ak.Record)) else ar
                    for ar in args
                )
                ar = [self._to_tt(ar) if hasattr(ar, "ak") else ar for ar in ar]
                out = op(tt, *ar, **kwargs)
                meta = PandasAwkwardAccessor._to_output(
                    ak.typetracer.length_zero_if_typetracer(out)
                )
            except (ValueError, TypeError):
                # could make our own fallback as follows, but dask will guess anyway
                # orig = self._obj.head()
                # ar = (ar.head() if hasattr(ar, "ak") else ar for ar in args)
                # meta = PandasAwkwardAccessor._to_output(op(orig.ak.array, *ar, **kwargs))
                meta = None

            def inner(data, _=DaskAwkwardAccessor):
                import awkward_pandas.pandas  # noqa: F401

                ar2 = (ar.ak.array if hasattr(ar, "ak") else ar for ar in args)
                out = op(data.ak.array, *ar2, **kwargs)
                return PandasAwkwardAccessor._to_output(out)

            return self._obj.map_partitions(inner, meta=meta)

        return run

    def __getattr__(self, item):
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        if func:
            orig = self._obj.head()

            @functools.wraps(func)
            def f(*others, **kwargs):
                def func2(data):
                    import awkward_pandas.pandas  # noqa: F401

                    # data and others are pandas objects here
                    return getattr(data.ak, item)(*others, **kwargs)

                return self._obj.map_partitions(func2, meta=func(orig))

        else:
            raise AttributeError(item)
        return f


register_series_accessor("ak")(DaskAwkwardAccessor)
register_dataframe_accessor("ak")(DaskAwkwardAccessor)
