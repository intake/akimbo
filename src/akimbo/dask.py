import functools
from typing import Iterable

import awkward as ak
import dask.dataframe as dd
from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from akimbo.mixin import Accessor as AkAccessor
from akimbo.mixin import df_methods, series_methods
from akimbo.pandas import PandasAwkwardAccessor


class DaskAwkwardAccessor(AkAccessor):
    """Perform awkward operations on a dask series or frame

    These operations are lazy, because of how dask works. Note
    that we use mapping operations here, so any action on
    axis==0 or 1 will produce results per partition, which
    you must then combine.

    To perform intra-partition operations, we recommend you
    use the ``.to_dask_awkward`` method.

    Correct arrow dtypes will be deduced when the input is
    also arrow, which is now the default for the dask
    "dataframe.dtype_backend" config options.
    """

    series_type = dd.Series
    dataframe_type = dd.DataFrame
    aggregations = False  # you need dask-awkward for that

    @staticmethod
    def _to_tt(data):
        # self._obj._meta.convert_dtypes(dtype_backend="pyarrow")
        data = data._meta if hasattr(data, "_meta") else data
        arr = PandasAwkwardAccessor.to_arrow(data)
        return ak.to_backend(ak.from_arrow(arr), "typetracer")

    def to_dask_awkward(self):
        """Convert to dask-awkard.Array object

        This make a s single complex awkward array type out of one or more columns.
        You would do this, in order to use dask-awkward's more advanced inter-
        partition aggregations and optimisation.
        See https://dask-awkward.readthedocs.io/

        c.f., dask_awkward.to_dataframe
        """
        import dask_awkward as dak

        tt = self._to_tt(self._obj)
        return dak.lib.core.new_array_object(
            self._obj.dask, divisions=self._obj.divisions, name=self._obj._name, meta=tt
        )

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
                meta = None

            def inner(data, _=DaskAwkwardAccessor):
                import akimbo.pandas  # noqa: F401

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
                    import akimbo.pandas  # noqa: F401

                    # data and others are pandas objects here
                    return getattr(data.ak, item)(*others, **kwargs)

                return self._obj.map_partitions(func2, meta=func(orig))

        else:
            raise AttributeError(item)
        return f

    def __dir__(self) -> Iterable[str]:
        attrs = (_ for _ in dir(self._obj._meta.ak.array) if not _.startswith("_"))
        meths = series_methods if self.is_series(self._obj) else df_methods
        return sorted(set(attrs) | set(meths))


register_series_accessor("ak")(DaskAwkwardAccessor)
register_dataframe_accessor("ak")(DaskAwkwardAccessor)
