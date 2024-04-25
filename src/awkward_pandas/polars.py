import functools

import awkward as ak
import polars as pl

from awkward_pandas.mixin import Accessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class AwkwardOperations(Accessor):
    series_type = pl.Series
    dataframe_type = pl.DataFrame

    @property
    def str(self):
        """String operations"""
        from awkward_pandas.strings import StringAccessor

        return StringAccessor(self)

    def __getattr__(self, item):
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        if func:

            @functools.wraps(func)
            def f(*others, **kwargs):
                others = [
                    other.ak.array
                    if isinstance(other, (pl.DataFrame, pl.Series))
                    else other
                    for other in others
                ]
                kwargs = {
                    k: v.ak.array if isinstance(v, (pl.DataFrame, pl.Series)) else v
                    for k, v in kwargs.items()
                }

                ak_arr = func(self.array, *others, **kwargs)
                if isinstance(ak_arr, ak.Array):
                    return self.to_output(ak_arr)
                return ak_arr

        else:
            raise AttributeError(item)
        return f

    def to_output(self, arr: ak.Array) -> pl.DataFrame | pl.Series:
        # Series Vs DataFrame?
        return pl.from_arrow(ak.to_arrow(arr, extensionarray=False))

    @property
    def array(self) -> ak.Array:
        return ak.from_arrow(self._obj.to_arrow())

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            return self.to_output(op(self.array, *args, **kwargs))

        return run

    def merge(self):
        # TODO: this is almost totally generic
        if not self.is_dataframe(self._obj):
            raise ValueError("Can only merge on a dataframe")
        out = {}
        for k in self._obj.columns:
            out[k] = ak.from_arrow(self._obj[k].to_arrow())
        arr = ak.Array(out)
        return self.to_output(arr)

    def unmerge(self):
        arr = self.array
        if not arr.fields:
            raise ValueError
        out = {k: self.to_output(arr[k]) for k in arr.fields}
        return self.dataframe_type(out)


AwkwardOperations._add_all()
