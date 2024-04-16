import functools
from typing import Callable, Iterable, Union

import awkward as ak
import polars as pl

from awkward_pandas.mixin import ArithmeticMixin


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class AwkwardOperations(ArithmeticMixin):
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __array_function__(self, *args, **kwargs):
        return self.array.__array_function__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] == "__call__":
            return args[0](self.array, *args[3:], **kwargs)
        raise NotImplementedError

    def __dir__(self) -> Iterable[str]:
        return [
            _
            for _ in (dir(ak))
            if not _.startswith(("_", "ak_")) and not _[0].isupper()
        ] + ["apply", "array"]

    def apply(self, fn: Callable) -> pl.DataFrame:
        """Perform function on all the values of the series"""
        out = fn(self.array)
        return ak_to_polars(out)

    def __getitem__(self, item):
        # scalars?
        out = self.array.__getitem__(item)
        result = ak_to_polars(out)
        return result

    @property
    def array(self):
        return ak.from_arrow(self._df.to_arrow())

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
                    return ak_to_polars(ak_arr)
                return ak_arr

        else:
            raise AttributeError(item)
        return f

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            return ak_to_polars(op(self.array, *args, **kwargs))

        return run

    _create_arithmetic_method = _create_op
    _create_comparison_method = _create_op
    _create_logical_method = _create_op


AwkwardOperations._add_all()


def ak_to_polars(arr: ak.Array) -> Union[pl.DataFrame, pl.Series]:
    return pl.from_arrow(ak.to_arrow(arr, extensionarray=False))
