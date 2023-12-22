import functools
from typing import Callable, Iterable

import awkward as ak
import polars as pl


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class AwkwardOperations:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __dir__(self) -> Iterable[str]:
        return [
            _
            for _ in (dir(ak))
            if not _.startswith(("_", "ak_")) and not _[0].isupper()
        ] + ["apply", "array", "to_column"]

    def apply(self, fn: Callable) -> pl.DataFrame:
        """Perform function on all the values of the series"""
        out = fn(self.array)
        result = pl.from_arrow(ak.to_arrow(out, extensionarray=False))
        return result

    def __getitem__(self, item):
        out = self.array.__getitem__(item)
        result = pl.from_arrow(ak.to_arrow(out, extensionarray=False))
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
                    other.ak.array if isinstance(other, pl.DataFrame) else other
                    for other in others
                ]

                ak_arr = func(self.array, *others, **kwargs)
                if isinstance(ak_arr, ak.Array):
                    return pl.from_arrow(ak.to_arrow(ak_arr, extensionarray=False))
                return ak_arr

        else:
            raise AttributeError

        return f
