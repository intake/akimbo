from typing import Callable

import awkward as ak
from cudf import DataFrame, Series

from awkward_pandas.ak_from_cudf import cudf_to_awkward as from_cudf
from awkward_pandas.mixin import Accessor


class CudfAwkwardAccessor(Accessor):
    series_type = Series
    dataframe_type = DataFrame

    @classmethod
    def _to_output(cls, arr):
        if isinstance(arr, ak.Array):
            return ak.to_cudf(arr)
        return arr

    @property
    def array(self) -> ak.Array:
        return from_cudf(self._obj)

    @property
    def str(self):
        """Nested string operations"""
        # need to find string ops within cudf
        raise NotImplementedError

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
