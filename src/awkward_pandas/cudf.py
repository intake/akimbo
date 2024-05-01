from cudf import DataFrame, Series
import awkward as ak

from awkward_pandas.mixin import Accessor
from awkward_pandas.ak_from_cudf import cudf_to_awkward as from_cudf


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


@property  # type:ignore
def ak_property(self):
    return CudfAwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
