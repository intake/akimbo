import awkward as ak
import polars as pl

from awkward_pandas.mixin import Accessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class AwkwardOperations(Accessor):
    series_type = pl.Series
    dataframe_type = pl.DataFrame

    def to_output(self, arr: ak.Array) -> pl.DataFrame | pl.Series:
        # Series Vs DataFrame?
        return pl.from_arrow(ak.to_arrow(arr, extensionarray=False))

    @property
    def array(self) -> ak.Array:
        return ak.from_arrow(self._obj.to_arrow())

    def merge(self):
        # TODO: this is almost totally generic
        if not self.is_dataframe(self._obj):
            raise ValueError("Can only merge on a dataframe")
        out = {}
        for k in self._obj.columns:
            out[k] = ak.from_arrow(self._obj[k].to_arrow())
        arr = ak.Array(out)
        return self.to_output(arr)


AwkwardOperations._add_all()
