import awkward as ak
import polars as pl

from awkward_pandas.mixin import Accessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class PolarsAwkwardAccessor(Accessor):
    series_type = pl.Series
    dataframe_type = pl.DataFrame

    def to_output(self, arr: ak.Array) -> pl.DataFrame | pl.Series:
        # Series Vs DataFrame?
        return pl.from_arrow(ak.to_arrow(arr, extensionarray=False))

    @property
    def arrow(self) -> ak.Array:
        return self._obj.to_arrow()
