import awkward as ak
import polars as pl

from akimbo.mixin import Accessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class PolarsAwkwardAccessor(Accessor):
    """Perform awkward operations on a polars series or dataframe"""

    series_type = pl.Series
    dataframe_type = pl.DataFrame

    @classmethod
    def _to_output(cls, arr):
        return pl.from_arrow(ak.to_arrow(arr, extensionarray=False))

    @classmethod
    def to_arrow(cls, data):
        return data.to_arrow()
