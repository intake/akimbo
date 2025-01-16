import polars as pl

from akimbo.mixin import Accessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class PolarsAwkwardAccessor(Accessor):
    """Perform awkward operations on a polars series or dataframe

    This is for *eager* operations. A Lazy version may eventually be made.
    """

    series_type = pl.Series
    dataframe_type = pl.DataFrame

    @classmethod
    def _arrow_to_series(cls, arr):
        return pl.from_arrow(arr)

    @classmethod
    def to_arrow(cls, data):
        return data.to_arrow()

    def pack(self):
        # polars already implements this directly
        return self._obj.to_struct()
