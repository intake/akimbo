import awkward as ak
import pandas as pd
import pyarrow as pa

from awkward_pandas.mixin import Accessor


@pd.api.extensions.register_series_accessor("ak")
@pd.api.extensions.register_dataframe_accessor("ak")
class PandasAwkwardAccessor(Accessor):
    series_type = pd.Series
    dataframe_type = pd.DataFrame

    @classmethod
    def to_arrow(cls, data):
        if cls.is_series(data):
            return pa.array(data)
        return pa.table(data)

    @classmethod
    def _to_output(cls, data):
        return pd.Series(
            pd.arrays.ArrowExtensionArray(ak.to_arrow(data, extensionarray=False))
        )

    def to_output(self, data=None):
        # override to apply index
        data = data if data is not None else self.array
        arr = pd.arrays.ArrowExtensionArray(ak.to_arrow(data, extensionarray=False))
        if self._obj is not None and len(arr) == len(self._obj.index):
            return pd.Series(arr, index=self._obj.index)
        else:
            return arr

    @staticmethod
    def _validate(_):
        # required by pandas
        return True
