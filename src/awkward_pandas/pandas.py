import awkward as ak
import pandas as pd
import pyarrow as pa

from awkward_pandas.mixin import Accessor


@pd.api.extensions.register_series_accessor("ak")
@pd.api.extensions.register_dataframe_accessor("ak")
class AwkwardAccessor(Accessor):
    @property
    def arrow(self):
        return pa.array(self._obj)

    def to_output(self, data):
        return pd.Series(
            pd.arrays.ArrowExtensionArray(ak.to_arrow(data, extensionarray=False))
        )

    @staticmethod
    def _validate(_):
        # required by pandas
        return True
