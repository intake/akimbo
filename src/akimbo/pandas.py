from collections.abc import Iterable

import awkward as ak
import pandas as pd
import pyarrow
import pyarrow as pa

from akimbo.mixin import EagerAccessor


@pd.api.extensions.register_series_accessor("ak")
@pd.api.extensions.register_dataframe_accessor("ak")
class PandasAwkwardAccessor(EagerAccessor):
    """Perform awkward operations on pandas data

    Nested structures are handled using arrow as the
    storage backend. If you use pandas object columns
    (python lists, dicts, strings), they will be converted
    on any access to a ``.ak`` method.
    """

    series_type = pd.Series
    dataframe_type = pd.DataFrame

    @classmethod
    def to_arrow(cls, data):
        if isinstance(data, ak.Array):
            return ak.to_arrow(data)
        if isinstance(data, ak.Record):
            return ak.to_arrow_table(data)
        if isinstance(data, (pyarrow.Array, pyarrow.Table)):
            return data
        if isinstance(data, cls.series_type):
            if getattr(data.dtype, "storage", "") == "pyarrow":
                return data.array.__arrow_array__()
            return pa.array(data)
        return pa.table(data)

    def to_output(self, data=None):
        # override to apply index
        data: ak.Array = data if data is not None else self.array
        if not isinstance(data, Iterable):
            return data
        arr = pd.arrays.ArrowExtensionArray(ak.to_arrow(data, extensionarray=False))
        if self._obj is not None and len(arr) == len(self._obj.index):
            return pd.Series(arr, index=self._obj.index)
        else:
            return pd.Series(arr)

    @staticmethod
    def _validate(_):
        # required by pandas
        return True
