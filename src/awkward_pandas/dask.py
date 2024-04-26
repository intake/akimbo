import functools

import awkward as ak
import dask.dataframe as dd
from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)

from awkward_pandas.mixin import Accessor as AkAccessor


class DaskAwkwardAccessor(AkAccessor):
    series_type = dd.Series
    dataframe_type = dd.DataFrame
    aggregations = (
        False  # you need dask-awkward for that, which we could optionally do here
    )

    def __getattr__(self, item):
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        if func:
            orig = self._obj.head()

            @functools.wraps(func)
            def f(*others, **kwargs):
                def func2(data):
                    return getattr(data.ak, item)(*others, **kwargs)

                return self._obj.map_partitions(func2, meta=func(orig))

        else:
            raise AttributeError(item)
        return f


register_series_accessor("ak")(DaskAwkwardAccessor)
register_dataframe_accessor("ak")(DaskAwkwardAccessor)
