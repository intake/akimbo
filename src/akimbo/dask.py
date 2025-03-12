import awkward as ak
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from dask.dataframe.accessor import (
    register_dataframe_accessor,
    register_series_accessor,
)

from akimbo.mixin import LazyAccessor
from akimbo.pandas import PandasAwkwardAccessor


class DaskAwkwardAccessor(LazyAccessor):
    """Perform awkward operations on a dask series or frame

    These operations are lazy, because of how dask works. Note
    that we use mapping operations here, so any action on
    axis==0 or 1 will produce results per partition, which
    you must then combine.

    To perform intra-partition operations, we recommend you
    use the ``.to_dask_awkward`` method.

    Correct arrow dtypes will be deduced when the input is
    also arrow, which is now the default for the dask
    "dataframe.dtype_backend" config options.
    """

    series_type = dd.Series
    dataframe_type = dd.DataFrame

    @staticmethod
    def _to_tt(data):
        # self._obj._meta.convert_dtypes(dtype_backend="pyarrow")
        data = data._meta if hasattr(data, "_meta") else data
        arr = PandasAwkwardAccessor.to_arrow(data)
        if isinstance(arr, pa.ChunkedArray) and len(arr) == 0:
            arr = arr.combine_chunks()
        return ak.to_backend(ak.from_arrow(arr), "typetracer")

    def to_dask_awkward(self):
        """Convert to dask-awkard.Array object

        This make a single complex awkward array type out of one or more columns.
        You would do this, in order to use dask-awkward's more advanced inter
        partition aggregations and column/buffer IO optimisation.
        See https://dask-awkward.readthedocs.io/

        c.f., dask_awkward.to_dataframe
        """
        import dask_awkward as dak

        tt = self._to_tt(self._obj)
        return dak.lib.core.new_array_object(
            self._obj.dask, divisions=self._obj.divisions, name=self._obj._name, meta=tt
        )

    def __getattr__(self, item):
        if self.subaccessor and isinstance(item, str):
            item = getattr(self.subaccessors[self.subaccessor], item)
        elif isinstance(item, str) and item in self.subaccessors:
            return DaskAwkwardAccessor(
                self._obj, subaccessor=item, behavior=self._behavior
            )

        def select(*inargs, where=None, **kwargs):
            orig = self._obj.head()

            def func2(data):
                import akimbo.pandas  # noqa: F401

                others = [
                    (k._obj if isinstance(k, DaskAwkwardAccessor) else k)
                    for k in inargs
                ]
                if isinstance(item, str):
                    # work on pandas API
                    return getattr(PandasAwkwardAccessor(data), item)(*others, **kwargs)
                else:
                    # ak to ak
                    arr = data.ak.array
                    # others =
                    if where:
                        part = arr[where]
                        arr = ak.with_field(arr, part, where)
                        others = [_[where] for _ in others]
                    else:
                        part = arr
                    out = item(part, *others, **kwargs)
                    if where:
                        out = ak.with_field(arr, out, where)
                    out = pd.arrays.ArrowExtensionArray(
                        ak.to_arrow(out, extensionarray=False)
                    )
                    return pd.Series(out)

            out0 = func2(orig)
            return self._obj.map_partitions(func2, meta=out0)

        return select

    def __dir__(self) -> list[str]:
        return sorted(super().__dir__() + ["to_dask_awkward"])


register_series_accessor("ak")(DaskAwkwardAccessor)
register_dataframe_accessor("ak")(DaskAwkwardAccessor)
