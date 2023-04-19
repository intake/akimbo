import pandas as pd

from awkward_pandas import AwkwardDtype

s = pd.Series(["hello"], dtype="awkward")


def register_dtype():
    from dask.dataframe.extensions import make_array_nonempty

    data = [[1], [1, 2, None]]

    @make_array_nonempty.register(AwkwardDtype)
    def _(x):
        return pd.Series(data, dtype="awkward")


try:
    from dask.dataframe.accessor import Accessor
except (ImportError, ModuleNotFoundError):
    Accessor = object


class DaskAwkwardAccessor(Accessor):
    _accessor_name = "ak"

    _accessor_methods = dir(s.ak)

    _accessor_properties = ()

    # TODO: dask-awkward could take over here
    for method in _accessor_methods:

        def _(self, *args, method=method, **kwargs):
            def __(s, method=method):
                return s.ak.__getattr__(method)(*args, **kwargs)

            return self._series.map_partitions(__)

        locals()[method] = _


try:
    import dask
    from dask.dataframe.extensions import register_series_accessor

    register_dtype()
    register_series_accessor("ak")(DaskAwkwardAccessor)
except (ImportError, ModuleNotFoundError):
    dask = False
