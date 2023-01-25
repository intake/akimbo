import functools
import inspect

import awkward as ak
import pandas as pd

from awkward_pandas.array import AwkwardExtensionArray
from awkward_pandas.dtype import AwkwardDtype
from awkward_pandas.strings import (
    all_bytes,
    all_strings,
    decode,
    dir_str,
    encode,
    get_func,
)

funcs = [n for n in dir(ak) if inspect.isfunction(getattr(ak, n))]


@pd.api.extensions.register_series_accessor("ak")
class AwkwardAccessor:
    def __init__(self, pandas_obj):
        if not self._validate(pandas_obj):
            raise AttributeError("ak accessor called on incompatible data")
        self._obj = pandas_obj
        self._arr = None

    @property
    def extarray(self):
        if self._arr is None:
            if isinstance(self._obj, AwkwardExtensionArray):
                self._arr = self._obj
            elif isinstance(self._obj.dtype, AwkwardDtype) and isinstance(
                self._obj, pd.Series
            ):
                # this is a pandas Series that contains an Awkward
                self._arr = self._obj.values
            elif isinstance(self._obj.dtype, AwkwardDtype):
                # a dask series - figure out what to do here
                raise NotImplementedError
            else:
                # this recreates series, possibly by iteration
                self._arr = AwkwardExtensionArray(self._obj)
        return self._arr

    @property
    def array(self):
        """Get underlying awkward array"""
        return self.extarray._data

    def __getitem__(self, items):
        """Extract components using awkward indexing"""
        ds = self.array.__getitem__(items)
        index = None
        if items[0]:
            if not isinstance(items[0], str) and not (
                isinstance(items[0], list) and isinstance(items[0][0], str)
            ):
                index = self._obj.index[items[0]]
        return pd.Series(AwkwardExtensionArray(ds), index=index)

    def to_column(self):
        """Convert awkward series to regular pandas type

        Will convert to numpy or string[pyarrow] if appropriate.
        May fail if the conversion cannot be done.
        """
        data = self.array
        if data.ndim > 1:
            raise ValueError
        # TODO: if all_strings(data) - accept ?str
        if data.layout.parameter("__array__") == "string":
            from pandas.core.arrays.string_arrow import ArrowStringArray

            new_ak_array = ak.to_arrow(
                data,
                string_to32=True,
                extensionarray=False,
            )
            return pd.Series(ArrowStringArray(new_ak_array))
        else:
            return pd.Series(ak.to_numpy(data))

    def to_columns(self, cull=True, extract_all=False, awkward_name="awkward-data"):
        """Extract columns from an awkward series

        Where the series is a record type, each field may become a regular
        pandas column.

        Parameters
        ----------
        cull: bool
            For those columns that we convert into regular ones, remove them
            from the original awkward series if True
        extract_all: bool
            If False (default), only extract columns that can turn into normal
            pandas columns. If True, all columns will be extracted, but those
            that cannot be converted retain "awkward" type
        awkward_name: str
            If there are leftover columns in the original series, in the
            resultant dataframe, these leftovers will get this column name

        Returns
        -------
        pd.DataFrame
        """
        s = self._obj
        fields = self.array.fields
        out = {}
        for field in fields:
            try:
                out[field] = s.ak[field].ak.to_column()
            except Exception:
                if extract_all:
                    out[field] = s.ak[field]
        if cull and extract_all:
            pass
        elif cull:
            n = s.name or awkward_name
            outfields = [_ for _ in fields if _ not in out]
            if outfields:
                out[n] = s.ak[outfields]
        else:
            out[s.name] = s
        return pd.DataFrame(out)

    def encode(self, encoding="utf-8"):
        """bytes -> string"""
        return pd.Series(AwkwardExtensionArray(encode(self.array)))

    def decode(self, encoding="utf-8"):
        """string -> bytes"""
        return pd.Series(AwkwardExtensionArray(decode(self.array)))

    @staticmethod
    def _validate(obj):
        return isinstance(
            obj, (AwkwardExtensionArray, ak.Array, ak.Record)
        ) or isinstance(obj.values, AwkwardExtensionArray)

    # def to_arrow(self):
    #    return self.array.to_arrow()

    # def cartesian(self, other, **kwargs):
    #    if isinstance(other, AwkwardExtensionArray):
    #        other = other._data
    #    return AwkwardExtensionArray(ak.cartesian([self.array, other], **kwargs))

    def __getattr__(self, item):
        """Call awkward namespace function on a series"""
        # replace with concrete implementations of all top-level ak functions
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        utf8 = all_strings(self.array.layout)
        byte = all_bytes(self.array.layout)
        if func:

            @functools.wraps(func)
            def f(*others, **kwargs):
                others = [
                    other._data
                    if isinstance(getattr(other, "_data", None), ak.Array)
                    else other
                    for other in others
                ]
                ak_arr = func(self.array, *others, **kwargs)
                # TODO: special case to carry over index and name information where output
                #  is similar to input, e.g., has same length
                if isinstance(ak_arr, ak.Array):
                    # TODO: perhaps special case here if the output can be represented
                    #  as a regular num/cupy array
                    return pd.Series(
                        AwkwardExtensionArray(ak_arr), index=self._obj.index
                    )
                return ak_arr

        elif utf8 or byte:
            func = get_func(item, utf8=utf8)
            if func is None:
                raise AttributeError

            @functools.wraps(func)
            def f(*args, **kwargs):
                import pyarrow

                if utf8:
                    data = pyarrow.chunked_array(
                        [
                            ak.to_arrow(
                                self.array,
                                extensionarray=False,
                                string_to32=True,
                                bytestring_to32=True,
                            )
                        ]
                    )
                else:
                    data = pyarrow.chunked_array(
                        [
                            ak.to_arrow(
                                self.decode().values._data,
                                extensionarray=False,
                                string_to32=True,
                                bytestring_to32=True,
                            )
                        ]
                    )

                arrow_arr = func(data, *args, **kwargs)
                # TODO: special case to carry over index and name information where output
                #  is similar to input, e.g., has same length
                ak_arr = ak.from_arrow(arrow_arr)
                if all_strings(ak_arr.layout) and not utf8:
                    # back to bytes-array
                    ak_arr = encode(ak.Array(ak_arr.layout.content))
                if isinstance(ak_arr, ak.Array):
                    # TODO: perhaps special case here if the output can be represented
                    #  as a regular num/cupy array

                    return pd.Series(
                        AwkwardExtensionArray(ak_arr), index=self._obj.index
                    )
                return ak_arr

        else:
            raise AttributeError

        return f

    def apply(self, fn):
        """Perform function on all the values of the series"""
        result = fn(self.array)
        if isinstance(result, ak.Array):
            return pd.Series(AwkwardExtensionArray(result))
        return result

    def __dir__(self):
        if self.array.layout.parameters.get("__array__") == "bytestring":
            extra = dir_str(utf8=False)
        elif self.array.layout.parameters.get("__array__") == "string":
            extra = dir_str(utf8=True)
        else:
            extra = []
        return (
            [
                _
                for _ in (dir(ak))
                if not _.startswith(("_", "ak_")) and not _[0].isupper()
            ]
            + ["to_column"]
            + extra
        )
