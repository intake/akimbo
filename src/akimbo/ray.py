import functools
from typing import Callable, Iterable

import awkward as ak
import numpy as np
import pyarrow as pa
import ray
import ray.data as rd

from akimbo.apply_tree import run_with_transform
from akimbo.datetimes import DatetimeAccessor
from akimbo.datetimes import match as match_dt
from akimbo.mixin import Accessor, match_any, numeric
from akimbo.strings import StringAccessor, match_string, strptime
from akimbo.utils import to_ak_layout


class RayStringAccessor(StringAccessor):
    def __init__(self, *_):
        pass

    def __getattr__(self, attr: str) -> callable:
        attr = self.method_name(attr)
        return getattr(ak.str, attr)

    @property
    def strptime(self):
        @functools.wraps(strptime)
        def run(*arrs, **kwargs):
            arr, *other = arrs
            return run_with_transform(arr, strptime, match_string, **kwargs)

        return run


class RayDatetimeAccessor:
    def __init__(self, *_):
        pass

    def __getattr__(self, item):
        if item in dir(DatetimeAccessor):
            fn = getattr(DatetimeAccessor, item)
            if hasattr(fn, "__wrapped__"):
                func = fn.__wrapped__  # arrow function
            else:
                raise AttributeError
        else:
            raise AttributeError

        @functools.wraps(func)
        def run(*arrs, **kwargs):
            arr, *other = arrs
            return run_with_transform(arr, func, match_dt, **kwargs)

        return run

    def __dir__(self):
        return dir(DatetimeAccessor)


class RayAccessor(Accessor):
    dataframe_type = rd.Dataset
    series_type = None  # only has "dataframe like"
    subaccessors = Accessor.subaccessors.copy()

    def __init__(self, obj, subaccessor=None, behavior=None):
        super().__init__(obj, behavior)
        self.subaccessor = subaccessor

    def to_arrow(self, data: rd.Dataset) -> pa.Table:
        batches = ray.get(data.to_arrow_refs())
        return pa.concat_tables(batches)

    def to_output(self, data=None):
        import pandas as pd

        data = self.to_arrow(data if data is not None else self._obj)
        data = data.to_pandas(types_mapper=pd.ArrowDtype)
        if list(data.columns) == ["_ak_series_"]:
            data = data["_ak_series_"]
        return data

    def __getattr__(self, item: str) -> rd.Dataset:
        if isinstance(item, str) and item in self.subaccessors:
            return RayAccessor(self._obj, subaccessor=item, behavior=self._behavior)

        def select(*inargs, subaccessor=self.subaccessor, where=None, **kwargs):
            if subaccessor:
                func0 = getattr(self.subaccessors[subaccessor](), item)
            elif callable(item):
                func0 = item
            else:
                func0 = None

            def f(batch):
                arr = ak.from_arrow(batch)
                if any(isinstance(_, str) and _ == "_ak_other_" for _ in inargs):
                    # binary input
                    other = arr[[_ for _ in arr.fields if _.startswith("_df2_")]]
                    # 5 == len("_df2_"); rename to original fields
                    other.layout._fields[:] = [k[5:] for k in other.fields]
                    arr = arr[[_ for _ in arr.fields if not _.startswith("_df2_")]]
                    if other.fields == ["_ak_series_"]:
                        other = other["_ak_series_"]
                    if where is not None:
                        other = other[where]
                    inargs0 = [other if str(_) == "_ak_other_" else _ for _ in inargs]
                else:
                    inargs0 = inargs
                    other = None
                if where:
                    arr0 = arr
                    arr = arr[where]
                if arr.fields == ["_ak_series_"]:
                    arr = arr["_ak_series_"]

                if callable(func0):
                    func = func0
                    args = (arr,)
                elif hasattr(arr, item) and callable(getattr(arr, item)):
                    func = getattr(arr, item)
                    args = ()
                elif subaccessor:
                    func = func0
                    args = (arr,)
                elif hasattr(ak, item):
                    func = getattr(ak, item)
                    args = (arr,)
                else:
                    raise KeyError(item)

                out = func(*args, *inargs0, **kwargs)
                if where:
                    out = ak.with_field(arr0, out, where)
                if not out.layout.fields:
                    out = ak.Array({"_ak_series_": out})
                return ak.to_arrow_table(
                    out,
                    extensionarray=False,
                    list_to32=True,
                    string_to32=True,
                    bytestring_to32=True,
                )

            f.__name__ = item.__name__ if callable(item) else item

            inargs = [_._obj if isinstance(_, type(self)) else _ for _ in inargs]
            n_others = sum(isinstance(_, self.dataframe_type) for _ in inargs)
            if n_others == 1:
                other = next(_ for _ in inargs if isinstance(_, self.dataframe_type))
                inargs = [
                    "_ak_other_" if isinstance(_, self.dataframe_type) else _
                    for _ in inargs
                ]
                obj = concat_columns_zip_index(self._obj, other)
            elif n_others > 1:
                raise NotImplementedError
            else:
                obj = self._obj
            arrow_type = obj.schema().base_schema
            arr = pa.table([[]] * len(arrow_type), schema=arrow_type)
            out1 = f(arr)
            out_schema = pa.table(out1).schema
            result = obj.map_batches(f, zero_copy_batch=True, batch_format="pyarrow")
            result._plan.cache_schema(out_schema)
            return result

        return select

    def __array_ufunc__(self, *args, where=None, out=None, **kwargs):
        # includes operator overload like df.ak + 1
        ufunc, call, inputs, *callargs = args
        if out is not None or call != "__call__":
            raise NotImplementedError

        return self.__getattr__(ufunc)(*callargs, where=where, **kwargs)

    def __getitem__(self, item) -> rd.dataset:
        def f(batch):
            arr = ak.from_arrow(batch)
            arr2 = arr.__getitem__(item)
            if not arr2.fields:
                arr2 = ak.Array({"_ak_series_": arr2})
            return ak.to_arrow_table(
                arr2,
                extensionarray=False,
                list_to32=True,
                string_to32=True,
                bytestring_to32=True,
            )

        arrow_type = self._obj.schema().base_schema
        if not isinstance(arrow_type, pa.Schema):
            # TODO: fix, for data via from_pandas or from_numpy
            raise ValueError("Use arrow types")
        arr = pa.table([[]] * len(arrow_type), schema=arrow_type)
        out1 = f(arr)
        out_schema = pa.table(out1).schema
        result = self._obj.map_batches(f, zero_copy_batch=True, batch_format="pyarrow")
        # this is what .schema(fetch_if_missing=True) does, but we already know
        # the value without compute
        result._plan.cache_schema(out_schema)
        return result

    def transform(
        self,
        fn: callable,
        *others,
        where=None,
        match=match_any,
        inmode="array",
        **kwargs,
    ):
        def f(arr, *others, **kwargs):
            return run_with_transform(
                arr, fn, match=match, others=others, inmode=inmode, **kwargs
            )

        return self.__getattr__(f)(*others, **kwargs)

    def apply(self, fn: Callable, *others, where=None, **kwargs):
        return self.__getattr__(fn)(*others, **kwargs)

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            args = [
                to_ak_layout(_) if isinstance(_, (str, int, float, np.number)) else _
                for _ in args
            ]
            return self.transform(op, *args, match=numeric)

        return run

    def __dir__(self) -> Iterable[str]:
        if self.subaccessor is not None:
            return dir(self.subaccessors[self.subaccessor](self))
        return super().__dir__()


RayAccessor.register_accessor("dt", RayDatetimeAccessor)
RayAccessor.register_accessor("str", RayStringAccessor)


def concat_columns_zip_index(df1: rd.Dataset, df2: rd.Dataset) -> rd.Dataset:
    """Add two DataFrames' columns into a single DF."""
    if df1.num_blocks != df2.num_blocks:
        # warn that this causes shuffle
        pass
    return df1.zip(df2.rename_columns({k: f"_df2_{k}" for k in df2.columns()}))


@property  # type:ignore
def ak_property(self):
    return RayAccessor(self)


rd.Dataset.ak = ak_property  # Ray has no Series
