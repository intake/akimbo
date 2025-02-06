import functools
from typing import Callable, Iterable

import awkward as ak
import numpy as np
import pyarrow as pa
import pyspark
from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

from akimbo.apply_tree import run_with_transform
from akimbo.datetimes import DatetimeAccessor
from akimbo.datetimes import match as match_dt
from akimbo.mixin import Accessor, match_any, numeric
from akimbo.pandas import pd
from akimbo.strings import StringAccessor, match_string, strptime
from akimbo.utils import to_ak_layout

sdf = pyspark.sql.DataFrame


class SparkStringAccessor(StringAccessor):
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


class SparkDatetimeAccessor:
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


class SparkAccessor(Accessor):
    """Operations on pyspark dataframes.

    This is a lazy backend, and operates partition-wise. It predicts the schema
    of each operation by running with an empty dataframe of the correct type.
    """

    subaccessors = Accessor.subaccessors.copy()
    dataframe_type = sdf

    def __init__(self, obj, subaccessor=None, behavior=None):
        super().__init__(obj, behavior)
        self.subaccessor = subaccessor

    def to_arrow(self, data) -> pa.Table:
        # collects data locally
        batches = data._collect_as_arrow()
        return pa.Table.from_batches(batches)

    def to_output(self, data=None) -> pd.DataFrame | pd.Series:
        # data is always arrow format internally
        data = self.to_arrow(data if data is not None else self._obj).to_pandas(
            types_mapper=pd.ArrowDtype
        )
        if list(data.columns) == ["_ak_series_"]:
            data = data["_ak_series_"]
        return data

    def __getattr__(self, item: str) -> sdf:
        if isinstance(item, str) and item in self.subaccessors:
            return SparkAccessor(self._obj, subaccessor=item, behavior=self._behavior)

        def select(*inargs, subaccessor=self.subaccessor, where=None, **kwargs):
            if subaccessor:
                func0 = getattr(self.subaccessors[subaccessor](), item)
            elif callable(item):
                func0 = item
            else:
                func0 = None

            def f(batches):
                for batch in batches:
                    arr = ak.from_arrow(batch)
                    if any(isinstance(_, str) and _ == "_ak_other_" for _ in inargs):
                        # binary input
                        arr, other = arr["_1"], arr["_df2"]
                        if other.fields == ["_ak_series_"]:
                            other = other["_ak_series_"]
                        if where is not None:
                            other = other[where]
                        inargs0 = [
                            other if str(_) == "_ak_other_" else _ for _ in inargs
                        ]
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
                    arrout = ak.to_arrow(
                        out,
                        extensionarray=False,
                        list_to32=True,
                        string_to32=True,
                        bytestring_to32=True,
                    )
                    yield pa.RecordBatch.from_struct_array(arrout)

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
            arrow_type = to_arrow_schema(obj.schema)
            arr = pa.table([[]] * len(arrow_type), schema=arrow_type)
            out1 = next(f([arr]))
            out_schema = pa.table(out1).schema
            return obj.mapInArrow(f, schema=from_arrow_schema(out_schema))

        return select

    def __array_ufunc__(self, *args, where=None, out=None, **kwargs):
        # includes operator overload like df.ak + 1
        ufunc, call, inputs, *callargs = args
        if out is not None or call != "__call__":
            raise NotImplementedError

        return self.__getattr__(ufunc)(*callargs, where=where, **kwargs)

    def __getitem__(self, item) -> sdf:
        def f(batches):
            for batch in batches:
                arr = ak.from_arrow(batch)
                arr2 = arr.__getitem__(item)
                if not arr2.fields:
                    arr2 = ak.Array({"_ak_series_": arr2})
                out = ak.to_arrow(
                    arr2,
                    extensionarray=False,
                    list_to32=True,
                    string_to32=True,
                    bytestring_to32=True,
                )
                yield pa.RecordBatch.from_struct_array(out)

        arrow_type = to_arrow_schema(self._obj.schema)
        arr = pa.table([[]] * len(arrow_type), schema=arrow_type)
        out1 = next(f([arr]))
        out_schema = pa.table(out1).schema
        return self._obj.mapInArrow(f, schema=from_arrow_schema(out_schema))

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


SparkAccessor.register_accessor("dt", SparkDatetimeAccessor)
SparkAccessor.register_accessor("str", SparkStringAccessor)


def concat_columns_zip_index(df1: sdf, df2: sdf) -> sdf:
    """Add two DataFrames' columns into a single DF.

    The is SQL-tricky, but at least it requires no python map/iteration!
    """
    if df1.rdd.getNumPartitions() != df2.rdd.getNumPartitions():
        # warn that this causes shuffle
        pass
    df1_ind = df1.rdd.zipWithIndex().toDF()
    df2_ind = df2.rdd.zipWithIndex().toDF().withColumnRenamed("_1", "_df2")
    return df1_ind.join(df2_ind, "_2", "left").sort("_2").drop("_2")


@property  # type:ignore
def ak_property(self):
    return SparkAccessor(self)


pyspark.sql.DataFrame.ak = ak_property  # spark has no Series
