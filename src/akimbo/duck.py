import functools
import operator
import uuid

import awkward as ak
import duckdb
import duckdb.duckdb.typing as dtyp
import pyarrow as pa

from akimbo.apply_tree import run_with_transform
from akimbo.datetimes import DatetimeAccessor
from akimbo.datetimes import match as match_dt
from akimbo.mixin import Accessor
from akimbo.strings import StringAccessor, match_string, strptime


# https://duckdb.org/docs/sql/query_syntax/from#positional-joins
# SELECT * FROM df1 POSITIONAL JOIN df2;
class DuckStringAccessor(StringAccessor):
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


class DuckDatetimeAccessor:
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


class DuckAccessor(Accessor):
    dataframe_type = duckdb.DuckDBPyRelation
    series_type = None  # only has "dataframe like"
    subaccessors = Accessor.subaccessors.copy()

    def __init__(self, obj, subaccessor=None, behavior=None):
        super().__init__(obj, behavior)
        self.subaccessor = subaccessor

    @classmethod
    def to_arrow(cls, data: duckdb.DuckDBPyRelation):
        return data.to_arrow_table()

    def to_output(self, data=None):
        import pandas as pd

        data = self.to_arrow(data or self._obj)
        if data.column_names == ["_ak_series_"]:
            data = data["_ak_series_"].to_pandas(types_mapper=pd.ArrowDtype)
        elif data.column_names == ["_ak_dataframe_"]:
            data = (
                data["_ak_dataframe_"].to_pandas(types_mapper=pd.ArrowDtype).ak.unpack()
            )
        return data

    def __getattr__(self, item: str) -> callable:
        if isinstance(item, str) and item in self.subaccessors:
            return DuckAccessor(self._obj, subaccessor=item, behavior=self._behavior)

        def select(
            *inargs, subaccessor=self.subaccessor, where=None, **kwargs
        ) -> duckdb.DuckDBPyRelation:
            if subaccessor:
                func0 = getattr(self.subaccessors[subaccessor](), item)
            elif callable(item):
                func0 = item
            else:
                func0 = None

            def f(batch: pa.ChunkedArray) -> pa.Table:
                # length 0 indicates this is a test run, and we want to know the output type: series/dataframe
                ret_type = len(batch) == 0
                # fixes bug: from_arrow fails on ChunkedArray with zero length
                arr = ak.from_arrow(batch if len(batch) else batch.chunks[0])
                arr0 = arr
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
                if arr.fields == ["_ak_series_"]:
                    arr = arr["_ak_series_"]
                elif arr.layout.is_option:
                    # A whole row cannot be NULL
                    arr = ak.Array(arr.layout.content)

                if callable(func0):
                    func = func0
                    args = (arr,)
                elif hasattr(arr, item) and callable(getattr(arr, item)):
                    func = getattr(arr, item)
                    args = ()
                elif hasattr(ak, item):
                    func = getattr(ak, item)
                    args = (arr,)
                else:
                    raise KeyError(item)

                out = func(*args, *inargs0, **kwargs)
                if where:
                    out = ak.with_field(arr0, out, where)
                outtype = "dataframe" if out.layout.is_record else "series"
                if len(out.fields) != 1:
                    out = ak.Array({"_ak_series_": out})
                out = ak.to_arrow_table(
                    out,
                    extensionarray=False,
                    list_to32=True,
                    string_to32=True,
                    bytestring_to32=True,
                )
                # must return one-column table
                if ret_type:
                    return out, outtype
                return out

            f.__name__ = item.__name__ if callable(item) else item

            inargs = [_._obj if isinstance(_, type(self)) else _ for _ in inargs]
            n_others = sum(isinstance(_, self.dataframe_type) for _ in inargs)
            if n_others == 1:
                raise NotImplementedError
                # other = next(_ for _ in inargs if isinstance(_, self.dataframe_type))
                # inargs = [
                #    "_ak_other_" if isinstance(_, self.dataframe_type) else _
                #    for _ in inargs
                # ]
                # obj = concat_columns_zip_index(self._obj, other)
            elif n_others > 1:
                raise NotImplementedError
            else:
                obj = self._obj

            if len(obj.dtypes) > 1:
                obj = (
                    self.pack()
                )  # no-compute operation, but input has to be a single column

            arrow_type = pa.schema(
                [(obj.columns[0], duckdb_to_pyarrow_type(obj.dtypes[0]))]
            )
            arr = pa.table([[]], schema=arrow_type)
            out1, outtype = f(arr[obj.columns[0]])  # ; used implicitly in next line

            out2 = duckdb.sql("SELECT * FROM out1").dtypes[0]
            fname = f"{item.__name__ if hasattr(item, '__name__') else item}_{uuid.uuid4().hex}"
            duckdb.create_function(
                fname, f, obj.types, out2, type=duckdb.functional.PythonUDFType.ARROW
            )
            # this operates on the one known column, so in the function the column
            # has been normalised out
            result = duckdb.sql(
                f"SELECT {fname}({obj.columns[0]}) as _ak_{outtype}_ FROM obj"
            )
            return result

        return select

    def __getitem__(self, item):
        return self.__getattr__(operator.getitem)(item)

    def pack(self, name="_ak_series_"):
        obj = self._obj
        inner = ",".join(f"{k} := {k}" for k in obj.columns)
        return duckdb.sql(f"SELECT struct_pack({inner}) as {name} FROM obj")

    def unpack(self):
        obj = self._obj
        assert [_.id for _ in obj.dtypes] == ["struct"]
        columns = obj.dtypes[0].children
        inner = ", ".join(
            f"struct_extract({obj.columns[0]}, '{c}') as {c}" for c in columns
        )
        return duckdb.sql(f"SELECT {inner} FROM obj")


def _pack_struct_type(obj):
    return duckdb.struct_type({c: t for c, t in zip(obj.columns, obj.dtypes)})


def _unpack_struct_type(obj):
    return dict(obj.dtypes[0].children)


def duckdb_to_pyarrow_type(duckdb_type: dtyp.DuckDBPyType) -> pa.DataType:
    """Convert complex schema"""
    type_map = {
        "tinyint": pa.int8(),
        "smallint": pa.int16(),
        "integer": pa.int32(),
        "bigint": pa.int64(),
        "utinyint": pa.uint8(),
        "usmallint": pa.uint16(),
        "uinteger": pa.uint32(),
        "ubigint": pa.uint64(),
        "float": pa.float32(),
        "real": pa.float32(),
        "double": pa.float64(),
        "boolean": pa.bool_(),
        "varchar": pa.string(),
        "text": pa.string(),
        "date": pa.date32(),
        "time": pa.time64("us"),
        # timez ?
        "timestamp": pa.timestamp("us"),
        "timestamp_ns": pa.timestamp("ns"),
        "timestam_ms": pa.timestamp("ms"),
        "timestamp_s": pa.timestamp("s"),
        # timestampz ?
        "blob": pa.binary(),  # same
    }

    if duckdb_type.id in type_map:
        return type_map[duckdb_type.id]

    # Handle DECIMAL types
    if duckdb_type.id == "decimal":
        return pa.decimal128(**dict(duckdb_type.children))

    # Handle array types
    if duckdb_type.id == "list":
        inner_type = duckdb_to_pyarrow_type(duckdb_type.children[0][1])
        return pa.list_(inner_type)

    # Handle struct types
    if duckdb_type.id == "struct":
        fields = {k: duckdb_to_pyarrow_type(v) for k, v in duckdb_type.children}
        return pa.struct(fields)

    # fixed-length-tuple?
    raise ValueError(f"Unsupported DuckDB type: {duckdb_type}")


DuckAccessor.register_accessor("dt", DuckDatetimeAccessor)
DuckAccessor.register_accessor("str", DuckStringAccessor)


@property  # type:ignore
def ak_property(self):
    return DuckAccessor(self)


duckdb.DuckDBPyRelation.ak = ak_property  # Duck has no Series
