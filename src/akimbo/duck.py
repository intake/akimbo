import duckdb
import duckdb.duckdb.typing as dtyp
import pyarrow as pa

from akimbo.mixin import Accessor


class DuckAccessor(Accessor):

    def to_arrow(cls, data: duckdb.DuckDBPyRelation):
        return data.to_arrow_table()

    def to_output(self, data=None):
        data = self.to_arrow(data or self._obj).to_pandas(types_mapper=pd.ArrowDtype)
        if list(data.columns) == ["_ak_series_"]:
            data = data["_ak_series_"]
        return data

    def make_func(self, func):
        def f(arr: pa.Table) -> pa.Table:
            ...

        # use func UUID
        duckdb.create_function("f", f, df.types, duckdb.type("out-type"),
                               type=duckdb.functional.PythonUDFType.ARROW)


def pyarrow_to_duckdb_type(pa_type: pa.DataType) -> dtyp.DuckDBPyType:
    tmap = {
        pa.int8: dtyp.TINYINT,
        pa.int16: dtyp.SMALLINT,
        pa.int32: dtyp.INTEGER,
        pa.int64: dtyp.BIGINT,
        pa.uint8: dtyp.UTINYINT,
        pa.uint16: dtyp.USMALLINT,
        pa.uint32: dtyp.UINTEGER,
        pa.uint64: dtyp.UBIGINT,
        pa.float32: dtyp.FLOAT,
        pa.float64: dtyp.DOUBLE,
        pa.Bool8Type: dtyp.BOOLEAN,
        pa.date32: dtyp.DATE,
        pa.date64: dtyp.DATE,
        pa.time32: dtyp.TIME,
        pa.time64: dtyp.TIME,
    }
    if type(pa_type) in tmap:
        return tmap[type(pa_type)]
    if isinstance(pa_type, pa.timestamp):
        time_tmap = {
            "s": dtyp.TIMESTAMP_S,
            "ms": dtyp.TIMESTAMP_MS,
            "ns": dtyp.TIMESTAMP_NS
        }
        if pa_type.unit in time_tmap: return time_tmap[pa_type.unit]

    # String types
    if pa.types.is_string(pa_type): return dtyp.VARCHAR
    if pa.types.is_large_string(pa_type): return dtyp.VARCHAR

    # Decimal
    if isinstance(pa_type, pa.Decimal128Type):
        return duckdb.decimal_type(pa_type.precision, pa_type.scale)

    # Binary
    if pa.types.is_binary(pa_type): return dtyp.BLOB
    if pa.types.is_large_binary(pa_type): return dtyp.BLOB

    # List
    if pa.types.is_list(pa_type):
        return duckdb.list_type(pa_type.value_type)

    # Struct
    if pa.types.is_struct(pa_type):
        return duckdb.struct_type({field.name: pyarrow_to_duckdb_type(field.type)})

    raise ValueError(f"Unsupported PyArrow type: {pa_type}")


def duckdb_to_pyarrow_type(duckdb_type: dtyp.DuckDBPyType) -> pa.DataType:
    # Basic type mapping
    type_map = {
        dtyp.TINYINT: pa.int8(),
        dtyp.SMALLINT: pa.int16(),
        dtyp.INTEGER: pa.int32(),
        dtyp.BIGINT: pa.int64(),
        dtyp.UTINYINT: pa.uint8(),
        dtyp.USMALLINT: pa.uint16(),
        dtyp.UINTEGER: pa.uint32(),
        dtyp.UBIGINT: pa.uint64(),
        dtyp.FLOAT: pa.float32(),
        dtyp.REAL: pa.float32(),
        dtyp.DOUBLE: pa.float64(),
        dtyp.BOOLEAN: pa.bool_(),
        dtyp.VARCHAR: pa.string(),
        dtyp.TEXT: pa.string(),
        dtyp.DATE: pa.date32(),
        dtyp.TIME: pa.time64(dtyp.us),
        dtyp.TIMESTAMP: pa.timestamp(dtyp.us),
        dtyp.BLOB: pa.binary()
    }

    if duckdb_type in type_map:
        return type_map[duckdb_type]

    # Handle DECIMAL types
    if duckdb_type.startswith(DECIMAL):
        precision, scale = map(int, duckdb_type.strip(DECIMAL()).split(,))
        return pa.decimal128(precision, scale)

    # Handle array types
    if duckdb_type.endswith([]):
        inner_type = duckdb_to_pyarrow_type(duckdb_type[:-2])
        return pa.list_(inner_type)

    # Handle struct types
    if duckdb_type.startswith(STRUCT):
        # Parse the struct definition
        struct_fields = duckdb_type.strip(STRUCT()).split(,)
        fields = []
        for field in struct_fields:
            name, field_type = field.strip().split( , 1)
            fields.append((name, duckdb_to_pyarrow_type(field_type)))
        return pa.struct(fields)

    raise ValueError(fUnsupported DuckDB type: {duckdb_type})
