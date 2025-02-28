from typing import Callable, Dict

import awkward as ak
import polars as pl
import pyarrow as pa

from akimbo.apply_tree import match_any
from akimbo.mixin import EagerAccessor, LazyAccessor


@pl.api.register_series_namespace("ak")
@pl.api.register_dataframe_namespace("ak")
class PolarsAwkwardAccessor(EagerAccessor):
    """Perform awkward operations on a polars series or dataframe

    This is for *eager* operations. A Lazy version may eventually be made.
    """

    series_type = pl.Series
    dataframe_type = pl.DataFrame

    @classmethod
    def to_arrow(cls, data):
        return data.to_arrow()

    def pack(self):
        # polars already implements this directly
        return self._obj.to_struct()

    def to_output(self, data=None):
        arr = data if data is not None else self._obj
        pa_arr = ak.to_arrow(arr, extensionarray=False)
        return pl.from_arrow(pa_arr)


@pl.api.register_lazyframe_namespace
class LazyPolarsAwkwardAccessor(LazyAccessor):
    dataframe_type = pl.LazyFrame
    series_type = None  # lazy is never series

    def transform(
        self, fn: Callable, *others, where=None, match=match_any, inmode="ak", **kwargs
    ):
        # TODO determine schema from first-run, with df.collect_schema()
        return pl.map_batches(
            (self._obj,) + others,
            lambda d: d.ak.transform(
                fn, match=match, inmode=inmode, **kwargs
            ).ak.unpack(),
            schema=None,
        )


def arrow_to_polars_type(arrow_type: pa.DataType) -> pl.DataType:
    type_mapping = {
        pa.int8(): pl.Int8,
        pa.int16(): pl.Int16,
        pa.int32(): pl.Int32,
        pa.int64(): pl.Int64,
        pa.uint8(): pl.UInt8,
        pa.uint16(): pl.UInt16,
        pa.uint32(): pl.UInt32,
        pa.uint64(): pl.UInt64,
        pa.float32(): pl.Float32,
        pa.float64(): pl.Float64,
        pa.string(): pl.String,
        pa.bool_(): pl.Boolean,
    }

    if arrow_type in type_mapping:
        return type_mapping[arrow_type]

    # parametrised types
    if pa.types.is_timestamp(arrow_type):
        return pl.Datetime(time_unit=arrow_type.unit, time_zone=arrow_type.tx)

    if pa.types.is_decimal(arrow_type):
        return pl.Decimal(precision=arrow_type.precision, scale=arrow_type.scale)

    # Handle list type
    if pa.types.is_list(arrow_type):
        value_type = arrow_to_polars_type(arrow_type.value_type)
        return pl.List(value_type)

    # Handle struct type
    if pa.types.is_struct(arrow_type):
        fields = {}
        for field in arrow_type:
            fields[field.name] = arrow_to_polars_type(field.type)
        return pl.Struct(fields)

    raise ValueError(f"Unsupported Arrow type: {arrow_type}")


def polars_to_arrow_type(polars_type: pl.DataType) -> pa.DataType:
    type_mapping = {
        pl.Int8: pa.int8(),
        pl.Int16: pa.int16(),
        pl.Int32: pa.int32(),
        pl.Int64: pa.int64(),
        pl.UInt8: pa.uint8(),
        pl.UInt16: pa.uint16(),
        pl.UInt32: pa.uint32(),
        pl.UInt64: pa.uint64(),
        pl.Float32: pa.float32(),
        pl.Float64: pa.float64(),
        pl.String: pa.string(),
        pl.Boolean: pa.bool_(),
        pl.Date: pa.date32(),
    }

    if polars_type in type_mapping:
        return type_mapping[polars_type]

    # parametrised types
    if isinstance(polars_type, pl.DataType):
        return pa.timestamp(polars_type.unit, polars_type.time_zone)

    if isinstance(polars_type, pl.Decimal):
        return pa.decimal128(polars_type.precision, polars_type.scale)

    # Handle list type
    if isinstance(polars_type, pl.List):
        value_type = polars_to_arrow_type(polars_type.inner)
        return pa.list_(value_type)

    # Handle struct type
    if isinstance(polars_type, pl.Struct):
        fields = []
        for name, dtype in polars_type.fields.items():
            arrow_type = polars_to_arrow_type(dtype)
            fields.append(pa.field(name, arrow_type))
        return pa.struct(fields)

    raise ValueError(f"Unsupported Polars type: {polars_type}")


def arrow_to_polars_schema(arrow_schema: pa.Schema) -> Dict[str, pl.DataType]:
    polars_schema = {}
    for field in arrow_schema:
        polars_schema[field.name] = arrow_to_polars_type(field.type)
    return polars_schema


def polars_to_arrow_schema(polars_schema: Dict[str, pl.DataType]) -> pa.Schema:
    fields = []
    for name, dtype in polars_schema.items():
        arrow_type = polars_to_arrow_type(dtype)
        fields.append(pa.field(name, arrow_type))
    return pa.schema(fields)
