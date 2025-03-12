from typing import Dict

import awkward as ak
import polars as pl
import pyarrow as pa

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


@pl.api.register_lazyframe_namespace("ak")
class LazyPolarsAwkwardAccessor(LazyAccessor):
    dataframe_type = pl.LazyFrame
    series_type = None  # lazy is never series

    def to_output(self, data=None):
        out = self._obj.collect()
        if out.columns == ["_ak_series_"]:
            out = out["_ak_series_"]
        return out

    def __getattr__(self, item: str, **flags) -> callable:
        if isinstance(item, str) and item in self.subaccessors:
            return LazyPolarsAwkwardAccessor(
                self._obj, subaccessor=item, behavior=self._behavior
            )

        def select(*inargs, subaccessor=self.subaccessor, where=None, **kwargs):
            if subaccessor and isinstance(item, str):
                func0 = getattr(self.subaccessors[subaccessor](), item)
            elif callable(item):
                func0 = item
            else:
                func0 = None

            def f(batch):
                arr = ak.from_arrow(batch.to_arrow())

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
                if where:
                    arr0 = arr
                    arr = arr[where]
                if arr.fields == ["_ak_series_"]:
                    arr = arr["_ak_series_"]

                out = func0(arr, *inargs0, **kwargs)
                if where:
                    out = ak.with_field(arr0, out, where)
                if not out.layout.fields:
                    out = ak.Array({"_ak_series_": out})
                arr = ak.to_arrow_table(out, extensionarray=False)
                return pl.DataFrame(arr, **flags)

            inargs = [_._obj if isinstance(_, type(self)) else _ for _ in inargs]
            n_others = sum(isinstance(_, self.dataframe_type) for _ in inargs)
            if n_others == 1:
                other = next(_ for _ in inargs if isinstance(_, self.dataframe_type))
                inargs = [
                    "_ak_other_" if isinstance(_, self.dataframe_type) else _
                    for _ in inargs
                ]
                obj = concat(self._obj, other)
            elif n_others > 1:
                raise NotImplementedError
            else:
                obj = self._obj
            arrow_type = polars_to_arrow_schema(obj.collect_schema())
            arr = pa.table([[]] * len(arrow_type), schema=arrow_type)
            out1 = f(pl.from_arrow(arr))

            return obj.map_batches(f, schema=out1.schema)

        return select

    def pack(self):
        return self._obj.select(
            pl.struct(*self._obj.collect_schema().names()).alias("_ak_series_")
        )

    def unpack(self):
        cols = self._obj.collect_schema().names()
        assert len(cols) == 1
        return self._obj.select(pl.col(cols[0]).struct.unnest())


def concat(*series: pl.LazyFrame) -> pl.LazyFrame:
    this, *others = series
    # don't actually expect more than one "others"
    return this.with_columns(
        [
            o.rename({c: f"_df{i + 2}_{c}" for c in o.collect_schema().names()})
            for i, o in enumerate(others)
        ]
    )


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
    if isinstance(polars_type, pl.Datetime):
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
        for name, dtype in dict(polars_type).items():
            arrow_type = polars_to_arrow_type(dtype)
            fields.append(pa.field(name, arrow_type))
        return pa.struct(fields)

    raise ValueError(f"Unsupported Polars type: {polars_type}")


def polars_to_arrow_schema(polars_schema: Dict[str, pl.DataType]) -> pa.Schema:
    fields = []
    for name, dtype in polars_schema.items():
        arrow_type = polars_to_arrow_type(dtype)
        fields.append(pa.field(name, arrow_type))
    return pa.schema(fields)
