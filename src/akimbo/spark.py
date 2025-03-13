import awkward as ak
import pyarrow as pa
import pyspark
from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

from akimbo.mixin import LazyAccessor

sdf = pyspark.sql.DataFrame


class SparkAccessor(LazyAccessor):
    """Operations on pyspark dataframes.

    This is a lazy backend, and operates partition-wise. It predicts the schema
    of each operation by running with an empty dataframe of the correct type.
    """

    dataframe_type = sdf

    def to_arrow(self, data) -> pa.Table:
        # collects data locally
        batches = data._collect_as_arrow()
        return pa.Table.from_batches(batches)

    def to_output(self, data=None):
        from akimbo.pandas import pd

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
            if subaccessor and isinstance(item, str):
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
