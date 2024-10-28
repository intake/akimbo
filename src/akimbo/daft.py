import awkward as ak
import daft
from daft.udf import StatelessUDF

# from akimbo.apply_tree import run_with_transform
from akimbo.mixin import Accessor


class DaftAccessor(Accessor):
    # this does not perform the operation, but makes a callable udf with
    # the right annotation
    def __getattr__(self, item):
        func = getattr(ak, item)
        return self._call_with_func(func)

    def _call_with_func(self, func):
        # todo: this should use run_with_transform

        def closure(*args, **kwargs) -> callable:
            not_series = [_ for _ in args if not isinstance(_, daft.DataFrame)]
            other_series = [_ for _ in args if isinstance(_, daft.DataFrame)]

            if not other_series:

                def f(x: daft.Series) -> daft.Series:
                    # unary
                    arr = ak.from_arrow(x.to_arrow())
                    ak_arr = ak.Array({x.name(): func(arr, *not_series, **kwargs)})
                    if isinstance(ak_arr, ak.Array):
                        # like _to_output
                        return daft.Series.from_arrow(
                            ak.to_arrow(ak_arr, extensionarray=False)
                        )
                    return ak_arr

            else:

                def f(x: daft.Series, y: daft.Series) -> daft.Series:
                    # binary
                    arr_x = ak.from_arrow(x.to_arrow())
                    arr_y = ak.from_arrow(y.to_arrow())

                    ak_arr = ak.Array(
                        {x.name(): func(arr_x, arr_y, *not_series, **kwargs)}
                    )
                    if isinstance(ak_arr, ak.Array):
                        # like _to_output
                        return daft.Series.from_arrow(
                            ak.to_arrow(ak_arr, extensionarray=False)
                        )
                    return ak_arr

            schema = self._obj.schema().to_pyarrow_schema()
            ak_schema = ak.from_arrow_schema(schema)
            arr = ak.Array(ak_schema.length_zero_array())  # typetracer??
            out = func(arr)
            outtype = daft.DataType.from_arrow_type(
                ak.to_arrow(out, extensionarray=False).type
            )
            udf = StatelessUDF(
                name=func.__name__,
                func=f,
                return_dtype=outtype,
                resource_request=None,
                batch_size=None,
            )
            return udf

        return closure

    def __array_ufunc__(self, *args, **kwargs):
        # ufuncs
        if args[1] == "__call__":
            return self._call_with_func(args[0])(self, **kwargs)
        raise NotImplementedError

    @classmethod
    def _create_op(cls, op):
        # dunder methods

        def run(self, *args, **kwargs):
            # closure =
            return cls._call_with_func()

        return run


@property  # type:ignore
def ak_property(self):
    return DaftAccessor(self)


daft.DataFrame.ak = ak_property
daft.Series.ak = ak_property
