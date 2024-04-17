import functools

from cudf.core.series import Series
from cudf import DataFrame
import awkward as ak

from awkward_pandas.mixin import ArithmeticMixin
from awkward_pandas.ak_from_cudf import cudf_to_awkward as from_cudf
from typing import Callable, Iterable


class AwkwardAccessor(ArithmeticMixin):

    def __init__(self, series: Series):
        self.array = from_cudf(series)

    def __array_function__(self, *args, **kwargs):
        return self.array.__array_function__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] == "__call__":
            return args[0](self.array, *args[3:], **kwargs)
        raise NotImplementedError

    def __dir__(self) -> Iterable[str]:
        return [
            _
            for _ in (dir(ak))
            if not _.startswith(("_", "ak_")) and not _[0].isupper()
        ] + ["apply", "array"]

    def apply(self, fn: Callable) -> Series:
        """Perform function on all the values of the series"""
        out = fn(self.array)
        return maybe_to_cudf(out)

    def __getitem__(self, item):
        # scalars?
        out = self.array.__getitem__(item)
        return maybe_to_cudf(out)

    def __getattr__(self, item):
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        if func:

            @functools.wraps(func)
            def f(*others, **kwargs):
                others = [
                    other.ak.array
                    if isinstance(other, (DataFrame, Series))
                    else other
                    for other in others
                ]
                kwargs = {
                    k: v.ak.array if isinstance(v, (DataFrame, Series)) else v
                    for k, v in kwargs.items()
                }

                ak_arr = func(self.array, *others, **kwargs)
                return maybe_to_cudf(ak_arr)

        else:
            raise AttributeError(item)
        return f

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            return maybe_to_cudf(op(self.array, *args, **kwargs))

        return run

    _create_arithmetic_method = _create_op
    _create_comparison_method = _create_op
    _create_logical_method = _create_op


def maybe_to_cudf(x):
    if isinstance(x, ak.Array):
        return ak.to_cudf(x)
    return x



AwkwardAccessor._add_all()


@property  # type:ignore
def ak_property(self):
    return AwkwardAccessor(self)


Series.ak = ak_property  # no official register function?
