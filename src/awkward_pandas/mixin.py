import functools
import operator
from typing import Callable, Iterable

import awkward as ak

methods = [
    _ for _ in (dir(ak)) if not _.startswith(("_", "ak_")) and not _[0].isupper()
] + ["apply", "array", "explode"]

df_methods = sorted(methods + ["merge"])
series_methods = sorted(methods + ["unmerge"])


def radd(left, right):
    return right + left


def rsub(left, right):
    return right - left


def rmul(left, right):
    return right * left


def rdiv(left, right):
    return right / left


def rtruediv(left, right):
    return right / left


def rfloordiv(left, right):
    return right // left


def rmod(left, right):
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    return right % left


def rdivmod(left, right):
    return divmod(right, left)


def rpow(left, right):
    return right**left


def rand_(left, right):
    return operator.and_(right, left)


def ror_(left, right):
    return operator.or_(right, left)


def rxor(left, right):
    return operator.xor(right, left)


class AbstractMethodError(NotImplementedError):
    pass


class ArithmeticMixin:
    @classmethod
    def _create_op(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _create_op(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _create_op(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        setattr(cls, "__add__", cls._create_op(operator.add))
        setattr(cls, "__radd__", cls._create_op(radd))
        setattr(cls, "__sub__", cls._create_op(operator.sub))
        setattr(cls, "__rsub__", cls._create_op(rsub))
        setattr(cls, "__mul__", cls._create_op(operator.mul))
        setattr(cls, "__rmul__", cls._create_op(rmul))
        setattr(cls, "__pow__", cls._create_op(operator.pow))
        setattr(cls, "__rpow__", cls._create_op(rpow))
        setattr(cls, "__mod__", cls._create_op(operator.mod))
        setattr(cls, "__rmod__", cls._create_op(rmod))
        setattr(cls, "__floordiv__", cls._create_op(operator.floordiv))
        setattr(cls, "__rfloordiv__", cls._create_op(rfloordiv))
        setattr(cls, "__truediv__", cls._create_op(operator.truediv))
        setattr(cls, "__rtruediv__", cls._create_op(rtruediv))
        setattr(cls, "__divmod__", cls._create_op(divmod))
        setattr(cls, "__rdivmod__", cls._create_op(rdivmod))

    @classmethod
    def _add_comparison_ops(cls) -> None:
        setattr(cls, "__eq__", cls._create_op(operator.eq))
        setattr(cls, "__ne__", cls._create_op(operator.ne))
        setattr(cls, "__lt__", cls._create_op(operator.lt))
        setattr(cls, "__gt__", cls._create_op(operator.gt))
        setattr(cls, "__le__", cls._create_op(operator.le))
        setattr(cls, "__ge__", cls._create_op(operator.ge))

    @classmethod
    def _add_logical_ops(cls) -> None:
        setattr(cls, "__and__", cls._create_op(operator.and_))
        setattr(cls, "__rand__", cls._create_op(rand_))
        setattr(cls, "__or__", cls._create_op(operator.or_))
        setattr(cls, "__ror__", cls._create_op(ror_))
        setattr(cls, "__xor__", cls._create_op(operator.xor))
        setattr(cls, "__rxor__", cls._create_op(rxor))

    @classmethod
    def _add_all(cls):
        cls._add_logical_ops()
        cls._add_arithmetic_ops()
        cls._add_comparison_ops()


class Accessor(ArithmeticMixin):
    def __init__(self, obj):
        self._obj = obj

    @property
    def series_type(self):
        raise NotImplementedError

    @property
    def dataframe_type(self):
        raise NotImplementedError

    def is_series(self, data):
        return isinstance(data, self.series_type)

    def is_dataframe(self, data):
        return isinstance(data, self.dataframe_type)

    def to_output(self, data):
        raise NotImplementedError

    def apply(self, fn: Callable):
        """Perform arbitrary function on all the values of the series"""
        return self.to_output(fn(self.array))

    def __getitem__(self, item):
        out = self.array.__getitem__(item)
        return self.to_output(out)

    def __dir__(self) -> Iterable[str]:
        return series_methods if self.is_series(self._obj) else df_methods

    def __array_function__(self, *args, **kwargs):
        return self.array.__array_function__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] == "__call__":
            return args[0](self.array, *args[3:], **kwargs)
        raise NotImplementedError

    @property
    def array(self) -> ak.Array:
        """Data as an awkward array"""
        raise NotImplementedError

    def merge(self, *args):
        """Create single record-type nested series from a dataframe"""
        raise NotImplementedError

    def unmerge(self):
        arr = self.array
        if not arr.fields:
            raise ValueError
        out = {k: self.to_output(arr[k]) for k in arr.fields}
        return self.dataframe_type(out)

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            return self.to_output(op(self.array, *args, **kwargs))

        return run

    def __getattr__(self, item):
        if item not in dir(self):
            raise AttributeError
        func = getattr(ak, item, None)

        if func:

            @functools.wraps(func)
            def f(*others, **kwargs):
                others = [
                    other.ak.array
                    if isinstance(other, (self.series_type, self.dataframe_type))
                    else other
                    for other in others
                ]
                kwargs = {
                    k: v.ak.array
                    if isinstance(v, (self.series_type, self.dataframe_type))
                    else v
                    for k, v in kwargs.items()
                }

                ak_arr = func(self.array, *others, **kwargs)
                if isinstance(ak_arr, ak.Array):
                    return self.to_output(ak_arr)
                return ak_arr

        else:
            raise AttributeError(item)
        return f
