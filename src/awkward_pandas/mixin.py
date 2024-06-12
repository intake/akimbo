import functools
import operator
from typing import Callable, Iterable

import awkward as ak

methods = [
    _ for _ in (dir(ak)) if not _.startswith(("_", "ak_")) and not _[0].isupper()
] + ["apply", "array", "explode", "dt", "str"]

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
    """Bring the awkward API to dataframes and series"""

    aggregations = True  # False means data is partitioned
    series_type = ()
    dataframe_type = ()

    def __init__(self, obj):
        self._obj = obj

    @classmethod
    def is_series(cls, data):
        return isinstance(data, cls.series_type)

    @classmethod
    def is_dataframe(cls, data):
        return isinstance(data, cls.dataframe_type)

    @classmethod
    def _to_output(cls, data):
        raise NotImplementedError

    def to_output(self, data=None):
        data = data if data is not None else self.array
        return self._to_output(data)

    def apply(self, fn: Callable, *args, **kwargs):
        """Perform arbitrary function on all the values of the series"""
        return self.to_output(fn(self.array, *args, **kwargs))

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
    def arrow(self) -> ak.Array:
        """Data as an arrow array"""
        return self.to_arrow(self._obj)

    @classmethod
    def to_arrow(cls, data):
        """Data as an arrow array"""
        raise NotImplementedError

    @property
    def array(self) -> ak.Array:
        """Data as an awkward array"""
        return ak.from_arrow(self.arrow)

    @property
    def str(self):
        """Nested string operations"""
        from awkward_pandas.strings import StringAccessor

        return StringAccessor(self)

    @property
    def dt(self):
        """Nested datetime operations"""
        from awkward_pandas.datetimes import DatetimeAccessor

        return DatetimeAccessor(self)

    def merge(self):
        """Make a single complex series out of the columns of a dataframe"""
        if not self.is_dataframe(self._obj):
            raise ValueError("Can only merge on a dataframe")
        out = {}
        for k in self._obj.columns:
            # TODO: partial merge when column names are like "record.field"
            out[k] = self._obj[k].ak.array
        arr = ak.Array(out)
        return self.to_output(arr)

    def unmerge(self):
        """Make dataframe out of a series of record type"""
        arr = self.array
        if not arr.fields:
            raise ValueError("Not array-of-records")
        # TODO: partial unmerge when (some) fields are records
        out = {k: self.to_output(arr[k]) for k in arr.fields}
        return self.dataframe_type(out)

    @classmethod
    def _create_op(cls, op):
        """Make functions to perform all the arithmetic, logical and comparison ops"""

        def run(self, *args, **kwargs):
            ar2 = (ar.ak.array if hasattr(ar, "ak") else ar for ar in args)
            ar3 = (ar.array if isinstance(ar, cls) else ar for ar in ar2)
            return self.to_output(op(self.array, *ar3, **kwargs))

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

    def __init_subclass__(cls, **kwargs):
        # auto add methods to all derivative classes
        cls._add_all()
