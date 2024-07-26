from __future__ import annotations

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
    subaccessors = {}

    def __init__(self, obj, behavior=None):
        self._obj = obj
        self._behavior = behavior

    @classmethod
    def is_series(cls, data):
        return isinstance(data, cls.series_type)

    @classmethod
    def is_dataframe(cls, data):
        return isinstance(data, cls.dataframe_type)

    @classmethod
    def _to_output(cls, data):
        # TODO: clarify protocol here; can data be in arrow already?
        raise NotImplementedError

    def to_output(self, data=None):
        data = data if data is not None else self.array
        if not isinstance(data, Iterable):
            return data
        return self._to_output(data)

    def apply(self, fn: Callable, where=None, **kwargs):
        """Perform arbitrary function on all the values of the series

        The function should take an ak array as input and produce an
        ak array or scalar.
        """
        if where:
            bits = tuple(where.split("."))
            arr = self.array
            part = arr.__getitem__(bits)
            out = fn(part, **kwargs)
            final = ak.with_field(arr, out, where=where)
        else:
            final = fn(self.array)
        return self.to_output(final)

    def __getitem__(self, item):
        out = self.array.__getitem__(item)
        return self.to_output(out)

    def __dir__(self) -> Iterable[str]:
        attrs = (_ for _ in dir(self.array) if not _.startswith("_"))
        meths = series_methods if self.is_series(self._obj) else df_methods
        return sorted(set(attrs) | set(meths))

    def with_behavior(self, behavior, where=()):
        """Assign a behavior to this array-of-records"""
        # TODO: compare usage with sub-accessors
        # TODO: implement where= (assign directly to ._paraneters["__record__"]
        #  of output's layout. In this case, behaviour is a dict of locations to apply to.
        #  and we can continually add to it (or accept a dict)
        # beh = self._behavior.copy()
        # if isinstance(behavior, dict):
        #    beh.update(behavior)
        # else:
        #    # str or type
        #    beh[where] = behaviour
        return type(self)(self._obj, behavior)

    with_name = with_behavior  # alias - this is the upstream name

    asa = with_behavior  # favoured name

    def __array_function__(self, *args, **kwargs):
        return self.array.__array_function__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        if args[1] == "__call__":
            return self.to_output(args[0](self.array, *args[3:], **kwargs))
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
        return ak.with_name(ak.from_arrow(self.arrow), self._behavior)

    @classmethod
    def register_accessor(cls, name, klass):
        # TODO: check clobber?
        cls.subaccessors[name] = klass

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

    def join(
        self,
        other,
        key: str,
        colname: str = "match",
        sort: bool = False,
        rkey: str | None = None,
        numba: bool = True,
    ):
        """DB ORM-style left join to other dataframe/series with nesting but no copy

        Related records of the ``other`` table will appear as a list under the new field
        ``colname`` for all matching keys. This is the speed and memory efficient way
        to doing a pandas-style merge/join, which explodes out the values to a much
        bigger memory footprint.

        Parameters
        ----------
        other: series or table
        key: name of the field in this table to match on
        colname: the field that will be added to each record. This field will exist even
            if there are no matches, in which case the list will be empty.
        sort: if False, assumes that they key is sorted in both tables. If True, an
            argsort is performed first, and the match is done by indexing. This may be
            significantly slower.
        rkey: if the name of the field to match on in different in the ``other`` table.
        numba: the matching algorithm will go much faster using numba. However, you can
            set this to False if you do not have numba installed.
        """
        from akimbo.io import join

        out = join(
            self.array, other.ak.array, key, colname=colname, sort=sort, rkey=rkey
        )
        return self.to_output(out)

    @classmethod
    def _create_op(cls, op):
        """Make functions to perform all the arithmetic, logical and comparison ops"""

        def run(self, *args, **kwargs):
            ar2 = (ar.ak.array if hasattr(ar, "ak") else ar for ar in args)
            ar3 = (ar.array if isinstance(ar, cls) else ar for ar in ar2)
            return self.to_output(op(self.array, *ar3, **kwargs))

        return run

    def __getattr__(self, item):
        arr = self.array
        if hasattr(arr, item) and callable(getattr(arr, item)):
            func = getattr(arr, item)
            args = ()
        elif item in self.subaccessors:
            return self.subaccessors[item](self)
        elif hasattr(ak, item):
            func = getattr(ak, item)
            args = (arr,)
        else:
            raise AttributeError(item)

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

            ak_arr = func(*args, *others, **kwargs)
            if isinstance(ak_arr, ak.Array):
                return self.to_output(ak_arr)
            return ak_arr

        return f

    def __init_subclass__(cls, **kwargs):
        # auto add methods to all derivative classes
        cls._add_all()
