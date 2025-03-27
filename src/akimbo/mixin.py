from __future__ import annotations

import functools
import operator
from typing import Callable

import awkward as ak
import numpy as np
import pyarrow.compute as pc

from akimbo.apply_tree import dec, match_any, numeric, run_with_transform
from akimbo.ops import join, rename
from akimbo.utils import to_ak_layout

methods = [
    _ for _ in (dir(ak)) if not _.startswith(("_", "ak_")) and not _[0].isupper()
]


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


class EagerAccessor(ArithmeticMixin):
    """Bring the awkward API to dataframes and series"""

    series_type = ()
    dataframe_type = ()
    subaccessors = {}
    funcs = {"cast": dec(pc.cast, inmode="arrow"), "join": join, "rename": rename}

    def __init__(self, obj, behavior=None, subaccessor=None):
        self.subaccessor = subaccessor
        self._obj = obj
        self._behavior = behavior

    def __call__(self, *args, behavior=None, **kwargs):
        return EagerAccessor(self._obj, behavior=behavior)

    def unexplode(self, *cols: tuple[str, ...], outname="grouped") -> ak.Array:
        """Repack "exploded" form dataframes into lists of structs

        This is the inverse of the regular dataframe explode() process.

        Uses arrow directly.
        """
        from akimbo.utils import rec_list_swap

        pa_arr = self.arrow
        # TODO: this does not work on cuDF as here we use arrow directly
        # TODO: pandas indexes are pre-grouped cat-like structures
        cols = list(cols)
        if set(cols) - set(pa_arr.column_names):
            raise ValueError(
                "One or more grouping column (%s) not in available columns %s",
                cols,
                pa_arr.column_names,
            )
        outcols = [(_, "list") for _ in pa_arr.column_names if _ not in cols]
        if not outcols:
            raise ValueError("Cannot group on all available columns")
        outcols2 = [f"{_[0]}_list" for _ in outcols]
        grouped = pa_arr.group_by(cols).aggregate(outcols)
        akarr = ak.from_arrow(grouped)
        akarr2 = akarr[outcols2]
        akarr2.layout._fields = [_[0] for _ in outcols]
        struct = rec_list_swap(akarr2)
        final = ak.with_field(akarr[cols], struct, outname)

        return self.to_output(final).ak.unpack()

    @classmethod
    def _create_op(cls, op):
        def run(self, *args, **kwargs):
            if self.subaccessor:
                # defer
                op2 = op(self.subaccessors[self.subaccessor](), None)
                return self.__getattr__(op2)(*args, **kwargs)
            args = [
                to_ak_layout(_) if isinstance(_, (str, int, float, np.number)) else _
                for _ in args
            ]
            return self.transform(op, *args, match=numeric, **kwargs)

        return run

    def __getitem__(self, item):
        return self.__getattr__(operator.getitem)(item)

    def __array_ufunc__(self, *args, where=None, out=None, **kwargs):
        # includes operator overload like df.ak + 1
        ufunc, call, inputs, *callargs = args
        if out is not None or call != "__call__":
            raise NotImplementedError

        return self.__getattr__(ufunc)(*callargs, where=where, **kwargs)

    def __dir__(self) -> list[str]:
        if self.subaccessor is not None:
            return dir(self.subaccessors[self.subaccessor]())
        cls_meth = [_ for _ in dir(self.dataframe_type) if not _.startswith("_")]
        this_meth = [_ for _ in dir(type(self)) if not _.startswith("_")]
        return sorted(methods + cls_meth + this_meth + list(self.funcs))

    def transform(
        self,
        fn: callable,
        *others,
        where=None,
        match=match_any,
        inmode="array",
        **kwargs,
    ):
        def f(arr, *others, **kwargs):
            return run_with_transform(
                arr, fn, match=match, others=others, inmode=inmode, **kwargs
            )

        return self.__getattr__(f)(*others, where=where, **kwargs)

    def apply(self, fn: Callable, *others, where=None, **kwargs):
        return self.__getattr__(fn)(*others, where=where, **kwargs)

    def with_behavior(self, behavior, where=()):
        """Assign a behavior to this array-of-records"""
        # TODO: compare usage with sub-accessors
        # TODO: implement where= (assign directly to ._paraneters["__record__"]
        #  of output's layout. In this case, behaviour is a dict of locations to apply to.
        #  and we can continually add to it (or accept a dict)
        return type(self)(self._obj, behavior)

    with_name = with_behavior  # alias - this is the upstream name

    asa = with_behavior  # favoured name

    def __array_function__(self, *args, **kwargs):
        return self.array.__array_function__(*args, **kwargs)

    @classmethod
    def register_accessor(cls, name: str, klass: type):
        # TODO: check clobber?
        cls.subaccessors[name] = klass

    def _pre(self, item):
        arr = self.array
        if callable(item):
            func = item
        elif item in self.funcs:
            func = self.funcs[item]
        elif hasattr(arr, item) and callable(getattr(arr, item)):
            func = getattr(type(arr), item)
        elif self.subaccessor:
            func = getattr(self.subaccessors[self.subaccessor](), item)
        elif hasattr(ak, item):
            func = getattr(ak, item)
        else:
            raise AttributeError(item)
        return func, (arr,)

    @classmethod
    def _to_arr(cls, other):
        # make eager ak arrays for any standard input
        if isinstance(other, (cls.series_type, cls.dataframe_type)):
            return other.ak.array
        if isinstance(other, cls):
            return other.array
        return other

    def __getattr__(self, item, frame=False):
        if not self.subaccessor and item in self.subaccessors:
            return type(self)(self._obj, subaccessor=item)
        func, args = self._pre(item)
        frame = isinstance(self._obj, self.dataframe_type)

        @functools.wraps(func)
        def f(*others, where=None, **kwargs):
            others = [self._to_arr(other) for other in others]
            kwargs = {k: self._to_arr(v) for k, v in kwargs.items()}
            if where:
                arr0 = args[0]
                ar0 = [_[where] for _ in args]
            else:
                arr0 = None
                ar0 = args
            ak_arr = func(*ar0, *others, **kwargs)
            if isinstance(ak_arr, ak.Array):
                if where:
                    ak_arr = ak.with_field(arr0, ak_arr, where)
                if frame and ak_arr.fields == ar0[0].fields:
                    return self.to_output(ak_arr).ak.unpack()
                return self.to_output(ak_arr)
            return ak_arr

        return f

    def __init_subclass__(cls, **kwargs):
        # auto add methods to all derivative classes
        cls._add_all()

    @property
    def array(self):
        return ak.from_arrow(self.arrow)

    @property
    def arrow(self):
        return self.to_arrow(self._obj)

    def pack(self):
        """Make a single complex series out of the columns of a dataframe"""
        if not isinstance(self._obj, self.dataframe_type):
            raise ValueError("Can only pack on a dataframe")
        out = {}
        for k in self._obj.columns:
            # TODO: partial pack when column names are like "record.field"
            out[k] = self._obj[k].ak.array
        arr = ak.Array(out)
        return self.to_output(arr)

    def unpack(self):
        """Make dataframe out of a series of record type"""
        # TODO: what to do when passed a dataframe, partial unpack of record fields?
        arr = self.array
        if not arr.fields:
            raise ValueError("Not array-of-records")
        out = {k: self.to_output(arr[k]) for k in arr.fields}
        return self.dataframe_type(out)

    def to_output(self, data=None):
        """Data returned as a series"""
        raise NotImplementedError

    def to_arrow(self, data=None):
        """Data as an arrow array"""
        raise NotImplementedError


class LazyAccessor(EagerAccessor):
    def __getattr__(self, item):
        raise NotImplementedError

    def unexplode(self):
        raise NotImplementedError

    def pack(self):
        raise NotImplementedError

    def unpack(self):
        raise NotImplementedError
