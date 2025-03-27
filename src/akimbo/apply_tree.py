from __future__ import annotations

import functools
import inspect
from typing import Callable, Literal

import awkward as ak
import pyarrow as pa


def match_any(*layout, **_):
    return True


def leaf(*layout, **_):
    """True for the lowest elements of any akwward layout tree"""
    return layout[0].is_leaf


def numeric(*layout, **_):
    return layout[0].is_leaf and layout[0].parameters.get("__array__", None) not in {
        "string",
        "char",
    }


def run_with_transform(
    arr: ak.Array,
    op: callable,
    match: callable = leaf,
    outtype: callable | None = None,
    inmode: str = "ak",
    others=(),
    match_kwargs: dict | None = None,
    **kw,
) -> ak.Array:
    def func(layout, **kwargs):
        from akimbo.utils import match_string

        if not isinstance(layout, tuple):
            layout = (layout,)
        if all(match(lay, **(match_kwargs or {})) for lay in layout):
            if inmode == "arrow":
                out = ak.str._apply_through_arrow(
                    op, *layout, **kw, **(match_kwargs or {})
                )
            elif inmode == "numpy":
                # works on numpy/cupy contents
                out = op(*(lay.data for lay in layout), **kw, **(match_kwargs or {}))
            elif inmode == "ak":
                out = op(*layout, **kw, **(match_kwargs or {}))
            else:
                out = op(
                    *(ak.Array(lay) for lay in layout), **kw, **(match_kwargs or {})
                )
            if callable(outtype):
                return outtype(out)
            elif isinstance(out, ak.Array):
                return out.layout
            else:
                return out
        if match_string(*layout):
            # non-string op may fail to descend into string
            return layout[0]

    return ak.transform(func, arr, *others, allow_records=True)


def dec(
    func: callable,
    match: Callable[[ak.contents.Content], bool] = leaf,
    outtype: Callable[[ak.contents.Content], ak.contents.Content] | None = None,
    inmode: Literal["arrow", "numpy", "ak", "other"] = "ak",
    match_kwargs=None,
):
    """Make a nested/ragged version of an operation to apply throughout a tree"""
    match_kw = match_kwargs

    @functools.wraps(func)
    def f(arr, *args, where=None, match_kwargs=None, **kwargs):
        others = []
        if args:
            sig = list(inspect.signature(func).parameters)[1:]
            for k, arg in zip(sig, args):
                if isinstance(arg, ak.Array):
                    others.append(arg)
                elif hasattr(arg, "ak") and hasattr(arg.ak, "array"):
                    others.append(arg.ak.array)
                elif isinstance(arg, pa.Array):
                    others.append(ak.from_arrow(arg))
                else:
                    kwargs.update({k: arg for k, arg in zip(sig, args)})
        if where:
            bits = tuple(where.split(".") if isinstance(where, str) else where)
            part = arr.__getitem__(bits)
            others = [
                o.__getitem__(bits) if isinstance(o, ak.Array) else o for o in others
            ]
        else:
            part = arr

        out = run_with_transform(
            part,
            func,
            match=match,
            outtype=outtype,
            inmode=inmode,
            others=others,
            match_kwargs=match_kwargs or match_kw,
            **kwargs,
        )
        if where:
            out = ak.with_field(arr, out, where=where)
        return out

    f.__doc__ = f"""Run vectorized functions on nested/ragged/complex array

where: None | str | Sequence[str, ...]
    if None, will attempt to apply the kernel throughout the nested structure,
    wherever correct types are encountered. If where is given, only the selected
    part of the structure will be considered, but the output will retain
    the original shape. A fieldname or sequence of fieldnames to descend into
    the tree are acceptable
match_kwargs: None | dict
    any extra field identifiers for matching a record as OK to process

{'--Kernel documentation follows from the original function--' if f.__doc__ else ''}

{f.__doc__ or str(f)}
"""

    return f
