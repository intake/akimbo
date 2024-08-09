import functools
import inspect
from typing import Sequence

import awkward as ak
import pyarrow as pa


class NoDtype:
    kind = ""


def leaf(*layout, **_):
    return layout[0].is_leaf


def run_with_transform(
    arr: ak.Array,
    op,
    match=leaf,
    outtype=None,
    inmode="arrow",
    others=(),
    match_kwargs=None,
    **kw,
) -> ak.Array:
    def func(layout, **kwargs):
        if not isinstance(layout, tuple):
            layout = (layout,)
        if match(*layout, **(match_kwargs or {})):
            if inmode == "arrow":
                out = ak.str._apply_through_arrow(
                    op, *layout, **kw, **(match_kwargs or {})
                )
            elif inmode == "numpy":
                # works on numpy/cupy contents
                out = op(*(lay.data for lay in layout), **kw, **(match_kwargs or {}))
            else:
                out = op(*layout, **kw, **(match_kwargs or {}))
            return outtype(out) if callable(outtype) else out

    return ak.transform(func, arr, *others)


def dec(func, match=leaf, outtype=None, inmode="arrow"):
    """Make a nested/ragged version of an operation to apply throughout a tree"""

    @functools.wraps(func)
    def f(self, *args, where=None, match_kwargs=None, **kwargs):
        if not (
            where is None
            or isinstance(where, str)
            or isinstance(where, Sequence)
            and all(isinstance(_, str) for _ in where)
        ):
            raise ValueError
        others = []
        if args:
            sig = list(inspect.signature(func).parameters)[1:]
            for k, arg in zip(sig, args):
                if isinstance(arg, ak.Array):
                    others.append(arg)
                elif isinstance(
                    arg, (self.accessor.series_type, self.accessor.dataframe_type)
                ):
                    others.append(arg.ak.array)
                elif isinstance(arg, pa.Array):
                    others.append(ak.from_arrow(arg))
                else:
                    kwargs.update({k: arg for k, arg in zip(sig, args)})
        if where:
            bits = tuple(where.split("."))
            arr = self.accessor.array
            part = arr.__getitem__(bits)
            others = [o.__getitem__(bits) for o in others]
            out = run_with_transform(
                part,
                func,
                match=match,
                outtype=outtype,
                inmode=inmode,
                others=others,
                match_kwargs=match_kwargs,
                **kwargs,
            )
            final = ak.with_field(arr, out, where=where)
            return self.accessor.to_output(final)
        else:
            return self.accessor.to_output(
                run_with_transform(
                    self.accessor.array,
                    func,
                    match=match,
                    outtype=outtype,
                    inmode=inmode,
                    others=others,
                    match_kwargs=match_kwargs,
                    **kwargs,
                )
            )

    f.__doc__ = f"""Run vectorized functions on nested/ragged/complex array

where: None | str | Sequence[str, ...]
    if None, will attempt to apply the kernel throughout the nested structure,
    wherever correct types are encountered. If where is given, only the selected
    part of the structure will be considered, but the output will retain
    the original shape. A fieldname or sequence of fieldnames to descend into
    the tree are acceptable
match_kwargs: None | dict
    any extra field identifiers for matching a record as OK to process

{'-Kernel documentation follows from the original function-' if f.__doc__ else ''}
===
{f.__doc__ or str(f)}
"""

    return f
