import functools
import inspect

import awkward as ak
import pyarrow as pa


class NoDtype:
    kind = ""


def leaf(*layout):
    return layout[0].is_leaf


def run(
    arr: ak.Array, op, match=leaf, outtype=None, inmode="arrow", others=(), **kw
) -> ak.Array:
    def func(layout, **kwargs):
        if not isinstance(layout, tuple):
            layout = (layout,)
        if match(*layout):
            if inmode == "arrow":
                out = ak.str._apply_through_arrow(op, *layout, **kw)
            elif inmode == "numpy":
                # works on numpy/cupy contents
                out = op(*(lay.data for lay in layout), **kw)
            else:
                out = op(*layout, **kw)
            return outtype(out) if callable(outtype) else out

    return ak.transform(func, arr, *others)


def dec(func, match=leaf, outtype=None, inmode="arrow"):
    """Make a nested/ragged version of an operation to apply throughout a tree"""

    @functools.wraps(func)
    def f(self, *args, **kwargs):
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

        return self.accessor.to_output(
            run(
                self.accessor.array,
                func,
                match=match,
                outtype=outtype,
                inmode=inmode,
                others=others,
                **kwargs,
            )
        )

    return f
