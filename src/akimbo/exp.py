"""Experimental code"""

import ast
import operator

import awkward as ak
import numba


@numba.njit
def is_arange(arr):
    """Does this look like an array made by arange

    Awkward index arrays look like this if there are no nulls and it's not from
    categorical/dictionaries.
    """
    last = -1
    for x in arr:
        if x != last + 1:
            return False
        last = x
    return True


op_map = {">": operator.gt, ">=": operator.ge, "==": operator.eq}


def query(arr: ak.Array, expr: str):
    """Perform a nested-pandas style query on inner dataframeish layer

    Example:
        >>> out = query(arr, "nested.t > 17")
    """
    field_selector, op, value = expr.split()
    value = ast.literal_eval(value)
    fields = tuple(field_selector.split("."))
    argument = ak.to_buffers(ak.flatten(arr.__getitem__(fields)))[-1]["node1-data"]
    bools = op_map[op](argument, value)  # dispatches to numpy

    outer = ak.copy(arr[fields[0]])
    inner0 = outer[fields[1]].layout
    prev = None
    inner = inner0

    while True:
        if len(inner) == len(bools):
            break
        inner, prev = inner._content, inner
    if isinstance(inner, ak.contents.UnmaskedArray):
        inner = inner.to_ByteMaskedArray(True)
        prev._content = inner
        inner._mask._data[:] = bools
    elif isinstance(inner, ak.contents.ByteMaskedArray):
        inner._mask._data &= bools
    elif isinstance(inner, ak.contents.IndexedOptionArray):
        inner._index._data = inner.index.data[bools]
    return ak.with_field(arr, inner0, fields)


def rec_list_swap(arr: ak.Array, field: str | None = None) -> ak.Array:
    """Make a record-of-lists into a list-of-records, assuming the lists have the same lengths"""
    record_of_lists = arr[field] if field else arr
    list_of_records = ak.zip(
        dict(zip(ak.fields(record_of_lists), ak.unzip(record_of_lists))), depth_limit=2
    )
    return ak.with_field(arr, list_of_records, field) if field else list_of_records
