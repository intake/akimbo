from __future__ import annotations

import awkward as ak


class NoAttributes:
    """Allows importing akimbo.cudf even if cudf isn't installed

    This is done so that sphinx can still build docs on non-GPU systems.
    """

    def __dir__(self):
        return []

    def __getattr__(self, item):
        if item == "__qualname__":
            return "akimbo.utils.DummyAttributesObject"
        if item == "__type_params__":
            return ()
        return self

    def __call__(self, *args, **kwargs):
        return self

    __name__ = "DummyAttributesObject"
    __doc__ = None
    __annotations__ = None


def to_ak_layout(ar, backend=None):
    if hasattr(ar, "ak"):
        return ar.ak.array
    elif hasattr(ar, "array"):
        return ar.array
    elif isinstance(ar, (ak.Array)):
        return ar
    else:
        return ak.Array(ak.to_layout(ar))


def match_string(*layout):
    return layout[0].is_list and layout[0].parameter("__array__") == "string"


def rec_list_swap(arr: ak.Array, field: str | None = None) -> ak.Array:
    """Make a record-of-lists into a list-of-records, assuming the lists have the same lengths"""
    record_of_lists = arr[field] if field else arr
    list_of_records = ak.zip(
        dict(zip(ak.fields(record_of_lists), ak.unzip(record_of_lists))), depth_limit=2
    )
    return ak.with_field(arr, list_of_records, field) if field else list_of_records
