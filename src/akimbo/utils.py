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


def to_ak_layout(ar):
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
