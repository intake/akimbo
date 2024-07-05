from __future__ import annotations

import functools
from collections.abc import Callable

import awkward as ak


def _encode(layout):
    if layout.is_record:
        [_encode(_) for _ in layout._contents]
    elif layout.is_list and layout.parameter("__array__") == "string":
        layout._parameters["__array__"] = "bytestring"
        layout.content._parameters["__array__"] = "byte"
    elif layout.is_option or layout.is_list:
        _encode(layout.content)


def encode(arr: ak.Array, encoding: str = "utf-8") -> ak.Array:
    if encoding.lower() not in ["utf-8", "utf8"]:
        raise NotImplementedError
    arr2 = ak.copy(arr)
    _encode(arr2.layout)
    return ak.Array(arr2)


def _decode(layout):
    if layout.is_record:
        [_decode(_) for _ in layout._contents]
    elif layout.is_list and layout.parameter("__array__") == "bytestring":
        layout._parameters["__array__"] = "string"
        layout.content._parameters["__array__"] = "char"
    elif layout.is_option or layout.is_list:
        _decode(layout.content)


def decode(arr: ak.Array, encoding: str = "utf-8") -> ak.Array:
    if encoding.lower() not in ["utf-8", "utf8"]:
        raise NotImplementedError
    arr2 = ak.copy(arr)  # to be mutated on creation
    _decode(arr2.layout)
    return ak.Array(arr2)


_SA_METHODMAPPING = {
    "endswith": "ends_with",
    "isalnum": "is_alnum",
    "isalpha": "is_alpha",
    "isascii": "is_ascii",
    "isdecimal": "is_decimal",
    "isdigit": "is_digit",
    "islower": "is_lower",
    "isnumeric": "is_numeric",
    "isprintable": "is_printable",
    "isspace": "is_space",
    "istitle": "is_title",
    "isupper": "is_upper",
    "startswith": "starts_with",
}
methods = [
    aname
    for aname in (dir(ak.str))
    if not aname.startswith(("_", "akstr_")) and not aname[0].isupper()
]


class StringAccessor:
    def __init__(self, accessor):
        self.accessor = accessor

    def encode(self, encoding: str = "utf-8"):
        """Encode Series of strings to Series of bytes. Leaves non-strings alone."""
        return self.accessor.to_output(encode(self.accessor.array, encoding=encoding))

    def decode(self, encoding: str = "utf-8"):
        """Decode Series of bytes to Series of strings. Leaves non-bytestrings alone.

        Validity of UTF8 is *not* checked.
        """
        return self.accessor.to_output(decode(self.accessor.array, encoding=encoding))

    @staticmethod
    def method_name(attr: str) -> str:
        return _SA_METHODMAPPING.get(attr, attr)

    def __getattr__(self, attr: str) -> Callable:
        attr = self.method_name(attr)
        fn = getattr(ak.str, attr)

        @functools.wraps(fn)
        def f(*args, **kwargs):
            arr = fn(self.accessor.array, *args, **kwargs)
            # idx = self.accessor._obj.index
            if isinstance(arr, ak.Array):
                return self.accessor.to_output(arr)
            return arr

        return f

    def __dir__(self) -> list[str]:
        return sorted(methods)
