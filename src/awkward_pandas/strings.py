from __future__ import annotations

import functools
from collections.abc import Callable

import awkward as ak
import pandas as pd

from awkward_pandas.array import AwkwardExtensionArray


def _encode(layout):
    if layout.is_record:
        [_encode(_) for _ in layout._contents]
    elif layout.is_list and layout.parameter("__array__") == "string":
        layout._parameters["__array__"] = "bytestring"
        layout.content._parameters["__array__"] = "byte"
    elif layout.is_option or layout.is_list:
        _encode(layout.content)


def encode(arr, encoding="utf-8"):
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


def decode(arr, encoding="utf-8"):
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


class StringAccessor:
    def __init__(self, accessor):
        self.accessor = accessor

    def encode(self, encoding: str = "utf-8") -> pd.Series:
        """bytes -> string"""
        return pd.Series(AwkwardExtensionArray(encode(self.accessor.array)))

    def decode(self, encoding: str = "utf-8") -> pd.Series:
        """string -> bytes"""
        return pd.Series(AwkwardExtensionArray(decode(self.accessor.array)))

    @staticmethod
    def method_name(attr: str) -> str:
        return _SA_METHODMAPPING.get(attr, attr)

    def __getattr__(self, attr: str) -> Callable:
        attr = StringAccessor.method_name(attr)
        fn = getattr(ak.str, attr)

        @functools.wraps(fn)
        def f(*args, **kwargs):
            arr = fn(self.accessor.array, *args, **kwargs)
            idx = self.accessor._obj.index
            if isinstance(arr, ak.Array):
                return pd.Series(AwkwardExtensionArray(arr), index=idx)
            return arr

        return f

    def __dir__(self) -> list[str]:
        return [
            aname
            for aname in (dir(ak.str))
            if not aname.startswith(("_", "akstr_")) and not aname[0].isupper()
        ]
