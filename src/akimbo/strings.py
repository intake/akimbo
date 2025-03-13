from __future__ import annotations

import functools

import awkward as ak
import pyarrow as pa
import pyarrow.compute as pc

from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor
from akimbo.utils import match_string


def _encode(layout):
    layout._parameters["__array__"] = "bytestring"
    layout.content._parameters["__array__"] = "byte"
    return layout


def match_bytestring(*layout):
    return layout[0].is_list and layout[0].parameter("__array__") == "bytestring"


def _decode(layout):
    layout._parameters["__array__"] = "string"
    layout.content._parameters["__array__"] = "char"
    return layout


_decode_f = dec(_decode, match=match_bytestring, inmode="ak")
_encode_f = dec(_encode, match=match_string, inmode="ak")
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


@functools.wraps(pc.strptime)
def strptime(*args, format="%FT%T", unit="us", error_is_null=True, **kw):
    """strptime with typical defaults set to reverse strftime"""
    out = pc.strptime(
        *args, format=format, unit=unit, error_is_null=error_is_null, **kw
    )
    return out


def repeat(arr, count):
    return pc.binary_repeat(arr, count)


def concat(arr, arr2, sep=""):
    return pc.binary_join_element_wise(
        arr.cast(pa.string()), arr2.cast(pa.string()), sep
    )


class StringAccessor:
    """String operations on nested/var-length data"""

    # TODO: implement dunder add (concat strings) and mul (repeat strings)
    #  - s.ak.str + "suffix" (and arguments swapped)
    #  - s.ak.str + s2.ak.str (with matching schemas)
    #  - s.ak.str * N (and arguments swapped)
    #  - s.ak.str * s (where each string maps to integers for variable repeats)

    def encode(self, arr, encoding: str = "utf-8"):
        """Encode Series of strings to Series of bytes. Leaves non-strings alone."""
        if encoding.lower() not in ["utf-8", "utf8"]:
            raise NotImplementedError
        return _encode_f(arr)

    def decode(self, arr, encoding: str = "utf-8"):
        """Decode Series of bytes to Series of strings. Leaves non-bytestrings alone.

        Validity of UTF8 is *not* checked.
        """
        if encoding.lower() not in ["utf-8", "utf8"]:
            raise NotImplementedError
        return _decode_f(arr)

    @staticmethod
    def method_name(attr: str) -> str:
        return _SA_METHODMAPPING.get(attr, attr)

    def __getattr__(self, attr: str) -> callable:
        attr = self.method_name(attr)
        return getattr(ak.str, attr)

    strptime = staticmethod(dec(strptime, match=match_string, inmode="arrow"))
    repeat = staticmethod(dec(repeat, match=match_string, inmode="arrow"))
    join_el = staticmethod(dec(concat, match=match_string, inmode="arrow"))

    def __add__(self, *_):
        return dec(concat, match=match_string, inmode="arrow")

    def __mul__(self, *_):
        return dec(repeat, match=match_string, inmode="arrow")

    def __dir__(self) -> list[str]:
        return sorted(methods + ["strptime", "encode", "decode"])


EagerAccessor.register_accessor("str", StringAccessor)
LazyAccessor.register_accessor("str", StringAccessor)
