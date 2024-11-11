from __future__ import annotations

import functools
from collections.abc import Callable

import awkward as ak
import pyarrow.compute as pc

from akimbo.apply_tree import dec
from akimbo.mixin import Accessor
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

# make sensible defaults for strptime
strptime = functools.wraps(pc.strptime)(
    lambda *args, format="%FT%T", unit="s", error_is_null=True, **kw: pc.strptime(
        *args, format=format, unit=unit, error_is_null=error_is_null
    )
)


class StringAccessor:
    """String operations on nested/var-length data"""

    # TODO: implement dunder add (concat strings) and mul (repeat strings)
    #  - s.ak.str + "suffix" (and arguments swapped)
    #  - s.ak.str + s2.ak.str (with matching schemas)
    #  - s.ak.str * N (and arguments swapped)
    #  - s.ak.str * s (where each string maps to integers for variable repeats)

    def __init__(self, accessor):
        self.accessor = accessor

    _encode_f = dec(_encode, match=match_string, inmode="ak")

    def encode(self, encoding: str = "utf-8"):
        """Encode Series of strings to Series of bytes. Leaves non-strings alone."""
        if encoding.lower() not in ["utf-8", "utf8"]:
            raise NotImplementedError
        return self._encode_f()

    _decode_f = dec(_decode, match=match_bytestring, inmode="ak")

    def decode(self, encoding: str = "utf-8"):
        """Decode Series of bytes to Series of strings. Leaves non-bytestrings alone.

        Validity of UTF8 is *not* checked.
        """
        if encoding.lower() not in ["utf-8", "utf8"]:
            raise NotImplementedError
        return self._decode_f()

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

    strptime = dec(strptime, match=match_string)

    def __dir__(self) -> list[str]:
        return sorted(methods + ["strptime"])


Accessor.register_accessor("str", StringAccessor)
