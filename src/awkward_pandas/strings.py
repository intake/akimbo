from __future__ import annotations

import awkward as ak


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


# def all_strings(layout):
#     if layout.is_record:
#         return all(all_strings(layout[field]) for field in layout.fields)
#     if layout.is_list or layout.is_option:
#         if layout.parameter("__array__") == "string":
#             return True
#         return all_strings(layout.content)
#     return layout.parameter("__array__") == "string"


# def all_bytes(layout):
#     if layout.is_record:
#         return all(all_strings(layout[field]) for field in layout.fields)
#     if layout.is_list or layout.is_option:
#         if layout.parameter("__array__") == "bytestring":
#             return True
#         return all_strings(layout.content)
#     return layout.parameter("__array__") == "bytestring"


# def get_split(utf8=True):
#     import pyarrow.compute

#     def f(stuff, sep=""):
#         if sep:
#             return pyarrow.compute.split_pattern(stuff, sep)
#         if utf8:
#             return pyarrow.compute.utf8_split_whitespace(stuff)
#         return pyarrow.compute.ascii_split_whitespace(stuff)

#     return f
