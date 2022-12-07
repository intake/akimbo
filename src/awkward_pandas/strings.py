from __future__ import annotations

import awkward as ak

string_methods = {}


def _make_string_methods(utf8=True):
    try:
        import pyarrow.compute
    except ImportError:
        return

    if utf8 not in string_methods:
        if utf8:
            string_methods[utf8] = {
                _[5:]
                for _ in pyarrow._compute.list_functions()
                if _.startswith("utf8_")
            }
        else:
            string_methods[utf8] = {
                _[6:]
                for _ in pyarrow._compute.list_functions()
                if _.startswith("ascii_")
            }
        string_methods[utf8] = {
            "is" + _[3:] if _.startswith("is_") else _ for _ in string_methods[utf8]
        }
    if "binary" not in string_methods:
        # not certain which of these expects a *list* of str
        string_methods["binary"] = {
            _[7:] for _ in pyarrow._compute.list_functions() if _.startswith("binary_")
        }
        string_methods["standard"] = {
            "startswith",
            "replace",
            "string_is_ascii",
            "replace_substring",
            "replace_substring_regex",
            "split_pattern",
            "extract_regex",
            "count_substring",
            "count_substring_regex",
            "endswith",
            "find_substring",
            "find_substring_regex",
            "index_in",
            "is_in",
            "match_like",
            "match_substring",
            "match_substring_regex",
            "encode",
            "decode",
            "split",
        }


def dir_str(utf8=True):
    _make_string_methods(utf8)
    return sorted(
        string_methods[utf8] | string_methods["binary"] | string_methods["standard"]
    )


def get_func(item, utf8=True):
    _make_string_methods(utf8)
    import pyarrow.compute

    if item in string_methods[utf8]:
        if item.startswith("is"):
            item = "is_" + item[2:]
        name = ["ascii_", "utf8_"][utf8] + item
    elif item in mapping:
        name = mapping[item]
        if callable(name):
            return name
    elif item == "split":
        return get_split(utf8)
    elif item in string_methods["binary"]:
        name = "binary_" + item
    elif item in string_methods["standard"]:
        name = item
    return getattr(pyarrow.compute, name, None)


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


def all_strings(layout):
    if layout.is_record:
        return all(all_strings(layout[field]) for field in layout.fields)
    if layout.is_list or layout.is_option:
        if layout.parameter("__array__") == "string":
            return True
        return all_strings(layout.content)
    return layout.parameter("__array__") == "string"


def all_bytes(layout):
    if layout.is_record:
        return all(all_strings(layout[field]) for field in layout.fields)
    if layout.is_list or layout.is_option:
        if layout.parameter("__array__") == "bytestring":
            return True
        return all_strings(layout.content)
    return layout.parameter("__array__") == "bytestring"


def get_split(utf8=True):
    import pyarrow.compute

    def f(stuff, sep=""):
        if sep:
            return pyarrow.compute.split_pattern(stuff, sep)
        if utf8:
            return pyarrow.compute.utf8_split_whitespace(stuff)
        return pyarrow.compute.ascii_split_whitespace(stuff)

    return f


mapping = {
    "startswith": "starts_with",
    "endswith": "ends_with",
}
