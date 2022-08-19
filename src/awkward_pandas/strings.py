import awkward._v2 as ak

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
    elif item in string_methods["binary"]:
        name = "binary_" + item
    elif item in string_methods["standard"]:
        name = item
    return getattr(pyarrow.compute, name, None)


def _encode(layout, continuation, **kwargs):
    if layout.is_ListType and layout.parameter("__array__") == "string":
        content = ak.with_parameter(
            layout.content, "__array__", "byte", highlevel=False
        )
        out = ak.with_parameter(layout, "__array__", "bytestring", highlevel=False)
        out._content = content  # the "out" Python object is a copy
        return out
    return layout


def encode(arr, encoding="utf-8"):
    if encoding.lower() not in ["utf-8", "utf8"]:
        raise NotImplementedError
    return ak.Array(arr.layout.recursively_apply(_encode))


def _decode(layout, continuation, encoding="utf-8", **kwargs):
    if layout.is_ListType and layout.parameter("__array__") == "bytestring":
        content = ak.with_parameter(
            layout.content, "__array__", "char", highlevel=False
        )
        out = ak.with_parameter(layout, "__array__", "string", highlevel=False)
        out._content = content  # the "out" Python object is a copy
        return out
    return layout


def decode(arr, encoding="utf-8"):
    if encoding.lower() not in ["utf-8", "utf8"]:
        raise NotImplementedError
    return ak.Array(arr.layout.recursively_apply(_decode))


mapping = {
    "startswith": "starts_with",
    "endswith": "ends_with",
}
