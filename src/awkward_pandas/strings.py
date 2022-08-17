
string_methods = {}

def _make_string_methods(utf8=True):
    try:
        import pyarrow.compute
    except ImportError:
        return

    if utf8 not in string_methods:
        if utf8:
            string_methods[utf8] = set(_[5:] for _ in pyarrow._compute.list_functions()
                                       if _.startswith("utf8_"))
        else:
            string_methods[utf8] = set(_[6:] for _ in pyarrow._compute.list_functions()
                                       if _.startswith("ascii_"))
    if "binary" not in string_methods:
        # not certain which of these expects a *list* of str
        string_methods['binary'] = set(_[7:] for _ in pyarrow._compute.list_functions()
                                       if _.startswith("binary_"))
        string_methods['standard'] = {
            'starts_with', 'replace', 'string_is_ascii', 'replace_substring',
            'replace_substring_regex', 'split_pattern', 'extract_regex',
            'count_substring', 'count_substring_regex', 'ends_with',
            'find_substring', 'find_substring_regex', 'index_in', 'is_in',
            'match_like', 'match_substring', 'match_substring_regex',
        }


def dir_str(utf8=True):
    _make_string_methods(utf8)
    return sorted(string_methods[utf8] | string_methods["binary"] | string_methods["standard"])


def get_func(item, utf8=True):
    _make_string_methods(utf8)
    import pyarrow.compute
    if item in string_methods[utf8]:
        name = ["ascii_", "utf8_"][utf8] + item
    elif item in string_methods["binary"]:
        name = "binary_" + item
    elif item in string_methods['standard']:
        name = item
    return getattr(pyarrow.compute, name, None)
