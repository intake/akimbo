from __future__ import annotations

import awkward as ak

import awkward_pandas.pandas


def read_parquet(
    url,
    extract=True,
    root_name="awkward",
    extract_all=False,
    **kwargs,
):
    """Read a Parquet dataset with nested data into a Series or DataFrame."""
    ds = ak.from_parquet(url, **kwargs)
    s = awkward_pandas.pandas.Accessor.to_output(None, ds)
    if extract:
        return s.ak.unmerge()
    return s


def read_json(
    url,
    extract=True,
    root_name="awkward",
    extract_all=False,
    **kwargs,
):
    """Read a JSON dataset with nested data into a Series or DataFrame."""
    ds = ak.from_json(
        url,
        line_delimited=True,
        **kwargs,
    )
    s = awkward_pandas.pandas.Accessor.to_output(None, ds)
    if extract:
        return s.ak.unmerge()
    return s
