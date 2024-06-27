from __future__ import annotations

import awkward as ak
import fsspec

import akimbo.pandas


def read_parquet(
    url: str,
    storage_options: dict | None = None,
    extract: bool = True,
    **kwargs,
):
    """Read a Parquet dataset with nested data into a pandas Series or DataFrame.

    This may cope with some deeply nested structures that pandas refuses
    to read by itself.

    You can pass a selection of columns to read (list of strings), and
    other columns will not be parsed into memory. Each of these labels
    may be a root of deeper-nested structs, or use "*" globbing.

    Parameters
    ----------
    url: data location
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    """
    ds = ak.from_parquet(url, storage_options=storage_options, **kwargs)
    s = akimbo.pandas.PandasAwkwardAccessor._to_output(ds)
    if extract:
        return s.ak.unmerge()
    return s


def read_json(
    url,
    storage_options: dict | None = None,
    extract=True,
    **kwargs,
):
    """Read a JSON dataset with nested data into a pandas Series or DataFrame.

    You can pass a selection of columns to read (list or jsonschema format),
    using ``schema=``, and
    other columns will not be parsed into memory. See the docs for
    ak.from_json for further details.

    Parameters
    ----------
    url: data location
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    """
    with fsspec.open(url, **storage_options) as f:
        ds = ak.from_json(
            f,
            line_delimited=True,
            **kwargs,
        )
    s = akimbo.pandas.PandasAwkwardAccessor._to_output(ds)
    if extract:
        return s.ak.unmerge()
    return s
