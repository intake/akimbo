from __future__ import annotations

import awkward as ak
import fsspec

import awkward_pandas.pandas


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
    other columns will not be parsed into memory.

    Parameters
    ----------
    url: data location
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    """
    # TODO: dispatch to backends, don't assume pandas as default
    ds = ak.from_parquet(url, storage_options=storage_options, **kwargs)
    s = awkward_pandas.pandas.PandasAwkwardAccessor._to_output(ds)
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

    You can pass a selection of columns to read (list or jsonschema format), and
    other columns will not be parsed into memory.

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
    s = awkward_pandas.pandas.PandasAwkwardAccessor._to_output(ds)
    if extract:
        return s.ak.unmerge()
    return s
