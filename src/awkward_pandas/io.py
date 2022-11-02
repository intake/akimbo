from __future__ import annotations

import awkward as ak
import pandas as pd

from awkward_pandas.array import AwkwardExtensionArray


def from_awkward(array: ak.Array, name: str | None = None) -> pd.Series:
    """Wrap an awkward Array in a pandas Series.

    Parameters
    ----------
    array : ak.Array
        Awkward array to wrap.
    name : str, optional
        Name for the series.

    Returns
    -------
    pandas.Series
        Resulting Series with dtype AwkwardDtype

    Examples
    --------
    >>> import awkward as ak
    >>> import awkward_pandas as akpd
    >>> a = ak.from_iter([[1, 2, 3], [4, 5], [6]])
    >>> s = akpd.from_awkward(a, name="my-array")
    0    [1, 2, 3]
    1       [4, 5]
    2          [6]
    Name: my-array, dtype: awkward

    """
    return pd.Series(AwkwardExtensionArray(array), name=name)


def read_parquet(
    url,
    extract=True,
    root_name="awkward",
    extract_all=False,
    **kwargs,
):
    """Read a Parquet dataset with nested data into a Series or DataFrame."""
    ds = ak.from_parquet(url, **kwargs)
    s = from_awkward(ds, name=root_name)
    if extract:
        return s.ak.to_columns(cull=True, extract_all=extract_all)
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
    s = from_awkward(ds, name=root_name)
    if extract:
        return s.ak.to_columns(cull=True, extract_all=extract_all)
    return s
