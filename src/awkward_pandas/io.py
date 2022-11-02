import awkward as ak
import pandas as pd

import awkward_pandas


def read_parquet(
    url,
    extract=True,
    root_name="awkward",
    extract_all=False,
    **kwargs,
):
    """Read a Parquet dataset with nested data into a Series or DataFrame.

    Parameters
    ----------
    url : str
        Location of the dataset.
    extract : bool
        ...
    root_name : str
        ...
    extract_all : bool
        ...

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Resulting Pandas object.

    """
    ds = ak.from_parquet(url, **kwargs)
    s = pd.Series(awkward_pandas.AwkwardExtensionArray(ds), name=root_name)
    if extract:
        return s.ak.to_columns(cull=True, extract_all=extract_all)
    return s


def read_json(
    source,
    extract=True,
    root_name="awkward",
    extract_all=False,
    **kwargs,
):
    """Read a Parquet dataset with nested data into a Series or DataFrame."""
    ds = ak.from_json(
        source,
        line_delimited=True,
        **kwargs,
    )
    s = pd.Series(awkward_pandas.AwkwardExtensionArray(ds), name=root_name)
    if extract:
        return s.ak.to_columns(cull=True, extract_all=extract_all)
    return s
