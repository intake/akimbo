from __future__ import annotations

import awkward as ak
import fsspec


def ak_to_series(ds, backend="pandas", extract=True):
    if backend == "pandas":
        import akimbo.pandas

        s = akimbo.pandas.PandasAwkwardAccessor._to_output(ds)
    elif backend == "polars":
        import akimbo.polars

        s = akimbo.polars.PolarsAwkwardAccessor._to_output(ds)
    elif backend == "dask":
        import akimbo.dask

        # TODO: actually don't use this, use dask-awkward, or dask.dataframe
        s = akimbo.polars.PolarsAwkwardAccessor._to_output(ds)
    else:
        raise ValueError("Backend must be in {'pandas', 'polars', 'dask'}")
    if extract:
        return s.ak.unmerge()
    return s


def read_parquet(
    url: str,
    storage_options: dict | None = None,
    extract: bool = True,
    backend: str = "pandas",
    **kwargs,
):
    """Read a Parquet dataset with nested data into a Series or DataFrame.

    This may cope with some deeply nested structures that pandas refuses
    to read by itself.

    You can pass a selection of columns to read (list of strings), and
    other columns will not be parsed into memory. Each of these labels
    may be a root of deeper-nested structs, or use "*" globbing.

    Parameters
    ----------
    url: data location
        Directory with data files, single file or glob pattern
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    backend: one of "pandas", "polars" or "dask"
    """
    # TODO: is this useful compared to pyarrow.parquet? Describe differences
    ds = ak.from_parquet(url, storage_options=storage_options, **kwargs)
    return ak_to_series(ds, backend, extract=extract)


def read_json(
    url: str,
    storage_options: dict | None = None,
    extract: bool = True,
    backend: str = "pandas",
    **kwargs,
):
    """Read a JSON dataset with nested data into a Series or DataFrame.

    You can pass a selection of columns to read (list or jsonschema format),
    using ``schema=``, and
    other columns will not be parsed into memory. See the docs for
    ak.from_json for further details.

    (examples to come)

    Parameters
    ----------
    url: data location (may include glob characters)
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    backend: one of "pandas", "polars" or "dask"
    """
    with fsspec.open_files(url, **(storage_options or {})) as f:
        ds = ak.concatenate([ak.from_json(_, line_delimited=True, **kwargs) for _ in f])
    return ak_to_series(ds, backend, extrcact=extract)


def read_avro(
    url: str,
    storage_options: dict | None = None,
    extract: bool = True,
    backend: str = "pandas",
    **kwargs,
):
    """Read AVRO structured data files

    Parameters
    ----------
    url: data location (may include glob characters)
    storage_options: any arguments for an fsspec backend
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    backend: one of "pandas", "polars" or "dask"
    """
    with fsspec.open(url, **(storage_options or {})) as f:
        ds = ak.from_avro_file(f, **kwargs)
    return ak_to_series(ds, backend, extract=extract)


def _merge(ind1, ind2, builder):
    """numba jittable laft join/merge index finder"""
    len2 = len(ind2)
    j = 0
    for i in ind1:
        builder.begin_list()
        while True:
            if j >= len2:
                break
            if i > ind2[j]:
                # ID not yet found
                j += 1
                continue
            if i < ind2[j]:
                # no more entrie
                break
            # hit
            while True:
                builder.append(j)
                j += 1
                if j >= len2 or i != ind2[j]:
                    break
        builder.end_list()


def join(
    table1: ak.Array,
    table2: ak.Array,
    key: str,
    colname: str = "match",
    sort: bool = False,
    rkey: str = None,
):
    """Make nested ORM-style left join on common key in two tables

    Assuming ``key`` is a field in each table, the output will look like
    ``table1`` but with an extra column ``colname`` containing a list of
    records from matching rows in ``table2``.
    """
    import numba

    rkey = rkey or key
    if sort:
        # indexed view is not cache friendly; real sort is better
        table1 = table1[ak.argsort(table1[key], axis=0)]
        table2 = table2[ak.argsort(table2[rkey], axis=0)]
    merge = numba.njit(cache=True)(_merge)
    builder = ak.ArrayBuilder()
    merge(table1[key], table2[key], builder)
    merge_index = builder.snapshot()
    indexed = table2[ak.flatten(merge_index)]
    counts = ak.num(merge_index)
    listy = ak.unflatten(indexed, counts)
    return ak.with_field(table1, listy, colname)
