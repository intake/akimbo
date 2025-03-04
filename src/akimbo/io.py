from __future__ import annotations

import awkward as ak
import fsspec
import numpy as np


def ak_to_series(ds, backend="pandas", extract=True):
    """Make backend-specific series from data"""
    if backend == "pandas":
        import akimbo.pandas

        s = akimbo.pandas.PandasAwkwardAccessor(None).to_output(ds)
    elif backend == "polars":
        import akimbo.polars

        s = akimbo.polars.PolarsAwkwardAccessor(None).to_output(ds)
    elif backend == "dask":
        import akimbo.dask

        # TODO: actually don't use this, use dask-awkward, or dask.dataframe
        s = akimbo.polars.PolarsAwkwardAccessor(None).to_output(ds)
    elif backend == "cudf":
        import akimbo.cudf

        s = akimbo.cudf.CudfAwkwardAccessor(None).to_output(ds)
    elif backend in ["ray", "spark"]:
        raise ValueError("Backend only supports dataframes, not series")

    else:
        raise ValueError("Backend must be in {'pandas', 'polars', 'dask'}")
    if extract and ds.fields:
        return s.ak.unpack()
    return s


# TODO: read_parquet should use native versions rather than convert. This version
#  is OK for pandas
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


# TODO: should be a map over input files, maybe with newline byte blocks
#  as in dask
def read_json(
    url: str,
    storage_options: dict | None = None,
    schema: dict | None = None,
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
    schema: if given, the JSONschema expected in the data; this allows for
        selecting only some part of the record structure, this saving on
        some parsing time and potentially a lot of memory footprint. Even if reading
        all the data, providing a schema will lead to better performance.
    extract: whether to turn top-level records into a dataframe. If False,
        will return a series.
    backend: one of "pandas", "polars" or "dask"
    """
    # TODO: implement columns=["field1", "field2.sub", ...] style schema
    #  using dak.lib.io.layout_to_jsonschema
    with fsspec.open_files(url, **(storage_options or {})) as f:
        ds = ak.concatenate(
            [ak.from_json(_, line_delimited=True, schema=schema, **kwargs) for _ in f]
        )
    return ak_to_series(ds, backend, extract=extract)


def get_json_schema(
    url: str, storage_options: dict | None = None, nbytes: int = 1_000_000, **kwargs
):
    """Get JSONSchema representation of the contents of a line-delimited JSON file

    Currently, requires dask_awkward to be installed, which in turn required dask

    Parameters
    ----------
    url: file location
    storage_options: passed to fsspec
    nbytes: how much of the file to read in infer the types. Must be at least one line, and
        should be representative of all the data.

    Returns
    -------
    JSONschema dictionary
    """
    from dask_awkward.lib.io.json import layout_to_jsonschema

    with fsspec.open(url, **(storage_options or {})) as f:
        data = f.read(nbytes).rsplit(b"\n", 1)[0]
    arr = ak.from_json(data, line_delimited=True, **kwargs)
    return layout_to_jsonschema(arr.layout)


# TODO: should be a map over input files, maybe with newline byte blocks
#  as in dask
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
    from awkward._connect.avro import ReadAvroFT
    from awkward.operations.ak_from_avro_file import _impl

    with fsspec.open(url, **(storage_options or {})) as f:
        # TODO: ak.from_avro_file broken with file-like
        reader = ReadAvroFT(f, limit_entries=None, debug_forth=False)
        ds = _impl(*reader.outcontents, highlevel=True, attrs=None, behavior=None)
    return ak_to_series(ds, backend, extract=extract)


def get_avro_schema(
    url: str,
    storage_options: dict | None = None,
):
    """Fetch ak form of the schema defined in given avro file"""
    from awkward._connect.avro import ReadAvroFT

    with fsspec.open(url, "rb", **(storage_options or {})) as f:
        reader = ReadAvroFT(f, limit_entries=1, debug_forth=False)
        form, length, container = reader.outcontents
    return form


# TODO: feather2/arrow format, get schema


_jitted = [None]


def join(
    table1: ak.Array,
    table2: ak.Array,
    key: str,
    colname: str = "match",
    sort: bool = False,
    rkey: str = None,
    numba=True,
):
    """Make nested ORM-style left join on common key in two tables

    Assuming ``key`` is a field in each table, the output will look like
    ``table1`` but with an extra column ``colname`` containing a list of
    records from matching rows in ``table2``.
    """
    rkey = rkey or key
    # assert key fields are 1D? allow optional?
    if sort:
        # indexed view is not cache friendly; real sort is better but copies
        table1 = table1[ak.argsort(table1[key], axis=0)]
        table2 = table2[ak.argsort(table2[rkey], axis=0)]
    if numba:
        if _jitted[0] is None:
            try:
                from numba import njit

                # per-session cache, cache=True doesn't work
                _jitted[0] = njit()(_merge)
            except ImportError:
                raise ImportError(
                    "numba is required for fast joins, but you can choose to run with"
                    " numba=False"
                )
        merge = _jitted[0]
    else:
        merge = _merge

    counts = np.empty(len(table1), dtype="uint64")
    # TODO: the line below over-allocates, can switch to something growable
    matches = np.empty(len(table2), dtype="uint64")
    # TODO: to_numpy(allow_missing) makes this a bit faster, but is not
    #  not GPU general
    counts, matches, ind = merge(table1[key], table2[key], counts, matches)
    matches.resize(int(ind), refcheck=False)
    indexed = table2[matches]
    listy = ak.unflatten(indexed, counts)
    return ak.with_field(table1, listy, colname)


def _merge(ind1, ind2, counts, matches):
    len2 = len(ind2)
    j = 0
    offind = 0
    matchind = 0
    last = 0
    for i in ind1:
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
                matches[matchind] = j
                j += 1
                matchind += 1
                if j >= len2 or i != ind2[j]:
                    break
        counts[offind] = matchind - last
        last = matchind
        offind += 1
    return counts, matches, matchind
