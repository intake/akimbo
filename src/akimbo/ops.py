import awkward as ak


def rename(arr: ak.Array, where: str, to: str) -> ak.Array:
    """Assign new field name to given location in the structure

    Parameters
    ----------
    arr: array to change
    where: str | tuple[str]
        location we would like to rename
    to: str
        new name
    """
    lay = ak.copy(arr.layout)
    where = list(where) if isinstance(where, tuple) else [where]
    parent = None
    bit = lay
    this = None
    while where:
        if getattr(bit, "contents", None):
            this = bit.fields.index(where.pop(0))
            parent, bit = bit, bit.contents[this]
        else:
            parent, bit = bit, bit.content
    parent.fields[this] = to
    return ak.Array(lay)


def join(
    arr: ak.Array,
    other: ak.Array,
    key: str,
    colname: str = "match",
    sort: bool = False,
    rkey: str | None = None,
    numba: bool = True,
):
    """DB ORM-style left join to other dataframe/series with nesting but no copy

    Related records of the ``other`` table will appear as a list under the new field
    ``colname`` for all matching keys. This is the speed and memory efficient way
    to doing a pandas-style merge/join, which explodes out the values to a much
    bigger memory footprint.

    Parameters
    ----------
    other: series or table
    key: name of the field in this table to match on
    colname: the field that will be added to each record. This field will exist even
        if there are no matches, in which case the list will be empty.
    sort: if False, assumes that they key is sorted in both tables. If True, an
        argsort is performed first, and the match is done by indexing. This may be
        significantly slower.
    rkey: if the name of the field to match on is different in the ``other`` table.
    numba: the matching algorithm will go much faster using numba. However, you can
        set this to False if you do not have numba installed.
    """
    from akimbo.io import join

    out = join(arr, other, key, colname=colname, sort=sort, rkey=rkey, numba=numba)
    return out
