from __future__ import annotations

import awkward_pandas.pandas  # noqa
from awkward_pandas.io import read_json, read_parquet
from awkward_pandas.version import version as __version__  # noqa

__all__ = (
    "read_parquet",
    "read_json",
)
