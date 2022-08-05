from __future__ import annotations

import awkward_pandas.accessor
from awkward_pandas._version import version as __version__
from awkward_pandas.array import AwkwardExtensionArray
from awkward_pandas.dtype import AwkwardDtype
from awkward_pandas.io import read_parquet
from awkward_pandas.utils import merge_columns


__all__ = (
    "AwkwardDtype",
    "AwkwardExtensionArray",
    "merge_columns",
    "read_parquet",
)
