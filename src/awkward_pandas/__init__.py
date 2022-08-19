from __future__ import annotations

import awkward_pandas.accessor
from awkward_pandas._version import version as __version__
from awkward_pandas.array import AwkwardExtensionArray, merge
from awkward_pandas.dtype import AwkwardDtype
from awkward_pandas.io import read_parquet

__all__ = ("AwkwardDtype", "AwkwardExtensionArray", "read_parquet", "merge")
