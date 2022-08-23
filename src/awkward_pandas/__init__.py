from __future__ import annotations

import awkward_pandas.accessor  # noqa
from awkward_pandas.array import AwkwardExtensionArray, merge
from awkward_pandas.dtype import AwkwardDtype
from awkward_pandas.io import read_parquet
from awkward_pandas.version import version as __version__  # noqa

__all__ = ("AwkwardDtype", "AwkwardExtensionArray", "read_parquet", "merge")
