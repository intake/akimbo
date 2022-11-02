from __future__ import annotations

import awkward_pandas.accessor  # noqa
from awkward_pandas.array import AwkwardExtensionArray
from awkward_pandas.dtype import AwkwardDtype
from awkward_pandas.io import from_awkward, read_json, read_parquet
from awkward_pandas.lib import merge
from awkward_pandas.version import version as __version__  # noqa

__all__ = (
    "AwkwardDtype",
    "AwkwardExtensionArray",
    "from_awkward",
    "merge",
    "read_parquet",
    "read_json",
)
