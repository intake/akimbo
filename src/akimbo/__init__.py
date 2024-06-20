from __future__ import annotations

from akimbo.io import read_json, read_parquet
from akimbo.version import version as __version__  # noqa

__all__ = (
    "read_parquet",
    "read_json",
)
