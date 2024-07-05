from __future__ import annotations

from awkward import behavior, mixin_class, mixin_class_method  # re-export

import akimbo.mixin as mixin
from akimbo.io import read_json, read_parquet
from akimbo.version import version as __version__  # noqa

__all__ = (
    "mixin",
    "read_parquet",
    "read_json",
    "behavior",
    "mixin_class",
    "mixin_class_method",
)
