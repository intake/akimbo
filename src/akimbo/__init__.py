from __future__ import annotations

from awkward import (  # re-export
    behavior,
    metadata_from_parquet,
    mixin_class,
    mixin_class_method,
)

import akimbo.datetimes as datetimes
import akimbo.mixin as mixin
import akimbo.strings as strings
from akimbo.io import join, read_avro, read_json, read_parquet
from akimbo.version import version as __version__  # noqa

__all__ = (
    "datetimes",
    "mixin",
    "join",
    "read_avro",
    "read_parquet",
    "read_json",
    "behavior",
    "mixin_class",
    "mixin_class_method",
    "metadata_from_parquet",
    "strings",
)
