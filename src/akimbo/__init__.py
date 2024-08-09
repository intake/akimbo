from __future__ import annotations

# re-exports
from awkward import behavior
from awkward import metadata_from_parquet as get_parquet_schema
from awkward import mixin_class, mixin_class_method

import akimbo.datetimes as datetimes
import akimbo.mixin as mixin
import akimbo.strings as strings
from akimbo.io import (
    get_avro_schema,
    get_json_schema,
    join,
    read_avro,
    read_json,
    read_parquet,
)
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
    "get_parquet_schema",
    "get_json_schema",
    "get_avro_schema",
    "strings",
)
