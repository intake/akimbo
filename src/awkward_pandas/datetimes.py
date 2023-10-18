from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import awkward as ak
import pandas as pd
import pyarrow.compute as pc

from awkward_pandas.array import AwkwardExtensionArray

if TYPE_CHECKING:
    from awkward_pandas.accessor import AwkwardAccessor


PYARROW_FUNCTIONS = [
    # CONVERSIONS
    "cast",
    "ceil_temporal",
    "floor_temporal",
    "round_temporal",
    "run_end_decode",
    "run_end_encode",
    "strftime",
    "strptime    ",
    # COMPONENT_EXTRACTION
    "day",
    "day_of_week",
    "day_of_year",
    "hour",
    "iso_week",
    "iso_year",
    "iso_calendar",
    "is_leap_year",
    "microsecond",
    "millisecond",
    "minute",
    "month",
    "nanosecond",
    "quarter",
    "second",
    "subsecond",
    "us_week",
    "us_year",
    "week",
    "year",
    "year_month_day",
    # DIFFERENCES
    "day_time_interval_between",
    "days_between",
    "hours_between",
    "microseconds_between",
    "milliseconds_between",
    "minutes_between",
    "month_day_nano_interval_between",
    "month_interval_between",
    "nanoseconds_between",
    "quarters_between",
    "seconds_between",
    "weeks_between",
    "years_between",
]


class DatetimeAccessor:
    def __init__(self, accessor: AwkwardAccessor) -> None:
        self.accessor = accessor

    def __dir__(self) -> list[str]:
        return sorted(
            [x for x in dir(type(self)) if not x.startswith("_")]
            + dir(super())
            + PYARROW_FUNCTIONS
        )

    def __getattr__(self, attr: str) -> Any:
        if attr not in PYARROW_FUNCTIONS:
            raise ValueError

        fn = getattr(pc, attr)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            arrow_array = ak.to_arrow(self.accessor.array, extensionarray=False)

            arrow_args, arrow_kwargs = [], {}
            for arg in args:
                if isinstance(arg, pd.Series) and arg.dtype == "awkward":
                    arrow_args.append(ak.to_arrow(arg.ak.array, extensionarray=False))
            for k, v in kwargs.items():
                if isinstance(v, pd.Series) and v.dtype == "awkward":
                    arrow_kwargs[k] = ak.to_arrow(v.ak.array, extensionarray=False)

            result = fn(arrow_array, *arrow_args, **arrow_kwargs)
            idx = self.accessor._obj.index
            return pd.Series(AwkwardExtensionArray(ak.from_arrow(result)), index=idx)

        return wrapper
