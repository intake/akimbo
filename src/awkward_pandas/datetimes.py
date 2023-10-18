from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import awkward as ak
import pandas as pd
import pyarrow.compute as pc

from awkward_pandas.array import AwkwardExtensionArray

if TYPE_CHECKING:
    from awkward_pandas.accessor import AwkwardAccessor


component_extraction = [
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
]


differences = [
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

    def __getattr__(self, attr: str) -> Any:
        if not (attr in component_extraction or attr in differences):
            raise ValueError

        fn = getattr(pc, attr)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            arrow_array = ak.to_arrow(self.accessor.array, extensionarray=False)
            result = ak.from_arrow(fn(arrow_array))
            idx = self.accessor._obj.index
            return pd.Series(AwkwardExtensionArray(result), index=idx)

        return wrapper
