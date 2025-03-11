import pyarrow.compute as pc

from akimbo.apply_tree import dec
from akimbo.mixin import EagerAccessor, LazyAccessor


def match(*layouts):
    return layouts[0].is_leaf and layouts[0].dtype.kind == "M"


methods = [
    "ceil_temporal",
    "day",
    "day_of_week",
    "day_of_year",
    "day_time_interval_between",
    "days_between",
    "floor_temporal",
    "hour",
    "hours_between",
    "is_leap_year",
    "iso_calendar",
    "iso_week",
    "iso_year",
    "microsecond",
    "microseconds_between",
    "millisecond",
    "milliseconds_between",
    "minute",
    "minutes_between",
    "month",
    "month_day_nano_interval_between",
    "month_interval_between",
    "nanosecond",
    "nanoseconds_between",
    "quarter",
    "quarters_between",
    "reound_temporal",
    "second",
    "seconds_between",
    "strftime",
    "subsecond",
    "us_week",
    "us_year",
    "week",
    "weeks_between",
    "year",
    "year_month_day",
    "years_between",
]


class DatetimeAccessor:
    def __getattr__(self, item):
        if item in methods:
            func = getattr(pc, item)
        else:
            raise AttributeError

        return dec(func, match=match, inmode="arrow")

    def __dir__(self):
        return methods


EagerAccessor.register_accessor("dt", DatetimeAccessor)
LazyAccessor.register_accessor("dt", DatetimeAccessor)
