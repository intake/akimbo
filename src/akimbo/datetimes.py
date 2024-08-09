import functools

import awkward as ak
import pyarrow.compute as pc

from akimbo.apply_tree import dec
from akimbo.mixin import Accessor


def match(*layouts):
    return layouts[0].is_leaf and layouts[0].dtype.kind == "M"


dec_t = functools.partial(dec, match=match)


class DatetimeAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    # listed below https://arrow.apache.org/docs/python/generated/
    # pyarrow.compute.ceil_temporal.html
    cast = dec(pc.cast)  # TODO: move to .ak
    ceil_temporal = dec_t(pc.ceil_temporal)
    floor_temporal = dec_t(pc.floor_temporal)
    reound_temporal = dec_t(pc.round_temporal)
    strftime = dec_t(pc.strftime)
    strptime = dec_t(pc.strptime)
    day = dec_t(pc.day)
    day_of_week = dec_t(pc.day_of_week)
    day_of_year = dec_t(pc.day_of_year)
    hour = dec_t(pc.hour)
    iso_week = dec_t(pc.iso_week)
    iso_year = dec_t(pc.iso_year)
    iso_calendar = dec_t(pc.iso_calendar)
    is_leap_year = dec_t(pc.is_leap_year)
    microsecond = dec_t(pc.microsecond)
    millisecond = dec_t(pc.millisecond)
    minute = dec_t(pc.minute)
    month = dec_t(pc.month)
    nanosecond = dec_t(pc.nanosecond)
    quarter = dec_t(pc.quarter)
    second = dec_t(pc.second)
    subsecond = dec_t(pc.subsecond)
    us_week = dec_t(pc.us_week)
    us_year = dec_t(pc.us_year)
    week = dec_t(pc.week)
    year = dec_t(pc.year)
    year_month_day = dec_t(pc.year_month_day)

    day_time_interval_between = dec_t(pc.day_time_interval_between)
    days_between = dec_t(pc.days_between)
    hours_between = dec_t(pc.hours_between)
    microseconds_between = dec_t(pc.microseconds_between)
    milliseconds_between = dec_t(pc.milliseconds_between)
    minutes_between = dec_t(pc.minutes_between)
    month_day_nano_interval_between = dec_t(pc.month_day_nano_interval_between)
    month_interval_between = dec_t(pc.month_interval_between)
    nanoseconds_between = dec_t(pc.nanoseconds_between)
    quarters_between = dec_t(pc.quarters_between)
    seconds_between = dec_t(pc.seconds_between)
    weeks_between = dec_t(pc.weeks_between)
    years_between = dec_t(pc.years_between)

    # TODO: strftime, strptime

    # TODO: timezone conversion


def _to_arrow(array):
    array = _make_unit_compatible(array)
    return ak.to_arrow(array, extensionarray=False)


def _make_unit_compatible(array):
    # TODO, actually convert units if not compatible
    return array


Accessor.register_accessor("dt", DatetimeAccessor)
