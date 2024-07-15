from __future__ import annotations

import functools
import inspect

import awkward as ak
import pyarrow.compute as pc

from akimbo.mixin import Accessor


def _run_unary(layout, op, kind=None, **kw):
    if layout.is_leaf and (kind is None or layout.dtype.kind == kind):
        return ak.str._apply_through_arrow(op, layout, **kw)
    if layout.is_list and layout.parameter("__array__") in ["bytestring", "string"]:
        return ak.str._apply_through_arrow(op, layout, **kw)


def run_unary(arr: ak.Array, op, kind=None, **kw) -> ak.Array:
    def func(x, **kwargs):
        return _run_unary(x, op, kind=kind, **kw)

    return ak.transform(func, arr)


def _run_binary(layout1, layout2, op, kind=None, **kw):
    if layout1.is_leaf and (kind is None or layout1.dtype.kind == kind):
        return ak.str._apply_through_arrow(op, layout1, layout2, **kw)


def run_binary(arr: ak.Array, other, op, kind=None, **kw) -> ak.Array:
    def func(arrays, **kwargs):
        x, y = arrays
        return _run_binary(x, y, op, kind=kind, **kw)

    return ak.transform(func, arr, other)


def dec(func, mode="unary", kind=None):
    # TODO: require kind= on functions that need timestamps

    if mode == "unary":
        # TODO: modily __doc__?
        @functools.wraps(func)
        def f(self, *args, **kwargs):
            if args:
                sig = list(inspect.signature(func).parameters)[1:]
                kwargs.update({k: arg for k, arg in zip(sig, args)})

            return self.accessor.to_output(
                run_unary(self.accessor.array, func, kind=kind, **kwargs)
            )

    elif mode == "binary":

        @functools.wraps(func)
        def f(self, other, *args, **kwargs):
            if args:
                sig = list(inspect.signature(func).parameters)[2:]
                kwargs.update({k: arg for k, arg in zip(sig, args)})

            return self.accessor.to_output(
                run_binary(
                    self.accessor.array, other.ak.array, func, kind=kind, **kwargs
                )
            )

    else:
        raise NotImplementedError
    return f


class DatetimeAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    cast = dec(pc.cast)
    ceil_temporal = dec(pc.ceil_temporal)
    floor_temporal = dec(pc.floor_temporal)
    reound_temporal = dec(pc.round_temporal)
    strftime = dec(pc.strftime)
    strptime = dec(pc.strptime)
    day = dec(pc.day)
    day_of_week = dec(pc.day_of_week)
    day_of_year = dec(pc.day_of_year)
    hour = dec(pc.hour)
    iso_week = dec(pc.iso_week)
    iso_year = dec(pc.iso_year)
    iso_calendar = dec(pc.iso_calendar)
    is_leap_year = dec(pc.is_leap_year)
    microsecond = dec(pc.microsecond)
    millisecond = dec(pc.millisecond)
    minute = dec(pc.minute)
    month = dec(pc.month)
    nanosecond = dec(pc.nanosecond)
    quarter = dec(pc.quarter)
    second = dec(pc.second)
    subsecond = dec(pc.subsecond)
    us_week = dec(pc.us_week)
    us_year = dec(pc.us_year)
    week = dec(pc.week)
    year = dec(pc.year)
    year_month_day = dec(pc.year_month_day)

    day_time_interval_between = dec(pc.day_time_interval_between, mode="binary")
    days_between = dec(pc.days_between, mode="binary")
    hours_between = dec(pc.hours_between, mode="binary")
    microseconds_between = dec(pc.microseconds_between, mode="binary")
    milliseconds_between = dec(pc.milliseconds_between, mode="binary")
    minutes_between = dec(pc.minutes_between, mode="binary")
    month_day_nano_interval_between = dec(
        pc.month_day_nano_interval_between, mode="binary"
    )
    month_interval_between = dec(pc.month_interval_between, mode="binary")
    nanoseconds_between = dec(pc.nanoseconds_between, mode="binary")
    quarters_between = dec(pc.quarters_between, mode="binary")
    seconds_between = dec(pc.seconds_between, mode="binary")
    weeks_between = dec(pc.weeks_between, mode="binary")
    years_between = dec(pc.years_between, mode="binary")


def _to_arrow(array):
    array = _make_unit_compatible(array)
    return ak.to_arrow(array, extensionarray=False)


def _make_unit_compatible(array):
    # TODO, actually convert units if not compatible
    return array


Accessor.register_accessor("dt", DatetimeAccessor)
