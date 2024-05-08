from __future__ import annotations

import functools
import inspect

import awkward as ak
import pyarrow.compute as pc


def _run_unary(layout, op, kind=None, **kw):
    if layout.is_leaf and (kind is None or layout.dtype.kind == kind):
        return ak.str._apply_through_arrow(op, layout, **kw)
    if layout.is_list and layout.parameter("__array__") in ["bytestring", "string"]:
        return ak.str._apply_through_arrow(op, layout, **kw)


def run_unary(arr: ak.Array, op, kind=None, **kw) -> ak.Array:
    def func(x, **kwargs):
        return _run_unary(x, op, kind=kind, **kw)

    return ak.transform(func, arr)


def dec(func, mode="unary"):
    # TODO: require kind= on functions that need timestamps

    if mode == "unary":
        # TODO: modily __doc__?
        @functools.wraps(func)
        def f(self, *args, **kwargs):
            if args:
                sig = list(inspect.signature(func).parameters)[1:]
                kwargs.update({k: arg for k, arg in zip(sig, args)})

            return self.accessor.to_output(
                run_unary(self.accessor.array, func, **kwargs)
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

    # the rest are binary
    def day_time_interval_between(self, end):
        raise NotImplementedError("TODO")

    def days_between(self, end):
        raise NotImplementedError("TODO")

    def hours_between(self, end):
        raise NotImplementedError("TODO")

    def microseconds_between(self, end):
        raise NotImplementedError("TODO")

    def milliseconds_between(self, end):
        raise NotImplementedError("TODO")

    def minutes_between(self, end):
        raise NotImplementedError("TODO")

    def month_day_nano_interval_between(self, end):
        raise NotImplementedError("TODO")

    def month_interval_between(self, end):
        raise NotImplementedError("TODO")

    def nanoseconds_between(self, end):
        return self.accessor.to_output(
            pc.nanoseconds_between(self.accessor.arrow, end.ak.arrow),
        )

    def quarters_between(self, end):
        raise NotImplementedError("TODO")

    def seconds_between(self, end):
        return self.accessor.to_output(
            pc.seconds_between(self.accessor.arrow, end.ak.arrow)
        )

    def weeks_between(
        self,
        end,
        /,
        *,
        count_from_zero=True,
        week_start=1,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def years_between(self, end):
        return self.accessor.to_output(
            pc.years_between(self.accessor.arrow, end.ak.arrow)
        )


def _to_arrow(array):
    array = _make_unit_compatible(array)
    return ak.to_arrow(array, extensionarray=False)


def _make_unit_compatible(array):
    # TODO, actually convert units if not compatible
    return array
