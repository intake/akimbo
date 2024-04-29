from __future__ import annotations

import awkward as ak
import pyarrow as pa
import pyarrow.compute as pc


def _run_unary(layout, op, kind=None, **kw):
    if layout.is_record:
        [_run_unary(_, op, kind=kind, **kw) for _ in layout._contents]
    elif layout.is_leaf and (kind is None or layout.dtype.kind == kind):
        layout._data = ak.str._apply_through_arrow(op, layout, **kw).data
    elif layout.is_option or layout.is_list:
        _run_unary(layout.content, op, kind=kind, **kw)


def run_unary(arr: ak.Array, op, kind=None, **kw) -> ak.Array:
    arr2 = ak.copy(arr)
    _run_unary(arr2.layout, op, kind=kind, **kw)
    return ak.Array(arr2)


class DatetimeAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    def cast(self, target_type=None, safe=None, options=None):
        """Cast values to given type

        This may be the easiest way to make time types from scratch

        Examples
        --------

        >>> import pandas as pd
        >>> import awkward_pandas.pandas
        >>> s = pd.Series([[0, 1], [1, 0], [2]])
        >>> s.ak.dt.cast("timestamp[s]")
        0    ['1970-01-01T00:00:00' '1970-01-01T00:00:01']
        1    ['1970-01-01T00:00:01' '1970-01-01T00:00:00']
        2                          ['1970-01-01T00:00:02']
        dtype: list<item: timestamp[s]>[pyarrow]
        """
        return self.accessor.to_output(
            run_unary(
                self.accessor.array,
                pc.cast,
                target_type=target_type,
                safe=safe,
                options=options,
            )
        )

    def ceil_temporal(
        self,
        /,
        multiple=1,
        unit="day",
        *,
        week_starts_monday=True,
        ceil_is_strictly_greater=False,
        calendar_based_origin=False,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def floor_temporal(
        self,
        /,
        multiple=1,
        unit="day",
        *,
        week_starts_monday=True,
        ceil_is_strictly_greater=False,
        calendar_based_origin=False,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def round_temporal(
        self,
        /,
        multiple=1,
        unit="day",
        *,
        week_starts_monday=True,
        ceil_is_strictly_greater=False,
        calendar_based_origin=False,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def run_end_decode(self, array):
        raise NotImplementedError("TODO")

    def run_end_encode(
        self,
        /,
        run_end_type=pa.int32(),
        *,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def strftime(
        self,
        /,
        format="%Y-%m-%dT%H:%M:%S",
        locale="C",
        *,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def strptime(
        self,
        /,
        format,
        unit,
        error_is_null=False,
        *,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def day(self):
        raise NotImplementedError("TODO")

    def day_of_week(
        self,
        /,
        *,
        count_from_zero=True,
        week_start=1,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def day_of_year(self):
        raise NotImplementedError("TODO")

    def hour(self):
        raise NotImplementedError("TODO")

    def iso_week(self):
        raise NotImplementedError("TODO")

    def iso_year(self):
        raise NotImplementedError("TODO")

    def iso_calendar(self):
        raise NotImplementedError("TODO")

    def is_leap_year(self):
        raise NotImplementedError("TODO")

    def microsecond(self):
        raise NotImplementedError("TODO")

    def millisecond(self):
        raise NotImplementedError("TODO")

    def minute(self):
        raise NotImplementedError("TODO")

    def month(self):
        raise NotImplementedError("TODO")

    def nanosecond(self):
        raise NotImplementedError("TODO")

    def quarter(self):
        raise NotImplementedError("TODO")

    def second(self):
        raise NotImplementedError("TODO")

    def subsecond(self):
        raise NotImplementedError("TODO")

    def us_week(self):
        raise NotImplementedError("TODO")

    def us_year(self):
        raise NotImplementedError("TODO")

    def week(
        self,
        /,
        *,
        week_starts_monday=True,
        count_from_zero=False,
        first_week_is_fully_in_year=False,
        options=None,
    ):
        raise NotImplementedError("TODO")

    def year(self):
        raise NotImplementedError("TODO")

    def year_month_day(self):
        raise NotImplementedError("TODO")

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
