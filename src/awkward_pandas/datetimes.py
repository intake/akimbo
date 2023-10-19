from __future__ import annotations

from typing import TYPE_CHECKING

import awkward as ak
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from awkward_pandas.array import AwkwardExtensionArray

if TYPE_CHECKING:
    from awkward_pandas.accessor import AwkwardAccessor


class DatetimeAccessor:
    def __init__(self, accessor: AwkwardAccessor) -> None:
        self.accessor = accessor

    def cast(self, target_type=None, safe=None, options=None, memory_pool=None):
        raise NotImplementedError("TODO")

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
        memory_pool=None,
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
        memory_pool=None,
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
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def run_end_decode(self, array, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def run_end_encode(
        self,
        /,
        run_end_type=pa.int32(),
        *,
        options=None,
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def strftime(
        self,
        /,
        format="%Y-%m-%dT%H:%M:%S",
        locale="C",
        *,
        options=None,
        memory_pool=None,
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
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def day(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def day_of_week(
        self,
        /,
        *,
        count_from_zero=True,
        week_start=1,
        options=None,
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def day_of_year(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def hour(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def iso_week(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def iso_year(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def iso_calendar(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def is_leap_year(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def microsecond(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def millisecond(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def minute(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def month(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def nanosecond(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def quarter(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def second(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def subsecond(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def us_week(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def us_year(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def week(
        self,
        /,
        *,
        week_starts_monday=True,
        count_from_zero=False,
        first_week_is_fully_in_year=False,
        options=None,
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def year(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def year_month_day(self, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def day_time_interval_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def days_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def hours_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def microseconds_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def milliseconds_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def minutes_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def month_day_nano_interval_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def month_interval_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def nanoseconds_between(self, end, /, *, memory_pool=None):
        arr, args, kwargs = _arrowize(self, end, memory_pool=memory_pool)
        return _as_series(pc.nanoseconds_between(arr, *args, **kwargs))

    def quarters_between(self, end, /, *, memory_pool=None):
        raise NotImplementedError("TODO")

    def seconds_between(self, end, /, *, memory_pool=None):
        arr, args, kwargs = _arrowize(self, end, memory_pool=memory_pool)
        return _as_series(pc.seconds_between(arr, *args, **kwargs))

    def weeks_between(
        self,
        end,
        /,
        *,
        count_from_zero=True,
        week_start=1,
        options=None,
        memory_pool=None,
    ):
        raise NotImplementedError("TODO")

    def years_between(self, end, /, *, memory_pool=None):
        arr, args, kwargs = _arrowize(self, end, memory_pool=memory_pool)
        return _as_series(pc.years_between(arr, *args, **kwargs))

    # def __getattr__(self, attr: str) -> Callable:
    #     if attr not in dir(self):
    #         raise AttributeError

    #     fn = getattr(pc, attr, None)

    #     if fn:

    #         @functools.wraps(fn)
    #         def wrapper(*args, **kwargs):
    #             try:
    #                 arrow_array = ak.to_arrow(self.accessor.array, extensionarray=False)
    #             except ArrowNotImplementedError("TODO") as err:
    #                 msg = (
    #                     "Could not convert data to arrow\n"
    #                     "Arrow requires datetime with units: "
    #                     "seconds, milliseconds, microseconds, nanoseconds"
    #                 )
    #                 raise ArrowNotImplementedError(msg)

    #             # parse args so that other series backed by awkward are
    #             # converted to arrow array objects.
    #             new_args = []
    #             for arg in args:
    #                 if isinstance(arg, pd.Series) and arg.dtype == "awkward":
    #                     new_args.append(ak.to_arrow(arg.ak.array, extensionarray=False))
    #                 else:
    #                     new_args.append(arg)

    #             # parse kwargs so that other series backed by awkward are
    #             # converted to arrow array objects.
    #             new_kwargs = {}
    #             for k, v in kwargs.items():
    #                 if isinstance(v, pd.Series) and v.dtype == "awkward":
    #                     new_kwargs[k] = ak.to_arrow(v.ak.array, extensionarray=False)
    #                 else:
    #                     new_kwargs[k] = v

    #             result = fn(arrow_array, *new_args, **new_kwargs)
    #             idx = self.accessor._obj.index
    #             return pd.Series(
    #                 AwkwardExtensionArray(ak.from_arrow(result)), index=idx
    #             )

    #     else:
    #         raise AttributeError

    #     return wrapper


def _to_arrow(array):
    array = _make_unit_compatible(array)
    return ak.to_arrow(array, extensionarray=False)


def _make_unit_compatible(array):
    # TODO, actually convert units if not compatible
    return array


def _arrowize(dta, *args, **kwargs):
    """Convert objects to arrow arrays.

    Parameters
    ----------
    dta : DatetimeAccessor
        The DatetimeAccessor with information about the main Series
        object that is of dtype :obj:`~awkward_pandas.AwkwardDtype`.
    *args : Any
        Arguments that should be converted to arrow if necessary. Any
        arguments that are Series backed by the
        :obj:`~awkward_pandas.AwkwardDtype` will have the underlying
        awkward array converted to an arrow array.
    **kwargs : Any
        Keyword arguments that should be converted to arrow if
        necessary. Any values that are Series backed by the
        :obj:`~awkward_pandas.AwkwardDtype` will have the underlying
        awkward array converted to an arrow array.

    Returns
    -------
    Array
        Primary awkward Series converted to arrow.
    tuple
        New arguments with necessary conversions.
    dict
        New keyword arguments with necessary conversions.

    """
    primary_as_arrow = _to_arrow(dta.accessor.array)

    # parse args so that other series backed by awkward are
    # converted to arrow array objects.
    new_args = []
    for arg in args:
        if isinstance(arg, pd.Series) and arg.dtype == "awkward":
            new_args.append(_to_arrow(arg.ak.array))
        else:
            new_args.append(arg)

    # parse kwargs so that other series backed by awkward are
    # converted to arrow array objects.
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, pd.Series) and v.dtype == "awkward":
            new_kwargs[k] = _to_arrow(v.ak.array)
        else:
            new_kwargs[k] = v

    return primary_as_arrow, new_args, new_kwargs


def _as_series(pyarrow_result):
    """Convert pyarrow Array back in to awkward Series.

    Parameters
    ----------
    pyarrow_result : pyarrow Array
        PyArray array that was the result of a pyarrow.compute call.

    Examples
    --------
    pd.Series
        Series of type :obj:`~awkward_pandas.AwkwardDtype`.

    """
    return pd.Series(AwkwardExtensionArray(ak.from_arrow(pyarrow_result)))
