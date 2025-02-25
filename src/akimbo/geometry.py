import numba
import numpy as np

# geoarrow definitions https://geoarrow.org/format.html
# note that if a geometry exists, no elements may be NULL; but
# in the awkward representation there may be (zero-null) optional layers


def cont(layout):
    while layout.is_option:
        # require unmasked?
        layout = layout.content
    return layout


def match_point_separated(layout, x="x", y="y", z="z", m="m"):
    fieldset = [x, y, z, m]
    return layout.is_record and layout.fields == fieldset[: len(layout.fields)]


def match_point_interleaved(layout):
    return layout.is_regular and 2 <= layout.size <= 4


def match_point(layout, **kw):
    return match_point_separated(layout, **kw) or match_point_interleaved(layout)


def match_line_string(layout, **kw):
    # layout.fields == ["vertices"]
    return layout.is_list and match_point(cont(layout), **kw)


def match_polygon(layout, **kw):
    # layout.fields == ["rings"] and layout.content.fields == ["vertices"]
    return (
        layout.is_list
        and cont(layout).is_list
        and match_point(cont(cont(layout)), **kw)
    )


def match_multipoint(layout, **kw):
    # layout.fields == ["points"]
    return layout.is_list and match_point(cont(layout), **kw)


def match_multipolygon(layout, **kw):
    # layout.fields == ["polygons"]
    return layout.is_list and match_polygon(cont(layout))


# TODO: multilinestring (aka. line collection)


@numba.njit(nogil=True, cache=True)
def bounds(coord):
    """
    Aggregation: compute max/min bounds for 1d  array

    If all points are nan, the return is (nan, nan)
    """
    # NB: (np.nanmin(coord), np.nanmax(coord)) is significantly faster,
    # but this should easily be made to work on the GPU
    # how would you apply this to find the bounds of each of lists of strings?
    xmin = np.inf
    xmax = -np.inf

    for x in coord:
        if np.isfinite(x):
            xmin = min(xmin, x)
            xmax = max(xmax, x)

    if not np.isfinite(xmin):
        xmin = xmax = np.nan

    return (xmin, xmax)


def bounding_box(arr):
    if arr.fields:
        return tuple(bounds(arr[c]) for c in arr.fields)  # should be dict?
    return tuple(bounds(arr[c]) for c in arr.fields)


@numba.njit(nogil=True, cache=True)
def _line_length(listx, listy):
    """of one 2d line"""
    total_len = 0.0
    x0 = listx[0]
    y0 = listy[0]

    for x1, y1 in zip(listx[1:], listy[1:]):
        if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
            # we just skip NULL points??
            total_len += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        x0 = x1
        y0 = y1
    return total_len


@numba.njit(nogil=True, cache=True)
def line_lengths(listlistx, listlisty):
    """of each 2d line in an array"""
    out = np.zeros(len(listlistx), dtype="float64")
    for i, (listx, listy) in enumerate(zip(listlistx, listlisty)):
        # this loop could be extracted or parallelised
        out[i] = _line_length(listx, listy)
    return out


@numba.njit(nogil=True, cache=True)
def _area(listx, listy):
    """of one 2d polygon

    CCW is +ve, CW is -ve
    """
    area = 0.0

    for i in range(len(listx) - 2):
        ix = listx[i + 1]
        jy = listy[i + 2]
        ky = listy[i]

        area += ix * (jy - ky)

    # wrap-around term for polygon
    firstx = listx[0]
    secondy = listy[1]
    lasty = listy[-1]
    area += firstx * (secondy - lasty)

    return area / 2.0


@numba.njit(nogil=True, cache=True)
def area(listlistx, listlisty):
    """of each 2d polygon in an array"""
    out = np.empty(len(listlistx), dtype="float64")
    for i in range(len(listlistx)):
        # this can be parallel
        out[i] = _area(listlistx[i], listlisty[i])
    return out


@numba.njit(nogil=True, cache=True)
def triangle_orientation(ax, ay, bx, by, cx, cy):
    """
    Orientation of single triangle

    Args:
        ax, ay: coords of first point
        bx, by: coords of second point
        cx, cy: coords of third point

    Returns:
        +1 if counter clockwise
         0 if colinear
        -1 if clockwise
    """
    ab_x, ab_y = bx - ax, by - ay
    ac_x, ac_y = cx - ax, cy - ay

    # compute cross product: ab x bc
    ab_x_ac = (ab_x * ac_y) - (ab_y * ac_x)

    if ab_x_ac > 0:
        # Counter clockwise
        return 1
    elif ab_x_ac < 0:
        # Clockwise
        return -1
    else:
        # Collinear
        return 0


@numba.njit(nogil=True, cache=True)
def orient_polygons(flatx, flaty, polygon_offsets, ring_offsets):
    """
    Reorient polygons so that exterior is in CCW order and interior rings (holes) CW

    This function mutates the values array.

    Because we do mutation here, input is numpy arrays, no ak, and must be
    extracted from the layout.
    """
    num_rings = len(ring_offsets) - 1

    # Compute expected orientation of rings
    expected_ccw = np.zeros(len(ring_offsets) - 1, dtype=np.bool_)
    expected_ccw[polygon_offsets[:-1]] = True  # outer ring of each set

    # Compute actual orientation of rings
    is_ccw = np.zeros(num_rings)
    for i in range(num_rings):
        is_ccw[i] = (
            _area(
                flatx[ring_offsets[i] : ring_offsets[i + 1]],
                flaty[ring_offsets[i] : ring_offsets[i + 1]],
            )
            >= 0
        )

    # Compute indices of rings to flip
    flip_inds = is_ccw != expected_ccw
    ring_starts = ring_offsets[:-1]
    ring_stops = ring_offsets[1:]
    flip_starts = ring_starts[flip_inds]
    flip_stops = ring_stops[flip_inds]

    for i in range(len(flip_starts)):
        flip_start = flip_starts[i]
        flip_stop = flip_stops[i]

        xs = flatx[flip_start:flip_stop]
        ys = flaty[flip_start:flip_stop]
        flatx[flip_start:flip_stop] = xs[::-1]
        flaty[flip_start:flip_stop] = ys[::-1]
