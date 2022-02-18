# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""
Clipping tools for point clouds and polygons.

The method poly_clip() is adapted from:
https://github.com/brycefrank/pyfor/blob/master/pyfor/clip.py

The method _point_inside_poly is adapted from:
https://github.com/sasamil/PointInPolygon_Py
"""
import numpy as np
from numba import jit
import numba
import logging

from ..utils import math_utils

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def rectangle_clip(points, rect):
    """
    Clip all points within a rectangle.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    rect : tuple of floats
        (x_min, y_min, x_max, y_max)

    Returns
    -------
    A boolean mask with True entries for all points within the rectangle.
    """
    clip_mask = ((points[:, 0] >= rect[0]) & (points[:, 0] <= rect[2])
                 & (points[:, 1] >= rect[1]) & (points[:, 1] <= rect[3]))
    return clip_mask


@jit(nopython=True, cache=True)
def box_clip(points, rect, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a 3D box.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    rect : tuple of floats
        (x_min, y_min, x_max, y_max)
    bottom : float (default: -inf)
        Bottom of the box.
    top : float (default: inf)
        Top of the box.

    Returns
    -------
    A boolean mask with True entries for all points within the 3D box.
    """
    box_mask = rectangle_clip(points, rect)
    box_mask = box_mask & ((points[:, 2] <= top) & (points[:, 2] >= bottom))
    return box_mask


@jit(nopython=True, cache=True)
def circle_clip(points, center, radius):
    """
    Clip all points within a circle (or unbounded cylinder).

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    center : tuple of floats (x, y)
        Center point of the circle.
    radius : float
        Radius of the circle.

    Returns
    -------
    A boolean mask with True entries for all points within the circle.
    """
    clip_mask = (np.power((points[:, 0] - center[0]), 2)
                 + np.power((points[:, 1] - center[1]), 2)
                 <= np.power(radius, 2))
    return clip_mask


@jit(nopython=True, cache=True)
def cylinder_clip(points, center, radius, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a cylinder.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    center : tuple of floats (x, y)
        Center point of the circle.
    radius : float
        Radius of the circle.
    bottom : float (default: -inf)
        Bottom of the cylinder.
    top : float (default: inf)
        Top of the cylinder.

    Returns
    -------
    A boolean mask with True entries for all points within the circle.
    """
    clip_mask = circle_clip(points, center, radius)
    clip_mask = clip_mask & ((points[:, 2] <= top) & (points[:, 2] >= bottom))
    return clip_mask


@jit(nopython=True, cache=True)
def _point_inside_poly(polygon, point):
    """
    Improved version of the Crossing Number algorithm that checks if a point is
    inside a polygon.
    Implementation taken from https://github.com/sasamil/PointInPolygon_Py
    """
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/below/right from
        # the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0]
                              or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = (dy * (polygon[jj][0] - polygon[ii][0])
                     / (dy-dy2) + polygon[ii][0])

                if point[0] > F:
                    # if line is left from the point - the ray moving towards
                    # left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and
            # dx*dx2<=0)
            elif (dy2 == 0
                  and (point[0] == polygon[jj][0]
                       or (dy == 0 and (point[0] - polygon[ii][0])
                           * (point[0] - polygon[jj][0]) <= 0))):
                return 2

        ii = jj
        jj += 1

    return intersections & 1


@jit(nopython=True, cache=True, parallel=True)
def is_inside(x, y, polygon):
    """
    Checks for each point in a list whether that point is inside a polygon.

    Parameters
    ----------
    x : list
        X-coordinates.
    y : list
        Y-coordinates.
    polygon : list of tuples
        Polygon as linear ring.

    Returns
    -------
    An array of shape (len(x),) with dtype bool, where each entry indicates
    whether the corresponding point is inside the polygon.
    """
    n = len(x)
    mask = np.empty((n,), dtype=numba.boolean)
    for i in numba.prange(n):
        mask[i] = _point_inside_poly(polygon, (x[i], y[i]))
    return mask


def poly_clip(points, poly):
    """
    Clip all points within a polygon.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    poly : shapely.geometry Polygon object
        Polygon to clip. Can have interior gaps.

    Returns
    -------
    A boolean mask with True entries for all points within the polygon.
    """
    # Convert to numpy to work with numba jit in nopython mode.
    exterior = np.array(poly.exterior.coords)
    interiors = [np.array(interior.coords) for interior in poly.interiors]

    clip_mask = np.zeros((len(points),), dtype=bool)

    # Clip exterior to include points.
    bbox_mask = rectangle_clip(
                    points, math_utils.compute_bounding_box(exterior))
    exterior_mask = is_inside(points[bbox_mask, 0], points[bbox_mask, 1],
                              exterior)
    bbox_inds = np.where(bbox_mask)[0]
    clip_mask[bbox_inds[exterior_mask]] = True

    # Clip interior(s) to exclude points.
    for interior in interiors:
        bbox_mask = rectangle_clip(
                        points, math_utils.compute_bounding_box(interior))
        interior_mask = is_inside(points[bbox_mask, 0], points[bbox_mask, 1],
                                  interior)
        bbox_inds = np.where(bbox_mask)[0]
        clip_mask[bbox_inds[interior_mask]] = False

    return clip_mask


def poly_box_clip(points, poly, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a 3D polygon with fixed height.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    poly : shapely.geometry Polygon object
        Polygon to clip. Can have interior gaps.
    bottom : float (default: -inf)
        Bottom height of the 3D polygon.
    top : float (default: inf)
        Top height of the 3D polygon.

    Returns
    -------
    A boolean mask with True entries for all points within the 3D polygon.
    """
    clip_mask = poly_clip(points, poly)
    clip_mask = clip_mask & ((points[:, 2] <= top) & (points[:, 2] >= bottom))
    return clip_mask
