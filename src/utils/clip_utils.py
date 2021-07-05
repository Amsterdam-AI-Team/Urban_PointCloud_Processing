"""
Clipping tools for point clouds and polygons.

The methods square_clip() and poly_clip() are taken from:
https://github.com/brycefrank/pyfor/blob/master/pyfor/clip.py

The method _point_inside_poly is adapted from:
https://github.com/sasamil/PointInPolygon_Py
"""
import numpy as np
from numba import jit
import numba
from shapely.geometry import Polygon
import pyclipper


def square_clip(xy_points, bounds):
    """
    Clips a square from a tuple describing the position of the square.
    :param points: A N x 2 numpy array of x and y coordinates, x is column 0
    :param bounds: A tuple of length 4, min and max y coordinates of the square
    :return: a list of indices of points within the square.
    """
    inds = np.where((xy_points['x'] >= bounds[0]) & (xy_points['x'] <=
                    bounds[2]) & (xy_points['y'] >= bounds[1]) &
                    (xy_points['y'] <= bounds[3]))[0]

    return inds


@jit(nopython=True)
def circle_clip(points, center, radius):
    """
    Clip all points within a circle (or unbounded cylinder).

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    center : tuple (x, y)
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


@jit(nopython=True)
def cylinder_clip(points, center, radius, bottom=-np.inf, top=np.inf):
    """
    Clip all points within a cylinder.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    center : tuple (x, y)
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


@jit(nopython=True)
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


@jit(nopython=True)
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
    # Can be parallelized by replacing this line with <for i in
    # numba.prange(ln):> and decorating the function with
    # <@njit(parallel=True)>
    for i in range(n):
        mask[i] = _point_inside_poly(polygon, (x[i], y[i]))
    return mask


def poly_clip(xy_points, poly):
    """
    Returns the indices of `points` that are within a given polygon. This
    differs from :func:`.ray_trace` in that it enforces a small "pre-clip"
    optimization by first clipping to the polygon bounding box. This function
    is directly called by :meth:`.Cloud.clip`.
    :param cloud: A cloud object.
    :param poly: A polygon list, with coordinates in the same CRS as the pc.
    :return: A 1D numpy array of indices corresponding to points within the
    given polygon.
    """
    shapely_poly = Polygon(poly)

    # Clip to bounding box
    bbox = shapely_poly.bounds
    pre_clip_inds = square_clip(xy_points, bbox)

    # Clip the preclip
    poly_coords = np.stack((shapely_poly.exterior.coords.xy[0],
                            shapely_poly.exterior.coords.xy[1]), axis=1)
    full_clip_mask = is_inside(xy_points['x'][pre_clip_inds],
                               xy_points['y'][pre_clip_inds],
                               poly_coords)
    clipped = pre_clip_inds[full_clip_mask]

    return clipped


def poly_offset(polygon, offset_meter):
    """
    Calculate coordinates of radius/offset around a polygon.

    Parameters
    ----------
    polygon : list
        One polygon in list format
    offset_meter : int
        Offset in meters (deflation, minus value, has bugs)

    Returns
    -------
    list
        Inflated polygon in a 1D list
    """
    clipper_offset = pyclipper.PyclipperOffset()
    coordinates_scaled = pyclipper.scale_to_clipper(polygon)

    clipper_offset.AddPath(coordinates_scaled, pyclipper.JT_SQUARE,
                           pyclipper.ET_CLOSEDPOLYGON)

    new_coordinates = clipper_offset.Execute(pyclipper
                                             .scale_to_clipper(offset_meter))

    polygon_offset = pyclipper.scale_from_clipper(new_coordinates)[0]

    return polygon_offset
