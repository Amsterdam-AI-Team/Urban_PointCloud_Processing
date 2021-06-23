import numpy as np
from numba import vectorize, bool_, float64
from shapely.geometry import Polygon
import pyclipper

# Some modifications are performed on the function square_clip, ray_trace and
# poly_clip, copied from:
# https://github.com/brycefrank/pyfor/blob/master/pyfor/clip.py


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


def ray_trace(x, y, poly):
    """
    Determines for some set of x and y coordinates, which of those coordinates
    is within `poly`. Ray trace is generally called as an internal function,
    see :func:`.poly_clip`
    :param x: A 1D numpy array of x coordinates.
    :param y: A 1D numpy array of y coordinates.
    :param poly: The coordinates of a polygon as a numpy array (i.e. from
    geo_json['coordinates']
    :return: A 1D boolean numpy array, true values are those points that are
    within `poly`.
    """
    @vectorize([bool_(float64, float64)])
    def ray(x, y):
        # where xy is a coordinate
        n = len(poly)
        inside = False
        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    return ray(x, y)


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

    full_clip_mask = ray_trace(xy_points['x'][pre_clip_inds],
                               xy_points['y'][pre_clip_inds], poly_coords)
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
