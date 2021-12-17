import numpy as np
from numba import jit
from scipy.spatial import ConvexHull


@jit(nopython=True)
def vector_angle(u, v=np.array([0., 0., 1.])):
    """
    Returns the angle in degrees between vectors 'u' and 'v'. If only 'u' is
    provided, the angle between 'u' and the vertical axis is returned.
    """
    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))
    clip = np.minimum(1, np.maximum(c, -1))
    return np.rad2deg(np.arccos(clip))


@jit(nopython=True)
def get_octree_level(points, grid_size):
    """Compute nearest octree level based on a desired grid_size."""
    dims = np.zeros((points.shape[1],))
    for d in range(points.shape[1]):
        dims[d] = np.max(points[:, d]) - np.min(points[:, d])
    max_dim = np.max(dims)
    if max_dim < 0.001:
        return 0
    octree_level = np.rint(-np.log(grid_size / max_dim) / (np.log(2)))
    if octree_level > 0:
        return np.int64(octree_level)
    return 1


@jit(nopython=True)
def compute_bounding_box(points):
    """
    Get the min/max values of a point list.

    Parameters
    ----------
    points : list
        List of x and y points that belong to a polygon

    Returns
    -------
    list
        Bounding box with outer points of a polygon
    """
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    return min_x, max_y, max_x, min_y


def convex_hull_poly(points):
    """Return convex hull as a closed ring."""
    hull = points[ConvexHull(points).vertices]
    return np.vstack((hull, hull[0]))


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    # Calculate center point and project onto rotated frame
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = np.dot([center_x, center_y], r)

    min_bounding_rect = np.zeros((4, 2))
    min_bounding_rect[0] = np.dot([x1, y2], r)
    min_bounding_rect[1] = np.dot([x2, y2], r)
    min_bounding_rect[2] = np.dot([x2, y1], r)
    min_bounding_rect[3] = np.dot([x1, y1], r)

    # Compute the dims of the min bounding rectangle
    dims = [(x1 - x2), (y1 - y2)]

    return min_bounding_rect, hull_points, min(dims), max(dims), center_point
