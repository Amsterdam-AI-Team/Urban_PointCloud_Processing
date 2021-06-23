import numpy as np


def _unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(rad)


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
    x_coord, y_coord = zip(*points)

    return min(x_coord), max(y_coord), max(x_coord), min(y_coord)
