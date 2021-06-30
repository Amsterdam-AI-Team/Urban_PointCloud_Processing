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

    return ((min(x_coord), max(y_coord)), (max(x_coord), min(y_coord)))


def poly_list_from_bbox(bbox):
    """Convert a bbox to a list of coordinates for PDOK queries."""
    ((x_min, y_max), (x_max, y_min)) = bbox
    poly_list = [[x_min, y_min], [x_min, y_max], [x_max, y_max],
                 [x_max, y_min], [x_min, y_min]]
    return poly_list


def point_in_bbox(point, bbox):
    """Checks whether a point is inside a bounding box."""
    ((x_min, y_max), (x_max, y_min)) = bbox
    return (x_min <= point[0] <= x_max) and (y_min <= point[1] <= y_max)


def poly_overlaps_bbox(poly_list, bbox):
    """Checks whether a polygon overlaps with a bounding box."""
    ((bx_min, by_max), (bx_max, by_min)) = bbox
    ((x_min, y_max), (x_max, y_min)) = compute_bounding_box(poly_list)
    return ((x_min < bx_max) and (x_max > bx_min)
            and (y_min < by_max) and (y_max > by_min))
