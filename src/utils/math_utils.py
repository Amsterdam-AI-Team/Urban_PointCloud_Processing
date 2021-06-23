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
