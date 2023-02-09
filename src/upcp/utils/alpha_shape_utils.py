# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""This module provides methods to generate a concave hull (alpha shape)."""

import numpy as np
import shapely.geometry as sg
from scipy.spatial import Delaunay


def alpha_shape(points, alpha=1.):
    """
    Return the concave polygon (alpha shape) of a set of points. Depending on
    the value of alpha, multiple polygons may be returned, and individual
    polygons may contain holes (interior rings).

    Parameters
    ----------
    points : np.array of shape (n,2)
        Points of which concave hull should be returned.
    alpha : float (default: 1.0)
        Alpha value, determines "concaveness". A value of 0 equals the convex
        hull.

        The derivation is as follows: to generate the concave hull,
        "circumcircles" are fitted to the points by triangulation. Whenever the
        radius of such a circle is larger than 1/alpha, a whole is generated.

    Returns
    -------
    A list of shapely.Polygon objects representing the concave hull.
    """
    edges = get_alpha_shape_edges(points, alpha=alpha, only_outer=True)
    concave_polys = generate_poly_from_edges(edges, points)
    return concave_polys


# https://stackoverflow.com/a/50159452
# CC BY-SA 4.0
def get_alpha_shape_edges(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer
    border or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j)
    are the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, (
                "Can't go twice over same directed edge right?")
            if only_outer:
                # if both neighboring triangles are in shape, it is not a
                # boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        int_var = s * (s - a) * (s - b) * (s - c)
        if int_var <= 0:
            # TODO something more clever?
            continue
        area = np.sqrt(int_var)
        circum_r = a * b * c / (4.0 * area)
        if circum_r < (1 / alpha):
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


# https://stackoverflow.com/a/50714300
# CC BY-SA 4.0
def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


# https://stackoverflow.com/a/50714300
# CC BY-SA 4.0
def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        # boundary_lst.append(boundary)
        boundary_lst.extend(split_loops(boundary))
    return boundary_lst


# Own work
def split_loops(boundary):
    pts = [j for i, j in boundary]
    u, c = np.unique(pts, return_counts=True)
    dups = u[c == 2]  # TODO: edge cases
    if len(dups) == 0:
        return [boundary]
    loops = []
    for dup in dups:
        locs = np.where(pts == dup)[0]
        if len(locs) != 2:
            continue  # TODO: loops within loops
        loop = pts[locs[0]:locs[1]]
        loop.append(dup)
        # The loop may potentially have loops itself. Add some recursion.
        loops.append([(loop[i], loop[i+1]) for i in range(len(loop)-1)])
        new_pts = pts[:locs[0]]
        new_pts.extend(pts[locs[1]:])
        pts = new_pts
    pts.append(pts[0])
    boundaries = [[(pts[i], pts[i+1]) for i in range(len(pts)-1)]]
    boundaries.extend(loops)
    return boundaries


# Own work
def boundary_to_poly(boundary, points):
    xs = []
    ys = []
    for i, j in boundary:
        xs.append(points[i, 0])
        ys.append(points[i, 1])
    xs.append(points[j, 0])
    ys.append(points[j, 1])
    poly = sg.Polygon([[x, y] for x, y in zip(xs, ys)])
    if not poly.is_valid:
        # TODO: temp fix for any left-over self-intersections
        poly = poly.buffer(0)
    return poly


# Own work
def generate_poly_from_edges(edges, points):
    def get_poly_with_hole(polys):
        biggest = np.argmax([p.area for p in polys])
        outer = polys.pop(biggest)
        inners = []
        for idx, poly in enumerate(polys):
            if outer.contains(poly):
                inners.append(idx)
                outer = outer - poly
        for index in sorted(inners, reverse=True):
            del polys[index]
        if type(outer) == sg.MultiPolygon:
            return outer.geoms
        else:
            return [outer]

    boundary_lst = stitch_boundaries(edges)
    polys = [boundary_to_poly(b, points) for b in boundary_lst
             if len(b) >= 3]
    outers = []
    while len(polys) > 0:
        outers.extend(get_poly_with_hole(polys))
    return outers
