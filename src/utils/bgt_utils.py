"""This module provides utility methods for BGT data."""

import ast
import pandas as pd

from ..utils.las_utils import get_bbox_from_tile_code


def get_polygons(bgt_file, tilecode):
    """Get the polygons from a bgt_file for a specific tilecode."""
    columns = ['BAG_ID', 'Polygon', 'x_min', 'y_max', 'x_max', 'y_min']
    ((bx_min, by_max), (bx_max, by_min)) = get_bbox_from_tile_code(tilecode)
    df = (pd.read_csv(bgt_file, header=0, names=columns)
          .query('(x_min < @bx_max) & (x_max > @bx_min)' +
                 ' & (y_min < @by_max) & (y_max > @by_min)'))
    polygons = [ast.literal_eval(poly) for poly in df.Polygon.values]
    return polygons


def get_points(bgt_file, tilecode, padding=0):
    """Get the bgt point objects from a bgt_file for a specific tilecode."""
    columns = ['Type', 'X', 'Y']
    ((bx_min, by_max), (bx_max, by_min)) = get_bbox_from_tile_code(
                                                tilecode, padding=padding)
    df = (pd.read_csv(bgt_file, header=0, names=columns)
          .query('(X < @bx_max) & (X > @bx_min)' +
                 ' & (Y < @by_max) & (Y > @by_min)'))
    points = [[row.Type, row.X, row.Y] for index, row in df.iterrows()]
    return points
