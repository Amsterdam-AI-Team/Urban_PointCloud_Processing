"""This module provides utility methods for BGT data."""

import ast
import os
import logging
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..utils.las_utils import get_bbox_from_tile_code

logger = logging.getLogger(__name__)


class BGTReader(ABC):
    """
    Abstract class for reading BGT data, either points or polygons.

    Parameters
    ----------
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    file_prefix : str (default: '')
        Prefix used to load the correct files; only used with bgt_folder.
    """
    @property
    @classmethod
    @abstractmethod
    def COLUMNS(cls):
        return NotImplementedError

    def __init__(self, bgt_file=None, bgt_folder=None, file_prefix=''):
        if (bgt_file is None) and (bgt_folder is None):
            print("Provide either a bgt_file or bgt_folder to load.")
            return None
        if (bgt_file is not None) and (bgt_folder is not None):
            print("Provide either a bgt_file or bgt_folder to load, not both")
            return None
        if (bgt_folder is not None) and (not os.path.isdir(bgt_folder)):
            print('The data folder specified does not exist')
            return None
        if (bgt_file is not None) and (not os.path.isfile(bgt_file)):
            print('The data file specified does not exist')
            return None

        super().__init__()
        self.file_prefix = file_prefix
        self.bgt_df = pd.DataFrame(columns=type(self).COLUMNS)

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            logger.error('No data folder or file specified. Aborting...')
            return None

    def _read_folder(self, path):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        file_match = self.file_prefix + '*.csv'
        frames = [pd.read_csv(file, header=0, names=type(self).COLUMNS)
                  for file in path.glob(file_match)]
        if len(frames) == 0:
            logger.error(f'No data files found in {path.as_posix()}.')
            return
        self.bgt_df = pd.concat(frames)

    def _read_file(self, path):
        """
        Read the contents of a file. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        self.bgt_df = pd.read_csv(path, header=0, names=type(self).COLUMNS)

    @abstractmethod
    def filter_tile(self, tilecode):
        """
        Returns data for the area represented by the given CycloMedia
        tile-code.
        """
        return NotImplementedError


class BGTPointReader(BGTReader):
    """
    Data files are assumed to be in CSV format and contain three columns:
    [bgt_type, x, y].
    """

    COLUMNS = ['bgt_type', 'x', 'y']

    def filter_tile(self, tilecode, bgt_types=[], exclude_types=[],
                    padding=0, return_types=False):
        """
        Return a list of points representing each of the objects found in the
        area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode, padding=padding)
        type_str = ''
        if len(bgt_types) >= 1:
            type_str += '(bgt_type == @bgt_types) & '
        if len(exclude_types) >= 1:
            type_str += '(bgt_type != @exclude_types) & '
        df = self.bgt_df.query(type_str +
                               '(x <= @bx_max) & (x >= @bx_min)' +
                               ' & (y <= @by_max) & (y >= @by_min)')
        bgt_points = list(df.to_records(index=False))

        if return_types:
            return bgt_points
        else:
            return [(x, y) for (_, x, y) in bgt_points]


class BGTPolyReader(BGTReader):
    """
    Data files are assumed to be in CSV format and contain six columns:
    [bgt_type, polygon, x_min, y_max, x_max, y_min].
    """

    COLUMNS = ['bgt_type', 'polygon', 'x_min', 'y_max', 'x_max', 'y_min']

    def filter_tile(self, tilecode, bgt_types=[], exclude_types=[],
                    padding=0, offset=0, merge=False):
        """
        Return a list of polygons found in the area represented by the given
        CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) =\
            get_bbox_from_tile_code(tilecode, padding=padding)
        type_str = ''
        if len(bgt_types) >= 1:
            type_str += '(bgt_type == @bgt_types) & '
        if len(exclude_types) >= 1:
            type_str += '(bgt_type != @exclude_types) & '
        df = self.bgt_df.query(type_str +
                               '(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        polygons = [ast.literal_eval(poly) for poly in df.polygon.values]
        if len(polygons) > 1 and merge:
            union = unary_union([Polygon(poly).buffer(offset)
                                for poly in polygons])
            if type(union) == Polygon:
                poly_offset = [union]
            else:
                poly_offset = list(union.geoms)
        else:
            poly_offset = [Polygon(poly).buffer(offset)
                           for poly in polygons]
        poly_valid = [poly.exterior.coords for poly in poly_offset
                      if len(poly.exterior.coords) > 1]
        return poly_valid


def get_polygons(bgt_file, tilecode, padding=0, offset=0, merge=False):
    """Get the polygons from a bgt_file for a specific tilecode."""
    return BGTPolyReader(bgt_file=bgt_file).filter_tile(
                        tilecode, padding=padding, offset=offset, merge=merge)


def get_points(bgt_file, tilecode, padding=0, return_types=True):
    """Get the bgt point objects from a bgt_file for a specific tilecode."""
    return BGTPointReader(bgt_file=bgt_file).filter_tile(
                        tilecode, padding=padding, return_types=return_types)
