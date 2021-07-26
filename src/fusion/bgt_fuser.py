"""BGT Data Fuser"""

import ast
import pandas as pd
from pathlib import Path
import os
import numpy as np
from abc import ABC, abstractmethod

from .abstract import AbstractFuser
from ..utils.clip_utils import poly_offset, poly_clip
from ..utils.las_utils import get_bbox_from_tile_code


class BGTFuser(AbstractFuser, ABC):
    """
    Abstract class for automatic labelling of points using BGT data.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
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

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix=''):
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

        super().__init__(label)
        self.file_prefix = file_prefix
        self.bgt_df = pd.DataFrame(columns=type(self).COLUMNS)

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            print('No data folder or file specified. Aborting...')
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
            print(f'No data files found in {path.as_posix()}.')
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
    def _filter_tile(self, tilecode):
        """
        Returns data for the area represented by the given CycloMedia
        tile-code.
        """
        return NotImplementedError


class BGTBuildingFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of building points using BGT data
    in the form of footprint polygons. Data files are assumed to be in CSV
    format and contain six columns: [ID, polygon, x_min, y_max, x_max, y_min].

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Data files are
        assumed to be prefixed by "bgt_buildings", unless otherwise specified.
        Either a file or a folder should be provided, but not both.
    file_prefix : str (default: 'bgt_building')
        Prefix used to load the correct files; only used with bgt_folder.
    building_offset : int (default: 0)
        The footprint polygon will be extended by this amount (in meters).
    """

    COLUMNS = ['BAG_ID', 'Polygon', 'x_min', 'y_max', 'x_max', 'y_min']

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix='bgt_buildings', building_offset=0,):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.building_offset = building_offset

        # TODO will this speedup the process?
        self.bgt_df.sort_values(by=['x_max', 'y_min'], inplace=True)

    def _filter_tile(self, tilecode):
        """
        Return a list of polygons representing each of the buildings found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode)
        df = self.bgt_df.query('(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        building_polygons = [ast.literal_eval(poly)
                             for poly in df.Polygon.values]

        return building_polygons

    def get_label_mask(self, tilecode, points, mask):
        """
        Returns the building mask for the given pointcloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        las_labels : array of shape (n_points, 1)
            All labels as int values

        Returns
        -------
        An array of shape (n_points,) with indices indicating which points
        should be labelled according to this fuser.
        """
        building_polygons = self._filter_tile(tilecode)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)

        building_mask = np.zeros((np.count_nonzero(mask),), dtype=bool)
        for polygon in building_polygons:
            # TODO if there are multiple buildings we could mask the points
            # iteratively to ignore points already labelled.
            building_with_offset = poly_offset(polygon, self.building_offset)
            building_points = poly_clip(points[mask, :], building_with_offset)
            building_mask = building_mask | building_points

        mask_indices = np.where(mask)[0]
        label_mask = np.zeros(len(points), dtype=bool)
        label_mask[mask_indices[building_mask]] = True

        print(f'BGT building fuser => processed (label={self.label}).')

        return label_mask


class BGTPointFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of point objects such as trees,
    street lights, and traffic signs using BGT data. Data files are assumed to
    be in CSV format and contain three columns: [Object type, X, Y].

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Data files are
        assumed to be prefixed by "bgt_points", unless otherwise specified.
        Either a file or a folder should be provided, but not both.
    file_prefix : str (default: 'bgt_points')
        Prefix used to load the correct files; only used with bgt_folder.
    """
    COLUMNS = ['Type', 'X', 'Y']

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix='bgt_points'):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)

        # TODO will this speedup the process?
        self.bgt_df.sort_values(by=['X', 'Y'], ignore_index=True, inplace=True)

    def _filter_tile(self, tilecode):
        """
        Return a list of polygons representing each of the buildings found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode)
        df = self.bgt_df.query('(X <= @bx_max) & (X >= @bx_min)' +
                               ' & (Y <= @by_max) & (Y >= @by_min)')
        bgt_points = list(df.to_records(index=False))

        return bgt_points

    def get_label_mask(self, tilecode, points, mask):
        """
        Returns the seed-point mask for the given pointcloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.

        Returns
        -------
        An array of shape (n_points,) with indices indicating which points
        should be labelled according to this fuser.
        """
        label_mask = np.zeros((len(points),), dtype=bool)

        # bgt_points = self._filter_tile(tilecode)

        # TODO
        # How to differentiate between different types of point objects?

        return label_mask
