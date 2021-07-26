"""AHN Data Fuser"""

import numpy as np
import os

from .abstract import AbstractFuser
from ..utils import ahn_utils as ahn_utils
from ..utils.interpolation import FastGridInterpolator


class AHNFuser(AbstractFuser):
    """
    Data Fuser class for automatic labelling of ground and building points
    using AHN data. The class can handle both pre-processed surfaces in .npz
    format (see preprocessing.ahn_preprocessing) as well as GeoTIFF files
    downloaded from PDOK.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    data_folder : str or Path
        Folder containing data files needed for this fuser.
    method : str (default: 'npz')
        Whether to use pre-processed .npz data ('npz') or GeoTIFF files
        ('geotiff').
    target : str (default: 'ground')
        If method='npz', this variable determines whether this fuser will label
        either 'ground' or 'building' points.
    epsilon : float (default: 0.2)
        Precision of the fuser.
    fill_gaps : bool (default: True)
        Whether to fill gaps in the AHN data. Only used when method='geotiff'.
    max_gap_size : int (default: 50)
        Max gap size for gap filling. Only used when method='geotiff'.
    smoothen : bool (default: True)
        Whether to smoothen edges in the AHN data. Only used when
        method='geotiff'.
    smooth_thickness : int (default: 1)
        Thickness for edge smoothening. Only used when method='geotiff'.
    """

    METHODS = ('npz', 'geotiff')
    TARGETS = ('ground', 'building')

    def __init__(self, label, data_folder,
                 method='npz', target='ground', epsilon=0.2,
                 fill_gaps=True, max_gap_size=50,
                 smoothen=True, smooth_thickness=1):
        super().__init__(label)
        if not os.path.isdir(data_folder):
            print('The data folder specified does not exist')
            return None
        if method not in self.METHODS:
            print(f'Method should be one of {self.METHODS}.')
            return None
        if target not in self.TARGETS:
            print(f'Target should be one of {self.TARGETS}.')
            return None
        if method == 'geotiff' and target == 'building':
            print(f'The combination of {method} and {target} is not valid.')
            return None

        self.data_folder = data_folder
        self.method = method
        self.target = target
        self.epsilon = epsilon
        if self.method == 'geotiff':
            self.reader = ahn_utils.GeoTIFFReader(data_folder)
            self.fill_gaps = fill_gaps
            self.max_gap_size = max_gap_size
            self.smoothen = smoothen
            self.smooth_thickness = smooth_thickness

    def _filter_tile(self, tilecode):
        """
        Returns an AHN tile dict for the area represented by the given
        CycloMedia tile-code.
        """
        if self.method == 'npz':
            return ahn_utils.load_ahn_tile(
                os.path.join(self.data_folder, 'ahn_' + tilecode + '.npz'))
        elif self.method == 'geotiff':
            return self.reader.filter_tile(tilecode)

    def get_label_mask(self, tilecode, points, mask):
        """
        Returns the label mask for the given pointcloud.

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
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        ahn_tile = self._filter_tile(tilecode)
        if self.target == 'ground':
            if self.method == 'geotiff':
                if self.fill_gaps:
                    ahn_utils.fill_gaps(ahn_tile,
                                        max_gap_size=self.max_gap_size,
                                        inplace=True)
                if self.smoothen:
                    ahn_utils.smoothen_edges(ahn_tile,
                                             thickness=self.smooth_thickness,
                                             inplace=True)
            surface = ahn_tile['ground_surface']
        elif self.target == 'building':
            surface = ahn_tile['building_surface']

        # Set-up and run interpolator.
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], surface)
        target_z = fast_z(points[mask, :])

        label_mask = np.zeros((len(points),), dtype=bool)
        if self.target == 'ground':
            label_mask[mask] = (np.abs(points[mask, 2] - target_z)
                                < self.epsilon)
        elif self.target == 'building':
            label_mask[mask] = points[mask, 2] < target_z + self.epsilon

        print(f'AHN [{self.method}] fuser => {self.target} processed '
              f'(label={self.label}).')

        return label_mask
