"""AHN Data Fuser"""

import numpy as np
import os

from .data_fuser import DataFuser
from ..preprocessing.ahn_preprocessing import load_ahn_tile
from ..utils.interpolation import FastGridInterpolator


class AHNFuser(DataFuser):
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
    """

    METHODS = ('npz', 'geotiff')
    TARGETS = ('ground', 'building')

    def __init__(self, label, data_folder,
                 method='npz', target='ground', epsilon=0.2):
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

    def filter_tile(self, tilecode):
        """Returns an AHN tile dict for the given CycloMedia tile-code."""
        if self.method == 'npz':
            return load_ahn_tile(os.path.join(self.data_folder,
                                              'ahn_' + tilecode + '.npz'))
        elif self.method == 'geotiff':
            # TODO implement this
            pass

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

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        ahn_tile = self.filter_tile(tilecode)
        pos = np.vstack((points['x'], points['y'])).T
        if self.target == 'ground':
            surface = ahn_tile['ground_surface']
        elif self.target == 'building':
            surface = ahn_tile['building_surface']

        # Set-up and run interpolator.
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], surface)
        target_z = fast_z(pos)

        if self.target == 'ground':
            mask = np.abs(points['z'] - target_z) < self.epsilon
        elif self.target == 'building':
            mask = points['z'] < target_z + self.epsilon
        return mask
