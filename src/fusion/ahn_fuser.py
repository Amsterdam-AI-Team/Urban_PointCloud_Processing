"""AHN Data Fuser"""

import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..labels import Labels

logger = logging.getLogger(__name__)


class AHNFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of ground and building points
    using AHN data. The class can handle both pre-processed surfaces in .npz
    format (see preprocessing.ahn_preprocessing) as well as GeoTIFF files
    downloaded from PDOK.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    ahn_reader : AHNReader object
        Used to read the AHN data.
    target : str (default: 'ground')
        If method='npz', this variable determines whether this fuser will label
        either 'ground' or 'building' points.
    epsilon : float (default: 0.2)
        Precision of the fuser.
    """

    TARGETS = ('ground', 'building')

    def __init__(self, label, ahn_reader, target='ground', epsilon=0.2):
        super().__init__(label)
        if target not in self.TARGETS:
            logger.error(f'Target should be one of {self.TARGETS}.')
            return None
        if ahn_reader.NAME == 'geotiff' and target == 'building':
            logger.error(
                f'The {ahn_reader.NAME} reader cannot supply {target} data.')
            return None

        self.ahn_reader = ahn_reader
        self.method = ahn_reader.NAME
        self.target = target
        self.epsilon = epsilon

    def get_label_mask(self, points, labels, mask, tilecode):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            Ignored by this fuser.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        logger.info(f'AHN [{self.method}/{self.target}] fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        if self.target == 'ground':
            target_z = self.ahn_reader.interpolate(
                tilecode, points[mask, :], mask, 'ground_surface')
        elif self.target == 'building':
            target_z = self.ahn_reader.interpolate(
                tilecode, points[mask, :], mask, 'building_surface')

        label_mask = np.zeros((len(points),), dtype=bool)
        if self.target == 'ground':
            label_mask[mask] = (np.abs(points[mask, 2] - target_z)
                                < self.epsilon)
        elif self.target == 'building':
            label_mask[mask] = points[mask, 2] < target_z + self.epsilon

        return label_mask
