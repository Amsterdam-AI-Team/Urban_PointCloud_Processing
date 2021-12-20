"""Building Fuser"""

import numpy as np
import logging

from ..utils import clip_utils
from ..labels import Labels
from ..abstract_processor import AbstractProcessor

logger = logging.getLogger(__name__)


class BGTBuildingFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of building points using BGT data
    in the form of footprint polygons.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_reader : BGTPolyReader object
        Used to load building polygons.
    offset : int (default: 0)
        The footprint polygon will be extended by this amount (in meters).
    padding : float (default: 0)
        Optional padding (in m) around the tile when searching for objects.
    ahn_reader : AHNReader object
        Optional, if provided AHN data will be used to set a maximum height for
        each building polygon.
    ahn_eps : float (default: 0.2)
        Precision for the AHN elevation cut-off for buildings.
    """

    def __init__(self, label, bgt_reader, offset=0, padding=0,
                 ahn_reader=None, ahn_eps=0.2):
        super().__init__(label)
        self.bgt_reader = bgt_reader
        self.offset = offset
        self.padding = padding
        self.ahn_reader = ahn_reader
        self.ahn_eps = ahn_eps

    def get_label_mask(self, points, labels, mask, tilecode):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            Ignored by this class.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        logger.info('BGT building fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        building_polygons = self.bgt_reader.filter_tile(
                                    tilecode, bgt_types=['pand'],
                                    padding=self.padding, offset=self.offset,
                                    merge=True)
        if len(building_polygons) == 0:
            logger.debug('No buildings found for tile, skipping.')
            return label_mask

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]

        building_mask = np.zeros((len(mask_ids),), dtype=bool)
        for polygon in building_polygons:
            # TODO if there are multiple buildings we could mask the points
            # iteratively to ignore points already labelled.
            clip_mask = clip_utils.poly_clip(points[mask, :], polygon)
            building_mask = building_mask | clip_mask

        if self.ahn_reader is not None:
            bld_z = self.ahn_reader.interpolate(
                tilecode, points[mask, :], mask, 'building_surface')
            bld_z_valid = np.isfinite(bld_z)
            ahn_mask = (points[mask_ids[bld_z_valid], 2]
                        <= bld_z[bld_z_valid] + self.ahn_eps)
            building_mask[bld_z_valid] = building_mask[bld_z_valid] & ahn_mask

        logger.debug(f'{len(building_polygons)} building polygons labelled.')

        label_mask[mask_ids[building_mask]] = True

        return label_mask
