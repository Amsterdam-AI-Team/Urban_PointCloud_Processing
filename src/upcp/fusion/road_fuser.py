# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Road Fuser"""

import numpy as np
import logging

from ..utils import clip_utils
from ..labels import Labels
from ..abstract_processor import AbstractProcessor

logger = logging.getLogger(__name__)


class BGTRoadFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of road points using BGT data
    in the form of polygons.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_reader : BGTPolyReader object
        Used to load road part polygons.
    bgt_types : str or list of [str] (default: all available)
        Indicates which road part types (BGT layers) to label as road. By
        default, all available types returned by the BGTPolyReader will be
        used.
    offset : int (default: 0)
        The road polygon will be extended by this amount (in meters).
    padding : float (default: 0)
        Optional padding (in m) around the tile when searching for objects.
    """

    ALL_TYPES = ['rijbaan_lokale_weg', 'parkeervlak', 'rijbaan_autoweg',
                 'rijbaan_autosnelweg', 'rijbaan_regionale_weg', 'ov-baan',
                 'fietspad']

    def __init__(self, label, bgt_reader, bgt_types=ALL_TYPES,
                 offset=0, padding=0):
        super().__init__(label)
        self.bgt_reader = bgt_reader
        if isinstance(bgt_types, str):
            bgt_types = [bgt_types]
        self.bgt_types = bgt_types
        self.offset = offset
        self.padding = padding

    def get_labels(self, points, labels, mask, tilecode):
        """
        Returns the labels for the given pointcloud.
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
        An array of shape (n_points,) with the updated labels.
        """
        logger.info('BGT road fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros(len(points), dtype=bool)

        road_polygons = self.bgt_reader.filter_tile(
                                tilecode, bgt_types=self.bgt_types,
                                padding=self.padding, offset=self.offset,
                                merge=True)
        if len(road_polygons) == 0:
            logger.debug('No road parts found in tile, skipping.')
            return label_mask

        # Already labelled ground points can be labelled as road.
        mask = labels == Labels.GROUND
        mask_ids = np.where(mask)[0]

        road_mask = np.zeros((len(mask_ids),), dtype=bool)
        for polygon in road_polygons:
            clip_mask = clip_utils.poly_clip(points[mask, :], polygon)
            road_mask = road_mask | clip_mask

        logger.debug(f'{len(road_polygons)} road polygons labelled.')
        logger.info(f'{np.count_nonzero(road_mask)} ground points relabelled.')

        label_mask[mask_ids[road_mask]] = True
        labels[label_mask] = self.label

        return labels
