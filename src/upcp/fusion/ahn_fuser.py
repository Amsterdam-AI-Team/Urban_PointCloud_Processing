# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""AHN Data Fuser"""

import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..region_growing import LabelConnectedComp
from ..labels import Labels
from ..utils import clip_utils
from ..utils import math_utils

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

    def __init__(self, label, ahn_reader, target='ground', epsilon=0.2,
                 refine_ground=True, params={}):
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
        self.refine_ground = refine_ground
        self.params = self._set_defaults(params)

    def _set_defaults(self, params):
        """Set defaults for parameters if not provided."""
        if 'bottom' not in params:
            params['bottom'] = 0.02
        if 'top' not in params:
            params['top'] = 0.5
        if 'grid_size' not in params:
            params['grid_size'] = 0.4
        if 'min_comp_size' not in params:
            params['min_comp_size'] = 50
        if 'buffer' not in params:
            params['buffer'] = 0.05
        return params

    def _refine_layer(self, points, points_z,
                      labels, ground_mask, target_label):
        """Process a layer of the region grower."""

        mask = np.zeros((len(points),), dtype=bool)
        label_ids = np.where(labels == target_label)[0]
        ground_ids = np.where(ground_mask)[0]

        lcc = LabelConnectedComp(
                    target_label, grid_size=self.params['grid_size'],
                    min_component_size=self.params['min_comp_size'])
        point_components = lcc.get_components(points[label_ids])

        cc_labels = np.unique(point_components)
        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)
            # Compute convex hull and add a small buffer
            poly = (math_utils
                    .convex_hull_poly(points[label_ids[cc_mask], 0:2])
                    .buffer(self.params['buffer']))
            # Select ground points within poly
            poly_mask = clip_utils.poly_clip(points[ground_mask], poly)
            mask[ground_ids[poly_mask]] = True
        return mask

    def _refine_ground(self, points, points_z, ground_mask,
                       labels, target_label):
        logger.info('Refining ground surface...')

        mask = ((points[:, 2] <= points_z + self.params['top'])
                & (points[:, 2] >= points_z - self.params['bottom']))

        ref_mask = np.zeros((len(points),), dtype=bool)
        if np.count_nonzero(labels[mask] == target_label) > 0:
            add_mask = self._refine_layer(
                                points[mask, :], points_z[mask], labels[mask],
                                ground_mask[mask], target_label)
            ref_mask[mask] = add_mask

        logger.info(f'{np.count_nonzero(ref_mask)} points removed.')
        return ref_mask

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
        An array of shape (n_points,) with the updated labels.
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
            ground_mask = (np.abs(points[mask, 2] - target_z) < self.epsilon)
            if self.refine_ground and (np.count_nonzero(ground_mask) > 0):
                logger.info(f'{np.count_nonzero(ground_mask)} points added.')
                tmp_labels = labels[mask].copy()
                tmp_labels[ground_mask] = self.label
                ref_mask = self._refine_ground(
                                    points[mask], target_z, ground_mask,
                                    tmp_labels, Labels.UNLABELLED)
                ground_mask = ground_mask & ~ref_mask
            label_mask[mask] = ground_mask
        elif self.target == 'building':
            label_mask[mask] = points[mask, 2] < target_z + self.epsilon

        labels[label_mask] = self.label
        return labels
