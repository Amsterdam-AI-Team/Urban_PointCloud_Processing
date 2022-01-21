import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..region_growing import LabelConnectedComp
from ..labels import Labels
from ..utils import clip_utils
from ..utils import math_utils

logger = logging.getLogger(__name__)


class RefineGround(AbstractProcessor):
    """

    """

    GROUND_LABELS = [Labels.GROUND,
                     Labels.ROAD]
    TARGET_LABELS = [Labels.BUILDING,
                     Labels.CAR,
                     Labels.TREE,
                     Labels.TRAFFIC_SIGN,
                     Labels.TRAFFIC_LIGHT,
                     Labels.STREET_LIGHT,
                     Labels.CITY_BENCH,
                     Labels.RUBBISH_BIN]
    POLY_BUFFER = 0.05
    GND_EPS = 0.02

    def __init__(self, label, ahn_reader, params=[]):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.params = self._set_defaults(params)

    def _set_defaults(self, params):
        """Set defaults for parameters if not provided."""
        if 'bottom' not in params:
            params['bottom'] = -0.05
        if 'top' not in params:
            params['top'] = np.inf
        if 'grid_size' not in params:
            params['grid_size'] = 0.2
        if 'min_comp_size' not in params:
            params['min_comp_size'] = 50
        return params

    def _process_layer(self, points, points_z,
                       labels, ground_mask, target_label):
        """Process a layer of the region grower."""

        self.polys = []

        mask = np.zeros((len(points),), dtype=bool)
        label_ids = np.where(labels == target_label)[0]
        ground_ids = np.where(ground_mask)[0]

        lcc = LabelConnectedComp(
                    target_label, grid_size=self.params['grid_size'],
                    min_component_size=self.params['min_comp_size'])
        point_components = lcc.get_components(points[label_ids])

        cc_labels = np.unique(point_components)
        cc_labels = set(cc_labels).difference((-1,))

        logger.debug(
                f'{len(cc_labels)} clusters found for label {target_label}.')

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)
            # Compute convex hull and add a small buffer
            poly = (math_utils
                    .convex_hull_poly(points[label_ids[cc_mask], 0:2])
                    .buffer(self.POLY_BUFFER))
            self.polys.append(poly)
            # Select ground points within poly and above ground + eps
            poly_mask = clip_utils.poly_clip(points[ground_mask], poly)
            local_hm = (points[ground_ids[poly_mask], 2] >
                        points_z[ground_ids[poly_mask]] + self.GND_EPS)
            mask[ground_ids[poly_mask][local_hm]] = True
            cnt = np.count_nonzero(ground_ids[poly_mask][local_hm])
            logger.debug(f'Added {cnt} points.')
        return mask

    def get_label_mask(self, points, labels, mask, tilecode):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            The labels corresponding to each point.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        logger.info(f'RefineGround for labels {self.label}')

        # We need to un-mask all points of the desired class label.
        mask = np.zeros((len(points),), dtype=bool)
        for lab in self.label:
            mask[labels == lab] = True
        # Un-mask ground points.
        for lab in self.GROUND_LABELS:
            mask[labels == lab] = True

        points_z = self.ahn_reader.interpolate(
                    tilecode, points[mask], None, 'ground_surface')
        height_mask = ((points[mask, 2] > points_z + self.params['bottom'])
                       & (points[mask, 2] <= points_z + self.params['top']))
        mask[mask] = height_mask

        ground_mask = np.zeros((np.count_nonzero(mask),), dtype=bool)
        for lab in self.GROUND_LABELS:
            ground_mask[labels[mask] == lab] = True

        return_mask = []

        for lab in self.label:
            label_mask = np.zeros((len(points),), dtype=bool)
            if np.count_nonzero(labels[mask] == lab) > 0:
                add_mask = self._process_layer(
                                        points[mask, :], points_z[height_mask],
                                        labels[mask], ground_mask, lab)
                label_mask[mask] = add_mask
                logger.info(f'Label {lab}: ' +
                            f'{np.count_nonzero(add_mask)} points added.')
            else:
                logger.info(f'Label {lab} not present, skipping.')
            return_mask.append(label_mask)

        return return_mask
