import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..region_growing import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.labels import Labels

logger = logging.getLogger(__name__)


class TopBottomLCC(AbstractProcessor):
    """
    Two-part region growing based on LabelConnectedComp. Region growing is
    applied separately to the top and bottom part of an object, so that
    different parameters can be used for both.

    Parameters
    ----------
    label : int
        Class label to use.
    ahn_reader : AHNReader object
        Used to get the correct elevation data.
    top_params : dict (see Notes)
        Parameters for the top-part of the region grower.
    bottom_params : dict (see Notes)
        Parameters for the bottom-part of the region grower.

    Notes
    -----
    The `top_params` and `bottom_params` and their defaults are as follows:

    'plane_height': REQUIRED
        Height above ground at which to start/stop (for top/bottom) growing.
    'octree_level': 9
        Octree level for LCC method, higher means more fine-grained.
    'min_comp_size': 100
        Minimum number of points for a component to be considered.
    'threshold': 0.5
        Minimum fraction of points in a component that are already labelled
        initially for the component to be added.
    """

    def __init__(self, label, ahn_reader,
                 top_params={}, bottom_params={}):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.top_params = self._set_defaults(top_params, 't')
        self.bottom_params = self._set_defaults(bottom_params, 'b')

    def _set_defaults(self, params, param_type):
        """Set defaults for parameters if not provided."""
        if 'plane_height' not in params:
            logger.error('You must supply `plane_height` parameter.')
            raise Exception
        if 'octree_level' not in params:
            params['octree_level'] = 9
        if 'min_comp_size' not in params:
            params['min_comp_size'] = 100
        if 'threshold' not in params:
            params['threshold'] = 0.5
        params['type'] = param_type
        return params

    def _filter_layer(self, points, points_z, labels, params):
        """Process either the top or bottom part of the region grower."""
        if params['type'] == 't':
            # Top-part, so we want points above points_z.
            height_mask_ids = np.where(
                        points[:, 2] > points_z + params['plane_height'])[0]
        else:
            # Bottom-part, so we want points below points_z.
            height_mask_ids = np.where(
                        points[:, 2] <= points_z + params['plane_height'])[0]
        mask = np.zeros((len(points),), dtype=bool)
        lcc = LabelConnectedComp(self.label, set_debug=True,
                                 octree_level=params['octree_level'],
                                 min_component_size=params['min_comp_size'],
                                 threshold=params['threshold'])
        lcc_mask = lcc.get_label_mask(points=points[height_mask_ids],
                                      labels=labels[height_mask_ids])
        mask[height_mask_ids[lcc_mask]] = True
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
        logger.info('TopBottomLCC ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}))')

        # We need to un-mask all points of the desired class label.
        mask_copy = mask.copy()
        mask_copy[labels == self.label] = True

        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                      ahn_tile['ground_surface'])
        points_z = fast_z(points[mask_copy, 0:2])

        label_mask = np.zeros((len(points),), dtype=bool)
        tmp_mask = np.zeros((np.count_nonzero(mask_copy),), dtype=bool)

        # Process the top part.
        layer_mask = self._filter_layer(points[mask_copy, :], points_z,
                                        labels[mask_copy], self.top_params)
        tmp_mask[layer_mask] = True

        # Process the bottom part.
        layer_mask = self._filter_layer(points[mask_copy, :], points_z,
                                        labels[mask_copy], self.bottom_params)
        tmp_mask[layer_mask] = True

        label_mask[mask_copy] = tmp_mask
        return label_mask & mask
