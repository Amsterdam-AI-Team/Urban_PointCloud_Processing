import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..region_growing import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.labels import Labels

logger = logging.getLogger(__name__)


class LayerLCC(AbstractProcessor):
    """
    Layered region growing based on LabelConnectedComp. Region growing is
    applied separately to each layer, specified by its bottom and top bounds,
    so that different parameters can be used for each layer.

    Parameters
    ----------
    label : int
        Class label to use.
    ahn_reader : AHNReader object
        Used to get the correct elevation data.
    reset_noise : bool (default: False)
        If set to true, points previously labelled as noise will be considered.
    params : dict or list of dicts (see Notes)
        Parameters for each layer of the region grower.

    Notes
    -----
    Each layer is specified by a parameters dict. Each layer/dict contains the
    following (default) parameters:

    'bottom': -inf
        Height above ground at which the layer starts.
    'top': inf
        Height above ground at which the layer stops.
    'octree_level': 9
        Octree level for LCC method, higher means more fine-grained.
    'min_comp_size': 100
        Minimum number of points for a component to be considered.
    'threshold': 0.5
        Minimum fraction of points in a component that are already labelled
        initially for the component to be added.
    """

    def __init__(self, label, ahn_reader, reset_noise=False, params=[]):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.reset_noise = reset_noise
        self.params = self._set_defaults(params)

    def _set_defaults(self, params):
        """Set defaults for parameters if not provided."""
        for layer in params:
            if 'bottom' not in layer:
                layer['bottom'] = -np.inf
            if 'top' not in layer:
                layer['top'] = np.inf
            if 'octree_level' not in layer:
                layer['octree_level'] = 9
            if 'min_comp_size' not in layer:
                layer['min_comp_size'] = 100
            if 'threshold' not in layer:
                layer['threshold'] = 0.5
        return params

    def _filter_layer(self, points, points_z, labels, params):
        """Process a layer of the region grower."""
        height_mask_ids = np.where(
                    (points[:, 2] > points_z + params['bottom'])
                    & (points[:, 2] <= points_z + params['top']))[0]
        mask = np.zeros((len(points),), dtype=bool)
        n_valid = np.count_nonzero(labels[height_mask_ids] == self.label)
        logger.debug(f'{n_valid} valid labels in this layer.')
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
        logger.info('LayerLCC ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}))')

        # We need to un-mask all points of the desired class label.
        mask_copy = mask.copy()
        mask_copy[labels == self.label] = True
        if self.reset_noise:
            noise_mask = labels == Labels.NOISE
            mask_copy[noise_mask] = True

        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                      ahn_tile['ground_surface'])
        points_z = fast_z(points[mask_copy, 0:2])

        label_mask = np.zeros((len(points),), dtype=bool)
        layer_mask = np.zeros((np.count_nonzero(mask_copy),), dtype=bool)
        labels_copy = labels[mask_copy].copy()

        for i, layer in enumerate(self.params):
            logger.debug(f'Layer {i}: {layer}')
            layer_mask = layer_mask | self._filter_layer(
                                            points[mask_copy, :], points_z,
                                            labels_copy, layer)
            labels_copy[layer_mask] = self.label

        label_mask[mask_copy] = layer_mask
        if self.reset_noise:
            return label_mask & (mask | noise_mask)
        else:
            return label_mask & mask
