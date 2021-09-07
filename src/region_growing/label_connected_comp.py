import numpy as np
import logging

# Two libraries necessary for the CloudCompare Python wrapper
# Installation instructions in notebook [5. Region growing.ipynb]
import pycc
import cccorelib

from ..abstract_processor import AbstractProcessor
from ..utils.labels import Labels

logger = logging.getLogger(__name__)


class LabelConnectedComp(AbstractProcessor):
    """
    Clustering based region growing implementation using the label connected
    components method from CloudCompare.

    Parameters
    ----------
    label : int (default: -1)
        Label to use when labelling the grown region.
    exclude_labels : list-like (default: [])
        Which labels to exclude (optional).
    octree_level : int (default: 9)
        Used to construct the underlying octree, larger means more
        fine-grained.
    min_component_size : int (default: 100)
        Minimum size of connected components to consider.
    threshold : float (default: 0.5)
        When labelling a cluster, at least this proportion of points should
        already be labelled.
    """
    def __init__(self, label=-1, exclude_labels=[], set_debug=False,
                 octree_level=9, min_component_size=100, threshold=0.1):
        super().__init__(label)
        """ Init variables. """
        self.octree_level = octree_level
        self.min_component_size = min_component_size
        self.threshold = threshold
        self.exclude_labels = exclude_labels
        self.debug = set_debug

    def _set_mask(self):
        """ Configure the points that we want to perform region growing on. """
        for exclude_label in self.exclude_labels:
            self.mask = self.mask & (self.point_labels != exclude_label)

    def _convert_input_cloud(self, points):
        """ Function to convert to CloudCompare point cloud. """
        # Be aware that CloudCompare stores coordinates on 32 bit floats.
        # To avoid losing too much precision you should 'shift' your
        # coordinates if they are 64 bit floats (which is the default in
        # python land)
        xs = (points[self.mask, 0]).astype(pycc.PointCoordinateType)
        ys = (points[self.mask, 1]).astype(pycc.PointCoordinateType)
        zs = (points[self.mask, 2]).astype(pycc.PointCoordinateType)
        point_cloud = pycc.ccPointCloud(xs, ys, zs)

        # (Optional) Create (if it does not exists already)
        # a scalar field where we store the Labels
        labels_sf_idx = point_cloud.getScalarFieldIndexByName('Labels')
        if labels_sf_idx == -1:
            labels_sf_idx = point_cloud.addScalarField('Labels')
        point_cloud.setCurrentScalarField(labels_sf_idx)

        self.labels_sf_idx = labels_sf_idx
        # You can access the x,y,z fields using self.point_cloud.points()
        self.point_cloud = point_cloud

    def _label_connected_comp(self):
        """ Perform the clustering algorithm: Label Connected Components. """
        (cccorelib
         .AutoSegmentationTools
         .labelConnectedComponents(self.point_cloud, level=self.octree_level))

        # Get the scalar field with labels and points coords as numpy array
        labels_sf = self.point_cloud.getScalarField(self.labels_sf_idx)
        self.point_components = labels_sf.asArray()

        # Filter based on min_component_size
        if self.min_component_size > 1:
            cc_labels, counts = np.unique(self.point_components,
                                          return_counts=True)
            cc_labels_filtered = cc_labels[counts < self.min_component_size]
            self.point_components[
                np.in1d(self.point_components, cc_labels_filtered)] = -1

    def _fill_components(self):
        """ Clustering based region growing process. When one initial seed
        point is found inside a component, make the whole component this
        label. """
        mask_indices = np.where(self.mask)[0]
        label_mask = np.zeros(len(self.mask), dtype=bool)

        cc_labels = np.unique(self.point_components)
        cc_labels = set(cc_labels).difference((-1,))

        cc_count = 0

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (self.point_components == cc)
            # cluster size
            # TODO is this still needed? (see line 77-80)
            cc_size = np.count_nonzero(cc_mask)
            if cc_size < self.min_component_size:
                continue
            # number of point in the cluster that are labelled as seed point
            seed_count = np.count_nonzero(
                self.point_labels[mask_indices[cc_mask]] == self.label)
            # at least X% of the cluster should be seed points
            if (float(seed_count) / cc_size) > self.threshold:
                label_mask[mask_indices[cc_mask]] = True
                cc_count += 1

        # Add label to the regions
        labels = self.point_labels
        labels[label_mask] = self.label

        logger.debug(f'Found {len(cc_labels)} clusters of ' +
                     f'>{self.min_component_size} points; ' +
                     f'{cc_count} added.')

        return label_mask

    def get_label_mask(self, points, labels, mask=None, tilecode=None):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            The labels corresponding to each point.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points. Can be
            overwritten by setting `exclude_labels` in the constructor.
        tilecode : str
            Ignored by this class.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        if not self.debug:
            logger.info('Clustering based Region Growing ' +
                        f'(label={self.label}, {Labels.get_str(self.label)}).')
        else:
            logger.debug('Clustering based Region Growing ' +
                         f'(label={self.label}, ' +
                         f'{Labels.get_str(self.label)}).')
        if self.label == -1:
            logger.warning('Label not set, defaulting to -1.')
        self.point_labels = labels
        if self.exclude_labels:
            self.mask = np.ones((len(points),), dtype=bool)
            self._set_mask()
        elif mask is None:
            self.mask = np.ones((len(points),), dtype=bool)
        else:
            self.mask = mask.copy()
            # We need to un-mask all points of the desired class label.
            self.mask[labels == self.label] = True

        self._convert_input_cloud(points)
        self._label_connected_comp()
        label_mask = self._fill_components()

        return label_mask

    def get_components(self, points, labels=None):
        """
        Simply get the components without caring about labels.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            The labels corresponding to each point. Optional, only used in
            combination with `exclude_labels`.

        Returns
        -------
        An array of shape (n_points,) with cluster labels for each point.
        Clusters with a size less than `min_component_size` are labelled as -1.
        """
        self.mask = np.ones((len(points),), dtype=bool)
        if labels is not None and self.exclude_labels:
            self.point_labels = labels
            self._set_mask()
        self._convert_input_cloud(points)
        self._label_connected_comp()
        return self.point_components
