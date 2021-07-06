import numpy as np

# Two libraries necessary for the CloudCompare Python wrapper
# Installation instructions in notebook [3. Clustering based region growing]
import pycc
import cccorelib

from .abstract import AbstractRegionGrowing


class LabelConnectedComp(AbstractRegionGrowing):
    """
    Clustering based region growing implementation using label connected comp.
    """
    def __init__(self, label, exclude_labels, octree_level=9,
                 min_component_size=100):
        super().__init__(label)
        """ Init variables. """
        self.octree_level = octree_level
        self.min_component_size = min_component_size

        self.exclude_labels = exclude_labels

    def _set_mask(self, las_labels):
        """ Configure the points that we want to perform region growing on. """
        mask = np.ones((len(las_labels),), dtype=bool)

        for exclude_label in self.exclude_labels:
            mask = mask & (las_labels != exclude_label)

        self.las_label = las_labels
        self.mask = mask

    def _convert_input_cloud(self, las):
        """ Function to convert to CloudCompare point cloud. """
        # Be aware that CloudCompare stores coordinates on 32 bit floats.
        # To avoid losing too much precision you should 'shift' your
        # coordinates if they are 64 bit floats (which is the default in
        # python land)
        xs = (las[self.mask, 0]).astype(pycc.PointCoordinateType)
        ys = (las[self.mask, 1]).astype(pycc.PointCoordinateType)
        zs = (las[self.mask, 2]).astype(pycc.PointCoordinateType)
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
        component_count = (cccorelib.AutoSegmentationTools
                           .labelConnectedComponents(self.point_cloud,
                                                     level=self.octree_level))
        print(f'There are {component_count} components found')

        # Get the scalar field with labels and points coords as numpy array
        labels_sf = self.point_cloud.getScalarField(self.labels_sf_idx)
        self.point_components = labels_sf.asArray()

    def _fill_components(self, threshold=0.1):
        """ Clustering based region growing process. When one initial seed
        point is found inside a component, make the whole component this
        label. """
        pre_seed_count = np.count_nonzero(self.las_label ==
                                          self.label)

        mask_indices = np.where(self.mask)[0]
        label_mask = np.zeros(len(self.mask), dtype=bool)

        cc_labels, counts = np.unique(self.point_components,
                                      return_counts=True)

        cc_labels_filtered = cc_labels[counts >= self.min_component_size]

        for cc in cc_labels_filtered:
            # select points that belong to the cluster
            cc_mask = (self.point_components == cc)
            # cluster size
            cc_size = np.count_nonzero(cc_mask)
            if cc_size < self.min_component_size:
                continue
            # number of point in teh cluster that are labelled as seed point
            seed_count = np.count_nonzero(
                self.las_label[mask_indices[cc_mask]] == self.label)
            # at least X% of the cluster should be seed points
            if (float(seed_count) / cc_size) > threshold:
                label_mask[mask_indices[cc_mask]] = True

        # Add label to the regions
        labels = self.las_label
        labels[label_mask] = self.label
        post_seed_count = np.count_nonzero(labels == self.label)

        # Calculate the number of points grown
        points_added = post_seed_count - pre_seed_count

        return label_mask, points_added

    def get_label_mask(self, points, las_labels):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points, 1)
            All labels as int values

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        self._set_mask(las_labels)
        self._convert_input_cloud(points)
        self._label_connected_comp()
        label_mask, points_added = self._fill_components()

        print(f'Clustering based Region Growing => {points_added} '
              f'points added (label={self.label}).')

        return label_mask
