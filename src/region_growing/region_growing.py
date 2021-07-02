import numpy as np
import open3d as o3d
import copy

from ..utils.math_utils import angle_between
from .abstract import AbstractRegionGrowing


class RegionGrowing(AbstractRegionGrowing):
    """
    Region growing implementation based on:
    https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html
    """
    def __init__(self, label, threshold_angle=20, threshold_curve=1.0,
                 max_nn=30, grow_region_knn=15, grow_region_radius=0.2):
        super().__init__(label)
        """ Init variables. """
        self.threshold_angle = threshold_angle
        self.threshold_curve = threshold_curve

        self.max_nn = max_nn
        self.grow_region_knn = grow_region_knn
        self.grow_region_radius = grow_region_radius

    def _set_mask(self, mask):
        self.mask = mask

    def _set_input_cloud(self, las, las_labels, seed_point_label):
        """ Function to convert to o3d point cloud. """
        coords = np.vstack((las['x'][self.mask], las['y'][self.mask], las['z'][self.mask])).transpose() # todo
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        self.pcd = pcd

        list_of_indices = np.where(las_labels[self.mask] == seed_point_label)[0]
        if len(list_of_indices) == 0:
            print('NOTE: Input point cloud does not contain any seed points.')
        self.list_of_seed_ids = list_of_indices.tolist()

        self.mask_indices = np.where(self.mask)[0]
        self.label_mask = np.zeros(len(self.mask), dtype=bool)

    def _compute_point_curvature(self, coords, pcd_tree, seed_point, method):
        """ Compute the curvature for a given a cluster of points. """
        if method == 'radius':
            _, idx, _ = (pcd_tree.search_radius_vector_3d(
                         seed_point, self.grow_region_radius))
        else:
            _, idx, _ = (pcd_tree.search_knn_vector_3d(
                         seed_point, self.grow_region_knn))

        neighbors = o3d.utility.Vector3dVector(coords[idx])
        pcd = o3d.geometry.PointCloud(neighbors)
        _, cov = pcd.compute_mean_and_covariance()
        eig_val, _ = np.linalg.eig(cov)

        return (eig_val[0]/(eig_val.sum()))

    def _region_growing(self, method='knn'):
        """
        The work of this region growing algorithm is based on the comparison
        of the angles between the points normals.

        The same can also be performed in Python using scipy.spatial.cKDTree
        with query_ball_tree or query.
        """
        # Compute the KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

        # Compute the normals for each point
        self.pcd.estimate_normals(search_param=o3d.geometry
                                  .KDTreeSearchParamHybrid(radius=self.grow_region_radius,
                                  max_nn=self.max_nn))

        seed_length = len(self.list_of_seed_ids)
        region = copy.deepcopy(self.list_of_seed_ids)

        # Initialize the indexes of all seed points as processed
        processed = np.full(len(self.pcd.points), False)
        processed[self.list_of_seed_ids] = True

        idx = 0
        while idx < len(self.list_of_seed_ids):
            seed_point = self.pcd.points[self.list_of_seed_ids[idx]]
            seed_normal = self.pcd.normals[self.list_of_seed_ids[idx]]

            # For every seed point, the algorithm finds its neighboring points
            if method == 'radius':
                k, neighbor_idx, _ = (pcd_tree.search_radius_vector_3d(
                                      seed_point, self.grow_region_radius))
            else:
                k, neighbor_idx, _ = (pcd_tree.search_knn_vector_3d(seed_point,
                                      self.grow_region_knn))

            # Remove index seed point itself
            neighbor_idx = neighbor_idx[1:k]

            for neighbor_id in neighbor_idx:
                # Is this point processed before?
                if processed[neighbor_id]:
                    continue

                # Compute angles between two n-dimensional vectors
                current_angle = angle_between(seed_normal,
                                              self.pcd.normals[neighbor_id])
                # The smoothness constraint in degrees
                if current_angle < self.threshold_angle:
                    region.append(neighbor_id)
                    processed[neighbor_id] = True

                    # Compute the curvature for a neighbor_id and its neighbors
                    curvature = (self._compute_point_curvature(
                                 np.asarray(self.pcd.points), pcd_tree,
                                 self.pcd.points[neighbor_id], method))

                    # Result is below threshold, we add it to the seed points
                    if curvature < self.threshold_curve:
                        self.list_of_seed_ids.append(neighbor_id)

            idx = idx+1

        print('There are {} points added'.format(len(region) - seed_length))

        self.label_mask[self.mask_indices[region]] = True

        return self.label_mask

    def get_label_mask(self, points, labels):
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
        seed_point_label = 2
        exclude_labels = [1, ]
        mask = np.zeros(len(labels), dtype=bool)
        mask = True
        for exclude_label in exclude_labels:
            mask = mask & (labels != exclude_label)

        self._set_mask(mask)
        self._set_input_cloud(points, labels, seed_point_label)
        labels = self._region_growing()

        return labels
