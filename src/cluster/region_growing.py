import numpy as np
import open3d as o3d
import copy

from ..utils.math_utils import angle_between


class RegionGrowing:
    def __init__(self, threshold_angle=20, threshold_curve=1.0, max_nn=30,
                 grow_region_knn=15, grow_region_radius=0.2):
        """ Init variables. """
        self.threshold_angle = threshold_angle
        self.threshold_curve = threshold_curve

        self.max_nn = max_nn
        self.grow_region_knn = grow_region_knn
        self.grow_region_radius = grow_region_radius

    def _get_label_indices(self, las_labels, seed_point_label):
        """ Function to get all indices of a label in a point cloud. """
        list_of_indices = np.where(las_labels == seed_point_label)[0]
        return list_of_indices

    def _convert_pcd(self, las, exclude_label):
        """ Stick the las coordinates together in a nx3 numpy array and
        convert it to a Open3D pcd format. """
        subset_pc = self._get_label_indices(las.label, exclude_label)

        coords = np.vstack((las.x[subset_pc], las.y[subset_pc],
                            las.z[subset_pc])).transpose()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        return pcd

    def set_input_cloud(self, las, unlabelled):
        """ Function to convert to o3d point cloud. """
        pcd = self._convert_pcd(las, unlabelled)
        self.pcd = pcd

    def set_initial_seed_points(self, las_labels, seed_point_label):
        """ Function to set list with indices of initial seed points. """
        list_of_indices = self._get_label_indices(las_labels, seed_point_label)
        if len(list_of_indices) == 0:
            print('NOTE: Input point cloud does not contain any seed points.')

        self.list_of_seed_ids = list_of_indices.tolist()

    def _compute_point_curvature(self, coords, pcd_tree, seed_point, method):
        """ Compute the curvature for a given a cluster of points. """
        if method == 'radius':
            _, idx, _ = pcd_tree.search_radius_vector_3d(seed_point,
                                                         self.grow_region_radius)
        else:
            _, idx, _ = pcd_tree.search_knn_vector_3d(seed_point,
                                                      self.grow_region_knn)

        neighbors = o3d.utility.Vector3dVector(coords[idx])
        pcd = o3d.geometry.PointCloud(neighbors)
        _, cov = pcd.compute_mean_and_covariance()
        eig_val, _ = np.linalg.eig(cov)

        return (eig_val[0]/(eig_val.sum()))

    def region_growing(self, method='knn'):
        """
        The work of this region growing algorithm is based on the comparison
        of the angles between the points normals.

        The same can also be performed in Python using scipy.spatial.cKDTree
        with query_ball_tree or query.
        """
        # Compute the KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

        # Compute the normals for each point
        self.pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=self.grow_region_radius, max_nn=self.max_nn))

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
                k, neighbor_idx, _ = pcd_tree.search_radius_vector_3d(seed_point,
                                                                      self.grow_region_radius)
            else:
                k, neighbor_idx, _ = pcd_tree.search_knn_vector_3d(seed_point,
                                                                   self.grow_region_knn)

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

                    # Here we compute the curvature for a neighbor_id and its neighbors.
                    curvature = self._compute_point_curvature(np.asarray(self.pcd.points),
                                                              pcd_tree,
                                                              self.pcd.points[neighbor_id],
                                                              method)

                    # If result is below threshold, we add it to the seed points
                    if curvature < self.threshold_curve:
                        self.list_of_seed_ids.append(neighbor_id)

            idx = idx+1

        print('There are {} points added'.format(len(region) - seed_length))

        return region
