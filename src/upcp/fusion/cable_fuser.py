# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Cable Fuser"""

import numpy as np
import logging
import warnings

from ..utils import clip_utils
from ..labels import Labels
from ..abstract_processor import AbstractProcessor
from ..utils.interpolation import FastGridInterpolator
from ..utils.ahn_utils import fill_gaps_intuitive
from ..utils.clip_utils import poly_clip, poly_box_clip
from ..utils.las_utils import get_bbox_from_tile_code

import open3d as o3d
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.optimize import OptimizeWarning


logger = logging.getLogger(__name__)


class CableFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of cable and suspended streetlight points.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bag_reader : BAGPolyReader object
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

    def __init__(self, label, cable_label, streetlight_label, ahn_reader,
                 bag_reader, bgt_tram_reader, min_cable_height=4.5, max_cable_height=12,
                 building_offset=1.25, voxel_size=, neigh_radius=, 
                 linearity_thres=, max_v_angle=, max_merge_angle=,
                 min_segment_length=,max_tramcable_height=,):
        super().__init__(label)
        self.cable_label = cable_label
        self.streetlight_label = streetlight_label
        self.ahn_reader = ahn_reader
        self.bag_reader = bag_reader
        self.bgt_tram_reader = bgt_tram_reader
        self.min_cable_height = min_cable_height
        self.max_cable_height = max_cable_height
        self.building_offset = building_offset
        self.voxel_size = voxel_size
        self.neigh_radius = neigh_radius
        self.linearity_thres = linearity_thres
        self.max_v_angle = max_v_angle
        self.max_merge_angle = max_merge_angle
        self.min_segment_length = min_segment_length
        self.cable_size = cable_size
        self.max_tramcable_height = max_tramcable_height

    def _vertical_segmentation(self, points, tilecode):
        '''Removes low and high height points from mask'''

        # Merge Ground and Artifact and Interpolate gaps of AHN tile
        ahn_tile = fill_gaps_intuitive(self.ahn_reader.filter_tile(tilecode))
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], ahn_tile['ground_surface'])
        ground_z = fast_z(points)

        # Segmentate points above and below min and max cable height
        vertical_seg_mask = (points[:, 2] > ground_z + self.min_cable_height) and \
                            (points[:, 2] < ground_z + self.max_cable_height)
                            

        return vertical_seg_mask

    def _bag_removal(self, points, tilecode, offset):
        """
        tilecode : str
            Tilecode to use for this filter.
        bgt_reader : BGTPolyReader object
            Used to load building polygons.
        ahn_reader : AHNReader object
            AHN data will be used to set a minimum height for each building polygon.
        offset : int (default: 0)
            The footprint polygon will be extended by this amount (in meters).
        """

        # Create non-building mask
        bag_seg_mask = np.ones(len(points), dtype=bool)

        # Read BAG
        building_polygons = self.bag_reader.filter_tile(
                                    tilecode, bgt_types=['pand'],
                                    padding=offset, offset=offset,
                                    merge=True)

        logger.debug(f'Buildings in tile {len(building_polygons)}')
        if len(building_polygons) == 0: 
            return bag_seg_mask

        # Remove buildings from mask
        tile_polygon = get_polygon_from_tile_code(tilecode)
        for polygon in building_polygons:
            if polygon.intersects(tile_polygon):
                clip_mask = poly_clip(points, polygon)
                bag_seg_mask[clip_mask] = False

        return bag_seg_mask

    def _point_features(self, pcd, radius, max_nn=50):
        
        pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        eig_val, eig_vec = np.linalg.eig(np.asarray(pcd.covariances))

        #: sort eigenvalues λ1 > λ2 > λ3
        index_array = np.flip(eig_val.argsort(), axis=1)
        eig_val = np.take_along_axis(eig_val, index_array, axis=1)

        linearity = np.nan_to_num((eig_val[:, 0] - eig_val[:, 1]) / eig_val[:, 0])

        principal_axis = eig_vec[np.arange(len(eig_vec)), :, index_array[:,0]]

        vertical_angle = np.degrees(np.arccos(np.clip(np.dot(principal_axis, [0.,0.,1.]), -1.0, 1.0)))
        vertical_angle = 90 - np.abs(vertical_angle - 90)

        return linearity, principal_axis, vertical_angle

    def _neighborhood_analysis(self, points, radius, linearity_thres, max_angle):

        pcd = to_open3d(points)
        linearity, principal_axis, vertical_angle = self._point_features(pcd, radius)
        candidate_mask = (linearity > linearity_thres) & (vertical_angle > (90-max_angle))

        return candidate_mask, principal_axis

    def _candidate_cable_points(self, points, voxel_size, radius, linearity_thres, max_angle):
        
        if voxel_size is not None:
            _, voxel_centers, inv_voxel_idx = self._voxelize(points, voxel_size)
            candidate_mask, principal_axis = self._neighborhood_analysis(voxel_centers, radius, linearity_thres, max_angle)
            
            # convert voxel features back to point features
            principal_axis = principal_axis[inv_voxel_idx]
            candidate_mask = candidate_mask[inv_voxel_idx]
        else:
            candidate_mask, principal_axis = self._neighborhood_analysis(points, radius, linearity_thres, max_angle)

        return candidate_mask, principal_axis

    def _grow_cables(self, points, cable_labels, principal_axis, radius):

        # Create KDTree
        unassigned_mask = cable_labels == -1
        kd_tree_unassigned = KDTree(points[unassigned_mask])
        kd_tree_cable = KDTree(points[~unassigned_mask])
        
        # Find neighbors of candidate points
        indices = kd_tree_cable.query_ball_tree(kd_tree_unassigned, r=radius)
        neighbors_n_parent = np.array([(j, i) for i in range(len(indices)) for j in indices[i]], dtype=int)
        if len(neighbors_n_parent) < 2:
            return cable_labels, principal_axis

        neighbors_idx = np.unique(neighbors_n_parent[:,0], return_index=True)[0]
        neighbors_idx = np.where(unassigned_mask)[0][neighbors_idx]
        
        # Assign candidate axis to non candidate neighbors
        nn_idx = kd_tree_cable.query(points[neighbors_idx], distance_upper_bound=radius)[1]
        grow_parent_idx = np.where(~unassigned_mask)[0][nn_idx]
        principal_axis[neighbors_idx] = principal_axis[grow_parent_idx]
        
        # Assign candidate labels to non candidate neighbors
        cable_labels[neighbors_idx] = cable_labels[grow_parent_idx]

        return cable_labels, principal_axis

    def _principal_vector(self, points):
        cov = to_open3d(points).compute_mean_and_covariance()[1]
        eig_val, eig_vec = np.linalg.eig(cov)
        return eig_vec[:, eig_val.argmax()]

    def _get_end_points(self, points, principal_axis):
        d_pts = np.dot(points[:,:2], principal_axis[:2])
        idx_a, idx_b = d_pts.argmin(), d_pts.argmax()
        return np.array([points[idx_a], points[idx_b]])

    def _cable_cluster_feature(self, points):

        # Compute cluster directions
        principal_axis = self._principal_vector(points) # questionable since clutter of points can influence..
        direction = np.abs(np.abs(np.degrees(np.arctan2(principal_axis[1],principal_axis[0]))) - 90)
        
        end_points = self._get_end_points(points, principal_axis)
        length = np.linalg.norm(end_points[0] - end_points[1])

        result = {
            'counts': len(points),
            'principal_axis': principal_axis,
            'dir': direction,
            'end_points': end_points,
            'length': length
        }
        return result

    def _cable_cluster_features(self, points, labels, exclude_labels=[]):
        cable_clusters = {}

        for label in np.unique(labels):
            if label not in exclude_labels:
                cl_points = points[labels==label]
                cable_clusters[label] = self._cable_cluster_feature(cl_points)

        return cable_clusters

    def _nearest_points(self, pts_a, pts_b):
        dist = cdist(pts_a, pts_b)
        idx_a, idx_b = np.unravel_index(np.argmin(dist), dist.shape)
        return idx_a, idx_b, dist.min()
        
    def _cluster_distance(self, cluster_a, cluster_b):
        a_end = cluster_a.get('end_points')
        b_end = cluster_b.get('end_points')
        return self._nearest_points(a_end, b_end)[2]

    def _cluster_merge_condition(self, points, ccl_dict, cl_labels, a, b, max_dist=4):
        a_end = ccl_dict[a].get('end_points')
        b_end = ccl_dict[b].get('end_points')
        idx_a, idx_b = self._nearest_points(a_end, b_end)[:2]
        O = a_end[idx_a]
        A = b_end[idx_b]

        # Option 1
        a_points_xy_axis = np.abs(np.dot((points[cl_labels==a,:2]-O[:2]), ccl_dict[a].get('principal_axis')[:2]))
        b = self._principal_vector(points[cl_labels==a][a_points_xy_axis < 1.5]) # must be unit vector

        # Option 2:
        # principal_axis of end point --> Not good for smaller radius...

        a = A - O
        a_1 = np.dot(a,b)*b
        a_2 = a-a_1 

        dir_dist = np.linalg.norm(a_1)
        offset_dist = np.linalg.norm(a_2)

        dist_bool = dir_dist < max_dist
        offset_bool = offset_dist < max(.2+ dir_dist * .1, .2)

        merge_bool = dist_bool and offset_bool

        return merge_bool, dir_dist, (O, A)

    def _catenary_merge(self, points, cl_labels, a, b, pt_a, pt_b, cable_width=.15):

        unmasked_idx = np.where(cl_labels<1)[0]
        merge_line = LineString([pt_a, pt_b])

        xy_clip_mask = poly_clip(points[(cl_labels==a)|(cl_labels==b),:2], merge_line.buffer(2, cap_style=3))
        fit_points = points[(cl_labels==a)|(cl_labels==b)][xy_clip_mask]
        principal_v = unit_vector((pt_b - pt_a)[:2])
        x_fit_points = np.dot(fit_points[:,:2], principal_v) 
        x_shift = np.min(x_fit_points)
        x_fit_points -= x_shift

        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                # Fit on cable
                popt, _ = curve_fit(catenary_func, x_fit_points, fit_points[:,2])

                # Evaluate fit on cable
                errors = abs(catenary_func(x_fit_points, *popt) - fit_points[:,2])
                fit_inliers = errors < self.cable_size
                fit_score = np.sum(fit_inliers) / len(fit_inliers)

                # Fit on gap
                xy_clip_mask = poly_clip(points[unmasked_idx,:2], merge_line.buffer(.1))
                gap_points = points[unmasked_idx[xy_clip_mask]]
                if len(gap_points) < 1:
                    return 0, 0, np.array([])
                x_gap_points = np.dot(gap_points[:,:2], principal_v) - x_shift

                # Evaluate fit on gap
                errors = abs(catenary_func(x_gap_points, *popt) - gap_points[:,2])
                gap_inliers = errors < cable_width/2
                gap_score = np.sum(gap_inliers) / len(gap_inliers)
                inlier_idx = unmasked_idx[xy_clip_mask][gap_inliers]

            except OptimizeWarning:
                # Do your other thing
                return 0, 0, None

        return fit_score, gap_score, inlier_idx

    def _box_merge(self, points, cl_labels, pt_a, pt_b, cable_width=.15):

        unmasked_idx = np.where(cl_labels<1)[0]

        merge_line = LineString([pt_a, pt_b]).buffer(cable_width)
        clip_mask = poly_box_clip(points[unmasked_idx], merge_line, bottom=np.min((pt_a[2], pt_b[2]))-cable_width, top=np.max((pt_a[2],pt_b[2]))+cable_width)

        merge_b = unit_vector(pt_b-pt_a)
        a = points[unmasked_idx[clip_mask]]-pt_a
        a_1 = np.dot(a, merge_b)[np.newaxis, :].T * merge_b
        a_2 = a-a_1
        dist = np.linalg.norm(a_2, axis=1)
        merge_idx = unmasked_idx[clip_mask][dist < cable_width/2]

        return merge_idx

    def _cable_merging(self, points, cl_labels, max_merge_angle=5):

        for cl in set(np.unique(cl_labels)).difference((-1,)):
            if np.sum(cl_labels==cl) < 4:
                cl_labels[cl_labels==cl] = -1

        ccl_dict = self._cable_cluster_features(points, cl_labels, [-1])
        cl_ordered = sorted(ccl_dict, key=lambda x: ccl_dict[x]["counts"], reverse=True)
        
        i = 0
        while len(cl_ordered) > i:
            main_cl = cl_ordered[i]
            candidate_cls = cl_ordered[i+1:]
            i += 1 

            search = True
            while search:
                search = False
                
                # Filter different direction
                direction = [ccl_dict[x].get('dir') for x in candidate_cls]
                angle_mask = np.abs(ccl_dict[main_cl].get('dir') - direction) < max_merge_angle
                selection_cls = np.asarray(candidate_cls)[angle_mask]

                if len(selection_cls) > 0:
                    # Filter range
                    ccls_dist = np.array([self._cluster_distance(ccl_dict[main_cl], ccl_dict[x]) for x in selection_cls])
                    ccls_ordered = ccls_dist.argsort()[:np.sum(ccls_dist<5)]
                    selection_cls = selection_cls[ccls_ordered]

                    for x in selection_cls:
                        valid_merge, dir_dist, (pt_a, pt_b) = self._cluster_merge_condition(points, ccl_dict, cl_labels, main_cl, x)
                        if valid_merge:
                            # Do your thing 
                            if dir_dist < 1:
                                inlier_mask = self._box_merge(points, cl_labels, pt_a, pt_b)
                            else:
                                fit_score, _, inlier_mask = self._catenary_merge(points, cl_labels, main_cl, x, pt_a, pt_b)
                                if fit_score < .8:
                                    continue
                            
                            # Assign new labels
                            cl_labels[inlier_mask] = main_cl
                            cl_labels[cl_labels==x] = main_cl

                            # Delete old cluster
                            ccl_dict.pop(x, None)
                            candidate_cls.remove(x)
                            cl_ordered.remove(x)

                            # Compute cluster features 
                            ccl_dict[main_cl] = self._cable_cluster_feature(points[cl_labels==main_cl])

                            search = True
                            break

        return cl_labels, ccl_dict

    def _detect_cables(self, points):
        '''Cable Detection'''

        # 1. Canndidate Points
        logger.debug(f'Selecting candidate cable points...')
        candidate_mask, principal_axis = self._candidate_cable_points(points, voxel_size=self.voxel_size, radius=self.neigh_radius, linearity_thres=self.linearity_thres, max_angle=self.max_v_angle)
        if np.sum(candidate_mask) == 0:
            return # TODO

        # 2. Clustering
        logger.debug('Clustering candidates...')
        cable_labels = np.full(len(points),-1)
        clustering = (DBSCAN(eps=self.neigh_radius, min_samples=1, p=2).fit(points[candidate_mask]))
        cable_labels[candidate_mask] = clustering.labels_

        # 3. Growing
        logger.info('Growing cables...')
        cable_labels, principal_axis = self._grow_cables(points, cable_labels, principal_axis, radius=self.grow_radius)

        # 4. Merging
        logger.info('Merging cables...')
        cable_labels, ccl_dict = self._cable_merging(points, cable_labels, max_merge_angle=self.max_merge_angle)
        
        # 5. Remove short cables
        short_clusters = [key for key in ccl_dict.keys() if ccl_dict[key]['length'] < self.min_segment_length]
        cable_labels[np.isin(cable_labels, short_clusters)] = -1

        return cable_labels

    def _cable_cut(self, points):
        """Create a new axis along the direction of a cable. Cable start is 0"""
        cable_dir = main_direction(points[:,:2])
        cable_dir_axis = np.dot(points[:,:2], cable_dir)
        cable_dir_axis -= cable_dir_axis.min()
        return cable_dir_axis

    def _linestring_cable_fit(self, points, cable_axis, binwidth_axis=.5):
        """
        Returns linetring fits for both z and xy projections.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        cable_axis : array of shape (n_points,)
            The cable directional axis values <d>.
        binwidth_z : float (default .75)
            The bindwithd used to calculate the statistic over.
        binwidth_axis : float (default .5)
            The bindwithd used to calculate the statistic over.

        Returns
        -------
        cable_zline : LineString
            linestring fit on Z axis projection.
        cable_axisline : LineString
            linestring fit on XY projection.
        """

        cable_max = cable_axis.max()

        # LineString fit XY projection
        bins = np.linspace(0 - (binwidth_axis/2), cable_max + (binwidth_axis/2), int(round(cable_max/binwidth_axis)+2))
        mean_x, _, _ = binned_statistic(cable_axis, points[:, 0], statistic='mean', bins=bins)
        mean_y, _, _ = binned_statistic(cable_axis, points[:, 1], statistic='mean', bins=bins)
        line_pts = np.vstack((mean_x, mean_y)).T
        line_pts = line_pts[~np.isnan(line_pts).any(axis=1)]
        cable_axisline = LineString(line_pts)

        return cable_axisline

    def _classify_tram_cables(self, points, cable_labels, tilecode):

        logger.debug('Classifying tram cables...')
        tram_cabel_labels = []

        # Load tramtracks
        tramtracks = self.bgt_tram_reader.filter_tile(
                    tilecode, padding=10)
        logger.debug(f'{len(tramtracks)} tramtracks in tile.')

        # Test cables
        if len(tramtracks) > 0:
            # AHN ground interpolation
            cable_mask = (cable_labels > -1)
            ground_z = self.ahn_reader.interpolate(
                        tilecode, points[cable_mask], cable_mask, 'ground_surface')

            tramtracks_polygon = unary_union(tramtracks).buffer(2.8+self.track_buffer).buffer(-2.8)

            # Check each cable for intersection with tramtrack
            for cl in set(np.unique(cable_labels[cable_mask])).difference((-1,)):
                # select points that belong to the cluster
                cl_mask = (cable_labels[cable_mask] == cl)

                target_z = ground_z[cl_mask]
                cl_pts = points[cable_mask][cl_mask]
                cable_axis = self._cable_cut(cl_pts)
                cable_axisline = self._linestring_cable_fit(cl_pts, cable_axis)
                
                # Rule based classification
                if tramtracks_polygon.intersects(cable_axisline.buffer(.5)):
                    cc_height = cl_pts[:, 2] - target_z
                    if cc_height.min() < self.max_tramcable_height:
                        tram_cabel_labels.append(cl)
        
        logger.debug(f'{len(tram_cabel_labels)} tram cables found.')

        return tram_cabel_labels

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
        logger.info('Cable and Suspended Streetlight fuser ' +
                    f'(cable label={self.cable_label}, streetlight label={self.streetlight_label}).')

        label_mask = np.copy(mask)

        # Vertical Segmentation
        vertical_seg_mask = self._vertical_segmentation(points[label_mask], tilecode)
        label_mask[label_mask] = vertical_seg_mask

        # BAG building removal
        bag_seg_mask = self._bag_removal(points[label_mask], tilecode, self.building_offset)
        label_mask[label_mask] = bag_seg_mask

        # Detect cables
        cable_labels = self._detect_cables(points[label_mask])
        # cable_labels.length == label_mask trues

        # Classify tram cables
        tram_cable_labels = self._classify_tram_cables(points[label_mask], cable_labels, tilecode)

        # Assign labels
        np.full(,CABLE)
        tramcable_mask = np.isin(cable_labels, tram_cable_labels) = TRAMCABLE
        nocable_mask = (cable_labels == -1) = NOISE



        new_mask_labels = np.copy(labels[mask])
        new_mask_labels[np.where(label_mask] = 

        logger.debug(f'{np.sum(labels[mask]!=new_mask_labels)} points labelled.')
        labels[mask] = new_mask_labels


        lanp.where(mask)[0][]

        return labels

def get_polygon_from_tile_code(tilecode, padding=0, width=50, height=50):
    bbox = get_bbox_from_tile_code(tilecode, padding, width, height)
    return Polygon([bbox[0],(bbox[0][0],bbox[1][1]), bbox[1],(bbox[1][0],bbox[0][1])])

def to_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def catenary_func(x, a, b, c):
    return a + c * np.cosh((x-b) / c)

def unit_vector(v1):
    """ Returns the unit vector of `v1`"""
    return v1 / np.linalg.norm(v1)

def main_direction(points):
    """ Returns the eigenvector corresponding to the largest eigenvalue of `points`"""
    cov = np.cov(points, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    dir_v = eig_vec[:,eig_val.argmax()]
    if dir_v[0] < 0:
        dir_v *= -1
    return dir_v