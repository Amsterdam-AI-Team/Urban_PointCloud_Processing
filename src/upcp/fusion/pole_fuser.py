# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Pole Fuser"""

import numpy as np
import logging
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic_2d

from ..abstract_processor import AbstractProcessor
from ..utils import clip_utils
from ..utils.interpolation import FastGridInterpolator
from ..labels import Labels

logger = logging.getLogger(__name__)


class BGTPoleFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of pole (point) objects such as
    trees, street lights, and traffic signs using BGT data.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_type : str
        Specify the 'type' of point object: 'boom', 'lichtmast', or
        'verkeersbord'
    bgt_reader : BGTPointReader object
        Used to load pole points.
    ahn_reader : AHNReader object
        AHNReader to retrieve elevation data.
    padding : float (default: 1)
        Optional padding (in m) around the tile when searching for objects.
    params : dict
        Parameters specific to the bgt_type (see notes).

    Notes
    -----
    The `params` and their defaults are as follows:

    'search_pad': 1.5
        Specify the padding (in m) around the BGT object in which to search for
        a match in the point cloud.
    'max_dist': 1.2
        Maximum distance (in m) between the expected location and the location
        of a potential match.
    'voxel_res': 0.2
        Resolution of voxels used when searching for a match.
    'seed_height': 1.75
        Height above ground at which to determine object dimensions.
    'min_height': 2.
        Minimum hieght for an object to be considered a match.
    'max_r': 0.5
        Maximum radius for a pole-like object to be considered a match.
    'min_points': 500
        Minimum number of points for a cluster to be considered.
    'z_min': 0.2
        Height above ground, above which to search for objects.
    'z_max': 2.7
        Height above ground, below which to search for objects.
    'r_mult': 1.5
        Multiplier for radius when performing the initial (cylinder-based)
        labelling.
    'label_height': 4.
        Maximum height for initial (cylinder-based) labelling.
    """

    def __init__(self, label, bgt_type, bgt_reader, ahn_reader=None,
                 padding=0, params={}):
        super().__init__(label)
        self.bgt_type = bgt_type
        self.bgt_reader = bgt_reader
        if ahn_reader is None:
            logger.warning('No ahn_reader specified. Assuming elevation=0.')
        self.ahn_reader = ahn_reader
        if padding > 0:
            # TODO: positive padding does not work with elevation data.
            logger.warning('Positive padding not yet implemented.')
            padding = 0.
        self.padding = padding
        if 'r_mult' not in params:
            params['r_mult'] = 1.5
        if 'label_height' not in params:
            params['label_height'] = 4.
        self.params = params

    def _find_point_cluster(self, points, point, plane_height,
                            plane_buffer=0.1, search_radius=1, max_dist=0.1,
                            min_points=1, max_r=0.5):
        """
        Find a cluster in the point cloud that includes / is close to a
        specified target point. The cluster is returned as a tuple (X, Y,
        Radius).

        For a description of parameters see "Notes" above.
        """
        search_ids = np.where(clip_utils.cylinder_clip(
                                points, point, search_radius,
                                bottom=plane_height-plane_buffer,
                                top=plane_height+plane_buffer))[0]
        if len(search_ids) < min_points:
            return np.empty((0, 3))
        # Cluster the potential seed points.
        clustering = (DBSCAN(eps=0.05, min_samples=5, p=2)
                      .fit(points[search_ids]))
        # Remove noise points.
        noise_mask = clustering.labels_ != -1
        # Get cluster labels and sizes.
        cc_labels, counts = np.unique(clustering.labels_, return_counts=True)
        if min_points > 1:
            # Only keep clusters with size at least min_points.
            cc_labels = cc_labels[counts >= min_points]
            noise_mask = noise_mask & [label in set(cc_labels)
                                       for label in clustering.labels_]
        # Create a list of cluster centers (x,y) and radius r.
        c_xyr_list = []
        for cl in set(cc_labels).difference((-1,)):
            c_mask = clustering.labels_ == cl
            (cx, cy) = np.mean(points[search_ids[c_mask], 0:2], axis=0)
            cr = np.max(np.max(points[search_ids[c_mask], 0:2], axis=0)
                        - np.min(points[search_ids[c_mask], 0:2], axis=0)) / 2
            point_in_cl = ((point[0] - cx)**2 + (point[1] - cy)**2
                           < (cr + max_dist)**2)
            if cr <= max_r and point_in_cl:
                c_xyr_list.append([cx, cy, cr])
                # TODO for now we ignore possible additional clusters.
                break
        return c_xyr_list

    def _find_seeds_for_point_objects(self, points, point_objects, fast_z=None,
                                      search_pad=1.5, max_dist=1.2,
                                      voxel_res=0.2, seed_height=1.5,
                                      min_height=2, min_points=500,
                                      max_r=0.5, z_min=0.2, z_max=2.7,
                                      **kwargs):
        """
        Locate a cluster of seed points that most likely matches each target
        point. Seed clusters are returned as a list of tuples (X, Y, Radius).

        For a description of parameters see "Notes" above.
        """
        seeds = []
        matches = dict()
        for ind, obj in enumerate(point_objects):
            # Assume obj = [x, y].
            if fast_z is None:
                ground_z = 0.
            else:
                # Get the ground elevation.
                ground_z = fast_z(np.array([obj]))

            # Define the "box" within which to search for candidates.
            search_box = (obj[0]-search_pad, obj[1]-search_pad,
                          obj[0]+search_pad, obj[1]+search_pad)
            box_ids = np.where(clip_utils.box_clip(points, search_box,
                                                   bottom=ground_z+z_min,
                                                   top=ground_z+z_max))[0]
            if len(box_ids) == 0:
                # Empty search box, no match.
                matches[obj] = None
                continue

            # Voxelize the search box and compute statistics for each column.
            # TODO this voxelization only works when box width / height are
            # multiples of voxel_res.
            x_edge = np.arange(search_box[0], search_box[2] + 0.01, voxel_res)
            y_edge = np.arange(search_box[1], search_box[3] + 0.01, voxel_res)
            min_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='min')
            max_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='max')
            med_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='median')
            count_z_bin = binned_statistic_2d(
                points[box_ids, 0], points[box_ids, 1], points[box_ids, 2],
                bins=[x_edge, y_edge], statistic='count')
            # Column height (max - min).
            height = max_z_bin.statistic - min_z_bin.statistic
            midpoint = (min_z_bin.statistic + max_z_bin.statistic) / 2
            # Check if midpoint and median are close. This is a rough
            # approximation of checking whether the z-distribution is uniform
            # (i.e. a pole).
            med_mid = np.abs(med_z_bin.statistic - midpoint) < 0.2 * height

            # Find target locations where all criteria are met.
            x_loc, y_loc = np.where((height > min_height)
                                    & (count_z_bin.statistic > min_points)
                                    & med_mid)
            if len(x_loc) == 0:
                # No candidates found.
                matches[obj] = None
                continue
            candidates = np.stack((x_edge[x_loc] + voxel_res/2,
                                   y_edge[y_loc] + voxel_res/2)).T
            # Distances of candidates to target point.
            dist = [np.linalg.norm(np.array(obj) - np.array([c]))
                    for c in candidates]
            # Candidate with minimum distance.
            c_prime = candidates[np.argmin(dist), :]
            if min(dist) <= max_dist:
                # Find a matching cluster.
                clusters = self._find_point_cluster(
                    points, c_prime, seed_height, max_r=max_r)
                if len(clusters) > 0:
                    # We simply take the first one (usually there is only one).
                    # TODO we could return a 'correspondence' so it's clear
                    # which objects were located.
                    seed = clusters[0]
                    seeds.append(seed)
                    matches[obj] = (seed[0], seed[1])
                else:
                    # No cluster found.
                    matches[obj] = None
            else:
                # No candidates found.
                matches[obj] = None
        return seeds, matches

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
        logger.info(f'BGT [{self.bgt_type}] point fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        bgt_points = self.bgt_reader.filter_tile(
                                    tilecode, bgt_types=[self.bgt_type],
                                    padding=self.padding, return_types=False)
        if len(bgt_points) == 0:
            logger.debug(f'No {self.bgt_type} objects found in tile, ' +
                         ' skipping.')
            return label_mask

        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                      ahn_tile['ground_surface'])

        # Find seed point clusters.
        seeds, matches = self._find_seeds_for_point_objects(
                            points[mask], bgt_points, fast_z, **self.params)
        for seed in seeds:
            # Label a cylinder based on the seed cluster.
            top_height = (fast_z(np.array([seed[0:2]]))
                          + self.params['label_height'])
            clip_mask = clip_utils.cylinder_clip(
                                        points[mask], np.array(seed[0:2]),
                                        self.params['r_mult']*seed[2],
                                        top=top_height)
            label_mask[mask] = label_mask[mask] | clip_mask

        match_str = ', '.join([f'{obj}->{cand}'
                               for (obj, cand) in matches.items()])
        logger.debug(f'{len(seeds)}/{len(bgt_points)} objects labelled.')
        logger.debug(f'Matches for [{self.bgt_type}]: {match_str}')

        return label_mask
