"""BGT Data Fuser"""

import numpy as np
import pandas as pd
import ast
import os
import logging
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.stats import binned_statistic_2d
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from abc import ABC, abstractmethod

from ..abstract_processor import AbstractProcessor
from ..utils import clip_utils
from ..utils.interpolation import FastGridInterpolator
from ..utils.las_utils import get_bbox_from_tile_code
from ..labels import Labels
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.math_utils import minimum_bounding_rectangle, euclid_distance

logger = logging.getLogger(__name__)


class BGTFuser(AbstractProcessor, ABC):
    """
    Abstract class for automatic labelling of points using BGT data.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    file_prefix : str (default: '')
        Prefix used to load the correct files; only used with bgt_folder.
    """
    @property
    @classmethod
    @abstractmethod
    def COLUMNS(cls):
        return NotImplementedError

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix=''):
        if (bgt_file is None) and (bgt_folder is None):
            print("Provide either a bgt_file or bgt_folder to load.")
            return None
        if (bgt_file is not None) and (bgt_folder is not None):
            print("Provide either a bgt_file or bgt_folder to load, not both")
            return None
        if (bgt_folder is not None) and (not os.path.isdir(bgt_folder)):
            print('The data folder specified does not exist')
            return None
        if (bgt_file is not None) and (not os.path.isfile(bgt_file)):
            print('The data file specified does not exist')
            return None

        super().__init__(label)
        self.file_prefix = file_prefix
        self.bgt_df = pd.DataFrame(columns=type(self).COLUMNS)

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            logger.error('No data folder or file specified. Aborting...')
            return None

    def _read_folder(self, path):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        file_match = self.file_prefix + '*.csv'
        frames = [pd.read_csv(file, header=0, names=type(self).COLUMNS)
                  for file in path.glob(file_match)]
        if len(frames) == 0:
            logger.error(f'No data files found in {path.as_posix()}.')
            return
        self.bgt_df = pd.concat(frames)

    def _read_file(self, path):
        """
        Read the contents of a file. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        self.bgt_df = pd.read_csv(path, header=0, names=type(self).COLUMNS)

    @abstractmethod
    def _filter_tile(self, tilecode):
        """
        Returns data for the area represented by the given CycloMedia
        tile-code.
        """
        return NotImplementedError


class BGTBuildingFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of building points using BGT data
    in the form of footprint polygons. Data files are assumed to be in CSV
    format and contain six columns: [ID, polygon, x_min, y_max, x_max, y_min].

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Data files are
        assumed to be prefixed by "bgt_buildings", unless otherwise specified.
        Either a file or a folder should be provided, but not both.
    file_prefix : str (default: 'bgt_building')
        Prefix used to load the correct files; only used with bgt_folder.
    building_offset : int (default: 0)
        The footprint polygon will be extended by this amount (in meters).
    padding : float (default: 0)
        Optional padding (in m) around the tile when searching for objects.
    ahn_reader : AHNReader object
        Optional, if provided AHN data will be used to set a maximum height for
        each building polygon.
    ahn_eps : float (default: 0.2)
        Precision for the AHN elevation cut-off for buildings.
    """

    COLUMNS = ['BAG_ID', 'Polygon', 'x_min', 'y_max', 'x_max', 'y_min']

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix='bgt_buildings', building_offset=0, padding=0,
                 ahn_reader=None, ahn_eps=0.2):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.building_offset = building_offset
        self.padding = padding
        self.ahn_reader = ahn_reader
        self.ahn_eps = ahn_eps

        # TODO will this speedup the process?
        self.bgt_df.sort_values(by=['x_max', 'y_min'], inplace=True)

    def _filter_tile(self, tilecode, merge=True):
        """
        Return a list of polygons representing each of the buildings found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode, padding=self.padding)
        df = self.bgt_df.query('(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        buildings = [ast.literal_eval(poly) for poly in df.Polygon.values]
        if len(buildings) > 1 and merge:
            poly_offset = list(cascaded_union(
                                [Polygon(bld).buffer(self.building_offset)
                                 for bld in buildings]))
        else:
            poly_offset = [Polygon(bld).buffer(self.building_offset)
                           for bld in buildings]
        poly_valid = [poly.exterior.coords for poly in poly_offset
                      if len(poly.exterior.coords) > 1]
        return poly_valid

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
        logger.info('BGT building fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        building_polygons = self._filter_tile(tilecode)
        if len(building_polygons) == 0:
            logger.debug('No buildings found in reference csv file.')
            return label_mask

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]

        building_mask = np.zeros((len(mask_ids),), dtype=bool)
        for polygon in building_polygons:
            # TODO if there are multiple buildings we could mask the points
            # iteratively to ignore points already labelled.
            clip_mask = clip_utils.poly_clip(points[mask, :], polygon)
            building_mask = building_mask | clip_mask

        if self.ahn_reader is not None:
            bld_z = self.ahn_reader.interpolate(
                tilecode, points[mask, :], mask, 'building_surface')
            bld_z_valid = np.isfinite(bld_z)
            ahn_mask = (points[mask_ids[bld_z_valid], 2]
                        <= bld_z[bld_z_valid] + self.ahn_eps)
            building_mask[bld_z_valid] = building_mask[bld_z_valid] & ahn_mask

        logger.debug(f'{len(building_polygons)} building polygons labelled.')

        label_mask[mask_ids[building_mask]] = True

        return label_mask


class BGTPoleFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of pole (point) objects such as
    trees, street lights, and traffic signs using BGT data. Data files are
    assumed to be in CSV format and contain three columns: [Object type, X, Y].

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_type : str
        Specify the 'type' of point object: 'boom', 'lichtmast', or
        'verkeersbord'
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Data files are
        assumed to be prefixed by "bgt_points", unless otherwise specified.
        Either a file or a folder should be provided, but not both.
    file_prefix : str (default: 'bgt_points')
        Prefix used to load the correct files; only used with bgt_folder.
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
    COLUMNS = ['Type', 'X', 'Y']

    def __init__(self, label, bgt_type, bgt_file=None,
                 bgt_folder=None, file_prefix='bgt_points', ahn_reader=None,
                 padding=0, params={}):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.bgt_type = bgt_type
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

        # TODO will this speedup the process?
        self.bgt_df.sort_values(by=['X', 'Y'], ignore_index=True, inplace=True)

    def _filter_tile(self, tilecode):
        """
        Return a list of points representing each of the objects found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode, padding=self.padding)
        df = self.bgt_df.query('(X <= @bx_max) & (X >= @bx_min)' +
                               ' & (Y <= @by_max) & (Y >= @by_min)')
        bgt_points = list(df.to_records(index=False))

        return [(x, y) for (t, x, y) in bgt_points if t == self.bgt_type]

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

        bgt_points = self._filter_tile(tilecode)
        if len(bgt_points) == 0:
            logger.debug(f'No {self.bgt_type} objects in reference csv file.')
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
        logger.debug('Matches for [{self.bgt_type}]: ' + match_str)

        return label_mask


class BGTStreetFurnitureFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of street furniture (point)
    objects such as trash cans and city benches using BGT data.
    Data files are assumed to be in CSV format and contain three columns:
    [Object type, X, Y].

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_type : str
        Specify the 'type' of point object: 'bank', 'afval_apart_plaats', or
        'afvalbak'
    ahn_reader : AHNReader object
        Elevation data reader.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Data files are
        assumed to be prefixed by "bgt_roads", unless otherwise specified.
        Either a file or a folder should be provided, but not both.
    file_prefix : str (default: 'bgt_roads')
        Prefix used to load the correct files; only used with bgt_folder.
    """

    COLUMNS = ['Type', 'X', 'Y']

    def __init__(self, label, bgt_type, ahn_reader,
                 bgt_file=None, bgt_folder=None,
                 file_prefix='bgt_street_furniture',
                 grid_size=0.05, min_component_size=1500,
                 padding=0, max_dist=1., params={}):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.bgt_type = bgt_type
        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.padding = padding
        self.max_dist = max_dist
        self.params = params

    def _filter_tile(self, tilecode):
        """
        Return a list of points representing each of the objects found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode, padding=self.padding)
        df = self.bgt_df.query('(X <= @bx_max) & (X >= @bx_min)' +
                               ' & (Y <= @by_max) & (Y >= @by_min)')
        bgt_points = list(df.to_records(index=False))

        return [(x, y) for (t, x, y) in bgt_points if t == self.bgt_type]

    def _label_street_furniture_like_components(self, points, ground_z,
                                                point_components, bgt_points):
        """
        Based on certain properties of street furniture objects we label
        clusters.
        """

        street_furniture_mask = np.zeros(len(points), dtype=bool)
        object_count = 0

        cc_labels = np.unique(point_components)

        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = ground_z[cc_mask]
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                cc_z = np.mean(valid_values)
                min_z = cc_z + self.min_height
                max_z = cc_z + self.max_height
                cluster_height = np.amax(points[cc_mask][:, 2])
                if min_z <= cluster_height <= max_z:
                    mbrect, _, mbr_width, mbr_length, center_point =\
                        minimum_bounding_rectangle(points[cc_mask][:, :2])
                    if (self.min_width < mbr_width < self.max_width and
                            self.min_length < mbr_length < self.max_length):
                        for bgt_point in bgt_points:
                            dist = euclid_distance(bgt_point, center_point)
                            if dist <= self.max_dist:
                                street_furniture_mask[cc_mask] = True
                                object_count += 1
                                break

        logger.debug(f'{object_count} {self.bgt_type} objects labelled.')
        return street_furniture_mask

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
        logger.info('Street furniture fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        bgt_points = self._filter_tile(tilecode)
        if len(bgt_points) == 0:
            logger.debug(f'No {self.bgt_type} objects found in reference ' +
                         'csv file.')
            return label_mask

        # Get the interpolated ground points of the tile
        ground_z = self.ahn_reader.interpolate(
                            tilecode, points[mask], mask, 'ground_surface')

        lcc = LabelConnectedComp(self.label, grid_size=self.grid_size,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask])

        # Label street_furniture like clusters
        street_furniture_mask = (self._label_street_furniture_like_components(
                                 points[mask], ground_z, point_components,
                                 bgt_points))
        label_mask[mask] = street_furniture_mask

        return label_mask
