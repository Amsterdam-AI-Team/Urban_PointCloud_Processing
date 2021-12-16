"""Car Fuser"""

import numpy as np
import logging
from shapely.geometry import Polygon

from ..abstract_processor import AbstractProcessor
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.math_utils import minimum_bounding_rectangle
from ..utils.clip_utils import poly_box_clip
from ..labels import Labels

logger = logging.getLogger(__name__)


class CarFuser(AbstractProcessor):
    """
    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    ahn_reader : AHNReader object
        Elevation data reader.
    bgt_reader : BGTPolyReader object
        Used to load road part polygons.
    """

    def __init__(self, label, ahn_reader, bgt_reader,
                 grid_size=0.05, min_component_size=5000,
                 overlap_perc=20, params={}, exclude_types=["fietspad"]):
        super().__init__(label)
        self.ahn_reader = ahn_reader
        self.bgt_reader = bgt_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.overlap_perc = overlap_perc
        self.params = params
        self.exclude_types = exclude_types

    def _label_car_like_components(self, points, ground_z, point_components,
                                   road_polygons, min_height, max_height,
                                   min_width, max_width, min_length,
                                   max_length):
        """ Based on certain properties of a car we label clusters.  """

        car_mask = np.zeros(len(points), dtype=bool)
        car_count = 0

        cc_labels = np.unique(point_components)

        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = ground_z[cc_mask]
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                cc_z = np.mean(valid_values)
                min_z = cc_z + min_height
                max_z = cc_z + max_height
                cluster_height = np.amax(points[cc_mask][:, 2])
                if min_z <= cluster_height <= max_z:
                    mbrect, _, mbr_width, mbr_length, _ =\
                        minimum_bounding_rectangle(points[cc_mask][:, :2])
                    poly = np.vstack((mbrect, mbrect[0]))
                    if (min_width < mbr_width < max_width and
                            min_length < mbr_length < max_length):
                        p1 = Polygon(poly)
                        for road_polygon in road_polygons:
                            p2 = Polygon(road_polygon)
                            do_overlap = p1.intersects(p2)
                            if do_overlap:
                                intersection_perc = (p1.intersection(p2).area
                                                     / p1.area) * 100
                                if intersection_perc > self.overlap_perc:
                                    car_mask = car_mask | poly_box_clip(
                                        points, poly, bottom=cc_z, top=max_z)
                                    car_count += 1
                                    break

        logger.debug(f'{car_count} cars labelled.')

        return car_mask

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
        logger.info('Car fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        road_polygons = self.bgt_reader.filter_tile(
                    tilecode, exclude_types=self.exclude_types, merge=False)
        if len(road_polygons) == 0:
            logger.debug('No road parts found for tile, skipping.')
            return label_mask

        # Get the interpolated ground points of the tile
        ground_z = self.ahn_reader.interpolate(
                            tilecode, points[mask], mask, 'ground_surface')

        lcc = LabelConnectedComp(self.label, grid_size=self.grid_size,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask])

        # Label car like clusters
        car_mask = self._label_car_like_components(points[mask], ground_z,
                                                   point_components,
                                                   road_polygons,
                                                   **self.params)
        label_mask[mask] = car_mask

        return label_mask
