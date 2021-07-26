import numpy as np
import os
from shapely.geometry import Polygon
from pathlib import Path
import pandas as pd
import ast

from .abstract import AbstractFuser
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.math_utils import minimum_bounding_rectangle
from ..utils.ahn_utils import load_ahn_tile
from ..utils.las_utils import get_bbox_from_tile_code


class CarFuser(AbstractFuser):
    def __init__(self, label, ahn_reader, octree_level=9,
                 min_component_size=100, max_above_ground=3,
                 bgt_file=None, bgt_folder=None, min_width_thresh=1.5,
                 max_width_thresh=2.55, min_length_thresh=2.0,
                 max_length_thresh=7.0):
        super().__init__(label)

        self.ahn_reader = ahn_reader
        self.octree_level = octree_level
        self.min_component_size = min_component_size
        self.max_above_ground = max_above_ground
        self.min_width_thresh = min_width_thresh
        self.max_width_thresh = max_width_thresh
        self.min_length_thresh = min_length_thresh
        self.max_length_thresh = max_length_thresh

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            print('No data folder or file specified. Aborting...')
            return None

    def _filter_tile(self, tilecode):
        """
        Returns an AHN tile dict for the area represented by the given
        CycloMedia tile-code. TODO also implement geotiff?
        """
        return load_ahn_tile(os.path.join(self.data_folder, 'ahn_' + tilecode
                                          + '.npz'))

    def _read_folder(self, path):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        file_match = "*.csv"
        frames = [pd.read_csv(file) for file in
                  path.glob(file_match)]
        self.bgt_df = pd.concat(frames)

    def _read_file(self, path):
        """
        Read the contents of a file. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        self.bgt_df = pd.read_csv(path)

    def _filter_road_area(self, bbox):
        """
        Return a list of polygons representing each of the road or parking
        spots found in the specified area.
        Parameters
        ----------
        bbox : tuple of tuples
            bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
        Returns
        -------
        dict
            Mapping building_id to a list of coordinates specifying the
            polygon.
        """
        ((bx_min, by_max), (bx_max, by_min)) = bbox
        df = self.bgt_df.query('(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')

        road_polygons = df['polygon'].apply(ast.literal_eval).tolist()

        return road_polygons

    def _fill_car_like_components(self, points, fast_z, point_components,
                                  road_polygons):
        """ Based on certain properties of a car we label clusters.  """

        car_mask = np.zeros(len(points), dtype=bool)

        cc_labels, counts = np.unique(point_components,
                                      return_counts=True)

        cc_labels_filtered = cc_labels[counts >= self.min_component_size]

        for cc in cc_labels_filtered:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = fast_z(points[cc_mask])
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                max_z_thresh = np.mean(valid_values) + self.max_above_ground

                max_z = np.amax(points[cc_mask][:, 2])
                if max_z < max_z_thresh:
                    _, hull_points, mbr_width, mbr_length =\
                        minimum_bounding_rectangle(
                            points[cc_mask][:, :2])

                    if (self.min_width_thresh < mbr_width <
                            self.max_width_thresh and self.min_length_thresh <
                            mbr_length < self.max_length_thresh):
                        # TODO use bounding rectangle instead?
                        p1 = Polygon(hull_points)
                        for road_polygon in road_polygons:
                            p2 = Polygon(road_polygon)

                            do_overlap = p1.intersects(p2)
                            if do_overlap:
                                car_mask[cc_mask] = True
                                break

        return car_mask

    def get_label_mask(self, tilecode, points, mask):
        """
        Returns the car mask for the given pointcloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        las_labels : array of shape (n_points, 1)
            All labels as int values

        Returns
        -------
        An array of shape (n_points,) with indices indicating which points
        should be labelled according to this fuser.
        """
        label_mask = np.zeros((len(points),), dtype=bool)

        bbox = get_bbox_from_tile_code(tilecode)  # TODO perform earlier, this is also performed in BGTBuildingFuser...

        road_polygons = self._filter_road_area(bbox)
        if not road_polygons:
            return label_mask

        # Get the interpolated ground points of the tile
        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        surface = ahn_tile['ground_surface']
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], surface)

        # Create lcc object and perform lcc
        lcc = LabelConnectedComp(self.label, octree_level=self.octree_level,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask])

        # Label car like clusters
        car_mask = self._fill_car_like_components(points[mask], fast_z,
                                                  point_components,
                                                  road_polygons)
        label_mask[mask] = car_mask

        print(f'Car fuser => processed (label={self.label}).')

        return label_mask
