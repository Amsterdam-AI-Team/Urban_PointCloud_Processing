"""BGT Data Fuser"""

import ast
import pandas as pd
from pathlib import Path
import numpy as np

from .abstract import AbstractFuser
from ..utils.clip_utils import poly_offset, poly_clip
from ..utils.las_utils import get_bbox_from_tile_code


class BGTBuildingFuser(AbstractFuser):
    """Convenience class for loading building polygons from Dutch BGT data."""

    def __init__(self, label, building_offset, bgt_file=None, bgt_folder=None):
        if (bgt_file is None) and (bgt_folder is None):
            print("Provide either a bgt_file or bgt_folder to load.")

        super().__init__(label)

        self.bgt_df = (pd.DataFrame(columns=['pand_id', 'pand_polygon',
                       'x_min', 'y_max', 'x_max', 'y_min'])
                       .set_index('pand_id'))

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            print('No data folder or file specified. Aborting...')
            return None

        # TODO will this speedup the process
        self.bgt_df.sort_values(by=['x_max', 'y_min'], inplace=True)

        self.building_offset = building_offset

    def _read_folder(self, path):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        file_match = "*.csv"
        frames = [pd.read_csv(file).set_index('pand_id') for file in
                  path.glob(file_match)]
        self.bgt_df = pd.concat(frames)

    def _read_file(self, path):
        """
        Read the contents of a file. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        self.bgt_df = pd.read_csv(path).set_index('pand_id')

    def _filter_building_area(self, bbox):
        """
        Return a list of polygons representing each of the buildings found in
        the specified area.

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
        building_polygons = {pand_id: ast.literal_eval(poly) for pand_id,
                             poly in zip(df.index, df.pand_polygon)}

        return building_polygons

    def get_label_mask(self, tilecode, points, mask, labels):
        """
        Returns the building mask for the given pointcloud.

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
        An array of shape (n_points,) with indices indicating which points
        should be labelled according to this fuser.
        """
        bbox = get_bbox_from_tile_code(tilecode)

        building_polygons = self._filter_building_area(bbox)

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)

        building_mask = np.zeros((np.count_nonzero(mask),), dtype=bool)
        for _, polygon in building_polygons.items():
            # TODO if there are multiple buildings we could mask the points
            # iteratively to ignore points already labelled.
            building_with_offset = poly_offset(polygon, self.building_offset)
            building_points = poly_clip(points[mask, :], building_with_offset)
            building_mask = building_mask | building_points

        mask_indices = np.where(mask)[0]
        label_mask = np.zeros(len(points), dtype=bool)
        label_mask[mask_indices[building_mask]] = True

        return label_mask
