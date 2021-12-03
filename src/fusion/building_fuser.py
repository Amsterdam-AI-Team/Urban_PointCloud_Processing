"""BGT Data Fuser"""

import numpy as np
import ast
import logging
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..utils import clip_utils
from ..utils.las_utils import get_bbox_from_tile_code
from ..labels import Labels
from ..fusion.bgt_fuser import BGTFuser

logger = logging.getLogger(__name__)


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
            poly_offset = list(unary_union(
                                [Polygon(bld).buffer(self.building_offset)
                                 for bld in buildings]).geoms)
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
