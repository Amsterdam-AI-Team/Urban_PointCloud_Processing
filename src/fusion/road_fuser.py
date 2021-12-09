"""BGT Data Fuser"""

import numpy as np
import ast
import logging
from shapely.geometry import Polygon

from ..utils import clip_utils
from ..utils.las_utils import get_bbox_from_tile_code
from ..labels import Labels
from ..fusion.bgt_fuser import BGTFuser

logger = logging.getLogger(__name__)


class BGTRoadFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of road points using BGT data
    in the form of polygons. Data files are assumed to be in CSV format and
    contain six columns: [bgt_name, polygon, x_min, y_max, x_max, y_min].

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
    road_offset : int (default: 0)
        The road polygon will be extended by this amount (in meters).
    padding : float (default: 0)
        Optional padding (in m) around the tile when searching for objects.
    """

    COLUMNS = ['bgt_name', 'Polygon', 'x_min', 'y_max', 'x_max', 'y_min']

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix='bgt_roads', road_offset=0, padding=0):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.road_offset = road_offset
        self.padding = padding

        # TODO will this speedup the process?
        self.bgt_df.sort_values(by=['x_max', 'y_min'], inplace=True)

    def _filter_tile(self, tilecode):
        """
        Return a list of polygons representing each of the roads found in
        the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode, padding=self.padding)
        df = self.bgt_df.query('(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        roads = [ast.literal_eval(poly) for poly in df.Polygon.values]

        poly_offset = [Polygon(rds).buffer(self.road_offset)
                       for rds in roads]
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
        logger.info('BGT road fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros(len(points), dtype=bool)

        road_polygons = self._filter_tile(tilecode)
        if len(road_polygons) == 0:
            logger.debug('No road parts found in reference csv file.')
            return label_mask

        # Already labelled ground points can be labelled as road.
        mask = np.ones((len(points),), dtype=bool)
        mask = mask & (labels == Labels.GROUND)
        mask_ids = np.where(mask)[0]

        road_mask = np.zeros((len(mask_ids),), dtype=bool)
        for polygon in road_polygons:
            clip_mask = clip_utils.poly_clip(points[mask, :], polygon)
            road_mask = road_mask | clip_mask

        logger.debug(f'{len(road_polygons)} road polygons labelled.')

        label_mask[mask_ids[road_mask]] = True

        return label_mask
