import numpy as np
from shapely.geometry import Polygon
import ast
import logging

from ..fusion.bgt_fuser import BGTFuser
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.math_utils import minimum_bounding_rectangle
from ..utils.las_utils import get_bbox_from_tile_code
from ..utils.clip_utils import poly_box_clip
from ..utils.labels import Labels

logger = logging.getLogger(__name__)


class CarFuser(BGTFuser):
    """
    Parameters
    ----------
    label : int
        Class label to use for this fuser.
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

    COLUMNS = ['bgt_name', 'polygon', 'x_min', 'y_max', 'x_max', 'y_min']

    def __init__(self, label, ahn_reader,
                 bgt_file=None, bgt_folder=None, file_prefix='bgt_roads',
                 octree_level=9, min_component_size=100, max_above_ground=3,
                 min_width_thresh=1.5, max_width_thresh=2.55,
                 min_length_thresh=2.0, max_length_thresh=7.0):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)

        self.ahn_reader = ahn_reader
        self.octree_level = octree_level
        self.min_component_size = min_component_size
        self.max_above_ground = max_above_ground
        self.min_width_thresh = min_width_thresh
        self.max_width_thresh = max_width_thresh
        self.min_length_thresh = min_length_thresh
        self.max_length_thresh = max_length_thresh

    def _filter_tile(self, tilecode):
        """
        Return a list of polygons representing each of the road segments found
        in the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) =\
            get_bbox_from_tile_code(tilecode)
        df = self.bgt_df.query('(x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        road_polygons = df['polygon'].apply(ast.literal_eval).tolist()

        return road_polygons

    def _fill_car_like_components(self, points, fast_z, point_components,
                                  road_polygons):
        """ Based on certain properties of a car we label clusters.  """

        car_mask = np.zeros(len(points), dtype=bool)
        car_count = 0

        cc_labels = np.unique(point_components)

        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = fast_z(points[cc_mask])
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                ground_z = np.mean(valid_values)
                max_z_thresh = ground_z + self.max_above_ground

                max_z = np.amax(points[cc_mask][:, 2])
                if max_z < max_z_thresh:
                    mbrect, _, mbr_width, mbr_length =\
                        minimum_bounding_rectangle(points[cc_mask][:, :2])
                    poly = np.vstack((mbrect, mbrect[0]))
                    if (self.min_width_thresh < mbr_width <
                            self.max_width_thresh and self.min_length_thresh <
                            mbr_length < self.max_length_thresh):
                        p1 = Polygon(poly)
                        for road_polygon in road_polygons:
                            p2 = Polygon(road_polygon)

                            do_overlap = p1.intersects(p2)
                            if do_overlap:
                                car_mask = car_mask | poly_box_clip(
                                    points, poly, bottom=ground_z, top=max_z)
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

        road_polygons = self._filter_tile(tilecode)
        if len(road_polygons) == 0:
            return label_mask

        # Get the interpolated ground points of the tile
        ahn_tile = self.ahn_reader.filter_tile(tilecode)
        fast_z = FastGridInterpolator(
                    ahn_tile['x'], ahn_tile['y'], ahn_tile['ground_surface'])

        # Create lcc object and perform lcc
        lcc = LabelConnectedComp(self.label, octree_level=self.octree_level,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask])

        # Label car like clusters
        car_mask = self._fill_car_like_components(points[mask], fast_z,
                                                  point_components,
                                                  road_polygons)
        label_mask[mask] = car_mask

        return label_mask
