import numpy as np
from shapely.geometry import Polygon
import ast
import logging

from ..fusion.bgt_fuser import BGTFuser
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.math_utils import minimum_bounding_rectangle
from ..utils.las_utils import get_bbox_from_tile_code
from ..labels import Labels

logger = logging.getLogger(__name__)


class BridgeFuser(BGTFuser):
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
                 grid_size=0.1, min_component_size=5000,
                 overlap_perc=20):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)

        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.overlap_perc = overlap_perc

    def _filter_tile(self, tilecode):
        """
        Return a list of polygons representing each of the road segments found
        in the area represented by the given CycloMedia tile-code.
        """
        ((bx_min, by_max), (bx_max, by_min)) =\
            get_bbox_from_tile_code(tilecode)
        df = self.bgt_df.query('(bgt_name != "parkeervlak") ' +
                               '& (x_min < @bx_max) & (x_max > @bx_min)' +
                               ' & (y_min < @by_max) & (y_max > @by_min)')
        road_polygons = df['polygon'].apply(ast.literal_eval).tolist() # TODO

        return road_polygons

    def _label_bridge_ground(self, points, point_components,
                             road_polygons):
        """ Based on AHN and BGT data we label clusters. TODO  """

        bridge_ground_mask = np.zeros(len(points), dtype=bool)
        bridge_part_count = 0

        cc_labels = np.unique(point_components)

        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = points[cc_mask] # TODO
            valid_values = target_z[np.isfinite(target_z)] # TODO

            if valid_values.size != 0:
                mbrect, _, mbr_width, mbr_length =\
                    minimum_bounding_rectangle(points[cc_mask][:, :2])
                poly = np.vstack((mbrect, mbrect[0]))
                p1 = Polygon(poly)
                for road_polygon in road_polygons:
                    p2 = Polygon(road_polygon)
                    do_overlap = p1.intersects(p2)
                    if do_overlap:
                        intersection_perc = (p1.intersection(p2).area
                                             / p1.area) * 100
                        if intersection_perc > self.overlap_perc:
                            bridge_ground_mask = bridge_ground_mask | cc_mask
                            bridge_part_count += 1
                            break
        logger.debug(f'{bridge_part_count} bridge part(s) labelled.')
        return bridge_ground_mask

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
        logger.info('Bridge part(s) fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        if mask is None:
            mask = np.ones((len(points),), dtype=bool)
        mask_ids = np.where(mask)[0]

        road_polygons = self._filter_tile(tilecode)

        label_mask = np.zeros((len(points),), dtype=bool)
        if len(road_polygons) == 0:
            # TODO print no road in this tile?
            return label_mask

        target_z = self.ahn_reader.interpolate(
            tilecode, points[mask, :], mask, 'bridge_surface')
        ahn_mask = (np.abs(points[mask_ids, 2] -
                    target_z) < 0.2)

        # If you dont want to get the complete bridge cluster
        return_ground = True
        if return_ground:
            label_mask[mask_ids[ahn_mask]] = True
            return label_mask

        lcc = LabelConnectedComp(self.label, grid_size=self.grid_size,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask_ids[ahn_mask]]) # TODO of gewoon mask gebruiken

        # Label bridge ground parts
        bridge_mask = self._label_bridge_ground(points[mask_ids[ahn_mask]],
                                                point_components,
                                                road_polygons)
        label_mask[mask_ids[ahn_mask]] = bridge_mask

        return label_mask
