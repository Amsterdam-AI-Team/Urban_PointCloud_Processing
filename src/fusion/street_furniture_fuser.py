import numpy as np
import logging

from ..fusion.bgt_fuser import BGTFuser
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.math_utils import minimum_bounding_rectangle, euclid_distance
from ..utils.las_utils import get_bbox_from_tile_code
from ..utils.labels import Labels

logger = logging.getLogger(__name__)


class StreetFurnitureFuser(BGTFuser):
    """
    Data Fuser class for automatic labelling of street furniture (point)
    objects such as trash bins and city benches using BGT data.
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
                 file_prefix='bgt_steet_furniture',
                 grid_size=0.1, min_component_size=5000,
                 min_height=1.2, max_height=2.2,
                 min_width=1.4, max_width=2.2,
                 min_length=3.0, max_length=6.0,
                 padding=0, max_dist=1):
        super().__init__(label, bgt_file, bgt_folder, file_prefix)
        self.bgt_type = bgt_type
        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width
        self.min_length = min_length
        self.max_length = max_length
        self.padding = padding

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
