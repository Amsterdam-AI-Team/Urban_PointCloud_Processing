# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

import numpy as np
import glob
import pathlib
import re
import os
import laspy
from shapely.geometry import Polygon


def get_tilecode_from_filename(filename):
    """Extract the tile code from a file name."""
    return re.match(r'.*(\d{4}_\d{4}).*', filename)[1]


def get_tilecodes_from_folder(las_folder, las_prefix=''):
    """Get a set of unique tilecodes for the LAS files in a given folder."""
    files = pathlib.Path(las_folder).glob(f'{las_prefix}*.laz')
    tilecodes = set([get_tilecode_from_filename(file.name) for file in files])
    return tilecodes


def get_bbox_from_tile_code(tile_code, padding=0, width=50, height=50):
    """
    Get the <X,Y> bounding box for a given tile code. The tile code is assumed
    to represent the lower left corner of the tile.

    Parameters
    ----------
    tile_code : str
        The tile code, e.g. 2386_9702.
    padding : float
        Optional padding (in m) by which the bounding box will be extended.
    width : int (default: 50)
        The width of the tile.
    height : int (default: 50)
        The height of the tile.

    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    tile_split = tile_code.split('_')

    # The tile code of each tile is defined as
    # 'X-coordinaat/50'_'Y-coordinaat/50'
    x_min = int(tile_split[0]) * 50
    y_min = int(tile_split[1]) * 50

    return ((x_min - padding, y_min + height + padding),
            (x_min + height + padding, y_min - padding))

def get_polygon_from_tile_code(tilecode, padding=0, width=50, height=50):

    bbox = get_bbox_from_tile_code(tilecode, padding, width, height)
    tile_polygon = Polygon([bbox[0],(bbox[0][0],bbox[1][1]), bbox[1],(bbox[1][0],bbox[0][1])])

    return tile_polygon


def get_bbox_from_las_file(laz_file, padding=0):
    """
    Get the <X,Y> bounding box for a given CycloMedia laz file, based on the
    filename.

    Parameters
    ----------
    laz_file : Path or str
        the .laz filename, e.g. filtered_2386_9702.laz
    padding : float
        Optional padding (in m) by which the bounding box will be extended.

    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    if type(laz_file) == str:
        laz_file = pathlib.Path(laz_file)
    tile_code = get_tilecode_from_filename(laz_file.name)

    return get_bbox_from_tile_code(tile_code, padding=padding)


def get_bbox_from_las_folder(folder_path, padding=0):
    """
    Get the <X,Y> bounding box for a given folder of CycloMedia LAS files.

    Parameters
    ----------
    folder_path : str or Path
        The folder name.
    padding : int (default: 0)
        Optional padding (in meters).

    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    x_min = y_min = 1e6
    x_max = y_max = 0
    file_types = ('.LAS', '.las', '.LAZ', '.laz')

    for file in [f for f in glob.glob(os.path.join(folder_path, '*'))
                 if f.endswith(file_types)]:
        bbox = get_bbox_from_las_file(file)
        x_min = min(x_min, bbox[0][0])
        x_max = max(x_max, bbox[1][0])
        y_min = min(y_min, bbox[1][1])
        y_max = max(y_max, bbox[0][1])

    return ((x_min-padding, y_max+padding), (x_max+padding, y_min-padding))


def read_las(las_file):
    """Read a las file and return the las object."""
    return laspy.read(las_file)


def label_and_save_las(las, labels, outfile):
    """Label a las file using the provided class labels and save to outfile."""
    assert len(labels) == las.header.point_count
    if 'label' not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="label", type="uint8",
                          description="Labels"))
    las.label = labels
    las.write(outfile)


def create_pole_las(outfile, point_objects, labels=0, z_step=0.1):
    """
    Create a LAS file based on a set of given point objects. The LAS file will
    contain columns of points visualising the given objects.

    Parameters
    ----------
    outfile : str
        Path to output file.
    point_objects : list
        Each entry represents one point object: (x, y, z, height) or
        (bottom, top) as ((x, y, z), (x, y, z))
    labels : int or list of integers (optional)
        Either provide one label for all point objects, or a list of labels
        (one for each object).
    z_step : float (default: 0.1)
        Resolution (step size) of the output columns in the z axis.
    """
    points = np.empty((0, 3))
    point_labels = []
    for i, obj in enumerate(point_objects):
        if type(obj[1]) == int:
            loc = obj[0]
            height = obj[1]
            steps = int(height / z_step)
            obj_points = [[loc[0], loc[1], z]
                          for z in np.linspace(loc[2], loc[2]+height, steps)]
        else:
            bottom = obj[0]
            top = obj[1]
            steps = int((top[2] - bottom[2]) / z_step)
            xs = np.linspace(bottom[0], top[0], steps)
            ys = np.linspace(bottom[1], top[1], steps)
            zs = np.linspace(bottom[2], top[2], steps)
            obj_points = np.stack((xs, ys, zs), axis=1)
        points = np.vstack((points, obj_points))
        if isinstance(labels, int):
            obj_label = labels
        else:
            obj_label = labels[i]
        point_labels.extend([obj_label]*len(obj_points))

    las = laspy.create(file_version="1.2", point_format=3)
    las.header.offsets = np.min(points, axis=0)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.add_extra_dim(laspy.ExtraBytesParams(name="label", type="uint8",
                                             description="Labels"))
    las.label = point_labels
    las.write(outfile)
