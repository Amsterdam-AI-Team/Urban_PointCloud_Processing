import numpy as np
import glob
import os
import pylas

from ..utils.labels import Labels


def get_bbox_from_las_file(laz_file):
    """
    Get the <X,Y> bounding box for a given CycloMedia laz file, based on the
    filename.

    Parameters
    ----------
    laz_file : str
        the .laz filename, e.g. filtered_2386_9702.laz

    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """

    filename_split = laz_file.split('.')[-2].split('_')

    # The file name of each file is structured as
    # "filtered_[X-coordinaat/50]_[Y-coordinaat/50]"
    x_min = int(filename_split[-2]) * 50
    y_min = int(filename_split[-1]) * 50

    return ((x_min, y_min + 50), (x_min + 50, y_min))


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


def save_las(mask_dict, las, laz_file, outfile=None, filter_noise=False):
    # TODO docstring
    if outfile is None:
        # Update .laz file in place.
        las.points['raw_classification'][mask_dict['ground_mask']] \
            = Labels.GROUND
        las.points['raw_classification'][mask_dict['building_mask']] \
            = Labels.BUILDING
        if filter_noise:
            las.points['raw_classification'][mask_dict['noise_mask']] \
                = Labels.NOISE
        las.write(laz_file)
    else:
        # Make a copy.
        las_labelled = pylas.create(point_format_id=3)
        las_labelled.header = las.header
        las_labelled.points = las.points
        las_labelled.points['raw_classification'][mask_dict['ground_mask']] \
            = Labels.GROUND
        las_labelled.points['raw_classification'][mask_dict['building_mask']] \
            = Labels.BUILDING
        if filter_noise:
            las_labelled.points['raw_classification'][mask_dict['noise_mask']]\
                = Labels.NOISE
        las_labelled.write(outfile)


def label_and_save_las(las, labels, outfile):
    """Label a las file using the provided class labels and save to outfile."""
    assert len(labels) == len(las.classification)
    las.points['raw_classification'] = labels
    las.write(outfile)


def convert_pcd(las):
    """
    Stick the las coords together in a nx3 numpy array and convert it to an
    Open3D pcd format.
    """
    import open3d as o3d

    coords = np.vstack((las.x, las.y, las.z)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    return pcd
