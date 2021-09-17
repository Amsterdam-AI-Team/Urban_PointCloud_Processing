import numpy as np
import logging
import os
import pathlib
from tqdm import tqdm

from ..region_growing import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.labels import Labels
from ..utils import las_utils

logger = logging.getLogger(__name__)


def get_label_stats(labels):
    """Returns a string describing statistics based on labels."""
    N = len(labels)
    labels, counts = np.unique(labels, return_counts=True)
    stats = f'Total: {N:25} points\n'
    for label, cnt in zip(labels, counts):
        name = Labels.get_str(label)
        perc = (float(cnt) / N) * 100
        stats += f'Class {label:2}, {name:14} ' +\
                 f'{cnt:7} points ({perc:4.1f} %)\n'
    return stats


def get_pole_dims(points, fast_z, decimals=2):
    """Returns the dimensions of a pole-like object."""
    x, y = np.median(points[:, 0:2], axis=0)
    ground_z = fast_z(np.array([[x, y]]))[0]
    if np.isnan(ground_z):
        z_vals = fast_z(points)
        if np.isnan(z_vals).all():
            logger.warn(f'Missing AHN data for point ({x}, {y}).')
        else:
            ground_z = np.nanmean(z_vals)
        if np.isnan(ground_z):
            ground_z = np.min(points[:, 2])
    height = np.max(points[:, 2]) - ground_z

    return (round(x, decimals), round(y, decimals),
            round(ground_z, decimals), round(height, decimals))


def get_pole_locations(points, labels, target_label, fast_z,
                       min_component_size=100, octree_level=5,
                       return_counts=False):
    """
    Returns a list of locations and dimensions of pole-like objects
    corresponding to the target_label in a given point cloud.

    Parameters
    ----------
    points : array with shape (n_points, 3)
        The point cloud.
    labels : array of shape (n_points,)
        The corresponding labels.
    target_label : int
        The label of the target class.
    fast_z : FastGridInterpolator
        Interpolator for the ground surface.
    min_component_size : int (default: 100)
        Minimum size of a component to be considered.
    octree_level : int (default: 6)
        Octree level for the LabelConnectedComp algorithm.
    return_counts : bool (default: False)
        Whether to return the number of points per object.

    Returns
    -------
    A list of tuples, one for each pole-like object: (x, y, z, height).
    """
    pole_locations = []
    mask = labels == target_label

    if np.count_nonzero(mask) > 0:
        point_components = (LabelConnectedComp(
                                octree_level=octree_level,
                                min_component_size=min_component_size)
                            .get_components(points[mask]))

        cc_labels, counts = np.unique(point_components, return_counts=True)
        cc_labels = cc_labels[(cc_labels != -1)
                              & (counts >= min_component_size)]

        logger.info(f'{len(cc_labels)} objects of class ' +
                    f'[{Labels.get_str(target_label)}] found.')

        for cc in cc_labels:
            cc_mask = (point_components == cc)
            logger.debug(f'Cluster {cc}: {np.count_nonzero(cc_mask)} points.')
            dims = get_pole_dims(points[mask][cc_mask], fast_z)
            if return_counts:
                dims = (*dims, np.count_nonzero(cc_mask))
            pole_locations.append(dims)
    return pole_locations


def get_pole_locations_pred(cloud_pc_folder, pred_pc_folder,
                            target_label, ahn_reader,
                            cloud_prefix='filtered', pred_prefix='pred',
                            min_component_size=100, return_counts=False,
                            hide_progress=False):

    locations = []

    cloud_files = list(pathlib.Path(cloud_pc_folder)
                       .glob(cloud_prefix + "_*.laz"))
    pred_files = list(pathlib.Path(pred_pc_folder)
                      .glob(pred_prefix + "_*.laz"))
    tilecodes = set([las_utils.get_tilecode_from_filename(f.name)
                     for f in cloud_files])
    tilecodes = list(tilecodes.intersection(set(
        [las_utils.get_tilecode_from_filename(f.name) for f in pred_files])))
    files_tqdm = tqdm(tilecodes, unit="file", disable=hide_progress,
                      smoothing=0)
    logger.debug(f'{len(tilecodes)} files found.')

    for tilecode in files_tqdm:
        files_tqdm.set_postfix_str(tilecode)
        logger.info(f'Processing tile {tilecode}...')

        pointcloud_pred = las_utils.read_las(os.path.join(
            pred_pc_folder, pred_prefix + '_' + tilecode + '.laz'))
        labels = pointcloud_pred.label

        if np.count_nonzero(labels == target_label) > 0:
            if ((cloud_pc_folder == pred_pc_folder)
                    and (cloud_prefix == pred_prefix)):
                pointcloud = pointcloud_pred
            else:
                pointcloud = las_utils.read_las(os.path.join(
                    cloud_pc_folder, cloud_prefix + '_' + tilecode + '.laz'))
            points = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z)).T
            ahn_tile = ahn_reader.filter_tile(tilecode)
            fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                          ahn_tile['ground_surface'])
            tile_locations = get_pole_locations(points, labels, target_label,
                                                fast_z, min_component_size,
                                                return_counts=return_counts)
            locations.extend([(*x, tilecode) for x in tile_locations])

    return locations
