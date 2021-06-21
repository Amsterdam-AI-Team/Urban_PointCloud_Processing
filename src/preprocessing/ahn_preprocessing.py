"""
This module provides methods to pre-process AHN data. In particular, starting
from an AHN point cloud, there are methods to clip specific tiles from this
point cloud, and further methods to extract ground and building surfaces from
each tile that can be used for automatic labelling of street-level pointclouds.

For an example, see notebooks/1.AHN_preprocessing.ipynb
"""

import numpy as np
import os
import pathlib
import pylas
import re
from tqdm import tqdm

from ..utils.las_utils import get_bbox_from_tile_code
from ..utils.interpolation import SpatialInterpolator

# AHN classification codes (see https://www.ahn.nl/4-classificatie)
# AHN_ARTIFACT is 'Kunstwerk' which includes 'vlonders/steigers, bruggen en ook
# portaalmasten van autosnelwegen'
AHN_OTHER = 1
AHN_GROUND = 2
AHN_BUILDING = 6
AHN_WATER = 9
AHN_ARTIFACT = 26


def clip_ahn_las_tile(ahn_cloud, las_file, out_folder='', buffer=1):
    """
    Clip a tile from the AHN cloud to match the dimensions of a given
    CycloMedia LAS tile, and save the result using the same naming convention.

    Parameters
    ----------
    ahn_cloud : pylas point cloud
        The full AHN point cloud. This is assumed to include the full area of
        the given CycloMedia tile.

    las_file : Path or str
        The CycloMedia tile on which the clip should be based.

    out_folder : str, optional
        Output folder to which the clipped file should be saved. Defaults to
        the current folder.

    buffer : int, optional (default: 1)
        Buffer around the CycloMedia tile (in m) to include, used for further
        processing (e.g. interpolation).
    """
    if type(las_file) == str:
        las_file = pathlib.Path(las_file)
    tile_code = re.match(r'.*(\d{4}_\d{4}).*', las_file.name)[1]

    ((x_min, y_max), (x_max, y_min)) = get_bbox_from_tile_code(tile_code)
    x_min -= buffer
    x_max += buffer
    y_min -= buffer
    y_max += buffer

    clip_idx = np.where((x_min <= ahn_cloud.x) & (ahn_cloud.x <= x_max)
                        & (y_min <= ahn_cloud.y) & (ahn_cloud.y <= y_max))[0]

    ahn_tile = pylas.create(point_format_id=ahn_cloud.header.point_format_id)
    ahn_tile.points = ahn_cloud.points[clip_idx]

    if out_folder != '':
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    ahn_tile.write(os.path.join(out_folder, 'ahn_' + tile_code + '.laz'))


def clip_ahn_las_folder(ahn_cloud, in_folder, out_folder=None, buffer=1,
                        resume=False, hide_progress=False):
    """
    Clip a tiles from the AHN cloud to match all CycloMedia LAS tiles in a
    given folder, and save the result using the same naming convention.

    Parameters
    ----------
    ahn_cloud : pylas point cloud
        The full AHN point cloud. This is assumed to include the full area of
        the given CycloMedia tiles.

    in_folder : Path or str
        The input folder (containing the point cloud tiles.)

    out_folder : Path or str, optional
        The output folder. Defaults to the input folder.

    buffer : int, optional (default: 1)
        Buffer around the CycloMedia tile (in m) to include, used for further
        processing (e.g. interpolation).

    resume : bool (default: False)
        Whether to resume, i.e. skip existing files in the output folder. If
        set to False, existing files will be overwritten.

    hide_progress : bool (default: False)
        Hide the progress bar.
    """
    if not os.path.isdir(in_folder):
        print('The input path specified does not exist')
        return None

    if type(in_folder) == str:
        in_folder = pathlib.Path(in_folder)

    if out_folder is None:
        out_folder = in_folder

    if out_folder != in_folder:
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in in_folder.glob('filtered_*')
             if f.name.endswith(file_types)]

    if resume:
        # Find which files have already been processed.
        done = set(file.name[-13:-4]
                   for file in pathlib.Path(out_folder).glob('ahn_*.laz'))
        files = [f for f in files if f.name[-13:-4] not in done]

    for file in tqdm(files, unit="file", disable=hide_progress):
        clip_ahn_las_tile(ahn_cloud, file, out_folder=out_folder,
                          buffer=buffer)


def _get_ground_surface(ahn_las, grid_x, grid_y, n_neighbors=8, max_dist=1,
                        power=2., fill_value=np.nan):
    """
    Use inverse distance weighted interpolation (IDW) to generate a ground
    surface (grid) from a given AHN cloud.

    For more information on IDW see:
    utils.interpolation.SpatialInterpolator

    Parameters
    ----------
    ahn_las : pylas point cloud
        The AHN point cloud.

    grid_x : list of floats
        X-values for the interpolation grid.

    grid_y : list of floats
        Y-values for the interpolation grid.

    n_neighbours : int (default: 8)
        Maximum number of neighbours to use for IDW.

    max_dist : float (default: 1.0)
        Maximum distance of neighbours to consider for IDW.

    power : float (default: 2.0)
        Power to use for IDW.

    fill_value : float (default: np.nan)
        Fill value to use for 'empty' grid cells for which no interpolation
        could be computed.

    Returns
    -------
    2d array of interpolation values for each <y,x> grid cell.
    """
    mask = ahn_las.classification == AHN_GROUND

    if np.count_nonzero(mask) <= 1:
        return np.full(grid_x.shape, np.nan, dtype='float16')

    coordinates = np.vstack((ahn_las.x[mask], ahn_las.y[mask])).T
    values = ahn_las.z[mask]
    positions = np.vstack((grid_x.reshape(-1), grid_y.reshape(-1))).T

    idw = SpatialInterpolator(coordinates, values, method='idw')
    ahn_gnd_grid = idw(positions, n_neighbors=n_neighbors, max_dist=max_dist,
                       power=power, fill_value=fill_value)

    return (np.around(ahn_gnd_grid.reshape(grid_x.shape), decimals=2)
            .astype('float16'))


def _get_building_surface(ahn_las, grid_x, grid_y, n_neighbors=8, max_dist=0.5,
                          fill_value=np.nan):
    """
    Use maximum-based interpolation to generate a building surface (grid) from
    a given AHN cloud.

    For more information on maximum-based interpolation see:
    utils.interpolation.SpatialInterpolator

    Parameters
    ----------
    ahn_las : pylas point cloud
        The AHN point cloud.

    grid_x : list of floats
        X-values for the interpolation grid.

    grid_y : list of floats
        Y-values for the interpolation grid.

    n_neighbours : int (default: 8)
        Maximum number of neighbours to use for interpolation.

    max_dist : float (default: 0.5)
        Maximum distance of neighbours to consider for interpolation.

    fill_value : float (default: np.nan)
        Fill value to use for 'empty' grid cells for which no interpolation
        could be computed.

    Returns
    -------
    2d array of interpolation values for each <y,x> grid cell.
    """
    mask = ahn_las.classification == AHN_BUILDING

    if np.count_nonzero(mask) <= 1:
        return np.full(grid_x.shape, np.nan, dtype='float16')

    coordinates = np.vstack((ahn_las.x[mask], ahn_las.y[mask])).T
    values = ahn_las.z[mask]
    positions = np.vstack((grid_x.reshape(-1), grid_y.reshape(-1))).T

    idw = SpatialInterpolator(coordinates, values, method='max')
    ahn_bd_grid = idw(positions, n_neighbors=n_neighbors, max_dist=max_dist,
                      fill_value=fill_value)

    return (np.around(ahn_bd_grid.reshape(grid_x.shape), decimals=2)
            .astype('float16'))


def process_ahn_las_tile(ahn_las_file, out_folder='', resolution=0.1):
    """
    Generate ground and building surfaces (grids) for a given AHN point cloud.
    The results are saved as .npz using the same filename convention.

    Parameters
    ----------
    ahn_las_file : Path or str
        The AHN point cloud file.

    out_folder : Path or str, optional
        The output folder. Defaults to the current folder.

    resolution : float (default: 0.1)
        The resolution (in m) for the surface grids.

    Returns
    -------
    Path of the output file.
    """
    if type(ahn_las_file) == pathlib.PosixPath:
        ahn_las_file = ahn_las_file.as_posix()
    tile_code = re.match(r'.*(\d{4}_\d{4}).*', ahn_las_file)[1]

    ((x_min, y_max), (x_max, y_min)) = get_bbox_from_tile_code(tile_code)

    ahn_las = pylas.read(ahn_las_file)

    # Create a grid with 0.1m resolution
    grid_y, grid_x = np.mgrid[y_max-resolution/2:y_min:-resolution,
                              x_min+resolution/2:x_max:resolution]

    ground_surface = _get_ground_surface(ahn_las, grid_x, grid_y)
    building_surface = _get_building_surface(ahn_las, grid_x, grid_y)

    filename = os.path.join(out_folder, 'ahn_' + tile_code + '.npz')
    np.savez_compressed(filename,
                        x=grid_x[0, :],
                        y=grid_y[:, 0],
                        ground=ground_surface,
                        building=building_surface)
    return filename
