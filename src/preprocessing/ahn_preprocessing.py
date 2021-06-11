import numpy as np
import os
import glob
import pylas
from tqdm import tqdm
from pathlib import Path

from ..utils.las_utils import get_bbox_from_las_file
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
    tile_code = las_file[-13:-4]
    ((x_min, y_max), (x_max, y_min)) = get_bbox_from_las_file(las_file)
    x_min -= buffer
    x_max += buffer
    y_min -= buffer
    y_max += buffer

    clip_idx = np.where((x_min <= ahn_cloud.x) & (ahn_cloud.x <= x_max)
                        & (y_min <= ahn_cloud.y) & (ahn_cloud.y <= y_max))[0]

    ahn_tile = pylas.create(point_format_id=ahn_cloud.header.point_format_id)
    ahn_tile.points = ahn_cloud.points[clip_idx]
    ahn_tile.write(os.path.join(out_folder, 'ahn_' + tile_code + '.laz'))


def _get_ground_surface(ahn_las, grid_x, grid_y, n_neighbors=8, max_dist=1,
                        power=2., fill_value=np.nan):
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
    tile_code = ahn_las_file[-13:-4]
    ((x_min, y_max), (x_max, y_min)) = get_bbox_from_las_file(ahn_las_file)

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


def process_folder(in_folder, out_folder=None, resolution=0.1, resume=False,
                   hide_progress=False):
    if out_folder is None:
        out_folder = in_folder

    if not os.path.isdir(in_folder):
        print('The input path specified does not exist')
        return None

    if out_folder != in_folder:
        Path(out_folder).mkdir(parents=True, exist_ok=True)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in glob.glob(os.path.join(in_folder, '*'))
             if f.endswith(file_types)]

    if resume:
        # Find which files have already been processed.
        done = set(file.name[-13:-4]
                   for file in Path(out_folder).glob('*.npz'))
        files = [f for f in files if f[-13:-4] not in done]

    processed = []

    for file in tqdm(files, unit="file", disable=hide_progress):
        # Process the las tile.
        outfile = process_ahn_las_tile(file, out_folder=out_folder,
                                       resolution=resolution)
        processed.append(outfile)

    return processed


def load_ahn_tile(ahn_file):
    ahn = np.load(ahn_file)
    ahn_tile = {'x': ahn['x'],
                'y': ahn['y'],
                'ground_surface': ahn['ground'],
                'building_surface': ahn['building']}
    return ahn_tile
