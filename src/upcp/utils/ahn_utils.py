# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""This module provides utility methods for AHN data."""

import numpy as np
import pandas as pd
import zarr
import copy
import warnings
import os
import logging
from abc import ABC, abstractmethod
from tifffile import TiffFile, imread
from pathlib import Path
from scipy import interpolate
from scipy.ndimage import measurements, generic_filter
from scipy.ndimage.morphology import binary_dilation

from ..utils.las_utils import get_bbox_from_tile_code
from ..utils.interpolation import FastGridInterpolator

logger = logging.getLogger(__name__)


class AHNReader(ABC):
    """Abstract class for reading AHN data."""

    @property
    @classmethod
    @abstractmethod
    def NAME(cls):
        return NotImplementedError

    def __init__(self, data_folder, caching):
        super().__init__()
        self.path = Path(data_folder)
        self.set_caching(caching)
        self._clear_cache()
        if not self.path.exists():
            print('Input folder does not exist.')
            raise ValueError

    @abstractmethod
    def filter_tile(self, tilecode):
        pass

    def set_caching(self, state):
        if not hasattr(self, 'caching') or (self.caching is not state):
            self.caching = state
            if not self.caching:
                self._clear_cache()
                logger.debug('Caching disabled.')
            else:
                logger.debug('Caching enabled.')

    def _clear_cache(self):
        self.cache = {'tilecode': ''}

    def cache_interpolator(self, tilecode, points, surface='ground_surface'):
        logger.info(f'Caching {surface} for tile {tilecode}.')
        self.set_caching(True)
        if self.cache['tilecode'] != tilecode:
            # Clear cache.
            self._clear_cache()
        ahn_tile = self.filter_tile(tilecode)
        if surface not in ahn_tile:
            logger.error(f'Unknown surface: {surface}.')
            raise ValueError
        fast_z = FastGridInterpolator(
            ahn_tile['x'], ahn_tile['y'], ahn_tile[surface])
        self.cache['tilecode'] = tilecode
        self.cache[surface] = fast_z(points)

    def interpolate(self, tilecode, points=None, mask=None,
                    surface='ground_surface'):
        if points is None and mask is None:
            logger.error('Must provide either points or mask.')
            raise ValueError
        if self.caching and mask is not None:
            # Try retrieving cache.
            if self.cache['tilecode'] == tilecode:
                if surface in self.cache:
                    return self.cache[surface][mask]
                else:
                    logger.debug(
                        f'Surface {surface} not in cache for tile {tilecode}.')
        elif self.caching:
            logger.debug('Caching enabled but no mask provided.')

        # No cache, fall back to FastGridInterpolator.
        if self.cache['tilecode'] != tilecode and points is None:
            logger.error(
                f'Tile {tilecode} not cached and no points provided.')
            raise ValueError

        ahn_tile = self.filter_tile(tilecode)
        if surface not in ahn_tile:
            logger.error(f'Unknown surface: {surface}.')
            raise ValueError
        fast_z = FastGridInterpolator(
            ahn_tile['x'], ahn_tile['y'], ahn_tile[surface])
        return fast_z(points)


class NPZReader(AHNReader):
    """
    NPZReader for AHN3 data. The data folder should contain the pre-processed
    .npz files.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing the .npz files.
    caching : bool (default: True)
        Enable caching of the current ahn tile and interpolation data.
    """

    NAME = 'npz'

    def __init__(self, data_folder, caching=True):
        super().__init__(data_folder, caching)

    def filter_tile(self, tilecode):
        """
        Returns an AHN tile dict for the area represented by the given
        CycloMedia tile-code. TODO also implement geotiff?
        """
        if self.caching:
            if self.cache['tilecode'] != tilecode:
                self._clear_cache()
                self.cache['tilecode'] = tilecode
            if 'ahn_tile' not in self.cache:
                self.cache['ahn_tile'] = load_ahn_tile(
                        os.path.join(self.path, 'ahn_' + tilecode + '.npz'))
            return self.cache['ahn_tile']
        else:
            return load_ahn_tile(
                        os.path.join(self.path, 'ahn_' + tilecode + '.npz'))


class GeoTIFFReader(AHNReader):
    """
    GeoTIFFReader for AHN3 data. The data folder should contain on or more 0.5m
    resolution GeoTIFF files with the original filename.

    Parameters
    ----------
    data_folder : str or Path
        Folder containing the GeoTIFF files.
    caching : bool (default: True)
        Enable caching of the current ahn tile and interpolation data.
    fill_gaps : bool (default: True)
        Whether to fill gaps in the AHN data. Only used when method='geotiff'.
    max_gap_size : int (default: 50)
        Max gap size for gap filling. Only used when method='geotiff'.
    smoothen : bool (default: True)
        Whether to smoothen edges in the AHN data. Only used when
        method='geotiff'.
    smooth_thickness : int (default: 1)
        Thickness for edge smoothening. Only used when method='geotiff'.
    """

    RESOLUTION = 0.5
    NAME = 'geotiff'

    def __init__(self, data_folder, caching=True,
                 fill_gaps=True, max_gap_size=50,
                 smoothen=True, smooth_thickness=1):
        super().__init__(data_folder, caching)
        self.fill_gaps = fill_gaps
        self.max_gap_size = max_gap_size
        self.smoothen = smoothen
        self.smooth_thickness = smooth_thickness
        self.ahn_df = (pd.DataFrame(columns=['Filename', 'Path',
                                             'Xmin', 'Ymax', 'Xmax', 'Ymin'])
                       .set_index('Filename'))
        self._readfolder()

    def _readfolder(self):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the bounding boxes of each available file to help with the
        area extraction.
        """
        file_match = "M_*.TIF"

        for file in self.path.glob(file_match):
            with TiffFile(file.as_posix()) as tiff:
                if not tiff.is_geotiff:
                    print(f'{file.as_posix()} is not a GeoTIFF file.')
                elif ((tiff.geotiff_metadata['ModelPixelScale'][0]
                       != self.RESOLUTION)
                      or (tiff.geotiff_metadata['ModelPixelScale'][1]
                          != self.RESOLUTION)):
                    print(f'{file.as_posix()} has incorrect resolution.')
                else:
                    (x, y) = tiff.geotiff_metadata['ModelTiepoint'][3:5]
                    (h, w) = tiff.pages[0].shape
                    x_min = x - self.RESOLUTION / 2
                    y_max = y + self.RESOLUTION / 2
                    x_max = x_min + w * self.RESOLUTION
                    y_min = y_max - h * self.RESOLUTION
                    self.ahn_df.loc[file.name] = [file.as_posix(),
                                                  x_min, y_max, x_max, y_min]
        if len(self.ahn_df) == 0:
            print(f'No GeoTIFF files found in {self.path.as_posix()}.')
        else:
            self.ahn_df.sort_values(by=['Xmin', 'Ymax'], inplace=True)

    def _get_df(self):
        """Return the DataFrame."""
        return self.ahn_df

    def _load_tile(self, tilecode, fill_value):
        """Extract one tile from the GeoTIFF data."""
        ((bx_min, by_max), (bx_max, by_min)) = \
            get_bbox_from_tile_code(tilecode)

        ahn_tile = {}

        # We first check if the entire area is within a single TIF tile.
        query_str = '''(Xmin <= @bx_min) & (Xmax >= @bx_max) \
                        & (Ymax >= @by_max) & (Ymin <= @by_min)'''
        target_frame = self.ahn_df.query(query_str)
        if len(target_frame) == 0:
            print(f'No data found for {tilecode}')
            return None
        else:
            # The area is within a single TIF tile, so we can easily return the
            # array.
            [path, x, y, w, h] = target_frame.iloc[0].values
            with imread(path, aszarr=True) as store:
                z_data = np.array(zarr.open(store, mode="r"))
            x_start = int((bx_min - x) / self.RESOLUTION)
            x_end = int((bx_max - x) / self.RESOLUTION)
            y_start = int((y - by_max) / self.RESOLUTION)
            y_end = int((y - by_min) / self.RESOLUTION)
            ahn_tile['x'] = np.arange(bx_min + self.RESOLUTION / 2,
                                      bx_max, self.RESOLUTION)
            ahn_tile['y'] = np.arange(by_max - self.RESOLUTION / 2,
                                      by_min, -self.RESOLUTION)
            ahn_tile['ground_surface'] = z_data[y_start:y_end, x_start:x_end]
            fill_mask = ahn_tile['ground_surface'] > 1e5
            ahn_tile['ground_surface'][fill_mask] = fill_value
            if self.fill_gaps:
                fill_gaps(
                    ahn_tile, max_gap_size=self.max_gap_size, inplace=True)
            if self.smoothen:
                smoothen_edges(
                    ahn_tile, thickness=self.smooth_thickness, inplace=True)
            return ahn_tile

    def filter_tile(self, tilecode, fill_value=np.nan):
        """
        Return a dictionary <X, Y, Z> representing the Z-values of the <X, Y>
        area corresponding to the given tilecode. The points are equally spaced
        with a resolution of 0.5m, heights are copied directly from the AHN
        GeoTIFF data. Missing data is filled with 'fill_value'.

        NOTE: This function assumes that the full tilecode is enclosed in a
        single AHN GEoTIFF tile. This assumption is valid for standard AHN data
        and CycloMedia tilecodes.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        fill_value : float or np.nan (default: np.nan)
            Value used to fill missing data.

        Returns
        -------
        A dict containing AHN Z-values for the requested area, as well as the X
        and Y coordinate axes.
        """
        if self.caching:
            if self.cache['tilecode'] != tilecode:
                self._clear_cache()
                self.cache['tilecode'] = tilecode
            if 'ahn_tile' not in self.cache:
                self.cache['ahn_tile'] = self._load_tile(tilecode, fill_value)
            return self.cache['ahn_tile']
        else:
            return self._load_tile(tilecode, fill_value)


def load_ahn_tile(ahn_file):
    """
    Load the ground and building surface grids in a given AHN .npz file and
    return the results as a dict with keys 'x', 'y', 'ground_surface' and
    'building_surface'.
    """
    if not os.path.isfile(ahn_file):
        msg = f'Tried loading {ahn_file} but file does not exist.'
        raise AHNFileNotFoundError(msg)

    ahn = np.load(ahn_file)
    ahn_tile = {'x': ahn['x'],
                'y': ahn['y'],
                'ground_surface': ahn['ground'].astype(float),
                'building_surface': ahn['building'].astype(float)}
    return ahn_tile


def _get_gap_coordinates(ahn_tile, max_gap_size=50, gap_flag=np.nan):
    """
    Helper method. Get the coordinates of gap pixels in the AHN data. The
    max_gap_size determines the maximum size of gaps (in AHN pixels) that will
    be considered.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_tile(.).
    max_gap_size : int (default: 50)
        The maximum size (in grid cells) for gaps to be considered.
    gap_flag : float (default: np.nan)
        Flag used for missing data.

    Returns
    -------
    An array of shape (n_pixes, 2) containing the [x, y] coordinates of the gap
    pixels.
    """
    # Create a boolean mask for gaps.
    if np.isnan(gap_flag):
        gaps = np.isnan(ahn_tile['ground_surface'])
    else:
        gaps = (ahn_tile['ground_surface'] == gap_flag)

    # Find connected components in the gaps mask and compute their sizes.
    gap_ids, num_gaps = measurements.label(gaps)
    ids = np.arange(num_gaps + 1)
    gap_sizes = measurements.sum(gaps, gap_ids, index=ids)

    # Collect all gap coordinates.
    gap_coords = np.empty(shape=(0, 2), dtype=int)
    for i in ids:
        if 0 < gap_sizes[i] <= max_gap_size:
            # The lower bound 0 is used to ignore the 'non-gap' cluster which
            # has size 0.
            gap_coords = np.vstack([gap_coords, np.argwhere(gap_ids == i)])
    return gap_coords


def fill_gaps(ahn_tile, max_gap_size=50, gap_flag=np.nan, inplace=False):
    """
    Fill gaps in the AHN ground surface by interpolation. The max_gap_size
    determines the maximum size of gaps (in AHN pixels) that will be
    considered. A copy of the AHN tile will be returned, unless 'inplace' is
    set to True in which case None will be returned.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_tile(.).
    max_gap_size : int (default: 50)
        The maximum size for gaps to be considered.
    gap_flag : float (default: np.nan)
        Flag used for missing data.
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with filled gaps is returned.
    Else, None is returned.
    """
    # Get the coodinates of gap pizels to consider.
    gap_coords = _get_gap_coordinates(ahn_tile, max_gap_size=max_gap_size,
                                      gap_flag=gap_flag)

    # Mask the z-data to exclude gaps.
    if np.isnan(gap_flag):
        mask = ~np.isnan(ahn_tile['ground_surface'])
    else:
        mask = ~(ahn_tile['ground_surface'] == gap_flag)

    # Get the interpolation values for the gaps.
    x = np.arange(0, len(ahn_tile['x']))
    y = np.arange(0, len(ahn_tile['y']))
    xx, yy = np.meshgrid(x, y)
    # TODO: method='cubic' is just a default, we should check what works best
    # for us.
    int_values = interpolate.griddata(
        points=(xx[mask], yy[mask]),
        values=ahn_tile['ground_surface'][mask].ravel(),
        xi=(gap_coords[:, 1], gap_coords[:, 0]),
        method='cubic')

    # Return the filled AHN tile.
    if not inplace:
        filled_ahn = copy.deepcopy(ahn_tile)
        filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values
        return filled_ahn
    else:
        ahn_tile['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values
        return None

def fill_gaps_intuitive(ahn_tile):
    """
    Fill nans in the AHN ground surface by interpolation. First, linear interpolation is used.
    The remaining gaps are filled using the z-value of nearest point. 
    The AHN tile will be returned

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_tile(.).
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with filled gaps is returned.
    Else, None is returned.
    """
    # Copy ahn_tile
    filled_ahn = copy.deepcopy(ahn_tile)

    # Get the coodinates of gap pixels to consider.
    gaps = np.isnan(filled_ahn['ground_surface'])
    if 'artifact_surface' in ahn_tile.keys():
        filled_ahn['ground_surface'][gaps] = filled_ahn['artifact_surface'][gaps]

    # Get the coodinates of gap pixels to consider.
    gaps = np.isnan(filled_ahn['ground_surface'])
    gap_coords = np.argwhere(gaps)

    # Get the interpolation values for the gaps.
    x = np.arange(0, len(filled_ahn['x']))
    y = np.arange(0, len(filled_ahn['y']))
    xx, yy = np.meshgrid(x, y)
    int_values = interpolate.griddata(
                        points=(xx[~gaps], yy[~gaps]),
                        values=filled_ahn['ground_surface'][~gaps].ravel(),
                        xi=(gap_coords[:, 1], gap_coords[:, 0]),
                        method='linear')

    # Return the filled AHN tile.
    filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] \
            = int_values

    # Get the interpolation values for the remaining gaps.
    gaps = np.isnan(filled_ahn['ground_surface'])
    gap_coords = np.argwhere(gaps)
    int_values = interpolate.griddata(
                        points=(xx[~gaps], yy[~gaps]),
                        values=filled_ahn['ground_surface'][~gaps].ravel(),
                        xi=(gap_coords[:, 1], gap_coords[:, 0]),
                        method='nearest')

    filled_ahn['ground_surface'][gap_coords[:, 0], gap_coords[:, 1]] = int_values
    return filled_ahn

def smoothen_edges(ahn_tile, thickness=1, gap_flag=np.nan, inplace=False):
    """
    Smoothen the edges of missing AHN ground surface data in the ahn_tile. In
    effect, this 'pads' the ground surface around gaps by the given 'thickness'
    and prevents small gaps around e.g. buildings when labelling a point cloud.
    A copy of the AHN tile will be returned, unless 'inplace' is set to True in
    which case None will be returned.

    Parameters
    ----------
    ahn_tile : dict
        E.g., output of GeoTIFFReader.filter_tile(.).
    thickness : int (default: 1)
        Thickness of the edge, for now only a value of 1 or 2 makes sense.
    gap_flag : float (default: np.nan)
        Flag used for missing data.
    inplace: bool (default: False)
        Whether or not to modify the AHN tile in place.

    Returns
    -------
    If inplace=false, a copy of the AHN tile with smoothened edges is returned.
    Else, None is returned.
    """
    if np.isnan(gap_flag):
        mask = ~np.isnan(ahn_tile['ground_surface'])
        z_data = ahn_tile['ground_surface']
    else:
        mask = ~(ahn_tile['ground_surface'] == gap_flag)
        z_data = copy.deepcopy(ahn_tile['ground_surface'])
        z_data[~mask] = np.nan

    # Find the edges of data gaps.
    edges = mask ^ binary_dilation(mask, iterations=thickness)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Compute smoothened AHN data by taking the mean of surrounding pixel
        # values (ignoring NaNs).
        # TODO: a 'thickness' of more than 2 would require a bigger footprint.
        smoother = generic_filter(z_data, np.nanmean,
                                  footprint=np.ones((3, 3), dtype=int),
                                  mode='constant', cval=np.nan)

    if inplace:
        ahn_tile['ground_surface'][edges] = smoother[edges]
        return None
    else:
        smoothened_ahn = copy.deepcopy(ahn_tile)
        smoothened_ahn['ground_surface'][edges] = smoother[edges]
        return smoothened_ahn


class AHNFileNotFoundError(Exception):
    """Exception raised for missing AHN files."""
