"""BGT Data Fuser"""

import pandas as pd
import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod

from ..abstract_processor import AbstractProcessor

logger = logging.getLogger(__name__)


class BGTFuser(AbstractProcessor, ABC):
    """
    Abstract class for automatic labelling of points using BGT data.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_file : str or Path or None (default: None)
        File containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    bgt_folder : str or Path or None (default: None)
        Folder containing data files needed for this fuser. Either a file or a
        folder should be provided, but not both.
    file_prefix : str (default: '')
        Prefix used to load the correct files; only used with bgt_folder.
    """
    @property
    @classmethod
    @abstractmethod
    def COLUMNS(cls):
        return NotImplementedError

    def __init__(self, label, bgt_file=None, bgt_folder=None,
                 file_prefix=''):
        if (bgt_file is None) and (bgt_folder is None):
            print("Provide either a bgt_file or bgt_folder to load.")
            return None
        if (bgt_file is not None) and (bgt_folder is not None):
            print("Provide either a bgt_file or bgt_folder to load, not both")
            return None
        if (bgt_folder is not None) and (not os.path.isdir(bgt_folder)):
            print('The data folder specified does not exist')
            return None
        if (bgt_file is not None) and (not os.path.isfile(bgt_file)):
            print('The data file specified does not exist')
            return None

        super().__init__(label)
        self.file_prefix = file_prefix
        self.bgt_df = pd.DataFrame(columns=type(self).COLUMNS)

        if bgt_file is not None:
            self._read_file(Path(bgt_file))
        elif bgt_folder is not None:
            self._read_folder(Path(bgt_folder))
        else:
            logger.error('No data folder or file specified. Aborting...')
            return None

    def _read_folder(self, path):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        file_match = self.file_prefix + '*.csv'
        frames = [pd.read_csv(file, header=0, names=type(self).COLUMNS)
                  for file in path.glob(file_match)]
        if len(frames) == 0:
            logger.error(f'No data files found in {path.as_posix()}.')
            return
        self.bgt_df = pd.concat(frames)

    def _read_file(self, path):
        """
        Read the contents of a file. Internally, a DataFrame is created
        detailing the polygons and bounding boxes of each building found in the
        CSV files in that folder.
        """
        self.bgt_df = pd.read_csv(path, header=0, names=type(self).COLUMNS)

    @abstractmethod
    def _filter_tile(self, tilecode):
        """
        Returns data for the area represented by the given CycloMedia
        tile-code.
        """
        return NotImplementedError
