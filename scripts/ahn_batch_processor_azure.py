#!/usr/bin/python

import glob
import os
from contextlib import contextmanager
from datetime import datetime
from functools import partial

from azureml.core import Dataset, Workspace
from src.preprocessing.ahn_preprocessing import process_ahn_las_tile
from tqdm.contrib.concurrent import process_map  # or thread_map

DEFAULT_RESOLUTION = 0.1
FILE_TYPES = ('.LAS', '.las', '.LAZ', '.laz')
INPUT_DATASET_PATH = "/UI/08-04-2021_095741_UTC"
OUTPUT_DATASET_PATH = "/sample/output"

def _process_file(file, out_folder):
    process_ahn_las_tile(
        file, 
        out_folder=out_folder,
        resolution=DEFAULT_RESOLUTION
    )

@contextmanager
def mount_volume(dataset):
    mount_context = dataset.mount()
    mount_context.start()
    yield mount_context
    mount_context.stop()


if __name__ == '__main__':
    # Get Azure resources
    ws = Workspace.from_config(".azure/config.json")
    def_blob_store = ws.get_default_datastore()
    input_dataset = Dataset.File.from_files((def_blob_store, INPUT_DATASET_PATH))

    # Mount Blob to container
    with mount_volume(input_dataset) as mount_context_in:
        # Get the target files
        files = [f for f in glob.glob(os.path.join(mount_context_in.mount_point, '*'))
                if f.endswith(FILE_TYPES)]
        
        # Chunk size can be used to reduce overhead for a large number of files.
        chunk = 1
        if len(files) > 100:
            chunk = 5
        if len(files) > 1000:
            chunk = 10

        # Run the workload
        os.mkdir("output")
        _process_file_to_dir = partial(_process_file, out_folder="output")
        r = process_map(_process_file_to_dir, files, chunksize=chunk)

        # Upload the result to
        def_blob_store.upload(
            "output",
             target_path=os.path.join(
                "/output",
                datetime.strftime(
                    datetime.utcnow(),
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )
        )
