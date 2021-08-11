#!/usr/bin/python

import argparse
import glob
import os
from contextlib import contextmanager
from datetime import datetime
from functools import partial

from azureml.core import Dataset, Workspace
from src.preprocessing.ahn_preprocessing import process_ahn_las_tile
from tqdm.contrib.concurrent import process_map  # or thread_map

DEFAULT_RESOLUTION = 0.1
FILE_TYPES = (".LAS", ".las", ".LAZ", ".laz")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", metavar="path", action="store",
                        type=str, required=True)
    parser.add_argument("--output_dataset_path", metavar="path", action="store",
                        type=str, required=True)
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    args = parser.parse_args()

    # Get Azure resources
    ws = Workspace.get(
        args.workspace_name,
        resource_group=args.resource_group,
        subscription_id=args.subscription_id
    )
    def_blob_store = ws.get_default_datastore()
    input_dataset = Dataset.File.from_files((def_blob_store, args.input_dataset_path))

    # Mount Blob to container
    with mount_volume(input_dataset) as mount_context_in:
        # Get the target files
        files = [f for f in glob.glob(os.path.join(mount_context_in.mount_point, "*"))
                if f.endswith(FILE_TYPES)]
        
        # Chunk size can be used to reduce overhead for a large number of files.
        chunk = 1
        if len(files) > 100:
            chunk = 5
        if len(files) > 1000:
            chunk = 10

        # Run the workload
        os.mkdir(args.output_dataset_path)
        _process_file_to_dir = partial(_process_file, out_folder=args.output_dataset_path)
        process_map(_process_file_to_dir, files, chunksize=chunk)

        # Upload the result to
        def_blob_store.upload(
            args.output_dataset_path,
            target_path=os.path.join(
                args.output_dataset_path,
                datetime.strftime(
                    datetime.utcnow(),
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )
        )
