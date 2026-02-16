from dataclasses import asdict
import os

import nd2


def path_checks(input_paths: list):
    "Simple checks on the file paths"
    for p in input_paths:
        # Checks on the file paths
        if not os.path.exists(p):
            raise FileNotFoundError(f"The file {p} does not exist")
        if not p.endswith(".nd2"):
            raise ValueError(f"The input file {p} is not an nd2 file")


def load_nd2(file_path):
    """Load an nd2 file and return the image data and metadata."""
    with nd2.ND2File(file_path) as nd2_file:
        image_data = nd2_file.asarray()
        metadata = asdict(nd2_file.metadata)
        if len(image_data.shape) != 4:
            raise ValueError(
                "The nd2 file does not have the expected shape (stack Z, channels C, height Y, width X)"
            )
        Z = image_data.shape[0]
        C = image_data.shape[1]
        Y = image_data.shape[2]
        X = image_data.shape[3]
    return image_data, metadata, Z, C, Y, X