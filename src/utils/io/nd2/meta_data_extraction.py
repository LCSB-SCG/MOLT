import json
import os
import nd2
from dataclasses import asdict

from src.utils.io.data_structure.files_and_paths.microscopy_metadata_path_operations import (
    get_microscopy_meta_data_path,
)
from src.utils.io.data_structure.files_and_paths.nd2_path_operations import (
    get_parts_of_nd2_path,
)


def get_metadata_as_dictionary(filepath: str):
    """
    Extracts the meta data from a given nd2 (Nikon NIS Elements) file and returns its meta data as a dictionary.

    :param filepath: The path to the nd2 file.
    :return: A dictionary containing the meta data of the nd2 file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")

    file = nd2.ND2File(filepath)
    meta_data = asdict(file.metadata)
    file.close()
    return meta_data


def save_microscopy_metadata_to_json(filepath: str):
    """
    Extracts the meta data from a given nd2 (Nikon NIS Elements) file and saves it as a json file.

    :param filepath: The path to the nd2 file.
    """
    meta_data = get_metadata_as_dictionary(filepath)

    # Create folder "metadata" in same folder as file
    folders = get_parts_of_nd2_path(filepath)
    json_file = get_microscopy_meta_data_path(*folders)
    print(f"Meta data at {json_file}")
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))

    # Save the meta data as a json file
    with open(json_file, "w") as f:
        json.dump(meta_data, f, indent=4)