import json
import os


from src.utils.io.json.conver_between_json_and_numpy import convert_numpy_objects_to_str


def write_to_json(
    data, filename, indent=4, create_dirs=True, convert_numpy=False
) -> None:
    """
    Write data to a json file.

    Args:
    data (dict): Data to write to the json file.
    filename (str): Path to the file to save the data to.
    """
    if convert_numpy:
        data = convert_numpy_objects_to_str(data)
    if create_dirs:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(json.dumps(obj=data, indent=indent, sort_keys=True))
