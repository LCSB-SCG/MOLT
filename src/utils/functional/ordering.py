from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_number_from_week_string,
)


def ordered_nested_dict(obj):
    """
    Generates an ordered nested tuple for an object.

    Args:
        obj (dict, list): nested object to be ordered

    Returns:
        tuple: nested ordered tuple. keys of dictionaries will at index 0 of each tuple.
    """
    if isinstance(obj, dict):
        return sorted((k, ordered_nested_dict(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered_nested_dict(x) for x in obj)
    else:
        return obj


def order_timeseries_timesteps(timeseries: list[str]):
    """
    Orders the list of timeseries ids temporally increasing.

    Args:
        timeseries (list[str]): list of timeseries ids
    """
    timeseries_ids = {extract_number_from_week_string(w): w for w in timeseries}
    return [timeseries_ids[ts_num] for ts_num in sorted(timeseries_ids.keys())]
