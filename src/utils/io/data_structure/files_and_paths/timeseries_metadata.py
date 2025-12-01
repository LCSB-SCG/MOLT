import os
from typing import Union

from src.utils.io.data_structure.files_and_paths.constants import (
    INSTANCES_ACROSS_CHANNELS_FOLDER,
)


def get_subject_roi_label_channel_time_series_info_path(
    data_path: str, subject_folder: str, roi: str, channel_id: str, label_setting: str
):
    """
    Get the exact file path for the channel time series info file.

    Args:
        data_path (str): The data path.
        subject_folder (str): The subject folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        label_setting (str): The label setting.
    """
    path = os.path.join(
        data_path,
        subject_folder,
        "time_series_info",
        roi,
        label_setting,
        str(channel_id),
        "time_series_info.json",
    )
    return path


def get_match_channel_a_settings_to_channel_b_settigs_folder_name(
    channel_a_id: Union[int, str],
    channel_b_id: Union[int, str],
    label_setting_a: str,
    label_setting_b: str,
):
    """
    Construct the folder name for the matching of channel a to channel b.

    Args:
        channel_a_id (int): The channel id.
        channel_b_id (int): The channel id.
        label_setting_a (str): The label setting.
        label_settings_b (str): The label setting.
        channel_ids_are_swapped (bool): If True, the channel ids are swapped.
    """
    return (
        f"{channel_a_id}"
        + "_("
        + label_setting_a
        + ")_"
        + f"{channel_b_id}"
        + "_("
        + label_setting_b
        + ")"
    )


def get_match_channel_a_to_channel_b_file_name(
    channel_a_id: str,
    channel_b_id: str,
) -> str:
    """
    Construct the file name for the matching of channel a to channel b.

    Args:
        channel_a_id (str): The channel id.
        channel_b_id (str): The channel id.

    Returns:
        str: The file name.
    """
    return f"{channel_a_id}_related_to_{channel_b_id}_timeseries.json"


def get_subject_roi_label_time_series_two_channel_mapping_info_path(
    data_path: str,
    subject_id: str,
    roi: str,
    channel_a_id: int,
    channel_b_id: int,
    label_setting_a: str,
    label_setting_b: str,
):
    """
    Get the exact file path for the json file containing the mapping of two channels for a given roi over the time series.

    Args:
        data_path (str): The data path.
        subject_folder (str): The subject folder.
        roi (str): The region of interest.
        channel_a_id (int): The channel id.
        channel_b_id (int): The channel id.
        label_setting_a (str): The label setting.
        label_settings_b (str): The label setting.
        channel_ids_are_swapped (bool): If True, the channel ids are swapped.
    """
    foldername = get_match_channel_a_settings_to_channel_b_settigs_folder_name(
        channel_a_id=channel_a_id,
        channel_b_id=channel_b_id,
        label_setting_a=label_setting_a,
        label_setting_b=label_setting_b,
    )

    filename = get_match_channel_a_to_channel_b_file_name(
        channel_a_id=channel_a_id,
        channel_b_id=channel_b_id,
    )

    path = os.path.join(
        data_path,
        subject_id,
        INSTANCES_ACROSS_CHANNELS_FOLDER,
        roi,
        foldername,
        filename,
    )
    return path
