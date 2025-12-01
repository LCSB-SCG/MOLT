import glob
import os
import re

from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_subject_folder_from_filepath,
    extract_week_from_filepath,
)
from src.utils.io.data_structure.files_and_paths.nifti_path_operations import (
    get_parts_of_nifti_path,
)
from src.utils.io.data_structure.files_and_paths.patterns import (
    LABEL_FOLDER_NAME_PATTERN,
    LFN_STRING,
)


def get_label_settings_folder_path(
    nifti_file_path,
    quantile=None,
    threshold=None,
    smoothing_sigma=None,
    min_volume_threshold=None,
    max_volume_threshold=None,
    binary_label_setting=None,
    path_only=False,
):

    if threshold is not None and quantile is not None:
        raise ValueError("Both threshold and quantile cannot be set.")

    # make settings folder
    if threshold is not None:
        settings_folder = f"t_{threshold}_s_{smoothing_sigma}_v_{min_volume_threshold}_{max_volume_threshold}"
    else:
        settings_folder = f"q_{quantile}_s_{smoothing_sigma}_v_{min_volume_threshold}_{max_volume_threshold}"

    path = get_label_path_from_nifti(nifti_file_path, settings_folder)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if path_only:
        return os.path.dirname(path)
    else:
        return path


def get_label_folder_name_from_filepath(filepath):
    """
    Extracts the label folder name from a filepath.

    Args:
        filepath (str): The filepath to extract the label folder name from.

    Returns:
        str: The label folder name or None if no label folder name was found.
    """
    parts = filepath.split(os.sep)
    for i, part in enumerate(parts):
        match = LABEL_FOLDER_NAME_PATTERN.match(part)
        if match:
            return part
    return None


def get_label_filename(
    roi: str, year: str, month: str, day: str, match_params: str, ending: str = "nii.gz"
) -> str:
    """
    Returns the label file name pattern for a given roi and channel id.

    Args:
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
        match_params (str): The target week needs to be provided if the returned metadata is for a matched instance file.
        ending (str): The file ending.
    """
    if year is None:
        year = "[0-9][0-9][0-9][0-9]"
    if month is None:
        month = "[0-9][0-9]"
    if day is None:
        day = "[0-9][0-9]"
    if match_params is not None:
        matched_param = f"_matched_{match_params}"
    else:
        matched_param = ""
    return f"{year}_{month}_{day}_{roi}_instances{matched_param}{ending}"


def get_subject_week_roi_label_folder(
    data_path: str, subject_folder: str, week: str, roi: str
):
    """
    Returns the subject label folder path.

    Args:
        data_path (str): The data path.
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
    """
    return os.path.join(data_path, subject_folder, week, "label", roi)


def get_label_path(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    setting: str,
    year: str,
    month: str,
    day: str,
    match_params: str = None,
):
    """
    Get the exact file path for a given subject, week, roi, channel, year, month and day for an instance segmentation image.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        setting (str): The setting folder name.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
        match_params (str): the target week in which space the labels have been matched
    """
    path = os.path.join(
        subject_folder,
        week,
        "label",
        roi,
        f"channel_{channel_id}" if "channel" not in channel_id else channel_id,
        setting,
        get_label_filename(roi, year, month, day, match_params),
    )
    return path


def get_unique_label_filepath(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    setting: str,
    year: str,
    month: str,
    day: str,
    match_params: str = None,
):
    """
    Retrieve the label file and check if the file exists and is unique.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        setting (str): The setting folder name.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
        match_params (str): the target week in which space the labels have been matched
    """
    if "channel" not in channel_id:
        channel_id = f"channel_{channel_id}"

    path = get_label_path(
        subject_folder,
        week,
        roi,
        channel_id,
        setting,
        year,
        month,
        day,
        match_params=match_params,
    )
    files = glob.glob(path)
    if len(files) == 0:
        raise ValueError(
            f"No label files found in {week} for {roi}, channel {channel_id}, setting {setting} and subject {os.path.basename(subject_folder)}"
        )
    elif len(files) > 1:
        raise ValueError(
            f"Multiple label files found in {week} for {roi}, channel {channel_id}, setting {setting} and subject {os.path.basename(subject_folder)}"
        )
    # the regex should be specific enough to only find one file
    return files[0]


def get_subject_week_roi_from_label_path(label_path):
    """
    Returns the subject, week and roi from a label path.

    Args:
        label_path (str): The path to the label file.

    Returns:
        subject_folder_path (str): path to the subject folder
        week_folder (str): week folder
        roi (str): region of interest
    """
    parts = label_path.split(os.sep)

    subject_folder_path = extract_subject_folder_from_filepath(label_path)
    week_folder = extract_week_from_filepath(label_path)
    roi = parts[-4]
    return subject_folder_path, week_folder, roi


def get_label_path_from_nifti(nifti_path, setting):
    """
    Get the label path from a nifti path.

    Args:
        nifti_path (str): The path to the nifti file.
        setting (str): The setting folder name.

    Returns:
        str: The path to the label file.
    """
    subject_folder_path, week_folder, roi, channel_id, year, month, day = (
        get_parts_of_nifti_path(nifti_path)
    )
    return get_label_path(
        subject_folder_path, week_folder, roi, channel_id, setting, year, month, day
    )


def get_metadata_filename(roi: str, year: str, month: str, day: str, match_params: str):
    """
    Returns the metadata filename pattern for a given roi and channel id.

    Args:
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
        match_params (str): The target week needs to be provided if the returned metadata is for a matched instance file.
    """
    if year is None:
        year = "[0-9][0-9][0-9][0-9]"
    if month is None:
        month = "[0-9][0-9]"
    if day is None:
        day = "[0-9][0-9]"
    if match_params is not None:
        matched_param = f"_matched_{match_params}"
    else:
        matched_param = "_meta.json"
    return f"{year}_{month}_{day}_{roi}_instances{matched_param}_meta.json"


def get_unique_metadata_filepath(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    setting: str,
    year: str,
    month: str,
    day: str,
    match_params: str = None,
):
    """
    Retrieve the metadata file and check if the file exists and is unique.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        setting (str): The setting folder name.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
        match_params (str): the target week to which the labels have been matched
    """
    if "channel" not in channel_id:
        channel_id = f"channel_{channel_id}"

    path = os.path.join(
        subject_folder,
        week,
        "label",
        roi,
        channel_id,
        setting,
        get_metadata_filename(roi, year, month, day, match_params),
    )
    paths = glob.glob(path)

    if len(paths) == 0:
        raise ValueError(
            f"No metadata files found in {week} for {roi}, channel {channel_id}, setting {setting} and subject {os.path.basename(subject_folder)}"
        )
    elif len(paths) > 1:
        raise ValueError(
            f"Multiple metadata files found in {week} for {roi}, channel {channel_id}, setting {setting} \
and subject {os.path.basename(subject_folder)}"
        )

    return paths[0]


def extract_threshold_from_label_setting(label_setting):
    """
    Extract the threshold from a label setting.

    Args:
        label_setting (str): The label setting.

    Example:
        label_setting = "t_500.0_s_1_v_10_100"
        extract_threshold_from_label_setting(label_setting)
        # Output: 500.0
        label_setting = "t_1000.0_s_1_v_10_100"
        extract_threshold_from_label_setting(label_setting)
        # Output: 1000.0
    """
    # Get the first match in the first group of regex:
    pattern = re.compile(LFN_STRING)
    return float(pattern.match(label_setting).group(1))
