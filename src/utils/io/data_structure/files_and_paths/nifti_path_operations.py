import glob
import os
import re

from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_date_from_filename,
    extract_subject_folder_from_filepath,
    extract_week_from_filepath,
)
from src.utils.io.data_structure.files_and_paths.patterns import (
    ROI_FROM_NIFTI_FILENAME_PATTERN,
)


def get_nifti_name(roi: str, channel_id: str, year=None, month=None, day=None):
    """
    Returns the nifti file name pattern for a given roi and channel id.

    Args:
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
    """
    if year is None:
        year = "[0-9][0-9][0-9][0-9]"
    if month is None:
        month = "[0-9][0-9]"
    if day is None:
        day = "[0-9][0-9]"
    if "channel" not in channel_id:
        channel_id = f"channel_{channel_id}"
    return f"{year}_{month}_{day}_{roi}_{channel_id}.nii.gz"


def get_nifti_path(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    year: str,
    month: str,
    day: str,
):
    """
    Get the exact file path for a given subject, week, roi, channel, year, month and day.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
    """
    path = os.path.join(
        subject_folder,
        week,
        "nifti",
        roi,
        f"channel_{channel_id}" if "channel" not in channel_id else channel_id,
        get_nifti_name(roi, channel_id, year, month, day),
    )
    return path


def get_unique_nifti_filepath(
    subject_folder: str, week: str, roi: str, channel_id: str
):
    """
    Retrieve the nifti file and check if the file exists and is unique.

    Args:
        weekfolder (str): path ot the week folder
        file_name (str): file name consistent in each week folder
    """
    if "channel" not in channel_id:
        channel_id = f"channel_{channel_id}"

    path = os.path.join(
        subject_folder, week, "nifti", roi, channel_id, get_nifti_name(roi, channel_id)
    )
    files = glob.glob(path)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found in {week} for {roi} and channel {channel_id} and subject {os.path.basename(subject_folder)}"
        )
    elif len(files) > 1:
        raise ValueError(
            f"Multiple files found in {week} for {roi} and channel {channel_id} and subject {os.path.basename(subject_folder)}"
        )
    # the regex should be specific enough to only find one file
    return files[0]


def extract_roi_from_nifti_filename(filename):
    """
    Extracts the region of interest from a filename.

    Args:
        filename (str): The filename to extract the roi from.

    Returns:
        str: The region of interest or None if no roi was found.
    """
    match = re.search(ROI_FROM_NIFTI_FILENAME_PATTERN, filename)
    if match:
        roi = match.group(1)
    else:
        roi = None
    return roi


def get_parts_of_nifti_path(filepath):
    """
    Returns the parts of a nifti file path.

    Args:
        filepath (str): The path to the nifti file.

    Returns:
        subject_folder_path (str): path to the subject folder
        week_folder (str): week folder
        roi (str): region of interest
        channel_id (str): channel id
        year (str): year of the image
        month (str): month of the image
        day (str): day of the image
    """
    parts = filepath.split(os.sep)

    subject_folder_path = extract_subject_folder_from_filepath(filepath)
    week_folder = extract_week_from_filepath(filepath)
    roi = extract_roi_from_nifti_filename(parts[-1])
    channel_id = parts[-2]
    year, month, day = extract_date_from_filename(parts[-1])
    return subject_folder_path, week_folder, roi, channel_id, year, month, day
