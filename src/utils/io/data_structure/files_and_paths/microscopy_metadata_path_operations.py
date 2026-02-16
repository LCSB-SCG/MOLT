import glob
import os


def get_microscopy_meta_data_name(roi: str, year: str, month: str, day: str):
    """
    Returns the meta data file name pattern for a given roi and channel id.

    Args:
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
    """
    return f"{year}_{month}_{day}_{roi}_meta.json"


def get_microscopy_meta_data_path(
    subject_folder: str,
    week: str,
    roi: str,
    year: str,
    month: str,
    day: str,
):
    """
    Get the exact file path for a given subject, week, roi, channel, year, month and day for a meta data file.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
    """
    path = os.path.join(
        subject_folder,
        week,
        "metadata",
        get_microscopy_meta_data_name(roi, year, month, day),
    )
    return path


def get_nd2_roi_microscopy_meta_data(data_path: str, subject, week, roi):
    """
    Get the meta data file path for a given subject, week and roi.

    Args:
        data_path (str): The data path.
        subject (str): The subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
    """
    path = os.path.join(
        data_path,
        subject,
        week,
        "metadata",
        get_microscopy_meta_data_name(roi, "*", "*", "*"),
    )
    paths = glob.glob(path)
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No meta data files found in {week} for {roi} and subject {subject}"
        )
    elif len(paths) > 1:
        raise ValueError(
            f"Multiple meta data files found in {week} for {roi} and subject {subject}"
        )
    return paths[0]
