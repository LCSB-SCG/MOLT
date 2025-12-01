import os


def get_rgb_name(roi: str, year: str, month: str, day: str):
    """
    Returns the rgb file name pattern for a given roi and channel id.

    Args:
        roi (str): The region of interest.
        channel_id (str): The channel id.
        year (str): The year of the image.
        month (str): The month of the image.
        day (str): The day of the image.
    """
    return f"{year}_{month}_{day}_{roi}.nii.gz"


def get_rgb_path(
    subject_folder: str,
    week: str,
    roi: str,
    year: str,
    month: str,
    day: str,
):
    """
    Get the exact file path for a given subject, week, roi, channel, year, month and day for an rgb image.

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
        "rgb",
        get_rgb_name(roi, year, month, day),
    )
    return path
