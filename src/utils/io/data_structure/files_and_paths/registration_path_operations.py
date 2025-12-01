import glob
import os
import re
from src.utils.io.data_structure.files_and_paths.constants import (
    AFFINE_TRANS,
    OVERLAP_NIFTI_NAME,
)
from src.utils.io.data_structure.files_and_paths.patterns import (
    REGISTRATION_FOLDER_NAME_PATTERN,
)


def get_registration_folder_from_unregisterd_file(
    image_path, target_week, create_dirs=False
):
    """
    Get the registration folder from an unregistered file path.

    Args:
        image_path (str): Path to the unregistered file.
        target_week (str): The week folder of the registered file.
        create_dirs (bool): If True, the registration folder will be created if it does not exist.

    Returns:
        str: Path to the registration folder.
    """
    path = os.path.abspath(os.path.normpath(image_path))
    path_parts = path.split(os.sep)

    # perform checks if the given path is actually an unregistered file
    if re.match(REGISTRATION_FOLDER_NAME_PATTERN, path_parts[-2]) is not None:
        raise ValueError(f"Given path is not an unregistered file: {image_path}")

    # get the registration folder path
    new_path = os.path.join(os.sep, *path_parts[:-1], f"registration_to_{target_week}")

    if create_dirs and not os.path.exists(new_path):
        os.makedirs(new_path)

    return new_path


def get_registered_file_name_from_unregisterd_file(image_path):
    """
    Get the registered file name from an unregistered file path.

    Args:
        image_path (str): Path to the unregistered file.

    Returns:
        str: The registered file name.
    """
    path = os.path.abspath(os.path.normpath(image_path))
    path_parts = path.split(os.sep)

    # perform checks if the given path is actually an unregistered file
    if re.match(REGISTRATION_FOLDER_NAME_PATTERN, path_parts[-2]) is not None:
        raise ValueError(f"Given path is not an unregistered file: {image_path}")

    # get the registered file name
    filename = path_parts[-1]
    new_filename = filename.replace(".nii", "_reg.nii")

    return new_filename


def get_registered_file_path_from_unregisterd_file(
    image_path, target_week, create_dirs=False
):
    """
    Get the registered file path from an unregistered file path.

    Args:
        image_path (str): Path to the unregistered file.
        create_dirs (bool): If True, the registration folder will be created if it does not exist.

    Returns:
        str: Path to the registered file.
    """
    folder = get_registration_folder_from_unregisterd_file(
        image_path, target_week, create_dirs
    )
    filename = get_registered_file_name_from_unregisterd_file(image_path)

    new_path = os.path.join(folder, filename)

    return new_path


def get_unique_registration_file(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    target_week: str,
    filename: str,
):
    """
    Get the exact file path for the filename in the registration folder.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        target_week (str): The target week folder.
    """
    if "channel" not in channel_id:
        channel_id = f"channel_{channel_id}"

    path = os.path.join(
        subject_folder,
        week,
        "nifti",
        roi,
        channel_id,
        f"registration_to_{target_week}",
        filename,
    )

    paths = glob.glob(path)
    if len(paths) == 0:
        raise ValueError(
            f"No {filename} found in {week} for {roi}, channel {channel_id} and subject {os.path.basename(subject_folder)}"
        )
    elif len(paths) > 1:
        raise ValueError(
            f"Multiple {filename} found in {week} for {roi}, channel {channel_id} and subject {os.path.basename(subject_folder)}"
        )
    return paths[0]


def get_unique_affine_transformation_path(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    target_week: str,
):
    """
    Get the exact file path for the affine transformation matrix in the registration folder.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        target_week (str): The target week folder.
    """
    return get_unique_registration_file(
        subject_folder=subject_folder,
        week=week,
        roi=roi,
        channel_id=channel_id,
        target_week=target_week,
        filename=AFFINE_TRANS,
    )


def get_unique_overlapping_region_path(
    subject_folder: str,
    week: str,
    roi: str,
    channel_id: str,
    target_week: str,
):
    """
    Get the exact file path for the overlapping region in the registration folder.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
        channel_id (str): The channel id.
        target_week (str): The target week folder.
    """
    return get_unique_registration_file(
        subject_folder=subject_folder,
        week=week,
        roi=roi,
        channel_id=channel_id,
        target_week=target_week,
        filename=OVERLAP_NIFTI_NAME,
    )


def get_unregistered_file(image_path):
    """
    Get the unregistered file path from a registered file path.

    Args:
        image_path (str): Path to the registered file.

    Returns:
        str: Path to the unregistered file.
    """
    path = os.path.abspath(os.path.normpath(image_path))
    path_parts = path.split(os.sep)

    # perform checks if the given path is actually a registered file
    if re.match(REGISTRATION_FOLDER_NAME_PATTERN, path_parts[-2]) is None:
        raise ValueError(f"Given path is not a registered file: {image_path}")

    # get the unregistered file path
    filename = path_parts[-1].replace("_reg", "")
    new_path = os.path.join(os.sep, *path_parts[:-2], filename)

    return new_path
