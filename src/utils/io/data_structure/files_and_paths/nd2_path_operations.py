import os
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_date_from_filename,
    extract_subject_folder_from_filepath,
    extract_subject_from_filepath,
    extract_week_from_filepath,
)
from src.utils.io.data_structure.files_and_paths.patterns import ROI_PATTERN


def extract_roi_from_nd2_filename(filename):
    """
    Extracts the region of interest from a nd2 filename.

    Args:
        filename (str): The filename to extract the roi from.

    Returns:
        str: The region of interest or None if no roi was found.
    """
    roi = filename[-7:-4]
    if ROI_PATTERN.match(roi):
        return roi
    else:
        raise ValueError(
            f"The filename {filename} does not contain a valid roi. The roi should be in the format '001'."
        )


def get_parts_of_nd2_path(filepath):
    """
    Returns the parts of a nd2 file path.

    Args:
        filepath (str): The path to the nd2 file.

    Returns:
        subject_folder_path (str): path to the subject folder
        week_folder (str): week folder
        roi (str): region of interest
        year (str): year of the image
        month (str): month of the image
        day (str): day of the image
    """
    parts = filepath.split(os.sep)

    subject_folder_path = extract_subject_folder_from_filepath(filepath)
    week_folder = extract_week_from_filepath(filepath)
    roi = extract_roi_from_nd2_filename(parts[-1])
    year, month, day = extract_date_from_filename(parts[-1])
    return subject_folder_path, week_folder, roi, year, month, day


def collect_and_check_path_information(input_paths: list):
    """
    Collects information on the subject, roi, timepoint, date and week from the input paths and validates that
    they are the same for paths in the list.
    """
    subject_id = extract_subject_from_filepath(input_paths[0])
    week = extract_week_from_filepath(input_paths[0])
    year, month, day = extract_date_from_filename(os.path.basename(input_paths[0]))
    roi = extract_roi_from_nd2_filename(os.path.basename(input_paths[0]))

    for p in input_paths:
        if subject_id != extract_subject_from_filepath(p):
            raise ValueError(
                f"The subject id {subject_id} does not match the subject id in {p} with {input_paths[0]}"
            )
        if week != extract_week_from_filepath(p):
            raise ValueError(
                f"The week {week} does not match the week in {p} with {input_paths[0]}"
            )
        if year != extract_date_from_filename(os.path.basename(p))[0]:
            raise ValueError(
                f"The year {year} does not match the year in {p} with {input_paths[0]}"
            )
        if month != extract_date_from_filename(os.path.basename(p))[1]:
            raise ValueError(
                f"The month {month} does not match the month in {p} with {input_paths[0]}"
            )
        if day != extract_date_from_filename(os.path.basename(p))[2]:
            raise ValueError(
                f"The day {day} does not match the day in {p} with {input_paths[0]}"
            )
        if roi != extract_roi_from_nd2_filename(os.path.basename(p)):
            raise ValueError(
                f"The roi {roi} does not match the roi in {p} with {input_paths[0]}"
            )

    return subject_id, week, year, month, day, roi
