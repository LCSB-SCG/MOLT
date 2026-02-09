import os
import re

from src.utils.io.data_structure.files_and_paths.patterns import (
    CHANNEL_PATTERN,
    DATE_PATTERN,
    MATCHING_CHANNEL_A_SET_TO_CHANNEL_B_SET_FOLDER_NAME,
    ROI_PATTERN,
    SUBJECT_ID_PATTERN,
    WEEKS_PATTERN,
)


def get_subject_folder(subject: str, data_path: str):
    """
    Constructs the path to the subject folder.
    """
    return os.path.join(data_path, subject)


def get_subject_roi_folder(
    subject_folder: str, week: str, roi: str, information_src: str
):
    """
    Returns the subject roi folder path.

    Args:
        subject_folder (str): The path to the subject folder.
        week (str): The week folder.
        roi (str): The region of interest.
    """
    return os.path.join(subject_folder, week, information_src, roi)


def get_available_subjects(data_path):
    """
    Get a list of all available subjects in the data path.

    Args:
        data_path (str): The path to the data folder.

    Returns:
        list: A list of all available subjects.
    """
    folders = os.listdir(data_path)

    if len(folders) == 0:
        raise ValueError("Directory is empty")

    subjects = []
    for f in folders:
        if SUBJECT_ID_PATTERN.match(f):
            subjects.append(f)

    return subjects


def get_available_rois(subject_folder: str):
    """
    Get a list of all available rois present in each weak in the subject folder .

    Args:
        subject_folder (str): The path to the subject folder.

    Returns:
        list: A list of all available rois.
    """
    weeks = get_timepoint_folders_names(subject_folder)
    all_rois = set()
    # collect all rois
    for week in weeks:
        nifti_folder = os.path.join(subject_folder, week, "nifti")
        if not os.path.exists(nifti_folder):
            continue
        folders = os.listdir(nifti_folder)
        for roi in folders:
            if is_roi_dir(roi):
                all_rois.add(roi)
    return sorted(list(all_rois))


def get_subject_roi_weeks(
    subject: str, roi: str = None, information_src: str = "nifti", data_path: str = None
) -> list:
    """
    Returns a list of all weeks numerically sorted for a given roi and subject.

    Args:
        subject (str): subject ID or subject folder path
        roi (str): region of interest
        information_src (str): source of the information (e.g. nifti, labels)
        data_path (str): path to the data folder
    """
    if not os.sep in subject:
        # if subject is an id = not containing os.sep
        if data_path is None:
            raise ValueError("No data path provided when subject is an id")
        subject_folder = get_subject_folder(subject, data_path=data_path)
    else:
        # subject is already the path
        subject_folder = subject
    weeks = get_timepoint_folders_names(subject_folder)
    if roi is None:
        # Return all weeks of the subject
        return weeks
    else:
        # Select only weeks that contain the roi
        roi_weeks = []
        for week in weeks:
            roi_folder = get_subject_roi_folder(
                subject_folder=subject_folder,
                week=week,
                roi=roi,
                information_src=information_src,
            )
            if os.path.exists(roi_folder):
                roi_weeks.append(week)
        return roi_weeks


def convert_week_int_to_week_string(week: int):
    """
    Returns the week string from a week number.

    Args:
        week (int): The week number.
    """
    return f"{week}_weeks"


def get_week_int_from_week_string(week: str):
    """
    Returns the week number as an integer from a week string.

    Args:
        week (str): The week string.
    """
    match = re.search(WEEKS_PATTERN, week)
    if match:
        return int(match.group(1))
    return None


def get_timepoint_folders_names(subject_folder: str):
    """
    Returns a list of all week folders in the source folder.

    Args:
        source (str): subject folder containing the week recording folders
    """
    # get all week folders
    all_folders = os.listdir(subject_folder)

    week_folders = {}
    has_baseline = False
    for folder in all_folders:
        if folder == "0_weeks":
            # check if there is a baseline folder
            has_baseline = True
            continue
        if "weeks" not in folder:
            # skip all folders that do not follow the data structure
            continue
        week_number = extract_number_from_week_string(folder)
        if week_number is None:
            # skip all folders that do not follow the data structure
            continue
        week_folders[week_number] = folder

    if not has_baseline:
        raise ValueError("No baseline folder found")
    else:
        week_folders[0] = "0_weeks"
        week_list = [week_folders[week] for week in sorted(week_folders.keys())]

    return week_list


def extract_subject_from_filepath(filepath):
    """
    Extracts the subject id from a nd2 filepath.

    Args:
        filepath (str): The filepath to extract the subject id from.

    Returns:
        str: The subject id or None if no subject id was found.
    """
    parts = filepath.split(os.sep)
    for part in parts:
        if SUBJECT_ID_PATTERN.match(part):
            return part
    return None


def extract_week_from_filepath(filepath):
    """
    Extracts the week number from a nd2 filepath.

    Args:
        filepath (str): The filepath to extract the week number from.

    Returns:
        str: The week number or None if no week number was found.
    """
    parts = filepath.split(os.sep)
    for part in parts:
        match = WEEKS_PATTERN.match(part)
        if match:
            return part
    return None


def extract_subject_folder_from_filepath(filepath):
    """
    Extracts the subject folder from a filepath.

    Args:
        filepath (str): The filepath to extract the subject folder from.

    Returns:
        str: The subject folder or None if no subject folder was found.
    """
    parts = filepath.split(os.sep)
    for i, part in enumerate(parts):
        if SUBJECT_ID_PATTERN.match(part):
            if filepath[0] == os.sep:
                return os.path.join(os.sep, *parts[: i + 1])
            else:
                return os.path.join(*parts[: i + 1])
    return None


def extract_number_from_week_string(input_string):
    """
    Extracts the first number from a string.

    Args:
        input_string (str): string to extract the number from
    """
    match = re.search(WEEKS_PATTERN, input_string)
    if match:
        return int(match.group(1))
    return None


def extract_date_from_filename(filename):
    """
    Extracts the date from a filename.

    Args:
        filename (str): The filename to extract the date from.

    Returns:
        str: The date or None if no date was found.
    """
    match = re.search(DATE_PATTERN, filename)
    if match:
        year = match.group(1)
        month = match.group(3)
        day = match.group(5)
        if len(year) == 2:
            year = "20" + year
        return year, month, day
    else:
        return None


def extract_roi_from_filepath(filepath):
    """
    Extracts the roi from a filename.

    Args:
        filename (str): The filename to extract the roi from.

    Returns:
        str: The roi or None if no roi was found.
    """
    parts = filepath.split(os.sep)
    for part in parts:
        if ROI_PATTERN.match(part):
            return part
    return None


def filter_images_by_roi(paths, roi):
    """
    Filters the list of images by their ROI in the filename and returns the filtered list.
    """
    filtered = []
    for path in paths:
        if re.search(r"\b\d{0,4}\_\d{0,2}\_\d{0,2}\_" + roi, path):
            filtered.append(path)
    return filtered


def is_roi_dir(dir: str):
    """
    Check if the given directory is a roi directory.
    """
    if os.sep in dir:
        dir = dir.split(os.sep)[-1]
    return bool(ROI_PATTERN.match(dir))


def is_channel_dir(dir: str):
    """
    Check if the given directory is a channel directory.
    """
    if os.sep in dir:
        dir = dir.split(os.sep)[-1]
    return bool(CHANNEL_PATTERN.match(dir))


def is_match_channel_a_to_channel_b_dir(dir: str):
    """
    Check if the given directory is a channel a to channel b directory.
    """
    if os.sep in dir:
        dir = dir.split(os.sep)[-1]
    return bool(MATCHING_CHANNEL_A_SET_TO_CHANNEL_B_SET_FOLDER_NAME.match(dir))


def get_all_subject_week_rois(
    data_path: str = None,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    if data_path is None:
        raise ValueError("No data path provided")

    subjects = get_available_subjects(data_path=data_path)
    subject_week_rois = {}

    for s in subjects:
        # gather weeks for each subject
        weeks = get_timepoint_folders_names(os.path.join(data_path, s))
        subject_week_rois[s] = {w: {} for w in weeks}
        # gather rois and associated files for each week
        for w in weeks:
            all_files = os.listdir(os.path.join(data_path, s, w, "nifti"))
            for f in all_files:
                if os.path.isdir(
                    os.path.join(data_path, s, w, "nifti", f)
                ) and is_roi_dir(f):
                    path = os.path.join(data_path, s, w, "nifti", f)
                    if f not in subject_week_rois[s][w].keys():
                        subject_week_rois[s][w][f] = [path]
                    else:
                        subject_week_rois[s][w][f].append(path)

    return subject_week_rois
