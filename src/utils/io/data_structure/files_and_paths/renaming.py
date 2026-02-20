import os

from src.utils.io.nd2.load_nd2 import load_nd2
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_date_from_filename,
)
from src.utils.io.data_structure.files_and_paths.patterns import SUBJECT_ID_PATTERN


def replace_space_with_underscore(path: str):
    """
    Replaces spaces in a string with underscores for all files and subdirectories in a directory.

    Args:
        path (str): The path to the directory.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if " " in file:
                os.rename(
                    os.path.join(root, file), os.path.join(root, file.replace(" ", "_"))
                )
        for dir in dirs:
            if " " in dir:
                os.rename(
                    os.path.join(root, dir), os.path.join(root, dir.replace(" ", "_"))
                )
            replace_space_with_underscore(os.path.join(root, dir.replace(" ", "_")))


def rename_week_folders(path: str):
    """
    Renames week folders to the following format:

    <week_number>_weeks
    """
    for root, dirs, _ in os.walk(path):
        if not SUBJECT_ID_PATTERN.match(root):
                continue
        else:
            for dir in dirs:
                # if not matches with the subject name
                print(f"Renaming {os.path.join(root, dir)}")
                week_id = dir.lower()
                print(week_id)
                print(f"maches: {week_id == 'bl' or week_id == 'baseline'}")

                if "weeks" not in week_id and "week" in week_id:
                    week_id = week_id.replace("week", "weeks")
                elif week_id == "bl" or week_id == "baseline":
                    week_id = "0_weeks"

                # insert underscore
                if "_" not in week_id:
                    week_id = week_id.replace("weeks", "_weeks")

                # do temporary renaming to name that differs in more then just case-sensitivity
                os.rename(os.path.join(root, dir), os.path.join(root, week_id + "_temp"))

                # rename to final name
                os.rename(
                    os.path.join(root, week_id + "_temp"), os.path.join(root, week_id)
                )
        for dir in dirs:
            rename_week_folders(os.path.join(root, dir))


def rename_files(path: str, channel_mapping: dict):
    """
    Renames files to the following format:

    <year>_<month>_<day>_<channel_names>_<image_id>.<file_extension>
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".nd2"):
                continue

            new_file_name = ""
            # Extract the date from the file name using a regular expression
            # Has the following format: yyyy.mm.dd or yy.mm.dd at the beginning of the file name
            year, month, day = extract_date_from_filename(file)

            _, metadata, _, _, _, _ = load_nd2(os.path.join(root, file))

            channels_string = [c["channel"]["name"] for c in metadata["channels"]]

            # for i, channel in enumerate(channels_string):
            #     if channel not in channel_mapping:
            #         raise ValueError(f"Channel {channel} not found in channel mapping.")
            # channels_string = sorted(channels_string, key=lambda x: channel_mapping.get(x, float('inf')))
            channels_string = "_".join(channels_string).replace(" ", "-")

            if year:
                new_file_name = f"{year}_{month}_{day}_"
            else:
                print(f"Date not found in file {file}.")
                continue
            new_file_name += channels_string + "_" + file[-7:]
            # Rename the file
            os.rename(os.path.join(root, file), os.path.join(root, new_file_name))
        for dir in dirs:
            rename_files(os.path.join(root, dir), channel_mapping=channel_mapping)


def remove_emtpy_dirs(path: str):
    """
    Removes all empty directories in a directory.

    Args:
        path (str): The path to the directory.
    """
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            path = os.path.join(root, dir)
            if len(os.listdir(path)) == 0:
                os.removedirs(path)
            else:
                remove_emtpy_dirs(path)


def remove_non_nd2_files(path: str, data_path: str = None):
    """
    Removes all files that are not .nd2 files in all subdirectories.

    Args:
        path (str): The path to the directory.
    """
    if data_path is None:
        data_path = path
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".nd2") and root != data_path:
                os.remove(os.path.join(root, file))
        for dir in dirs:
            remove_non_nd2_files(os.path.join(root, dir))
