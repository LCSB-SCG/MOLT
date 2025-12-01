import numpy as np
import pandas as pd

from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_number_from_week_string,
)


def pandas_df_from_mapping(data, y_log_scale=False, no_weeks=False):
    """
    Creates a pandas dataframe from the subject-roi-instance-week-measurement data dict.

    Args:
        data: Dictionary of the data.
        y_log_scale: If True, the y-axis is log scaled.
        no_weeks: If True, the dataframe does not contain the weeks column e.g. for color dataframes that contain a color per instance id.
    """
    if no_weeks:
        df = pd.DataFrame(columns=["subject_id", "roi", "ts_id", "value"])
    else:
        df = pd.DataFrame(columns=["subject_id", "roi", "ts_id", "weeks", "value"])

    for subject_id in data.keys():
        for roi in data[subject_id].keys():
            if no_weeks:
                for instance_id, value in data[subject_id][roi].items():
                    if "average" in instance_id:
                        continue
                    df.loc[len(df)] = {
                        "subject_id": subject_id,
                        "roi": roi,
                        "ts_id": instance_id,
                        "value": np.log(value) if y_log_scale else value,
                    }
            else:
                for instance_id, week_dict in data[subject_id][roi].items():
                    if "average" in instance_id:
                        continue
                    for week, value in week_dict.items():
                        df.loc[len(df)] = {
                            "subject_id": subject_id,
                            "roi": roi,
                            "ts_id": instance_id,
                            "weeks": extract_number_from_week_string(week),
                            "value": np.log(value) if y_log_scale else value,
                        }
    return df
