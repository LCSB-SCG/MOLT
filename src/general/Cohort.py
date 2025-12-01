import os
import warnings
from typing import Union

from src.general.Config import Config
from src.utils.io.data_structure.files_and_paths.constants import (
    AFFINE_TRANS,
    DEFORMABLE_TRANS,
    INSTANCES_ACROSS_CHANNELS_FOLDER,
    OVERLAP_NIFTI_NAME,
)
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_number_from_week_string,
    get_available_rois,
    get_available_subjects,
    get_subject_roi_weeks,
    is_channel_dir,
    is_match_channel_a_to_channel_b_dir,
    is_roi_dir,
)
from src.utils.io.data_structure.files_and_paths.nifti_path_operations import (
    extract_roi_from_nifti_filename,
)
from src.utils.io.data_structure.files_and_paths.patterns import (
    CHANNEL_A_RELATED_TO_CHANNEL_B_FILE_NAME,
    CHANNEL_PATTERN,
    GRAPH_LINEAGE_FOLDER_NAME,
    ROI_PATTERN,
    WEEKS_PATTERN,
)
from src.utils.io.data_structure.files_and_paths.timeseries_metadata import (
    get_match_channel_a_settings_to_channel_b_settigs_folder_name,
    get_match_channel_a_to_channel_b_file_name,
)


class Cohort:
    """
    A class to represent a cohort of subjects. The class is used to gather all available subjects and their associated
    weeks, rois and channels, along side the meta data.
    """

    supported_information_sources = [
        "nifti",
        "microscopy_metadata",
        "rgb",
        "label",
        INSTANCES_ACROSS_CHANNELS_FOLDER,
        "lineage_graph",
        "lineage",
        "tracked",
    ]
    allowed_label_file_types = [
        "instances",
        "binary",
        "meta",
        "padded_instances_for_registration",
        "lineage",
        "tracked",
    ]

    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.data_path = config["general"]["data_path"]
        self.results_path = config["general"]["result_folder"]
        self.information_sources = config["general"]["information_sources"]
        self.subjects = config["general"]["only_use_this_subjects"]

        for info_source in self.information_sources:
            assert info_source in self.supported_information_sources, (
                f"Information source {info_source} is not supported. "
                f"Supported information sources are {self.supported_information_sources}"
            )

        self.base_information_source = config["general"]["base_information_source"]
        self.update_data()

    def __str__(self):
        string_rep = ""
        for subject in self.get_subjects():
            string_rep += f"Subject: {subject}\n"
            for roi in self.get_subject_rois(subject):
                string_rep += f"\tROI: {roi}\n"
                for roi_info_source in self.get_subject_roi_ionformation_sources(
                    subject=subject, roi=roi
                ):
                    string_rep += f"\t\t\tInformation source: {roi_info_source}\n"
                    for f in self.data[subject][roi][roi_info_source]:
                        string_rep += f"\t\t\t\tMatching: {f}\n"
                for week in self.get_subject_roi_weeks(subject_id=subject, roi=roi):
                    string_rep += f"\t\tWeek: {week}\n"
                    for info_source in self.get_subject_roi_week_information_sources(
                        subject=subject, roi=roi, week=week
                    ):
                        string_rep += f"\t\t\tInformation source: {info_source}\n"
                        for (
                            channel
                        ) in self.get_subject_roi_week_information_source_channels(
                            subject=subject,
                            roi=roi,
                            week=week,
                            information_source=info_source,
                        ):
                            string_rep += f"\t\t\t\tChannel: {channel}\n"
                            if info_source == "label":
                                for label_setting in self.data[subject][roi][week][
                                    info_source
                                ][channel]:
                                    string_rep += (
                                        f"\t\t\t\t\tLabel setting: {label_setting}\n"
                                    )
                                    try:
                                        for reg_to in self.data[subject][roi][week][
                                            info_source
                                        ][channel][label_setting]["registrations_to"]:
                                            string_rep += f"\t\t\t\t\t\tRegistrations to: {reg_to}\n"
                                    except KeyError:
                                        pass
                            elif info_source == "nifti":
                                try:
                                    for reg_to in self.data[subject][roi][week][
                                        info_source
                                    ][channel]["registrations_to"]:
                                        string_rep += (
                                            f"\t\t\t\t\tRegistrations to: {reg_to}\n"
                                        )
                                except KeyError:
                                    pass

        return string_rep

    def update_data(self):
        """
        Collects all the paths to available information of the subjects, weeks,
        rois and channels.

        The Cohort will be a dictionary where the final value of each branch is the path to the file.

        The generak structure of the dictionary is as follows:
            subject_id:
                roi:
                    week:
                        information_source:
                            ...
        """
        self.data = {}
        self.label_with_matched_instances_over_time = []

        if self.subjects is None:
            subjects = get_available_subjects(data_path=self.data_path)
        else:
            subjects = self.subjects
        subject_roi_weeks_src = {s: {} for s in subjects}

        for s in subjects:
            # gather rois for each subject
            rois = get_available_rois(subject_folder=os.path.join(self.data_path, s))
            for r in rois:
                subject_roi_weeks_src[s][r] = {}
                # collect the instances across channels matching json files for each roi
                if "lineage_graph" in self.information_sources:
                    folder_path = os.path.join(self.results_path, s, "lineage_graphs")
                    info_data = self.collect_lineage_graph_file(
                        subject=s, roi=r, gt_file=False, with_associated_ids=False
                    )
                    if info_data is not None:
                        subject_roi_weeks_src[s][r]["lineage_graph"] = info_data
                    info_data = self.collect_lineage_graph_file(
                        subject=s, roi=r, gt_file=True, with_associated_ids=False
                    )
                    if info_data is not None:
                        subject_roi_weeks_src[s][r][
                            "lineage_graph_ground_truth"
                        ] = info_data
                    info_data = self.collect_lineage_graph_file(
                        subject=s, roi=r, gt_file=False, with_associated_ids=True
                    )
                    if info_data is not None:
                        subject_roi_weeks_src[s][r][
                            "lineage_graph_with_associated_gt_ids"
                        ] = info_data
                    info_data = self.collect_lineage_graph_file(
                        subject=s, roi=r, gt_file=True, with_associated_ids=True
                    )
                    if info_data is not None:
                        subject_roi_weeks_src[s][r][
                            "lineage_graph_ground_truth_with_associated_tracked_ids"
                        ] = info_data

                if INSTANCES_ACROSS_CHANNELS_FOLDER in self.information_sources:
                    folder_path = os.path.join(
                        self.data_path, s, INSTANCES_ACROSS_CHANNELS_FOLDER
                    )
                    info_data = self.collect_tracking_and_matchings(
                        tracking_folder=folder_path, roi=r
                    )
                    if info_data is not None:
                        subject_roi_weeks_src[s][r][
                            INSTANCES_ACROSS_CHANNELS_FOLDER
                        ] = info_data

                # gather weeks for each roi
                weeks = get_subject_roi_weeks(
                    subject=os.path.join(self.data_path, s),
                    roi=r,
                    information_src=self.base_information_source,
                )

                # gather rois and associated files for each week
                for w in weeks:
                    subject_roi_weeks_src[s][r][w] = {}

                    # gather information sources for each week
                    for info_source in self.information_sources:
                        folder = os.path.join(self.data_path, s, w, info_source)
                        if info_source == "nifti":
                            info_data = self.collect_nifit_files(
                                nifti_folder=folder, roi=r
                            )
                        elif info_source == "microscopy_metadata":
                            info_data = self.collect_metadata(
                                metadata_folder=folder, roi=r
                            )
                        elif info_source == "rgb":
                            info_data = self.collect_rgb(rgb_folder=folder, roi=r)
                        elif info_source == "label":
                            info_data = self.collect_labels(label_folder=folder, roi=r)
                        else:
                            info_data = None

                        if info_data is not None:
                            subject_roi_weeks_src[s][r][w][info_source] = info_data
        self.data = subject_roi_weeks_src

    def collect_labels(self, label_folder, roi):
        """
        Collects all the labels from a given path.
        """
        label_data = {}
        # Iterate through the roi folders
        roi_label_folder = os.path.join(label_folder, roi)
        if not os.path.exists(roi_label_folder):
            return None

        if is_roi_dir(roi):
            label_data = {}
            # iterate through the channels
            for channel in os.listdir(roi_label_folder):
                if channel.startswith("vessels"):
                    warnings.warn(
                        message="Ignoring vessel channel, as it is not used for the analysis for now."
                    )
                    continue
                elif is_channel_dir(channel):
                    label_data[channel] = {}
                    # Iterate through the available label setting folder
                    for label_setting in os.listdir(
                        os.path.join(roi_label_folder, channel)
                    ):
                        label_data[channel][label_setting] = {}
                        label_data[channel][label_setting]["registrations_to"] = {}
                        # Iterate through the available files
                        for file in os.listdir(
                            os.path.join(roi_label_folder, channel, label_setting)
                        ):
                            path = os.path.join(
                                roi_label_folder, channel, label_setting, file
                            )
                            # Save if the matched instances over time are available
                            if os.path.isdir(path) and ("registration" in file):
                                reg_to = file[16:]

                                if WEEKS_PATTERN.match(reg_to) is None:
                                    if path.endswith("preceding_timepoint"):
                                        reg_to = "preceding_timepoint"
                                    else:
                                        raise ValueError(
                                            f"Registration folder naclearme {file} does not match the expected pattern."
                                        )

                                label_data[channel][label_setting]["registrations_to"][
                                    reg_to
                                ] = {}
                                for reg_file in os.listdir(path):
                                    if reg_file.endswith("_reg.nii.gz"):
                                        label_data[channel][label_setting][
                                            "registrations_to"
                                        ][reg_to]["reg_instances"] = os.path.join(
                                            path, reg_file
                                        )
                                if (
                                    label_data[channel][label_setting][
                                        "registrations_to"
                                    ][reg_to]
                                    == {}
                                ):
                                    # in case the registration folder is empty
                                    label_data[channel][label_setting][
                                        "registrations_to"
                                    ].pop(reg_to)
                            else:
                                if "matched" in file:
                                    label_data[channel][label_setting][
                                        "matched_instances_over_time_available"
                                    ] = True
                                    self.label_with_matched_instances_over_time.append(
                                        os.path.join(
                                            roi_label_folder, channel, label_setting
                                        )
                                    )

                                    if file.endswith(
                                        "instances_matched_by_union.nii.gz"
                                    ):
                                        label_data[channel][label_setting][
                                            "instances_matched_by_union"
                                        ] = path

                                else:
                                    label_data[channel][label_setting][
                                        "matched_instances_over_time_available"
                                    ] = False

                                # Save the path to the file
                                if file.endswith("_padded_for_reg.nii.gz"):
                                    label_data[channel][label_setting][
                                        "padded_instances_for_registration"
                                    ] = path
                                elif file.endswith(".nii.gz"):
                                    label_data[channel][label_setting][
                                        file[15:-7]
                                    ] = path
                                elif file.endswith("_meta.json"):
                                    label_data[channel][label_setting][
                                        file[15:-5]
                                    ] = path
                        if label_data[channel][label_setting]["registrations_to"] == {}:
                            # in case the registration folder is empty
                            label_data[channel][label_setting].pop("registrations_to")
        return label_data

    def collect_nifit_files(self, nifti_folder, roi):
        """
        Collect all the nifti files from a given path.
        """
        nifti_data = {}
        roi_folder = os.path.join(nifti_folder, roi)
        if not os.path.exists(roi_folder):
            return None
        if is_roi_dir(roi_folder):
            nifti_data = {}
            for channel in os.listdir(roi_folder):
                nifti_data[channel] = {}
                nifti_data[channel]["registrations_to"] = {}
                nifti_data[channel]["unregistered"] = None  # Single file
                for file in os.listdir(os.path.join(roi_folder, channel)):
                    path = os.path.join(roi_folder, channel, file)
                    if os.path.isdir(path) and "registration" in file:
                        # save the registrated files
                        reg_to = file[16:]
                        nifti_data[channel]["registrations_to"][reg_to] = {}
                        for reg_file in os.listdir(path):
                            if reg_file.endswith("_reg.nii.gz"):
                                nifti_data[channel]["registrations_to"][reg_to][
                                    "reg_nifti"
                                ] = os.path.join(path, reg_file)
                            elif reg_file == AFFINE_TRANS:
                                nifti_data[channel]["registrations_to"][reg_to][
                                    AFFINE_TRANS
                                ] = os.path.join(path, reg_file)
                            elif reg_file == DEFORMABLE_TRANS:
                                nifti_data[channel]["registrations_to"][reg_to][
                                    DEFORMABLE_TRANS
                                ] = os.path.join(path, reg_file)
                            elif reg_file == OVERLAP_NIFTI_NAME:
                                nifti_data[channel]["registrations_to"][reg_to][
                                    OVERLAP_NIFTI_NAME
                                ] = os.path.join(path, reg_file)
                        if nifti_data[channel]["registrations_to"][reg_to] == {}:
                            # in case the registration folder is empty
                            nifti_data[channel]["registrations_to"].pop(reg_to)
                    elif file.endswith(".nii.gz"):
                        # save the unregistered files
                        nifti_data[channel]["unregistered"] = path

        return nifti_data

    def collect_metadata(self, metadata_folder, roi):
        """
        Collect the microscopy metadata for a specific subject-week-roi.
        """
        metadata_data = {}

        if not os.path.exists(metadata_folder):
            return None

        for file in os.listdir(metadata_folder):
            if file.endswith("meta.json"):
                file_roi = file[11:14]
                if file_roi == roi:
                    path = os.path.join(metadata_folder, file)
                    metadata_data = path
                    break

        if metadata_data == {}:
            return None
        else:
            return metadata_data

    def collect_rgb(self, rgb_folder: str, roi: str):
        """
        Collect all the rgb image for a specific week, roi and subject.
        """
        if not os.path.exists(rgb_folder):
            return None
        for file in os.listdir(rgb_folder):
            if file.endswith(".nii.gz"):
                file_roi = extract_roi_from_nifti_filename(filename=file)
                if file_roi == roi:
                    return os.path.join(rgb_folder, file)
        return None

    def collect_tracking_and_matchings(self, tracking_folder: str, roi: str):
        """
        Collect all the tracking and matching information for a specific week, roi and subject.
        """
        tracking_data = {}
        if not os.path.exists(tracking_folder):
            return None
        if roi in os.listdir(tracking_folder) and is_roi_dir(roi):
            for f in os.listdir(os.path.join(tracking_folder, roi)):
                if is_match_channel_a_to_channel_b_dir(f):
                    tracking_data[f] = {}
                    for file in os.listdir(os.path.join(tracking_folder, roi, f)):
                        if CHANNEL_A_RELATED_TO_CHANNEL_B_FILE_NAME.match(file):
                            tracking_data[f][file[:-5]] = os.path.join(
                                tracking_folder, roi, f, file
                            )
        return tracking_data

    def collect_lineage_graph_file(
        self, subject: str, roi: str, gt_file: bool, with_associated_ids: bool = False
    ) -> Union[str, None]:
        """
        Collect the lineage graph file for a specific subject and roi.
        """
        folder_name = os.path.join(self.results_path, GRAPH_LINEAGE_FOLDER_NAME)
        if not gt_file:
            if not with_associated_ids:
                # the tracked graph WITHOUT the associated ground truth ids
                file_path = os.path.join(folder_name, f"{subject}_{roi}_graph.graphml")
            else:
                # the tracked graph WITH the associated ground truth ids
                file_path = os.path.join(
                    folder_name, f"{subject}_{roi}_graph_added_gt_ids.graphml"
                )
        else:
            if not with_associated_ids:
                # the ground truth graph WITOUT the associated tracked ids
                file_path = os.path.join(
                    folder_name, f"{subject}_{roi}_lineage_gt.graphml"
                )
            else:
                # the ground truth graph WITH the associated tracked ids
                file_path = os.path.join(
                    folder_name, f"{subject}_{roi}_lineage_gt_added_tracked_ids.graphml"
                )
        if os.path.exists(file_path):
            return file_path
        else:
            return None

    def get_label_paths_with_matched_instances_available(self):
        """
        Returns the label paths for the matched instances over time.
        """
        return self.label_with_matched_instances_over_time

    def get_subjects(self):
        """
        Returns a list of available subjects.

        Returns:
            list: List of available subjects.
        """
        return list(self.data.keys())

    def get_subject_rois(self, subject_id):
        """
        Returns a list of available rois for a given subject.

        Args:
            subject_id (str): The subject to get the rois for.

        Returns:
            list: List of available rois for the subject.
        """
        rois = []
        for roi in self.data[subject_id]:
            if bool(ROI_PATTERN.match(roi)):
                rois.append(roi)
        return rois

    def get_all_weeks(self):
        """
        Collect all weeks available in the cohort.
        """
        weeks = {}
        subject_weeks = set()
        for subject in self.get_subjects():
            for roi in self.get_subject_rois(subject_id=subject):
                s_r_w = self.get_subject_roi_weeks(subject_id=subject, roi=roi)
                subject_weeks = subject_weeks.union(set(s_r_w))
        for w in subject_weeks:
            weeks[extract_number_from_week_string(w)] = w
        return [weeks[w] for w in sorted(weeks.keys())]

    def get_subject_roi_weeks(self, subject_id, roi=None):
        """
        Returns a list of available weeks for a given subject and roi or all the subject weeks.

        Args:
            subject (str): The subject to get the weeks for.
            roi (str, optional): The roi to get the weeks for. Defaults to None.

        Returns:
            list: List of time-ordered available weeks for the subject sorted in by time.
        """
        return get_subject_roi_weeks(
            subject=subject_id,
            roi=roi,
            information_src=self.base_information_source,
            data_path=self.data_path,
        )

    def get_subject_roi_ionformation_sources(self, subject, roi):
        """
        Returns a list of available information sources for a given subject and roi.

        Args:
            subject (str): The subject to get the information sources for.
            roi (str): The roi to get the information sources for.

        Returns:
            list: List of available information sources for the subject.
        """
        info_sources = self.data[subject][roi].keys()
        # remove all keys that are not information sources
        return [i for i in info_sources if i in self.supported_information_sources]

    def get_subject_roi_week_information_sources(self, subject, roi, week):
        """
        Returns a list of available information sources for a given subject, roi and week.

        Args:
            subject (str): The subject to get the information sources for.
            roi (str): The roi to get the information sources for.
            week (str): The week to get the information sources for.

        Returns:
            list: List of available information sources for the subject.
        """
        info_sources = self.data[subject][roi][week].keys()
        # remove all keys that are not information sources
        return [i for i in info_sources if i in self.supported_information_sources]

    def get_subject_roi_week_information_source_channels(
        self, subject, roi, week, information_source
    ):
        """
        Returns a list of available channels for a given subject, roi, week and information source.

        Args:
            subject (str): The subject to get the channels for.
            roi (str): The roi to get the channels for.
            week (str): The week to get the channels for.
            information_source (str): The information source to get the channels for.

        Returns:
            list: List of available channels for the subject.
        """
        channels = []
        for channel in self.data[subject][roi][week][information_source]:
            if bool(CHANNEL_PATTERN.match(channel)):
                channels.append(channel)
        return channels

    def get_across_channel_instance_matching_files(
        self,
        subject: str,
        roi: str,
        channel: str,
        label_setting: str,
        matched_channel: str,
        matched_channel_setting: str,
    ) -> list[str]:
        """
        Get the instance matching files for a specific subject and specific roi.
        """
        match_name = get_match_channel_a_settings_to_channel_b_settigs_folder_name(
            channel_a_id=channel,
            channel_b_id=matched_channel,
            label_setting_a=label_setting,
            label_setting_b=matched_channel_setting,
        )
        if (
            match_name
            in self.data[subject][roi][INSTANCES_ACROSS_CHANNELS_FOLDER].keys()
        ):
            pass
        else:
            raise ValueError(
                f"Matching channel {channel} to {matched_channel} not found for {subject}, {roi}"
            )
        filename = get_match_channel_a_to_channel_b_file_name(
            channel_a_id=channel, channel_b_id=matched_channel
        )[:-5]
        return [
            self.data[subject][roi][INSTANCES_ACROSS_CHANNELS_FOLDER][match_name][
                filename
            ]
        ]

    def get_lineage_graph_file(self, subject: str) -> Union[str, None]:
        """
        Get the lineage graph file for a specific subject.
        """
        folder_name = os.path.join(self.results_path, GRAPH_LINEAGE_FOLDER_NAME)
        file_path = os.path.join(folder_name, f"{subject}_graph.graphml")
        if not os.path.exists(folder_name):
            raise ValueError(
                f"Lineage graph folder {folder_name} does not exist. "
                "Please run the lineage graph generation first."
            )
        return file_path

    def get_information_source_files_as_list(
        self,
        information_source: str,
        channel: str,
        subject_ids: Union[str, list[str], None] = None,
        rois: Union[str, list[str], None] = None,
        weeks: Union[str, list[str], None] = None,
        registered: bool = False,
        registration_to: str = None,
        reg_file_type: str = AFFINE_TRANS,
        label_setting: str = None,
        file_type: str = "instances",
        matched_channel: Union[str, None] = None,
        matched_channel_setting: Union[str, None] = None,
    ) -> list[str]:
        """
        The main function to querry cohort files.
        Returns a list of all available files, it will only ever return one type of file.
        Subject, roi, and week can have multiple values. All other paramters can only have one value otherwise their
        would be multiple files with semantically mixed information.

        Args:
            subject (str, optional): The subject to get the nifti files for. Defaults to None.
            roi (str, optional): The roi to get the nifti files for. Defaults to None.
            week (str, optional): The week to get the nifti files for. Defaults to None.
            registered (bool, optional): If True, returns the registered nifti files. Defaults to False.
            channel (str, optional): The channel to get the nifti files for. Defaults to "*".

        Returns:
            list: List of all available nifti files.
        """
        # Assert the parameters
        if registered:
            allowed_file_types = [
                "reg_instances",
                "reg_nifti",
                AFFINE_TRANS,
                DEFORMABLE_TRANS,
                OVERLAP_NIFTI_NAME,
            ]
            # Accessing registered files
            if registration_to is None:
                raise ValueError("If registered is True, registration_to must be set.")
            if information_source not in ["nifti", "label"]:
                raise ValueError(
                    "If registered is True, information_source must be either 'nifti' or 'label'."
                )
            if reg_file_type not in allowed_file_types:
                raise ValueError(f"File type must be one of {allowed_file_types}")
        else:
            if information_source == "label":
                if label_setting is None:
                    raise ValueError(
                        "If information_source is 'label', label_setting must be set."
                    )
                if file_type not in self.allowed_label_file_types:
                    raise ValueError(
                        f"File type must be in {self.allowed_label_file_types}"
                    )
            elif information_source == INSTANCES_ACROSS_CHANNELS_FOLDER:
                # Accessing instances across channels matching files can be return for a specific subject and roi
                if matched_channel is None:
                    raise ValueError(
                        "If information_source is 'across_channel_instance_matching', matched_channel must be set."
                    )
                if matched_channel_setting is None:
                    raise ValueError(
                        "If information_source is 'across_channel_instance_matching', matched_channel_setting must be set."
                    )
                if label_setting is None:
                    raise ValueError(
                        "If information_source is 'across_channel_instance_matching', label_setting must be set."
                    )
                if rois is None or (type(rois) == list and len(rois) != 1):
                    raise ValueError(
                        "If information_source is 'across_channel_instance_matching', rois must be set."
                    )
                if subject_ids is None or (
                    type(subject_ids) == list and len(subject_ids) != 1
                ):
                    raise ValueError(
                        "If information_source is 'across_channel_instance_matching', subject_ids must be set."
                    )
            elif information_source == "lineage_graph":
                if file_type not in [
                    "with_associated_gt_ids",
                    "with_associated_tracked_ids",
                    None,
                ]:
                    raise ValueError(
                        "If information_source is 'lineage_graph', file_type must be either 'with_associated_gt_ids', 'with_associated_tracked_ids' or None."
                    )

        file_list = []
        for s in self.get_subjects():
            if (subject_ids is not None) and (s not in subject_ids):
                # Skip if the subject is not the one we are looking for if a subject is specified
                continue
            for r in self.get_subject_rois(subject_id=s):
                if information_source == "lineage_graph":
                    if label_setting == "ground_truth":
                        gt_file = True
                        if file_type == "with_associated_tracked_ids":
                            with_associated_ids = True
                        elif file_type is None:
                            with_associated_ids = False
                        else:
                            raise ValueError(
                                "Ground Truth lineage graph can only be returned with or without the associated TRACKED ids."
                            )
                    else:
                        gt_file = False
                        if file_type == "with_associated_gt_ids":
                            with_associated_ids = True
                        elif file_type is None:
                            with_associated_ids = False
                        else:
                            raise ValueError(
                                "Tracked lineage graph can only be returned with or without the associated GROUND TRUTH ids."
                            )

                    file_list.append(
                        self.collect_lineage_graph_file(
                            subject=s,
                            roi=r,
                            gt_file=gt_file,
                            with_associated_ids=with_associated_ids,
                        )
                    )
                    continue

                if (rois is not None) and (r not in rois):
                    # Skip if the roi is not the one we are looking for if a roi is specified
                    continue
                if information_source == INSTANCES_ACROSS_CHANNELS_FOLDER:
                    return self.get_across_channel_instance_matching_files(
                        subject=s,
                        roi=r,
                        channel=channel,
                        label_setting=label_setting,
                        matched_channel=matched_channel,
                        matched_channel_setting=matched_channel_setting,
                    )

                for w_idx, w in enumerate(
                    self.get_subject_roi_weeks(subject_id=s, roi=r)
                ):
                    if (weeks is not None) and (w not in weeks):
                        # Skip if the week is not the one we are looking for if a week is specified
                        continue
                    for c in self.get_subject_roi_week_information_source_channels(
                        subject=s, roi=r, week=w, information_source=information_source
                    ):
                        if (channel is not None) and (c != channel):
                            # Skip if the channel is not the one we are looking for if a channel is specified
                            continue
                        if registered:

                            if information_source == "label" and w == registration_to:
                                # the original file equals a registration to itself
                                file_list.append(
                                    self.data[s][r][w][information_source][c][
                                        label_setting
                                    ][
                                        "padded_instances_for_registration"  # remove the "reg_" prefix
                                    ]
                                )
                                continue
                            if information_source == "label":
                                # Accessing registered label files --> access label_setting folder
                                reg_files = self.data[s][r][w][information_source][c][
                                    label_setting
                                ]["registrations_to"]
                            else:
                                # Accessing registered nifti files --> access registrations_to folder
                                reg_files = self.data[s][r][w][information_source][c][
                                    "registrations_to"
                                ]

                            for reg_to in reg_files:
                                if (registration_to is not None) and (
                                    reg_to not in registration_to
                                ):
                                    # Skip if the registration is not the one we are looking for if a registration is specified
                                    continue
                                # Append the file to the list
                                file_list.append(reg_files[reg_to][reg_file_type])
                        else:
                            if information_source == "label":
                                if (
                                    label_setting
                                    in self.data[s][r][w][information_source][c]
                                ):
                                    file_list.append(
                                        self.data[s][r][w][information_source][c][
                                            label_setting
                                        ][file_type]
                                    )
                                else:
                                    raise ValueError(
                                        f"Label setting {label_setting} not found for {s}, {r}, {w}, {c}"
                                    )
                            elif information_source == "nifti":
                                file_list.append(
                                    self.data[s][r][w][information_source][c][
                                        "unregistered"
                                    ]
                                )
        return file_list

    def update(self):
        return Cohort(config=self.config)
