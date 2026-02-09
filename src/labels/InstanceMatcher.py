import json
import logging
import os
import threading

import networkx as nx
import nibabel as nib
import numpy as np

from src.general.Cohort import Cohort
from src.general.Config import Config
from src.labels.functional.instance_labeling import collect_meta_data
from src.labels.functional.instance_matching import (
    apply_matching,
    collect_vams_information_for_matched_ids,
    match_instances_by_binary_union,
    match_instances_by_intersection_of_two_images,
    match_instances_by_pairwise_binary_union,
    validate_matched_by_union_images,
)
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_number_from_week_string,
    extract_week_from_filepath,
)
from src.utils.io.data_structure.files_and_paths.timeseries_metadata import (
    get_subject_roi_label_channel_time_series_info_path,
    get_subject_roi_label_time_series_two_channel_mapping_info_path,
)
from src.utils.io.nifti.write_to_nifti import write_to_nifti
from src.utils.cell_tracking_challenge.ctc_format import write_graph_to_ctc, write_to_tiff
from utils.common import logging_to_stdout

allowed_modes = ["pairwise", "global"]
allowed_registration_to = ["first_timepoint", "preceding_timepoint"]


class InstanceMatcher:
    def __init__(self, config, cohort) -> None:
        self.config: Config = config
        self.cohort: Cohort = cohort
        self.max_num_threads = self.config["InstanceMatcher"]["max_num_threads"]
        self.mode = (
            self.config["InstanceMatcher"]["mode"]
            if "mode" in self.config["InstanceMatcher"]
            else "global"
        )
        self.reg_to = (
            self.config["InstanceMatcher"]["registration_to"]
            if "registration_to" in self.config["InstanceMatcher"]
            else "first_timepoint"
        )
        self.threads = []

        if self.reg_to not in allowed_registration_to:
            raise ValueError(
                f"Registration to {self.reg_to} is not allowed. Allowed values are: {allowed_registration_to}"
            )
        if self.mode not in allowed_modes:
            raise ValueError(
                f"Mode {self.mode} is not allowed. Allowed modes are: {allowed_modes}"
            )

        if self.max_num_threads > 1:
            logging.info(
                f"InstanceMatcher will run in parallel with {self.max_num_threads} threads."
            )
        else:
            logging.info("InstanceMatcher will run sequentially.")

    def run(self):
        """
        Runs the matching algorithm to match instances in the same channel labels over time for each region of
        interest of each subject in the cohort.
        """
        # run the per instance tracking with the same channel
        if "label_settings_per_channel" in self.config["InstanceMatcher"]:
            matchings_to_run = self._collect_matching_tasks()

            # split the task into chunks
            matchings_per_thread = np.array_split(
                ary=matchings_to_run,
                indices_or_sections=min(self.max_num_threads, len(matchings_to_run)),
            )

            for mpt in matchings_per_thread:
                mpt = mpt.tolist()
                t = threading.Thread(
                    target=self._run_thread,
                    args=(mpt,),
                )
                self.threads.append(t)
                t.start()

            for t in self.threads:
                # Wait for all threads to finish
                t.join()

        self.cohort.update_data()

        # run the across channel matching
        if "across_channel_matching_settings" in self.config["InstanceMatcher"]:
            across_channel_matchings_to_run = (
                self._collect_across_channel_matching_tasks()
            )

            # split the task into chunks
            across_channel_matchings_per_thread = np.array_split(
                ary=across_channel_matchings_to_run,
                indices_or_sections=min(
                    self.max_num_threads, len(across_channel_matchings_to_run)
                ),
            )

            for acmpt in across_channel_matchings_per_thread:
                acmpt = acmpt.tolist()
                t = threading.Thread(
                    target=self._run_across_channel_matching_thread,
                    args=(acmpt,),
                )
                self.threads.append(t)
                t.start()

            for t in self.threads:
                # Wait for all threads to finish
                t.join()

    def _run_thread(self, match_tasks_for_thread: list[dict]):
        """
        Runs the tasks for the thread sequentially within single thread.

        Args:
            match_tasks_for_thread (list[dict]): The tasks to run for the thread.
        """
        for task_idx, task in enumerate(match_tasks_for_thread):
            logging.info(
                f"Running task {task_idx} out of {len(match_tasks_for_thread)}: Matching for {task['subject_folder']} - {task['roi']} - {task['channel_id']}"
            )

            self._run_matching_from_paths(
                subject_id=task["subject_folder"],
                roi=task["roi"],
                channel_id=task["channel_id"],
                label_settings_folder=task["label_settings_folder"],
                unregistered_label_paths=task["unregistered_label_paths"],
                registered_label_paths=task["registered_label_paths"],
                week_numbers=task["week_numbers"],
                validate=True,
            )

    def _run_across_channel_matching_thread(self, match_tasks_for_thread: list[dict]):
        """
        Runs the tasks for the thread sequentially within single thread.

        Args:
            match_tasks_for_thread (list[dict]): The tasks to run for the thread.
        """
        for task_idx, task in enumerate(match_tasks_for_thread):
            logging.info(
                f"Running task {task_idx} out of {len(match_tasks_for_thread)}: Matching for {task['subject_id']} - {task['roi']} - {task['channel_id_a']} - {task['channel_id_b']}"
            )

            self._run_across_channel_matching_from_paths(
                subject_id=task["subject_id"],
                roi=task["roi"],
                images_channel_a=task["images_channel_a"],
                images_channel_b=task["images_channel_b"],
                channel_id_a=task["channel_id_a"],
                channel_id_b=task["channel_id_b"],
                channel_a_settings=task["channel_a_settings"],
                channel_b_settings=task["channel_b_settings"],
                week_numbers=task["week_numbers"],
                verbose=task["verbose"],
            )

    def _collect_matching_tasks(self) -> list:
        """
        Collects all matching tasks to run as list for parallel processing.
        """
        matchings_to_run = []
        for subject_id in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject_id):
                for label_settings_per_channel in self.config["InstanceMatcher"][
                    "label_settings_per_channel"
                ]:
                    for channel_id, label_setting in label_settings_per_channel.items():
                        weeks = self.cohort.get_subject_roi_weeks(
                            subject_id=subject_id, roi=roi
                        )
                        week_numbers = [
                            extract_number_from_week_string(input_string=week)
                            for week in weeks
                        ]
                        unregisterd_labels = (
                            self.cohort.get_information_source_files_as_list(
                                information_source="label",
                                channel=channel_id,
                                subject_ids=[subject_id],
                                rois=[roi],
                                weeks=weeks,
                                registered=False,
                                label_setting=label_setting,
                                file_type="instances",
                            )
                        )

                        if self.reg_to == "first_timepoint":
                            # register to the first timepoint
                            registration_to = weeks[0]
                        elif self.reg_to == "preceding_timepoint":
                            # register to the preceding timepoint
                            registration_to = (
                                "preceding_timepoint"  # remove the first week
                            )

                        registered_labels = (
                            self.cohort.get_information_source_files_as_list(
                                information_source="label",
                                channel=channel_id,
                                subject_ids=[subject_id],
                                rois=[roi],
                                weeks=weeks,
                                registered=True,
                                registration_to=registration_to,
                                label_setting=label_setting,
                                reg_file_type="reg_instances",
                            )
                        )

                        matchings_to_run.append(
                            {
                                "subject_folder": subject_id,
                                "roi": roi,
                                "channel_id": channel_id,
                                "label_settings_folder": label_setting,
                                "unregistered_label_paths": unregisterd_labels,
                                "registered_label_paths": registered_labels,
                                "week_numbers": week_numbers,
                            }
                        )
        logging.info(
            f"Collected {len(matchings_to_run)} matching tasks for {len(self.cohort.get_subjects())} subjects."
        )
        return matchings_to_run

    def _collect_across_channel_matching_tasks(self) -> list:
        """
        Collects all cross channel matching tasks to run as list for parallel processing.
        """
        matchings_to_run = []
        for subject_id in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject_id):
                for across_channel_matching_setting in self.config["InstanceMatcher"][
                    "across_channel_matching_settings"
                ]:
                    channel_a_id, channel_a_setting = across_channel_matching_setting[
                        "channel_a"
                    ]
                    channel_b_id, channel_b_setting = across_channel_matching_setting[
                        "channel_b"
                    ]

                    weeks = self.cohort.get_subject_roi_weeks(
                        subject_id=subject_id, roi=roi
                    )
                    week_numbers = [
                        extract_number_from_week_string(input_string=week)
                        for week in weeks
                    ]

                    images_channel_a = self.cohort.get_information_source_files_as_list(
                        information_source="label",
                        channel=channel_a_id,
                        subject_ids=[subject_id],
                        rois=[roi],
                        weeks=weeks,
                        registered=False,
                        label_setting=channel_a_setting,
                        file_type="instances_matched_by_union",
                    )

                    images_channel_b = self.cohort.get_information_source_files_as_list(
                        information_source="label",
                        channel=channel_b_id,
                        subject_ids=[subject_id],
                        rois=[roi],
                        weeks=weeks,
                        registered=False,
                        label_setting=channel_b_setting,
                        file_type="instances_matched_by_union",
                    )

                    matchings_to_run.append(
                        {
                            "subject_id": subject_id,
                            "roi": roi,
                            "images_channel_a": images_channel_a,
                            "images_channel_b": images_channel_b,
                            "channel_id_a": channel_a_id,
                            "channel_id_b": channel_b_id,
                            "channel_a_settings": channel_a_setting,
                            "channel_b_settings": channel_b_setting,
                            "week_numbers": week_numbers,
                            "verbose": True,
                        }
                    )
        logging.info(
            f"Collected {len(matchings_to_run)} across channel matching tasks for {len(self.cohort.get_subjects())} subjects."
        )
        return matchings_to_run

    def _run_matching_from_paths(
        self,
        subject_id: str,
        roi: str,
        channel_id: str,
        label_settings_folder: str,
        unregistered_label_paths: list[str],
        registered_label_paths: list[str],
        week_numbers: list[int] = None,
        validate: bool = False,
    ) -> None:
        """
        Matches the instances in the unregistered images to the registered images.

        Args:
            subject_id (str): The subject folder.
            roi (str): The region of interest.
            channel_id (str): The channel id.
            unregistered_label_paths (list[str]): The paths to the unregistered images.
            registered_label_paths (list[str]): The paths to the registered images.
            week_numbers (list[int]): The week numbers of the images.

        Returns:
            list[np.ndarray]: The matched images.

        """
        # load the images
        registered_instance_imgs = [
            nib.load(filename=img).get_fdata() for img in registered_label_paths
        ]
        unregistered_instance_imgs = [
            nib.load(filename=img).get_fdata() for img in unregistered_label_paths
        ]

        # Also make a folder for the ctc results, all the files in this folder are only sym_links to the result files
        # to not waste memory but still have a folder with the correct format for evaluation locally (not sending to the 
        # cell tracking challenge). The evaluation software allows only results folder to be present for evaluation e.g.
        # 01_RES. As we will compare multiple results this is not suited for us so we will sym_link the dataset folders
        # in a results folder per tested algorithm and dataset.
        if "Cells_" in subject_id:
            # Create folder to add files or links
            ctc_ds_folder = os.path.join(self.config["general"]["result_folder"], "ctc")
            ctc_sub_id = subject_id.replace("Cells_", "")
            ctc_sub_RES = os.path.join(ctc_ds_folder, ctc_sub_id + "_RES")

            # Sym_link paths
            ctc_sub_path_src = os.path.join(self.config["general"]["data_path"], ctc_sub_id)
            ctc_sub_path_dst = os.path.join(ctc_ds_folder, ctc_sub_id)
            ctc_sub_ERR_SEG_src = os.path.join(self.config["general"]["data_path"], ctc_sub_id + "_ERR_SEG")
            ctc_sub_ERR_SEG_dst = os.path.join(ctc_ds_folder, ctc_sub_id + "_ERR_SEG")
            ctc_sub_GT_src = os.path.join(self.config["general"]["data_path"], ctc_sub_id + "_GT")
            ctc_sub_GT_dst = os.path.join(ctc_ds_folder, ctc_sub_id + "_GT")
            ctc_sub_ST_src = os.path.join(self.config["general"]["data_path"], ctc_sub_id + "_ST")
            ctc_sub_ST_dst = os.path.join(ctc_ds_folder, ctc_sub_id + "_ST")

            # Create the folder in which we add results
            for p in [ctc_ds_folder, ctc_sub_RES]:
                os.makedirs(p, exist_ok=True)
            # Sym_link to the original dataset
            for p_src, p_dst in zip(
                [ctc_sub_path_src, ctc_sub_ERR_SEG_src, ctc_sub_GT_src, ctc_sub_ST_src],
                [ctc_sub_path_dst, ctc_sub_ERR_SEG_dst, ctc_sub_GT_dst, ctc_sub_ST_dst]):
                if os.path.islink(p_dst):
                    os.unlink(p_dst)
                os.symlink(p_src, p_dst)

        if self.mode == "global":
            # match the instances
            matches, union_instances, _ = match_instances_by_binary_union(
                img_list=registered_instance_imgs,
                background_instance=self.config["general"]["background_instance"],
            )

            write_to_nifti(
                nifti_data=union_instances,
                dtype=union_instances.dtype,
                filename=unregistered_label_paths[0].replace(
                    ".nii.gz", "_union_instances.nii.gz"
                ),
            )

            # apply the matches
            matched_lineage_images = []
            for idx, img in enumerate(unregistered_instance_imgs):
                matched_lineage_images.append(
                    apply_matching(
                        matches=matches[idx],
                        image=img,
                        background_instance=self.config["general"][
                            "background_instance"
                        ],
                        ignore_label=self.config["general"]["ignore_label"],
                    )
                )
        else:
            if validate:
                validate = False
                logging.warning("Validation not supported for pairwise matching.")

            # pairwise matching
            lineage_mapping, tcui_mapping, graph = (
                match_instances_by_pairwise_binary_union(
                    img_list=registered_instance_imgs,
                    background_instance=self.config["general"]["background_instance"],
                    unreg_img_list=unregistered_instance_imgs,
                    enforce_no_merge=self.config["InstanceMatcher"]["enforce_no_merge"],
                    enforce_no_split=self.config["InstanceMatcher"]["enforce_no_split"],
                )
            )

            # save the graph as graphml file
            graph_path = os.path.join(
                self.config["general"]["result_folder"],
                "lineage_graphs",
                f"{subject_id}_{roi}_graph.graphml",
            )
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_graphml(graph, graph_path)
            # Write it in the ctc format for use with their comparison tools.
            if "Cells_" in subject_id:
                ctc_graph_path = os.path.join(ctc_sub_RES,"res_track.txt")
                write_graph_to_ctc(graph, ctc_graph_path)

            # apply the lineage matching
            matched_lineage_images = []
            for idx, img in enumerate(unregistered_instance_imgs):
                matched_lineage_images.append(
                    apply_matching(
                        matches=lineage_mapping[idx],
                        image=img,
                        background_instance=self.config["general"][
                            "background_instance"
                        ],
                        ignore_label=self.config["general"]["ignore_label"],
                    )
                )

            # apply the lineage mapping to the matched images
            matched_tcui_images = []
            for idx, img in enumerate(unregistered_instance_imgs):
                matched_tcui_images.append(
                    apply_matching(
                        matches=tcui_mapping[idx],
                        image=img,
                        background_instance=self.config["general"][
                            "background_instance"
                        ],
                        ignore_label=self.config["general"]["ignore_label"],
                    )
                )

        if validate:
            # apply the matches to the registered images for validation, that the matching is correct
            matched_registered_images = []
            for idx, img in enumerate(registered_instance_imgs):
                matched_registered_images.append(
                    apply_matching(
                        matches=lineage_mapping[idx],
                        image=img,
                        background_instance=self.config["general"][
                            "background_instance"
                        ],
                        ignore_label=self.config["general"]["ignore_label"],
                    )
                )

                write_to_nifti(
                    nifti_data=matched_registered_images[-1],
                    dtype=matched_registered_images[-1].dtype,
                    filename=registered_label_paths[idx].replace(
                        ".nii.gz", "_matched_by_union.nii.gz"
                    ),
                )

            # validate the matched images to never have overlapping instances with different ids across time
            validate_matched_by_union_images(
                matched_images=matched_registered_images,
                union_instances=union_instances,
                background_instance=self.config["general"]["background_instance"],
            )

        # collect the statistics for the matched instance ids
        # for each matched image in the series we need to keep track of the
        # instances that are the result of vanished, emerged, split or merged instances
        stats_matched_instances = collect_vams_information_for_matched_ids(
            matched_images=matched_lineage_images,
            original_images=unregistered_instance_imgs,
            background_instance=self.config["general"]["background_instance"],
            week_numbers=week_numbers,
        )
        # save it
        ts_path = get_subject_roi_label_channel_time_series_info_path(
            data_path=self.config["general"]["data_path"],
            subject_folder=subject_id,
            roi=roi,
            channel_id=channel_id,
            label_setting=label_settings_folder,
        )
        os.makedirs(os.path.dirname(ts_path), exist_ok=True)

        print("Folder to save the stats: " + ts_path)
        with open(ts_path, "w") as f:
            f.write(json.dumps(stats_matched_instances, indent=4, sort_keys=True))

        # save the matched lineage images
        for idx, img in enumerate(matched_lineage_images):
            write_to_nifti(
                nifti_data=img,
                dtype=img.dtype,
                filename=unregistered_label_paths[idx].replace(
                    "instances.nii.gz", "lineage.nii.gz"
                ),
            )

        # save the matched tcui images
        for idx, img in enumerate(matched_tcui_images):
            nifti_path = unregistered_label_paths[idx].replace(
                    "instances.nii.gz", "tracked.nii.gz"
                )
            write_to_nifti(
                nifti_data=img,
                dtype=img.dtype,
                filename=nifti_path
            )
            if "Cells_" in subject_id:
                timepoint = int(extract_week_from_filepath(nifti_path).replace("_weeks", ""))
                tiff_path = os.path.join(ctc_sub_RES, f"mask{timepoint:03d}.tif")
                write_to_tiff(tiff_data=img, filename=tiff_path)


        # save the meta data
        for idx, img in enumerate(matched_lineage_images):
            meta_data = collect_meta_data(img)
            meta_path = unregistered_label_paths[idx].replace(
                "instances.nii.gz", "lineage_meta.json"
            )
            with open(meta_path, "w") as f:
                f.write(json.dumps(meta_data, indent=4, sort_keys=True))

    def _run_across_channel_matching_from_paths(
        self,
        subject_id: str,
        roi: str,
        images_channel_a,
        images_channel_b,
        channel_id_a,
        channel_id_b,
        channel_a_settings,
        channel_b_settings,
        week_numbers=None,
        verbose=False,
    ):
        """
        Matches the instances in the two channels of the same week.

        Args:
            images_channel_a (list[np.ndarray]): The paths to the images of channel a.
            images_channel_b (list[np.ndarray]): The paths to the images of channel b.
            channel_id_a (str): The channel id of channel a.
            channel_id_b (str): The channel id of channel b.
            channel_a_settings (str): The label settings of channel a.
            channel_b_settings (str): The label settings of channel b.
            week_numbers (list[int]): The week numbers of the images.
            verbose (bool): If True, print the progress of the matching.
        """
        assert len(images_channel_b) == len(images_channel_a)
        if week_numbers is not None:
            assert len(week_numbers) == len(images_channel_a)
        else:
            week_numbers = list(range(len(images_channel_a)))

        matches_a_to_b = {}  # channel_a_instance_id: {
        #     "matched_instances_channel_b_per_week": [[#channel_b_ids]],
        #     "week_numbers": [week_numbers]
        # }
        matches_b_to_a = {}  # channel_b_instance_id: {
        # "matched_instances_channel_a_per_week": [[#channel_a_ids]],
        # "week_numbers": [week_numbers]
        # }

        for idx, (img_a, img_b) in enumerate(zip(images_channel_a, images_channel_b)):
            if verbose:
                logging.info(
                    "Matching images "
                    + str(idx + 1)
                    + " of "
                    + str(len(images_channel_a))
                )
            channel_matches = match_instances_by_intersection_of_two_images(
                image_channel_a=nib.load(img_a).get_fdata(),
                image_channel_b=nib.load(img_b).get_fdata(),
                channel_id_a=channel_id_a,
                channel_id_b=channel_id_b,
                background_instance=self.config["general"]["background_instance"],
            )

            if verbose:
                logging.info("Channel A to B: " + str(channel_matches.a_to_b))
                logging.info("Channel B to A: " + str(channel_matches.b_to_a))

            # save the matches from the channel a to b over the time series as single dict
            for instance_a in channel_matches.a_to_b.keys():
                if instance_a not in matches_a_to_b.keys():
                    matches_a_to_b[instance_a] = {
                        f"matched_instances_per_week": [],
                        "week_numbers": [],
                    }
                matches_a_to_b[instance_a][f"matched_instances_per_week"].append(
                    list(channel_matches.a_to_b[instance_a])
                )
                matches_a_to_b[instance_a]["week_numbers"].append(week_numbers[idx])

            # save the matches from the channel b to a over the time series as single dict
            for instance_b in channel_matches.b_to_a.keys():
                if instance_b not in matches_b_to_a.keys():
                    matches_b_to_a[instance_b] = {
                        f"matched_instances_per_week": [],
                        "week_numbers": [],
                    }
                matches_b_to_a[instance_b][f"matched_instances_per_week"].append(
                    list(channel_matches.b_to_a[instance_b])
                )
                matches_b_to_a[instance_b]["week_numbers"].append(week_numbers[idx])

        # save the matches to json files
        a_to_b_path = get_subject_roi_label_time_series_two_channel_mapping_info_path(
            data_path=self.config["general"]["data_path"],
            subject_id=subject_id,
            roi=roi,
            label_setting_a=channel_a_settings,
            label_setting_b=channel_b_settings,
            channel_a_id=channel_id_a,
            channel_b_id=channel_id_b,
        )

        b_to_a_path = get_subject_roi_label_time_series_two_channel_mapping_info_path(
            data_path=self.config["general"]["data_path"],
            subject_id=subject_id,
            roi=roi,
            label_setting_a=channel_b_settings,
            label_setting_b=channel_a_settings,
            channel_a_id=channel_id_b,
            channel_b_id=channel_id_a,
        )

        os.makedirs(os.path.dirname(a_to_b_path), exist_ok=True)
        os.makedirs(os.path.dirname(b_to_a_path), exist_ok=True)

        with open(a_to_b_path, "w") as f:
            f.write(json.dumps(matches_a_to_b, indent=4, sort_keys=True))

        with open(b_to_a_path, "w") as f:
            f.write(json.dumps(matches_b_to_a, indent=4, sort_keys=True))
        return matches_a_to_b, matches_b_to_a


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging_to_stdout()
    config_path = "/workspaces/MOLT/data/CellTracking/Fluo-N2DL-HeLa/config.json"
    config = Config(config_path=config_path)
    cohort = Cohort(config)
    im = InstanceMatcher(config=config, cohort=cohort)
    im.run()