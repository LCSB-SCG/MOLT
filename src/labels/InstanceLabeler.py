import logging
import os
import threading

import nibabel as nib
import numpy as np

from src.general.Cohort import Cohort
from src.general.Config import Config
from src.labels.functional.instance_labeling import (
    create_binary_label,
    indentify_instances_and_meta,
    threshold_volume_filter,
)
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_date_from_filename,
    extract_roi_from_filepath,
)
from src.utils.io.data_structure.files_and_paths.label_path_operations import (
    get_label_filename,
    get_label_settings_folder_path,
)
from src.utils.io.json.write_to_json import write_to_json
from src.utils.io.nifti.write_to_nifti import write_to_nifti


class InstanceLabeler:
    def __init__(self, config: Config, cohort: Cohort) -> None:
        self.config = config
        self.cohort = cohort
        self.max_num_threads = self.config["InstanceLabeler"]["max_num_threads"]
        self.threads = []

        if self.max_num_threads > 1:
            logging.info(
                f"InstanceLabeler will run in parallel with {self.max_num_threads} threads."
            )
        else:
            logging.info("InstanceLabeler will run sequentially.")

    def run(self) -> None:
        """
        Run the instance labeling for the entire cohort either in parallel or sequentially.
        """
        settings = self.config["InstanceLabeler"]["settings"]

        for setting_idx, setting in enumerate(settings):
            logging.info(
                f"Running instance labeling for setting {setting_idx + 1}/{len(settings)}"
            )

            if "instance_segmentation_from_binary" not in setting:
                setting["instance_segmentation_from_binary"] = False

            if "binary_label_setting" not in setting:
                setting["binary_label_setting"] = None

            if setting["instance_segmentation_from_binary"]:
                # Collect all binary presegmented nifti files
                all_nifti_files = self.cohort.get_information_source_files_as_list(
                    registered=False,
                    information_source="label",
                    channel=setting["channel"],
                    label_setting=setting["binary_label_setting"],
                    file_type="binary",
                )
            else:
                # Collect all nifti files
                all_nifti_files = self.cohort.get_information_source_files_as_list(
                    registered=False,
                    information_source="nifti",
                    channel=setting["channel"],
                )

            # Split the nifti files into chunks for parallel processing
            nifti_files_per_thread = np.array_split(
                ary=all_nifti_files,
                indices_or_sections=min(self.max_num_threads, len(all_nifti_files)),
            )

            for nfpt in nifti_files_per_thread:
                # Start a thread for each chunk
                t = threading.Thread(
                    target=self.create_and_save_label_for_files,
                    args=(
                        nfpt.tolist(),
                        setting,
                        setting["instance_segmentation_from_binary"],
                        (
                            setting["binary_label_setting"]
                            if "binary_label_setting" in setting
                            else None
                        ),
                    ),
                )
                self.threads.append(t)
                t.start()

            for t in self.threads:
                # Wait for all threads to finish
                t.join()

        self.cohort.update_data()

    def create_and_save_label_for_files(
        self,
        nifti_files,
        setting,
        instance_segmentation_from_binary=False,
        binary_label_setting=None,
    ) -> None:
        for idx, nifti_file in enumerate(nifti_files):
            logging.info(f"Processing file {idx + 1} out of {len(nifti_files)}")
            nifti_file_name = os.path.basename(nifti_file)
            year, month, day = extract_date_from_filename(filename=nifti_file_name)
            roi = extract_roi_from_filepath(filepath=nifti_file)
            label_file_name = get_label_filename(
                roi=roi,
                year=year,
                month=month,
                day=day,
                ending="",
                match_params=None,
            )
            if binary_label_setting is None:
                output_folder = get_label_settings_folder_path(
                    nifti_file_path=nifti_file,
                    quantile=None,
                    threshold=setting["threshold"],
                    smoothing_sigma=setting["smoothing_sigma"],
                    min_volume_threshold=setting["volume_min"],
                    max_volume_threshold=setting["volume_max"],
                    path_only=True,
                )
            else:
                output_folder = os.path.dirname(nifti_file)
                if os.path.basename(output_folder) != binary_label_setting:
                    raise ValueError(
                        f"Binary label setting {binary_label_setting} does not match the folder name {os.path.basename(output_folder)}."
                    )
            output_filename = os.path.join(output_folder, label_file_name)
            os.makedirs(output_folder, exist_ok=True)

            self.create_and_save_instances_for_image(
                setting=setting,
                image_path=nifti_file,
                output_path=output_filename,
                instance_segmentation_from_binary=instance_segmentation_from_binary,
            )

    def create_and_save_instances_for_image(
        self, setting, image_path, output_path, instance_segmentation_from_binary=False
    ) -> None:
        # Load the image
        image = nib.load(image_path).get_fdata()
        # Create the binary label
        if instance_segmentation_from_binary:
            bin_label = image
        else:
            bin_label = create_binary_label(
                image=image,
                smoothing_sigma=setting["smoothing_sigma"],
                threshold=setting["threshold"],
                keep_top_quantile=setting["keep_top_quantile"],
            )
        # Identify the instances and meta data
        instances, meta_data = indentify_instances_and_meta(bin_label)
        # Filter the instances by volume
        instances, meta_data = threshold_volume_filter(
            instances=instances,
            meta_data=meta_data,
            background_instance=self.config["general"]["background_instance"],
            min_volume_threshold=setting["volume_min"],
            max_volume_threshold=setting["volume_max"],
        )

        # Save the instances and meta data
        write_to_nifti(
            nifti_data=instances, dtype=np.uint16, filename=output_path + ".nii.gz"
        )
        write_to_json(data=meta_data, filename=output_path + "_meta.json")

        # Padded with background instance to get rid of instance splitting due to applying the registration later
        # Pad the array by its full size to the left, right, top, bottom, front, and back
        x_pad = int(
            instances.shape[0] * float(self.config["InstanceLabeler"]["padding_factor"])
        )
        y_pad = int(
            instances.shape[1] * float(self.config["InstanceLabeler"]["padding_factor"])
        )
        if len(instances.shape) == 2:
            z_pad = 0
        else:
            z_pad = int(
                instances.shape[2]
                * float(self.config["InstanceLabeler"]["padding_factor"])
            )

        if instances.ndim == 2:
            padding = ((x_pad, x_pad), (y_pad, y_pad))
        elif instances.ndim == 3:
            padding = ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad))

        instances_padded = np.pad(
            array=instances,
            pad_width=padding,
            mode="constant",
            constant_values=self.config["general"]["background_instance"],
        )
        write_to_nifti(
            nifti_data=instances_padded,
            dtype=np.int16,
            filename=output_path + "_padded_for_reg.nii.gz",
        )

        logging.info("Saving: " + output_path + "_padded_for_reg.nii.gz")
