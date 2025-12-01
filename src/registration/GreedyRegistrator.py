import logging
import os
import shutil
import subprocess

import nibabel as nib
import numpy as np

from src.general.Cohort import Cohort
from src.general.Config import Config
from src.utils.io.data_structure.files_and_paths.constants import (
    AFFINE_TRANS,
    DEFORMABLE_TRANS,
    OVERLAP_NIFTI_NAME,
)
from src.utils.io.data_structure.files_and_paths.registration_path_operations import (
    get_registered_file_path_from_unregisterd_file,
    get_registration_folder_from_unregisterd_file,
)
from src.utils.io.nifti.write_to_nifti import write_to_nifti


class GreedyRegistrator:
    """
    This class is used to register two point clouds using the greedy algorithm.
    """

    allowed_metrics = ["WNCC"]
    # allowed patch radius for arbitrary dimension
    allowed_patch_radius = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowed_modes = ["all_to_first_week_of_roi", "all_to_preceding_timepoint"]
    allowed_registration_types = ["rigid", "affine", "deformable"]

    def __init__(self, config: Config, cohort: Cohort):
        """
        Args:
            mode (str): mode of the registration, either "all_to_first_week_of_roi":
                - "all_to_first_week_of_roi" registers each week to the first week of the roi.
            max_iterations (list[int], optional): number of iterations per level. Defaults to [100, 50, 10].
            metric (str, optional): metric to use. Defaults to "WNCC".
            patch_radius (list[int], optional): radius of the kernels. Defaults to [2, 2, 2].
            ia_image_centers (bool, optional): align image centers before registration. Defaults to True.
            registration_type (str, optional): type of registration, either "rigid", "affine" or "deformable".
                Defaults to "affine".
            dimensions (int, optional): number of dimensions int the image. Defaults to 3.
        """
        self.cohort: Cohort = cohort
        self.data_path = config["general"]["data_path"]
        self.config = config["GreedyRegistrator"]
        self.channel_to_be_registered = self.config["settings"]["channel_to_be_registered"]
        self.mode = self.config["settings"]["mode"]
        self.max_iterations = self.config["settings"]["max_iterations"]
        self.metric = self.config["settings"]["metric"]
        self.patch_radius = self.config["settings"]["patch_radius"]
        self.ia_image_centers = self.config["settings"]["ia_image_centers"]
        self.registration_type = self.config["settings"]["registration_type"]

        if self.mode not in self.allowed_modes:
            raise ValueError(
                f"The registration_to must be one of the following: {str(self.allowed_modes)}"
            )
        elif self.mode == "all_to_preceding_timepoint":
            self.reg_fn = self.register_all_to_preceding_timepoints_of_roi
            self.apply_reg_fn = (
                self.apply_registrations_for_all_timepoints_to_preceding_timepoints_of_roi
            )
        elif self.mode == "all_to_first_week_of_roi":
            self.reg_fn = self.register_all_timepoints_to_first_week_of_roi
            self.apply_reg_fn = self.apply_registration_for_all_timepoints_to_first_week_of_roi

        if self.registration_type == "rigid":
            raise NotImplementedError("Rigid registration is not implemented yet.")
        elif self.registration_type == "affine":
            self.deformable = False
        elif self.registration_type == "deformable":
            self.deformable = True
        else:
            raise ValueError(
                "The registration type must be one of the following: "
                + str(self.allowed_registration_types)
            )

        self.set_deformable_registration(deformable=self.deformable)
        self.dimensions = str(self.config["settings"]["dimensions"])

        if self.metric not in self.allowed_metrics:
            raise ValueError(
                f"The metric must be one of the following: {str(self.allowed_metrics)}"
            )

        for i in range(len(self.patch_radius)):
            if self.patch_radius[i] not in self.allowed_patch_radius:
                raise ValueError(
                    f"The patch radius must be one of the following: {str(self.allowed_patch_radius)}"
                )

        if self.mode not in self.allowed_modes:
            raise ValueError(f"The mode must be one of the following: {str(self.allowed_modes)}")

        logging.info("Running greedy registration with the following parameters:")
        logging.info("Mode: " + self.mode)
        logging.info("Max iterations: " + str(self.max_iterations))
        logging.info("Metric: " + self.metric)
        logging.info("Patch radius: " + str(self.patch_radius))
        logging.info("Align image centers first: " + str(self.ia_image_centers))
        logging.info(
            "Deformable registration (if false only affine transformation): " + str(self.deformable)
        )

    def run(self):
        """

        Args:
            register_main_channel (bool): whether to register the main channel
        """
        # Run the registration for the main channel
        if not self.config["settings"]["only_apply_already_available_registration"]:
            logging.info("Running affine registrations")
            logging.info(f"All subjects: {self.cohort.get_subjects()}")
            for subject in self.cohort.get_subjects():
                logging.info("Registering subject: " + subject)
                logging.info(f"ROIs: {self.cohort.get_subject_rois(subject_id=subject)}")
                for roi in self.cohort.get_subject_rois(subject_id=subject):
                    logging.info("Registering ROI: " + roi)
                    self.reg_fn(
                        subject_id=subject,
                        roi=roi,
                        channel_id=self.channel_to_be_registered,
                        affine=True,
                        deformable=False,
                    )
            self.cohort.update_data()

            if self.deformable:
                # deformable registration need to be run after affine registration
                logging.info("Running deformable registrations based on the affine registrations")
                for subject in self.cohort.get_subjects():
                    logging.info("Deformable registering subject: " + subject)
                    logging.info(f"ROIs: {self.cohort.get_subject_rois(subject_id=subject)}")
                    for roi in self.cohort.get_subject_rois(subject_id=subject):
                        logging.info("Deformable registering ROI: " + roi)
                        self.reg_fn(
                            subject_id=subject,
                            roi=roi,
                            channel_id=self.channel_to_be_registered,
                            affine=False,
                            deformable=True,
                        )
                self.cohort.update_data()

        # Apply the registration to the other channels
        if self.config["settings"]["apply_to_other_channels"] is not None:
            logging.info("Applying registration to other channels")
            logging.info("All subjects: " + str(self.cohort.get_subjects()))
            for subject in self.cohort.get_subjects():
                logging.info("Applying registration for subject: " + subject)
                logging.info(f"ROIs: {self.cohort.get_subject_rois(subject_id=subject)}")
                for roi in self.cohort.get_subject_rois(subject_id=subject):
                    logging.info("Applying registration for ROI: " + roi)
                    logging.info(
                        f"Applying registration of channel {self.channel_to_be_registered} to other channels: {self.config['settings']['apply_to_other_channels']}"
                    )
                    for channel in self.config["settings"]["apply_to_other_channels"]:
                        logging.info("Applying registration for channel: " + channel)
                        self.apply_reg_fn(
                            subject_id=subject,
                            roi=roi,
                            channel_id=channel,
                        )
            self.cohort.update_data()

        if self.config["settings"]["apply_to_labels"] is not None:
            logging.info("Applying registration to labels")
            logging.info("All subjects: " + str(self.cohort.get_subjects()))
            for subject in self.cohort.get_subjects():
                logging.info("Applying registration for subject: " + subject)
                logging.info(f"ROIs: {self.cohort.get_subject_rois(subject_id=subject)}")
                for roi in self.cohort.get_subject_rois(subject_id=subject):
                    logging.info("Applying registration for ROI: " + roi)
                    for channel in self.config["settings"]["apply_to_labels"]:
                        logging.info(
                            f"Applying registration of {self.channel_to_be_registered} for labels of {channel}"
                        )
                        for label_setting in self.config["settings"]["apply_to_labels"][channel]:
                            logging.info(
                                f"Applying registration for label setting: {label_setting}"
                            )
                            self.apply_reg_fn(
                                subject_id=subject,
                                roi=roi,
                                channel_id=channel,
                                label_setting=label_setting,
                            )
            self.cohort.update_data()

    def register_all_timepoints_to_first_week_of_roi(
        self,
        subject_id: str,
        roi: str,
        channel_id: str,
        affine: bool = True,
        deformable: bool = False,
    ):
        """
        Registers all timepoints to the first_week_of_roi (equal to the first timepoint of measurment).

        Args:
            subject_folder (str): path to the subject folder
            subject_id (str): subject folder containing the week recording folders
            roi (str): id of the region of interest. e.g 001
            channel_id (str): id of the channel, e.g. 1 or 2
        """
        assert affine or deformable, "Either affine or deformable registration must be True"
        assert not (
            affine and deformable
        ), "Only one of affine or deformable registration can be True"
        # get all week folders
        timepoint_folders = self.cohort.get_subject_roi_weeks(subject_id=subject_id, roi=roi)
        timepoint_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=timepoint_folders,
            channel=channel_id,
            information_source="nifti",
            registered=False,
        )
        if affine:
            reg_file_type = AFFINE_TRANS
        else:
            reg_file_type = DEFORMABLE_TRANS

        transformation_files = []
        for tp_file, tp_id in zip(timepoint_files[1:], timepoint_folders[1:]):
            logging.info("Registering week " + tp_id + " to the first_week_of_roi")
            transformation_files.append(
                self.register(
                    source_file=tp_file,
                    target_file=timepoint_files[0],
                    source_id=tp_id,
                    target_id=timepoint_folders[0],
                    reg_file_type=reg_file_type,
                )
            )

        if affine:
            # Calculate the overlapping region
            result_path = get_registration_folder_from_unregisterd_file(
                image_path=timepoint_files[0],
                target_week=timepoint_folders[0],
                create_dirs=True,
            )

            result_path = os.path.join(result_path, OVERLAP_NIFTI_NAME)

            logging.info("Calculating the overlapping region and saving it to " + result_path)
            self.calculate_overlapping_region(
                transformation_files=transformation_files,
                target=timepoint_files[0],
                result_path=result_path,
            )
            transformation_files = [[t] for t in transformation_files]
            # Rewarp the overlapping region to the unward images for each time step
            self.warp_overlapping_region_back_to_image_space(
                transformation_files=transformation_files,
                week_files=timepoint_files[1:],
                overlap_file=result_path,
                target_week_id=timepoint_folders[0],
            )

    def register_all_to_preceding_timepoints_of_roi(
        self,
        subject_id: str,
        roi: str,
        channel_id: str,
        affine: bool = True,
        deformable: bool = False,
        *args,
        **kwargs,
    ):
        """
        Registers all timepoints to their previous timepoint (equal to the first timepoint of measurment).

        Args:
            subject_folder (str): path to the subject folder
            subject_id (str): subject folder containing the week recording folders
            roi (str): id of the region of interest. e.g 001
            channel_id (str): id of the channel, e.g. 1 or 2
        """
        assert affine or deformable, "Either affine or deformable registration must be True"
        assert not (
            affine and deformable
        ), "Only one of affine or deformable registration can be True"
        # get all week folders
        timepoint_folders = self.cohort.get_subject_roi_weeks(subject_id=subject_id, roi=roi)
        timepoint_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=timepoint_folders,
            channel=channel_id,
            information_source="nifti",
            registered=False,
        )

        if affine:
            reg_file_type = AFFINE_TRANS
        else:
            reg_file_type = DEFORMABLE_TRANS

        # copy the image for the first timepoint to the output folder
        dest = get_registered_file_path_from_unregisterd_file(
            image_path=timepoint_files[0],
            target_week="preceding_timepoint",
            create_dirs=True,
        )
        if not os.path.exists(dest):
            shutil.copyfile(
                timepoint_files[0],
                get_registered_file_path_from_unregisterd_file(
                    image_path=timepoint_files[0],
                    target_week="preceding_timepoint",
                    create_dirs=True,
                ),
            )

        transformation_files = []
        for tp_file, tp_id, prev_tp_file, prev_tp_id in zip(
            timepoint_files[1:],
            timepoint_folders[1:],
            timepoint_files[:-1],
            timepoint_folders[:-1],
        ):
            logging.info("Registering week " + tp_id + " to the first_week_of_roi")
            transformation_files.append(
                self.register(
                    source_file=tp_file,
                    target_file=prev_tp_file,
                    source_id=tp_id,
                    target_id="preceding_timepoint",
                    reg_file_type=reg_file_type,
                )
            )

        # as we started with the second timepoint, we have no registration folder for the first timepoint
        # create one for conistency with the other registrations
        first_week_folder = get_registration_folder_from_unregisterd_file(
            image_path=timepoint_files[0],
            target_week="preceding_timepoint",
            create_dirs=True,
        )
        os.makedirs(first_week_folder, exist_ok=True)

    def apply_registration_for_all_timepoints_to_first_week_of_roi(
        self, subject_id: str, roi: str, channel_id: str, label_setting: str = None
    ):
        """
        Applies the registration to all other channels and segmentations for all weeks.
        """
        # get all week folders
        week_folders = self.cohort.get_subject_roi_weeks(subject_id=subject_id, roi=roi)
        week_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=week_folders,
            channel=channel_id,
            information_source="nifti" if label_setting is None else "label",
            label_setting=label_setting,
            file_type=(None if label_setting is None else "padded_instances_for_registration"),
        )

        target_week = week_folders[0]
        target_file = week_files[0]

        transformation_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=week_folders[1:],
            channel=self.channel_to_be_registered,
            information_source="nifti",
            registered=True,
            registration_to=target_week,
            reg_file_type=AFFINE_TRANS,
        )

        if len(transformation_files) != len(week_files) - 1:
            raise ValueError(
                "The number of transformation files must be the same as the number of week files - 1 (no transformation for the first week). It seams not all files were registered."
            )

        transformation_files = [[t] for t in transformation_files]  # make it a list of lists

        output_files = [
            get_registered_file_path_from_unregisterd_file(
                image_path=image_path, target_week=target_week, create_dirs=True
            )
            for image_path in week_files[1:]
        ]

        for week_file, transformation_file, output_file in zip(
            week_files[1:], transformation_files, output_files
        ):
            logging.info(
                f"Applying registration to {week_file} of channel {channel_id} of {roi} of {subject_id} to {target_week}"
            )
            logging.info(f"Transformation file: {transformation_file}")
            if label_setting is None:
                # Apply the registration to the nifti files --> With linear interpolation
                self.apply_registration(
                    target=target_file,
                    transformation_files=transformation_file,
                    file=week_file,
                    output_file=output_file,
                    segmentation_file=None,
                    output_segmentation_file=None,
                    inverse_flags=None,
                )
            else:
                # Apply the registration to the label files --> With majority voting for the labels (like nearest neighbor)
                self.apply_registration(
                    target=target_file,
                    transformation_files=transformation_file,
                    file=None,
                    output_file=None,
                    segmentation_file=week_file,
                    output_segmentation_file=output_file,
                    inverse_flags=None,
                )

    def apply_registrations_for_all_timepoints_to_preceding_timepoints_of_roi(
        self, subject_id: str, roi: str, channel_id: str, label_setting: str = None
    ):
        """
        Applies the registration to all other channels and segmentations for all weeks.
        """
        # get all week folders
        timepoint_folders = self.cohort.get_subject_roi_weeks(subject_id=subject_id, roi=roi)
        timepoint_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=timepoint_folders,
            channel=channel_id,
            information_source="nifti" if label_setting is None else "label",
            label_setting=label_setting,
            file_type=(None if label_setting is None else "padded_instances_for_registration"),
        )

        target_week = "preceding_timepoint"

        affine_trans_files = self.cohort.get_information_source_files_as_list(
            subject_ids=[subject_id],
            rois=[roi],
            weeks=timepoint_folders[1:],
            channel=self.channel_to_be_registered,
            information_source="nifti",
            registered=True,
            registration_to=target_week,
            reg_file_type=AFFINE_TRANS,
        )

        if self.deformable:
            deformable_trans_files = self.cohort.get_information_source_files_as_list(
                subject_ids=[subject_id],
                rois=[roi],
                weeks=timepoint_folders[1:],
                channel=self.channel_to_be_registered,
                information_source="nifti",
                registered=True,
                registration_to=target_week,
                reg_file_type=DEFORMABLE_TRANS,
            )
            transformation_files = [
                [deformable_file, affine_file]
                for deformable_file, affine_file in zip(deformable_trans_files, affine_trans_files)
            ]
        else:
            transformation_files = [[t] for t in affine_trans_files]  # make it a list of lists

        if len(transformation_files) != len(timepoint_files) - 1:
            raise ValueError(
                "The number of transformation files must be the same as the number of week files - 1 (no transformation for the first week). It seams not all files were registered."
            )

        output_files = [
            get_registered_file_path_from_unregisterd_file(
                image_path=image_path, target_week=target_week, create_dirs=True
            )
            for image_path in timepoint_files[1:]
        ]

        # copy the image for the first timepoint to the output folder
        shutil.copyfile(
            timepoint_files[0],
            get_registered_file_path_from_unregisterd_file(
                image_path=timepoint_files[0],
                target_week="preceding_timepoint",
                create_dirs=True,
            ),
        )

        for tp_file, prev_tp_file, transformation_file, output_file in zip(
            timepoint_files[1:],
            timepoint_files[:-1],
            transformation_files,
            output_files,
        ):
            logging.info(
                f"Applying registration to {tp_file} of channel {channel_id} of {roi} of {subject_id} to {target_week}"
            )
            logging.info(f"Transformation file: {transformation_file}")
            if label_setting is None:
                # Apply the registration to the nifti files --> With linear interpolation
                self.apply_registration(
                    target=prev_tp_file,
                    transformation_files=transformation_file,
                    file=tp_file,
                    output_file=output_file,
                    segmentation_file=None,
                    output_segmentation_file=None,
                    inverse_flags=None,
                )
            else:
                # Apply the registration to the label files --> With majority voting for the labels (like nearest neighbor)
                self.apply_registration(
                    target=prev_tp_file,
                    transformation_files=transformation_file,
                    file=None,
                    output_file=None,
                    segmentation_file=tp_file,
                    output_segmentation_file=output_file,
                    inverse_flags=None,
                )

    def register(
        self,
        source_file: str,
        target_file: str,
        source_id: str,
        target_id: str,
        reg_file_type: str,
    ):
        """
        Registers a single file to the target.
        Args:
            source_file (str): file to be registered
            target_file (str): target file
            source_id (str): id of the source file, e.g. the week number "3_weeks"
            target_id (str): id of the target file, e.g. the week number "0_weeks"
            reg_file_type (str): type of the registration file, either affine or deformable file types
        """
        # save the registration in the source folder
        output_folder = get_registration_folder_from_unregisterd_file(
            image_path=source_file, target_week=target_id, create_dirs=True
        )

        affine_trans_file_path = os.path.join(output_folder, AFFINE_TRANS)
        deformable_trans_file_path = os.path.join(output_folder, DEFORMABLE_TRANS)

        if reg_file_type == AFFINE_TRANS:
            logging.info("Affine transformation file path: " + affine_trans_file_path)

            output_file_path = get_registered_file_path_from_unregisterd_file(
                image_path=source_file, target_week=target_id
            )

            logging.info("Output file path: " + output_file_path)

            command = self.create_affine_command(
                source_file=source_file,
                target_file=target_file,
                transformation_file_path=affine_trans_file_path,
            )

        elif reg_file_type == DEFORMABLE_TRANS:
            logging.info("1st Affine transformation file path: " + affine_trans_file_path)
            logging.info("2nd Deformable transformation file path: " + deformable_trans_file_path)

            output_file_path = get_registered_file_path_from_unregisterd_file(
                image_path=source_file, target_week=target_id
            )

            logging.info("Output file path: " + output_file_path)

            command = self.create_deformable_command(
                source_file=source_file,
                target_file=target_file,
                affine_file_path=affine_trans_file_path,
                deformable_file_path=deformable_trans_file_path,
            )

        os.makedirs("results/log", exist_ok=True)
        with open("results/log/stdout.log", "w") as log_stdout:
            subprocess.call(command, shell=True, stdout=log_stdout, stderr=log_stdout)

        if not self.deformable and reg_file_type == AFFINE_TRANS:
            # only do affine and apply the registration
            self.apply_registration(
                file=source_file,
                target=target_file,
                output_file=output_file_path,
                transformation_files=[affine_trans_file_path],
            )
            return affine_trans_file_path
        elif self.deformable and reg_file_type == DEFORMABLE_TRANS:
            # do deformable registration and and affine is already done, apply the registration
            self.apply_registration(
                file=source_file,
                target=target_file,
                output_file=output_file_path,
                transformation_files=[
                    deformable_trans_file_path,
                    affine_trans_file_path,
                ],
            )
            return deformable_trans_file_path
        elif self.deformable and reg_file_type == AFFINE_TRANS:
            # do deformable registration and affine is not done yet, only return the affine transformation file without applying it
            return affine_trans_file_path
        return None

    def create_affine_command(
        self, source_file: str, target_file: str, transformation_file_path: str
    ):
        """
        Creates the affine command for the greedy algorithm.

        Args:
            source_file (str): file to be registered
            target_file (str): target file
            transformation_file_path (str): path to the transformation file
        """
        command_affine = (
            "greedy -d "
            + self.dimensions
            + " -a -i "
            + target_file
            + " "
            + source_file
            + " -o "
            + transformation_file_path
        )
        command_affine += (
            " -m "
            + self.metric
            + " "
            + str(self.patch_radius[0])
            + "x"
            + str(self.patch_radius[1])
            + "x"
            + str(self.patch_radius[2])
        )
        command_affine += (
            " -n "
            + str(self.max_iterations[0])
            + "x"
            + str(self.max_iterations[1])
            + "x"
            + str(self.max_iterations[2])
        )
        if self.ia_image_centers:
            command_affine += " -ia-image-centers "
        if self.config["settings"]["threads"] is not None:
            command_affine += " -threads " + str(self.config["settings"]["threads"])

        logging.info(
            "Calculating greedy registration with the following command: " + command_affine
        )
        return command_affine

    def create_deformable_command(
        self,
        source_file: str,
        target_file: str,
        affine_file_path: str,
        deformable_file_path: str,
    ):
        """
        Creates the deformable command for the greedy algorithm.

        Args:
            source_file (str): file to be registered
            target_file (str): target file
            affine_file_path (str): path to the affine transformation file
            deformable_file_path (str): path to the deformable transformation file
            
        Example Command:
            greedy -d 3 \
                -m WNCC 2x2x2 \
                -i fixed.nii.gz moving.nii.gz \
                -it affine.mat \
                -o warp.nii.gz \
                -sv -n 100x50x10
        """
        command_def = (
            "greedy -d "
            + self.dimensions
            + " -m "
            + self.metric
            + " "
            + str(self.patch_radius[0])
            + "x"
            + str(self.patch_radius[1])
            + "x"
            + str(self.patch_radius[2])
            + " -i "
            + target_file
            + " "
            + source_file
            + " -o "
            + deformable_file_path
            + " -it "
            + affine_file_path
            + " -sv -n "
            + str(self.max_iterations[0])
            + "x"
            + str(self.max_iterations[1])
            + "x"
            + str(self.max_iterations[2])
        )
        if self.config["settings"]["threads"] is not None:
            command_def += " -threads " + str(self.config["settings"]["threads"])

        logging.info("Calculating greedy registration with the following command: " + command_def)
        return command_def

    def set_deformable_registration(self, deformable: bool):
        """
        Sets the registration to deformable or not.

        Args:
            deformable (bool): whether the registration should be deformable or not
        """
        self.deformable = deformable
        if deformable:
            self.output_filename = DEFORMABLE_TRANS
        else:
            self.output_filename = AFFINE_TRANS

    def apply_registration(
        self,
        target: str,
        transformation_files: "list[str]",
        file: str = None,
        output_file: str = None,
        segmentation_file: str = None,
        output_segmentation_file: str = None,
        inverse_flags=None,
    ):
        """
        Applies a transformation to a file.


        Args:
            file (str): file to be transformed / moving file
            target (str): target / fixed file
            output_file (str): output file path
            transformation_file (list[str]): transformation file path, from last transformation to first
            label_interpolation (bool, optional): Whether to interpolate the labels if the file is a
                segmentation / label file. Defaults to False.
            segmentation_file (str): segmentation file to be transformed
            output_segmentation_file (str): output segmentation file path
            inverse_flags (list[bool], optional): list of flags indicating whether the transformation should
                be inverted. Defaults to None = No transformation will be inverted.

        Example Command:
            greedy -d 3 \
                -rf fixed.nii.gz \
                -rm moving.nii.gz resliced.nii.gz \
                -ri LABEL 0.2vox \
                -rm moving_seg.nii.gz resliced_seg.nii.gz \
                -r warp.nii.gz affine.mat
        """
        if file is None and segmentation_file is None:
            raise ValueError("At least one of file or segmentation_file must be given")

        if file is not None and output_file is None:
            raise ValueError("If file is given, output_file must be given as well")

        if segmentation_file is not None and output_segmentation_file is None:
            raise ValueError(
                "If segmentation_file is given, output_segmentation_file must be given as well"
            )

        if inverse_flags is not None and len(inverse_flags) != len(transformation_files):
            raise ValueError(
                "The length of the inverse_flags must be the same as the length of the transformation_files"
            )
        elif inverse_flags is not None:
            # Convert to notation for greedy
            inverse_flags = [",-1" if flag else "" for flag in inverse_flags]
        elif inverse_flags is None:
            inverse_flags = [""] * len(transformation_files)

        command = "greedy -d " + self.dimensions
        command += " -rf " + target
        if file is not None:
            command += " -ri LINEAR "
            command += " -rm " + file + " " + output_file
        if segmentation_file is not None:
            command += " -ri NN"  # LABEL 0.2vox"
            command += " -rm " + segmentation_file + " " + output_segmentation_file
        if self.config["settings"]["threads"] is not None:
            command += " -threads " + str(self.config["settings"]["threads"])
        command += " -r "
        for transformation, inv in zip(transformation_files, inverse_flags):
            command += transformation + inv + " "

        logging.info("Applying the registration with the following command: " + command)
        with open("results/log/stdout.log", "w") as log_stdout:
            subprocess.call(command, shell=True, stdout=log_stdout)

    def calculate_overlapping_region(self, transformation_files, target, result_path):
        """
        Calculates the overlapping region of the registered weeks.

        Args:
            transformation_files (list[str]): list of transformation files
            target (str): target file
            result_path (str): path to the result folder
        """
        shape = nib.load(target).get_fdata().shape
        tmp_paths = []
        for idx, transformation in enumerate(transformation_files):
            # create a nifti file with only ones
            path = os.path.dirname(transformation)
            reg_path = os.path.join(path, "tmp_reg_one_only.nii")
            path = os.path.join(path, "tmp_one_only.nii")
            tmp_paths.append(reg_path)
            GreedyRegistrator.create_nifti_with_ones_only(path, shape)
            # apply the transformation to the file
            self.apply_registration(
                target=target,
                transformation_files=[transformation],
                segmentation_file=path,
                output_segmentation_file=reg_path,
            )
            os.remove(path)

        # calculate the overlapping region
        overlapping_region = np.ones(shape)
        for path in tmp_paths:
            overlapping_region *= nib.load(path).get_fdata()
            # delete file
            os.remove(path)

        write_to_nifti(overlapping_region, np.uint8, result_path)

    def warp_overlapping_region_back_to_image_space(
        self, transformation_files, week_files, overlap_file, target_week_id
    ):
        """
        Unwarps the overlapping region in target space to the unwarped images for each time step.

        Args:
            transformation_files (list[list[str]]): list of transformation files for each week file
            week_files (list[str]): list of week files
            overlap_file (str): path to the overlapping region
            region_id (str): id of the region
        """
        inverse_transformation_flags = [[True] * len(tf) for tf in transformation_files]
        output_paths = [
            os.path.join(
                get_registration_folder_from_unregisterd_file(
                    week_file, target_week=target_week_id
                ),
                OVERLAP_NIFTI_NAME,
            )
            for week_file in week_files
        ]
        for op, tf, inv, wf in zip(
            output_paths, transformation_files, inverse_transformation_flags, week_files
        ):
            logging.info(" Applying registration to the overlapping region for week ")
            logging.info(f"target: {wf}")
            logging.info(f"transformation: {tf}")
            logging.info(f"Inserve: {inv}")
            logging.info(f"output: {op}")
            logging.info(f"Overlap file: {overlap_file}")
            self.apply_registration(
                target=wf,
                transformation_files=tf,
                segmentation_file=overlap_file,
                output_segmentation_file=op,
                inverse_flags=inv,
            )

    ### STATIC METHODS ###

    @staticmethod
    def create_nifti_with_ones_only(path: str, shape: tuple[int]):
        """
        Creates a nifti file with only ones, except for 0s at the borders.

        0s at the borders, since objects at the border should not be accounted for
        as it can change the underlying biological signal.

        Args:
            path (str): path to the file
            shape (tuple[int]): shape of the data of the file
        """
        shape = [s - 2 for s in shape]
        data = np.ones(shape)
        data = np.pad(data, 1, "constant", constant_values=0)
        write_to_nifti(data, np.uint8, path)
