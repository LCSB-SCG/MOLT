import glob
import logging
import os
import threading


from src.general.Config import Config
from src.utils.io.data_structure.files_and_paths.renaming import (
    remove_emtpy_dirs,
    remove_non_nd2_files,
    rename_files,
    rename_week_folders,
    replace_space_with_underscore,
)
from src.preprocessing.functional.nd2_to_nifti import nd2_to_nifti
import numpy as np

from src.utils.io.data_structure.files_and_paths.patterns import (
    SUBJECT_ID_PATTERN,
    WEEKS_PATTERN,
)


class DataPreProcessor:
    def __init__(self, config: Config) -> None:
        self.config = config

    def preprocess_data(self):
        # First enforce the correct data structure
        logging.info("Replacing spaces with underscores")
        replace_space_with_underscore(path=self.config["general"]["data_path"])
        logging.info("Renaming the files")
        # rename_files(
        #     path=self.config["general"]["data_path"],
        #     channel_mapping=self.config["general"]["channel_mapping"],
        # )
        logging.info("Removing non-nd2 files")
        remove_non_nd2_files(path=self.config["general"]["data_path"])
        logging.info("Removing empty directories")
        remove_emtpy_dirs(path=self.config["general"]["data_path"])
        logging.info("Renaming week folders")
        rename_week_folders(self.config["general"]["data_path"])

        # Combine the nd2 files into a single nifti file and save the combined metadata
        logging.info(
            "Combining nd2 files: All the files should end in the 'xxx.nd2' format, where xxx is the roi number"
        )
        self.combine_nd2_files()

        # # Exract the metadata of the microscopy images
        # self.extract_microscope_meta_from_nd2_files()

        # # Turn the nd2 files into nifti files
        # self.convert_nd2_files_to_nifti()

    def collect_subject_timepoint_roi_files(self, subject, timepoint, roi):
        return glob.glob(
            os.path.join(
                self.config["general"]["data_path"], subject, timepoint, f"*{roi}*.nd2"
            )
        )

    def get_filepaths_from_patterns(self, subject, timepoint):
        """
        Fetches all the file paths.

        Returns:
            List of lists of file paths. Each list contains the file paths for a specific pattern.
            The list should all have the same length and the same index should correspond to the same roi.
        """
        paths_to_files = []
        for roi in range(self.config["DataPreProcessor"]["max_num_rois"]):
            roi = str(roi).zfill(3)
            files = self.collect_subject_timepoint_roi_files(subject, timepoint, roi)
            if files != []:
                paths_to_files.append(files)
        return paths_to_files

    def combine_nd2_files(self, verbose=True):
        """
        Combines multiple available nd2 files into a single file save nifti using multiple threads.
        """

        def _process_roi_images(subject, timepoint, roi_images):
            # Get reference images for intensity matching from reference timepoint
            if self.config["DataPreProcessor"]["ref_timepoint"] is not None:
                possible_ref_images = self.collect_subject_timepoint_roi_files(
                    subject,
                    self.config["DataPreProcessor"]["ref_timepoint"],
                    roi=roi_images[0][-7:-4],
                )
            else:
                possible_ref_images = None
            logging.info(
                f"Subject: {subject}, timepoint: {timepoint}, roi: {roi_images[0][-7:-4]}, pos. ref. images {possible_ref_images}"
            )

            # Convert nd2 files to nifti format with specified parameters
            nd2_to_nifti(
                data_path=self.config["general"]["data_path"],
                input_paths=roi_images,
                auto_adjust_contrast=self.config["DataPreProcessor"][
                    "auto_adjust_contrast"
                ],
                channel_settings=self.config["DataPreProcessor"]["channel_settings"],
                channel_mapping=self.config["general"]["channel_mapping"],
                channel_selection_conditions=self.config["DataPreProcessor"][
                    "channel_selection_conditions"
                ],
                possible_ref_images=possible_ref_images,
                channels_for_intensity_matching=self.config["DataPreProcessor"][
                    "channel_for_instensity_matching"
                ],
                x_scale=self.config["DataPreProcessor"]["x_scale"],
                y_scale=self.config["DataPreProcessor"]["y_scale"],
                z_scale=self.config["DataPreProcessor"]["z_scale"],
            )

        # Create list of tasks to process
        tasks = []
        # Iterate through all subjects in data path
        for subject in os.listdir(self.config["general"]["data_path"]):
            # Skip if not matching subject ID pattern
            if SUBJECT_ID_PATTERN.match(subject) is None:
                continue
            # Iterate through timepoints for each subject
            for timepoint in os.listdir(
                os.path.join(self.config["general"]["data_path"], subject)
            ):
                # Skip if not matching weeks pattern
                if WEEKS_PATTERN.match(timepoint) is None:
                    continue
                if verbose:
                    logging.info(f"Combining nd2 files for {subject} at {timepoint}")

                # Get file paths for current subject and timepoint
                paths_to_files = self.get_filepaths_from_patterns(subject, timepoint)
                # Add tasks for each ROI
                for roi_images in paths_to_files:
                    tasks.append((subject, timepoint, roi_images))

        # Distribute tasks across threads
        num_threads = self.config["DataPreProcessor"]["num_threads"]
        tasks_per_thread = [tasks[i::num_threads] for i in range(num_threads)]

        # Create and start threads
        threads = []
        for thread_tasks in tasks_per_thread:
            if not thread_tasks:
                continue
            t = threading.Thread(
                target=lambda tasks: [_process_roi_images(*task) for task in tasks],
                args=(thread_tasks,),
            )
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

    def convert_nd2_files_to_nifti(self, verbose=True):
        """
        Convert the nd2 files to nifti files.
        """

        def _convert_list_of_images(file_list):
            """
            Helper function to run a single thread to convert a list of images
            """
            for idx, input_path in enumerate(file_list):
                logging.info(f"Processing file {idx + 1}/{len(file_list)}")
                nd2_to_nifti(
                    data_path=self.config["general"]["data_path"],
                    input_paths=str(input_path),
                    auto_adjust_contrast=self.config["DataPreProcessor"][
                        "auto_adjust_contrast"
                    ],
                    channel_settings=self.config["DataPreProcessor"][
                        "channel_settings"
                    ],
                )

        files = glob.glob(
            os.path.join(self.config["general"]["data_path"], "**", "*.nd2"),
            recursive=True,
        )

        if verbose:
            logging.info("Files:")
            for i in files:
                logging.info(i)

        image_list_per_thread = np.array_split(
            files, self.config["DataPreProcessor"]["num_threads"]
        )

        threads = []
        for thread in range(self.config["DataPreProcessor"]["num_threads"]):
            logging.info(
                f"Starting thread {thread + 1}/{self.config['DataPreProcessor']['num_threads']}"
            )
            t = threading.Thread(
                target=_convert_list_of_images, args=(image_list_per_thread[thread],)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()