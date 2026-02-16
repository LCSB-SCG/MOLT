import json
import os
from typing import Dict, List, Optional

import matplotlib as mpl
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import cm
from moviepy.editor import ImageSequenceClip
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import requests
from src.general.Config import Config
from src.general.Cohort import Cohort
from src.utils.io.data_structure.files_and_paths.general_path_operations import (
    extract_number_from_week_string,
)
from src.utils.io.nifti.write_to_nifti import write_to_nifti
import zipfile
import shutil
import logging
from scipy import ndimage

dl_dir = "tmp_downloads"
data_dir = "data"
config_template = "assets/config_celltracking_template.json"


class CellTrackingDataset:
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        if config_path is None and config is None:
            raise ValueError("config or config_path must be provided")
        elif config_path is not None and config is not None:
            raise ValueError("Only one of config or config_path must be provided")
        if config is not None:
            self.config = config
        else:
            assert config_path is not None
            self.config = Config(config_path=config_path)
        self.cohort = Cohort(self.config)

    @staticmethod
    def setup_datasets():

        d1 = {
            "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip",
            "name": "Fluo-N2DH-GOWT1",
        }
        d2 = {
            "url": "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip",
            "name": "Fluo-N2DL-HeLa",
        }
        d3 = {
            "url": "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip",
            "name": "PhC-C2DL-PSC",
        }
        datasets = [d1, d2, d3]
        # CellTrackingDataset.download_datasets(datasets)

        for d in datasets:
            d_path = os.path.join("data", "CellTracking", d["name"])
            d_set = CellTrackingDataset(config_path=os.path.join(d_path, "config.json"))
            logging.info(f"Structuring dataset {d['name']}")
            d_set.structure_dataset()
            # logging.info("- Extracting tracking files.")
            # d_set.man_track_to_graph_and_lineage()
            # logging.info(f"- Removing unnecessary files.")
            # d_set.remove_original_format_files(dataset_path=d_path)

    @staticmethod
    def remove_original_format_files(dataset_path):
        to_be_rm = [
            "01",
            "01_ERR_SEG",
            "01_GT",
            "01_ST",
            "02",
            "02_ERR_SEG",
            "02_GT",
            "02_ST",
        ]

        for tbrm in to_be_rm:
            shutil.rmtree(os.path.join(dataset_path, tbrm))

    def load_tiff_image(self, filepath):
        """
        Loads a TIFF image from the specified file path.

        Args:
            filepath (str): The path to the TIFF image file.

        Returns:
            numpy.ndarray: A NumPy array representing the image,
                        or None if an error occurs.
        """
        try:
            img = Image.open(filepath)
            img_array = np.array(img)
            return img_array
        except FileNotFoundError:
            logging.info(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            logging.info(f"Error loading TIFF image: {e}")
            return None

    def structure_dataset(self, erosion=True, vol_filter=True):
        sub_ids = ["01", "02"]
        issue_counter = 0
        for sub_id in sub_ids:
            name = f"Cells_{sub_id}"
            tiff_path = os.path.join(self.config["general"]["data_path"], sub_id)
            tracking_path = os.path.join(
                self.config["general"]["data_path"], f"{sub_id}_GT", "TRA"
            )
            seg_path = os.path.join(
                self.config["general"]["data_path"], f"{sub_id}_ST", "SEG"
            )

            cellpose_path = os.path.join(
                self.config["general"]["data_path"], f"{sub_id}_Cellpose"
            )

            # Move the manual tracking file to the sub_id folder
            subject_path = os.path.join(
                self.config["general"]["data_path"], f"Cells_{sub_id}"
            )
            os.makedirs(subject_path, exist_ok=True)
            shutil.copy(
                src=os.path.join(tracking_path, "man_track.txt"),
                dst=os.path.join(subject_path, "man_track_001.txt"),
            )

            num_files = len(os.listdir(tiff_path))

            logging.info("Number of files: ", str(num_files))
            for idx in range(0, num_files):
                load_tiff_path = os.path.join(tiff_path, f"t{idx:03d}.tif")
                load_tracking_path = os.path.join(
                    tracking_path, f"man_track{idx:03d}.tif"
                )
                load_seg_path = os.path.join(seg_path, f"man_seg{idx:03d}.tif")
                if os.path.exists(cellpose_path):
                    load_cellpose_path = os.path.join(cellpose_path, f"mask{idx:03d}.tif")
                else:
                    load_cellpose_path = ""

                # Create the nifti paths
                out_orig_img_path = os.path.join(
                    self.config["general"]["data_path"],
                    name,
                    f"{idx}_weeks",
                    "nifti",
                    "001",
                    "channel_1",
                    "1901_01_01_001_channel_1.nii.gz",
                )
                
                out_gt_seg_path = os.path.join(
                    self.config["general"]["data_path"],
                    name,
                    f"{idx}_weeks",
                    "label",
                    "001",
                    "channel_1",
                    "ground_truth",
                    "1901_01_01_001_instances.nii.gz",
                )
                out_bin_seg_path = os.path.join(
                    self.config["general"]["data_path"],
                    name,
                    f"{idx}_weeks",
                    "label",
                    "001",
                    "channel_1",
                    "manual",
                    "1901_01_01_001_binary.nii.gz",
                )
                
                if os.path.exists(cellpose_path):
                    out_cellpose_path = os.path.join(
                        self.config["general"]["data_path"],
                        name,
                        f"{idx}_weeks",
                        "label",
                        "001",
                        "channel_1",
                        "cellpose",
                        "1901_01_01_001_instances.nii.gz",
                    )
                    out_cellpose_bin_path = os.path.join(
                        self.config["general"]["data_path"],
                        name,
                        f"{idx}_weeks",
                        "label",
                        "001",
                        "channel_1",
                        "cellpose_bin",
                        "1901_01_01_001_binary.nii.gz",
                    )
                else:
                    out_cellpose_path = ""
                    out_cellpose_bin_path = ""
                    
                os.makedirs(os.path.dirname(out_gt_seg_path), exist_ok=True)
                os.makedirs(os.path.dirname(out_bin_seg_path), exist_ok=True)
                os.makedirs(os.path.dirname(out_orig_img_path), exist_ok=True)
                if os.path.exists(cellpose_path):
                    os.makedirs(os.path.dirname(out_cellpose_path), exist_ok=True)
                    os.makedirs(os.path.dirname(out_cellpose_bin_path), exist_ok=True)

                # Load and save the original image
                img = self.load_tiff_image(load_tiff_path)
                write_to_nifti(
                    nifti_data=img, dtype=np.float32, filename=out_orig_img_path
                )
                

                # We will process 1. ground truth annotions and 2. automatic segmentations for cellpose
                # 1. to test the performance of the algorithm without the influence of the segmentation performance
                # 2. to test the performance of the algorithm with the influence of the segmentation erros from automatic segmentations, which is more similar to the real use case of the algorithm
                
                # 1.
                # Load and save the ground truth tracking (these are not full segmentations)
                use_tra_img_as_gt = False
                if use_tra_img_as_gt:
                    gt_tracking = self.load_tiff_image(load_tracking_path)
                else: 
                    gt_tracking = self.load_tiff_image(load_seg_path)

                write_to_nifti(
                    nifti_data=gt_tracking,
                    dtype=np.float32,
                    filename=out_gt_seg_path,
                )

                # also save as binary and eroded for input to the tracking
                new_img = self._process_segmentation(
                    gt_tracking, 
                    apply_erosion=erosion, 
                    apply_vol_filter=vol_filter
                )

                write_to_nifti(
                    nifti_data=new_img, dtype=np.float32, filename=out_bin_seg_path
                )

                # 2. Load and save the cellpose segmentation if exists
                if os.path.exists(cellpose_path):
                    cellpose_seg = self.load_tiff_image(load_cellpose_path)
                    write_to_nifti(
                        nifti_data=cellpose_seg,
                        dtype=np.float32,
                        filename=out_cellpose_path,
                    )
                    cellpose_bin = self._process_segmentation(
                        cellpose_seg, apply_erosion=erosion, apply_vol_filter=vol_filter
                    )
                    write_to_nifti(
                        nifti_data=cellpose_bin,
                        dtype=np.float32,
                        filename=out_cellpose_bin_path,
                    )

                logging.info(
                    f"Processed {name} {idx} {load_tiff_path} -> {out_orig_img_path}"
                )
        logging.info(
            f"Finished structuring dataset. Found {issue_counter} issues with tracking instances overlapping with ground truth instances."
        )


    def _process_segmentation(self, seg, apply_erosion=True, apply_vol_filter=True):
        if not apply_erosion:
            return (seg > 0).astype(np.uint8)

        new_img = np.zeros_like(seg)
        for i in np.unique(seg):
            if i == 0:
                continue
            mask = seg == i

            # Check if this instance touches any other instance (8-connected in 2D, 26-connected in 3D)
            dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((3,)*seg.ndim))
            touches_other = np.any((dilated_mask & (seg != 0) & (seg != i)))

            if touches_other:
                # Apply erosion only if touching another instance
                for _ in range(1):
                    new_mask = ndimage.binary_erosion(mask, iterations=1).astype(np.uint8)
                    if np.sum(new_mask) == 0:
                        break
                    # Ensure the eroded mask remains a single connected component
                    label, num_cc = ndimage.label(new_mask)
                    if num_cc > 1 and apply_vol_filter:
                        largest_cc = np.argmax(np.bincount(label.flat)[1:]) + 1
                        new_mask = (label == largest_cc).astype(np.uint8)
                    mask = new_mask
            new_img += mask

        new_img[new_img > 0] = 1  # Convert to binary
        return new_img


    def associate_tracked_instances_with_gt(self):
        # Placeholder for ground truth collection logic
        sr_tracked_to_gt = {}
        for s in self.cohort.get_subjects():
            sr_tracked_to_gt[s] = {}
            for r in self.cohort.get_subject_rois(subject_id=s):

                logging.info(f"Processing ROI {r} for subject {s}")
                gt_instances = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[s],
                    file_type="instances",
                    label_setting="ground_truth",
                )
                tracked = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[s],
                    file_type="tracked",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                instances = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[s],
                    file_type="instances",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                graph_path = self.cohort.get_information_source_files_as_list(
                    information_source="lineage_graph",
                    channel="channel_1",
                    subject_ids=[s],
                    rois=r,
                    file_type=None,
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )[0]
                graph = nx.read_graphml(graph_path)
                sr_tracked_to_gt[s][r] = {}

                for img_idx, (gt_inst, tracked, inst) in tqdm(
                    enumerate(zip(gt_instances, tracked, instances))
                ):
                    # load the images
                    gt_inst = nib.load(gt_inst).get_fdata()
                    tracked = nib.load(tracked).get_fdata()
                    inst = nib.load(inst).get_fdata()

                    # get the overlapping ids of the ground truth and the calculated instances to add them to the graph
                    for i in np.unique(inst):
                        if i == 0:
                            continue
                        formated_id = f"{img_idx}_{i}"
                        i_volume = np.sum(inst == i)
                        unique_gt_tracked_ids = np.unique(gt_inst[inst == i])
                        unique_gt_tracked_ids = unique_gt_tracked_ids[
                            unique_gt_tracked_ids != 0
                        ]  # Exclude background

                        unique_gt_instance_ids = np.unique(gt_inst[inst == i])
                        unique_gt_instance_ids = unique_gt_instance_ids[
                            unique_gt_instance_ids != 0
                        ]  # Exclude background
                        # Calculate the ious between gt and tracked instances
                        if len(unique_gt_tracked_ids) == 0:
                            gt_with_highest_iou = None
                            highest_iou = 0
                        else:
                            ious = []
                            for gt_id in unique_gt_tracked_ids:
                                gt_mask = gt_inst == gt_id
                                inst_mask = inst == i
                                intersection = np.sum(
                                    np.logical_and(gt_mask, inst_mask)
                                )
                                union = np.sum(np.logical_or(gt_mask, inst_mask))
                                iou = intersection / union if union > 0 else 0
                                ious.append((gt_id, iou))
                            # Find the ground truth instance with the highest IoU
                            gt_with_highest_iou, highest_iou = max(
                                ious, key=lambda x: x[1]
                            )

                        if formated_id in graph.nodes:
                            graph.nodes[formated_id]["gt_instance_ids"] = ",".join(
                                map(str, unique_gt_instance_ids.tolist())
                            )
                            graph.nodes[formated_id]["highest_iou"] = highest_iou
                            graph.nodes[formated_id]["volume"] = i_volume
                            graph.nodes[formated_id]["gt_with_highest_iou"] = (
                                f"{img_idx}_{gt_with_highest_iou}"
                                if gt_with_highest_iou
                                else "NONE"
                            )

                    # get the associations between the tracked instances and the ground truth instances
                    for i in np.unique(tracked):
                        if i == 0:
                            continue
                        if i not in sr_tracked_to_gt[s][r]:
                            sr_tracked_to_gt[s][r][i] = {
                                "matched_gt": [],
                                "timepoints": [],
                            }
                        unique_gts = np.unique(gt_inst[tracked == i])
                        sr_tracked_to_gt[s][r][i]["matched_gt"].append(
                            unique_gts[unique_gts != 0].tolist()
                        )
                        sr_tracked_to_gt[s][r][i]["timepoints"].append(img_idx)

                # Save the graph with the ground truth ids
                nx.write_graphml(
                    graph, graph_path.replace(".graphml", "_added_gt_ids.graphml")
                )

        # Save the associations both associated with the ground truth and tracked instances
        json_tracked_to_gt_path = os.path.join(
            self.config["general"]["result_folder"], "tracked_to_gt.json"
        )
        with open(json_tracked_to_gt_path, "w") as f:
            json.dump(sr_tracked_to_gt, f, indent=4)

    def associate_gt_with_tracked_instances(self):
        """
        Associates the ground truth instances with the tracked instances.
        This is done by comparing the ground truth instances with the tracked instances
        and finding the best match based on the highest IoU.
        """
        gt_to_tracked = {}
        for subject in self.cohort.get_subjects():
            gt_to_tracked[subject] = {}
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                logging.info(f"Processing ROI {roi} for subject {subject}")
                gt_instances = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="instances",
                    label_setting="ground_truth",
                )
                tracked = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="tracked",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                instances = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="instances",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                graph_path = self.cohort.get_information_source_files_as_list(
                    information_source="lineage_graph",
                    channel="channel_1",
                    subject_ids=[subject],
                    rois=roi,
                    file_type=None,
                    label_setting="ground_truth",
                )[0]
                logging.info(f"Loading graph from {graph_path}")
                graph = nx.read_graphml(path=graph_path)

                gt_to_tracked[subject][roi] = {}

                for img_idx, (gt_inst, tracked, inst) in tqdm(
                    enumerate(zip(gt_instances, tracked, instances))
                ):
                    # load the images
                    gt_inst = nib.load(gt_inst).get_fdata()
                    tracked = nib.load(tracked).get_fdata()
                    inst = nib.load(inst).get_fdata()

                    # get the overlapping ids of the ground truth and the calculated instances to add them to the graph
                    for gt_i in np.unique(gt_inst):
                        if gt_i == 0:
                            continue
                        gt_mask = gt_inst == gt_i
                        formated_id = f"{img_idx}_{gt_i}"
                        i_volume = np.sum(gt_mask)
                        unique_tracked_ids = np.unique(tracked[gt_mask])
                        unique_tracked_ids = unique_tracked_ids[unique_tracked_ids != 0]

                        unique_inst_ids = np.unique(inst[gt_mask])
                        unique_inst_ids = unique_inst_ids[unique_inst_ids != 0]

                        # Calculate the ious between gt and tracked instances
                        # this assumes that the predicted segmentations is aligned with the ground truth
                        # and that the tracked instances are in the same space as the ground truth and not larger
                        if len(unique_inst_ids) == 0:
                            instance_with_highest_iou = None
                            percent_of_gt = 0
                        else:
                            num_voxel_overlap = []
                            for inst_id in unique_inst_ids:
                                inst_mask = inst == inst_id
                                intersection = np.sum(
                                    np.logical_and(gt_mask, inst_mask)
                                )
                                num_voxel_overlap.append(
                                    (inst_id, np.sum(gt_mask) / intersection)
                                )
                            # Find the tracked instance with the highest IoU
                            instance_with_highest_iou, percent_of_gt = max(
                                num_voxel_overlap, key=lambda x: x[1]
                            )

                        graph.nodes[formated_id]["tracked_instance_ids"] = ",".join(
                            map(str, unique_inst_ids.tolist())
                        )
                        graph.nodes[formated_id][
                            "%_vol_intersection_vol"
                        ] = percent_of_gt
                        graph.nodes[formated_id]["volume"] = i_volume
                        graph.nodes[formated_id]["tracked_with_%"] = (
                            f"{img_idx}_{instance_with_highest_iou}"
                            if instance_with_highest_iou
                            else "NONE"
                        )
                    # get the associations between the ground truth instances and the tracked instances
                    for gt_i in np.unique(gt_inst):
                        if gt_i == 0:
                            continue
                        if gt_i not in gt_to_tracked[subject][roi]:
                            gt_to_tracked[subject][roi][gt_i] = {
                                "matched_tracked": [],
                                "timepoints": [],
                            }
                        unique_tracked = np.unique(tracked[gt_mask])
                        gt_to_tracked[subject][roi][gt_i]["matched_tracked"].append(
                            unique_tracked[unique_tracked != 0].tolist()
                        )
                        gt_to_tracked[subject][roi][gt_i]["timepoints"].append(img_idx)
                # Save the graph with the tracked ids
                nx.write_graphml(
                    graph,
                    graph_path.replace(
                        "lineage_gt.graphml", "lineage_gt_added_tracked_ids.graphml"
                    ),
                )
                # Save the associations both associated with the ground truth and tracked instances
                json_gt_to_tracked_path = os.path.join(
                    self.config["general"]["result_folder"], "gt_to_tracked.json"
                )
                with open(json_gt_to_tracked_path, "w") as f:
                    json.dump(gt_to_tracked, f, indent=4)

    def man_track_to_graph_and_lineage(self):
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                lineage = nx.DiGraph()
                counter_lineage = 0
                gt_to_lineage = {}
                gt_start = {}
                gt_end = {}
                with open(
                    os.path.join(
                        self.config["general"]["data_path"],
                        subject,
                        f"man_track_{roi}.txt",
                    ),
                    "r",
                ) as f:
                    lines = f.readlines()
                    # order the lines by start timepoint
                    lines.sort(key=lambda x: int(x.split()[1]))
                    for lines in lines:
                        gt_id, start_timepoint, end_timepoint, parent_id = map(
                            int, lines.strip().split()
                        )
                        gt_id = float(gt_id)
                        parent_id = float(parent_id)
                        gt_start[gt_id] = start_timepoint
                        gt_end[gt_id] = end_timepoint
                        if parent_id == 0:
                            counter_lineage += 1
                            gt_to_lineage[gt_id] = counter_lineage
                            lineage.add_node(f"{start_timepoint}_{gt_id}")
                            nx.set_node_attributes(
                                lineage,
                                {
                                    f"{start_timepoint}_{gt_id}": {
                                        "id": f"{start_timepoint}_{gt_id}",
                                        "lineage_id": counter_lineage,
                                        "start_timepoint": start_timepoint,
                                        "end_timepoint": end_timepoint,
                                    }
                                },
                            )
                            for i in range(start_timepoint + 1, end_timepoint + 1):
                                lineage.add_node(f"{i}_{gt_id}")
                                lineage.add_edge(
                                    f"{i-1}_{gt_id}",
                                    f"{i}_{gt_id}",
                                )
                                nx.set_node_attributes(
                                    lineage,
                                    {
                                        f"{i}_{gt_id}": {
                                            "id": f"{i}_{gt_id}",
                                            "lineage_id": counter_lineage,
                                            "start_timepoint": start_timepoint,
                                            "end_timepoint": end_timepoint,
                                        }
                                    },
                                )
                        else:
                            gt_to_lineage[gt_id] = gt_to_lineage[parent_id]
                            lineage.add_edge(
                                f"{gt_end[parent_id]}_{parent_id}",
                                f"{start_timepoint}_{gt_id}",
                            )
                            lineage.add_node(f"{start_timepoint}_{gt_id}")
                            nx.set_node_attributes(
                                lineage,
                                {
                                    f"{start_timepoint}_{gt_id}": {
                                        "id": f"{start_timepoint}_{gt_id}",
                                        "lineage_id": gt_to_lineage[parent_id],
                                        "start_timepoint": start_timepoint,
                                        "end_timepoint": end_timepoint,
                                    }
                                },
                            )
                            for i in range(start_timepoint + 1, end_timepoint + 1):
                                lineage.add_node(f"{i}_{gt_id}")
                                lineage.add_edge(
                                    f"{i-1}_{gt_id}",
                                    f"{i}_{gt_id}",
                                )
                                nx.set_node_attributes(
                                    lineage,
                                    {
                                        f"{i}_{gt_id}": {
                                            "id": f"{i}_{gt_id}",
                                            "lineage_id": gt_to_lineage[parent_id],
                                            "start_timepoint": start_timepoint,
                                            "end_timepoint": end_timepoint,
                                        }
                                    },
                                )

                lineage_path = os.path.join(
                    self.config["general"]["result_folder"],
                    "lineage_graphs",
                    f"{subject}_{roi}_lineage_gt.graphml",
                )
                nx.write_graphml(lineage, lineage_path)

    def tracked_to_video(self):
        background_color = self.config["Cell_Tracking_Dataset"]["video_setting"][
            "background_color"
        ]
        background_color = (
            background_color["r"],
            background_color["g"],
            background_color["b"],
        )
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                logging.info(f"Creating video for subject {subject}, ROI {roi}")
                tracked = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="tracked",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                lineage = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    rois=roi,
                    file_type="lineage",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )
                tracked_video_name = f"{subject}_{roi}_tracked_video.mp4"
                lineage_video_name = f"{subject}_{roi}_lineage_video.mp4"
                for frames_files, video_name in zip(
                    [tracked, lineage], [tracked_video_name, lineage_video_name]
                ):
                    video_frames = []
                    inst_to_color = {}
                    seen_colors = set(background_color)
                    for img_idx, frame_file in tqdm(
                        enumerate(frames_files), desc="Processing frames"
                    ):
                        tracked_data = nib.load(frame_file).get_fdata()
                        frame = np.zeros(
                            (tracked_data.shape[0], tracked_data.shape[1], 3),
                            dtype=np.uint8,
                        )
                        for i in np.unique(tracked_data):
                            if i in inst_to_color.keys():
                                color = inst_to_color[i]
                            else:
                                color = background_color
                                while color in seen_colors:
                                    color = (
                                        np.random.randint(0, 255),
                                        np.random.randint(0, 255),
                                        np.random.randint(0, 255),
                                    )
                                inst_to_color[i] = color
                                seen_colors.add(color)
                            if i == 0:
                                continue
                            mask = tracked_data == i
                            # Create a color for the instance (e.g., random color)
                            frame[mask] = color

                        # Convert to a suitable format for video (e.g., 8-bit grayscale)
                        video_frames.append(frame.astype(np.uint8))

                    # Save the video frames as a video file
                    video_path = os.path.join(
                        self.config["general"]["result_folder"],
                        "tracked_videos",
                        video_name,
                    )
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    clip = ImageSequenceClip(video_frames, fps=1)
                    clip.write_videofile(video_path, codec="libx264", audio=False)

    def gt_to_video(self):
        background_color = self.config["Cell_Tracking_Dataset"]["video_setting"][
            "background_color"
        ]
        background_color = (
            background_color["r"],
            background_color["g"],
            background_color["b"],
        )
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                logging.info(f"Creating video for subject {subject}, ROI {roi}")
                gt_instances = self.cohort.get_information_source_files_as_list(
                    information_source="label",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="instances",
                    label_setting="ground_truth",
                )
                gt_video_name = f"{subject}_{roi}_gt_video.mp4"
                video_frames = []
                inst_to_color = {}
                seen_colors = set(background_color)
                for img_idx, frame_file in tqdm(
                    enumerate(gt_instances), desc="Processing frames"
                ):
                    gt_data = nib.load(frame_file).get_fdata()
                    frame = np.zeros(
                        (gt_data.shape[0], gt_data.shape[1], 3), dtype=np.uint8
                    )
                    for i in np.unique(gt_data):
                        if i in inst_to_color.keys():
                            color = inst_to_color[i]
                        else:
                            color = background_color
                            while color in seen_colors:
                                color = (
                                    np.random.randint(0, 255),
                                    np.random.randint(0, 255),
                                    np.random.randint(0, 255),
                                )
                            inst_to_color[i] = color
                            seen_colors.add(color)
                        if i == 0:
                            continue
                        mask = gt_data == i
                        # Create a color for the instance (e.g., random color)
                        frame[mask] = color

                    # Convert to a suitable format for video (e.g., 8-bit grayscale)
                    video_frames.append(frame.astype(np.uint8))

                # Save the video frames as a video file
                video_path = os.path.join(
                    self.config["general"]["result_folder"],
                    "gt_videos",
                    gt_video_name,
                )
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                clip = ImageSequenceClip(video_frames, fps=1)
                clip.write_videofile(video_path, codec="libx264", audio=False)

    def microscopy_to_video(self):
        background_color = self.config["Cell_Tracking_Dataset"]["video_setting"][
            "background_color"
        ]
        background_color = (
            background_color["r"],
            background_color["g"],
            background_color["b"],
        )
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                logging.info(f"Creating video for subject {subject}, ROI {roi}")
                microscopy_images = self.cohort.get_information_source_files_as_list(
                    information_source="nifti",
                    channel="channel_1",
                    subject_ids=[subject],
                    file_type="channel_1",
                )
                microscopy_video_name = f"{subject}_{roi}_microscopy_video.mp4"
                video_frames = []
                for img_idx, frame_file in tqdm(
                    enumerate(microscopy_images), desc="Processing frames"
                ):
                    img_data = nib.load(frame_file).get_fdata()
                    # Normalize the image data to 0-255
                    img_data = (img_data - np.min(img_data)) / (
                        np.max(img_data) - np.min(img_data)
                    )
                    # VIRISDIS colormap from matplotlib
                    colored_img = cm.viridis(img_data)
                    # Convert to a suitable format for video (e.g., 8-bit grayscale)
                    video_frames.append(
                        colored_img[:, :, :3] * 255
                    )  # Convert to RGB and scale to 0-255

                # Save the video frames as a video file
                video_path = os.path.join(
                    self.config["general"]["result_folder"],
                    "microscopy_videos",
                    microscopy_video_name,
                )
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                clip = ImageSequenceClip(video_frames, fps=1)
                clip.write_videofile(video_path, codec="libx264", audio=False)

    @staticmethod
    def temporal_consistency(pred_graph, gt_graph):
        """
        Checks the temporal consistency of the predicted graph against the ground truth graph.
        """
        # Every GT instance needs to match with at least one tracked instance over time
        # Every tracked instance needs to match with at least one GT instance over time
        # The temporal consistency is calculated on the edges of the graph
        # the gt instance with the highest iou is considered as the match
        # We consider the consistency between two timepoints to calculate the accuracy
        # TP:
        #   a) tracked instance has one successor and both are associated with the same GT instance (a_n_c=1 and x_n_c=1)
        # TN:
        #   b) tracked instance has no successor and GT instance has no successor (a_n_c=0 and x_n_c=0)
        #   c) tracked instance has more than one successor and GT instance has same number of
        #       successors (a_n_c>1 and x_n_c>1 and a_n_c=x_n_c)
        # FP:
        #   d) tracked instance has one successor but both are associated with different
        #       GT instances (a_n_c=1 and a=x and a_child=y)
        #   e) tracked instance has one successor and GT instance has no successor (a_n_c=1 and x_n_c=0)
        #   f) tracked instance has one successor and GT instance has more than one successor (a_n_c=1 and x_n_c>1)
        # FN:
        #   g) tracked instance has no successor but GT instance has one successor (a_n_c=0 and x_n_c=1)
        #   h) tracked instance has no successor but GT instance has more than one successor (a_n_c=0 and x_n_c>1)
        #   i) tracked instance has more than one successor and GT instance has one successor (a_n_c>1 and x_n_c=1)
        #   j) tracked instance has more than one successor and GT instance has no successor (a_n_c>1 and x_n_c=0)
        #   k) tracked instance has more than one successor and GT instance has more than one successor but
        #       different number of successors (a_n_c>1 and x_n_c>1 and a_n_c!=x_n_c)

        a_cons_x_cons = 0  # a)
        a_leaf_x_leaf = 0  # b)
        a_split_x_split_same_num_childs = 0  # c)
        a_cons_x_incons = 0  # d)
        a_cons_x_leaf = 0  # e)
        a_cons_x_split = 0  # f)
        a_leaf_x_cons = 0  # g)
        a_leaf_x_split = 0  # h)
        a_split_x_cons = 0  # i)
        a_split_x_leaf = 0  # j)
        a_split_x_split_diff_num_childs = 0  # k)

        # Go ttrough all nodes of the predicted graph and check if they match with the ground truth graph
        for node in pred_graph.nodes:
            node_tracked_id = pred_graph.nodes[node]["tracked_id"]
            associated_gt_node = pred_graph.nodes[node]["gt_with_highest_iou"]
            gt_out_degree = gt_graph.out_degree[associated_gt_node]
            pred_out_degree = pred_graph.out_degree[node]
            if pred_out_degree == 0:
                if gt_out_degree == 0:
                    a_leaf_x_leaf += 1
                elif gt_out_degree == 1:
                    a_leaf_x_cons += 1
                elif gt_out_degree > 1:
                    a_leaf_x_split += 1
            if pred_out_degree == 1:
                child_node = list(pred_graph.successors(node))[0]
                child_tracked_id = pred_graph.nodes[child_node]["tracked_id"]
                if (
                    child_tracked_id != node_tracked_id
                ):  # [2:] cut away temporal information
                    raise ValueError(
                        "This case should not be possible. Pred nodes with one child should have the same tracked id."
                    )
                if gt_out_degree == 1:
                    gt_child_node = list(gt_graph.successors(associated_gt_node))[0]
                    if associated_gt_node.split("_")[1] == gt_child_node.split("_")[1]:
                        a_cons_x_cons += 1
                    else:
                        a_cons_x_incons += 1
                        logging.info(
                            f"a cons ({child_tracked_id}) x incons ({associated_gt_node.split('_')[1]} and {gt_child_node.split('_')[1]})"
                        )
                elif gt_out_degree == 0:
                    a_cons_x_leaf += 1
                elif gt_out_degree > 1:
                    a_cons_x_split += 1

            elif pred_out_degree > 1:
                if gt_out_degree == 1:
                    a_split_x_cons += 1
                elif gt_out_degree == 0:
                    a_split_x_leaf += 1
                elif gt_out_degree > 1:
                    if pred_out_degree == gt_out_degree:
                        a_split_x_split_same_num_childs += 1
                    else:
                        a_split_x_split_diff_num_childs += 1

        count_dict = {
            "a_cons_x_cons": a_cons_x_cons,
            "a_cons_x_split": a_cons_x_split,
            "a_cons_x_leaf": a_cons_x_leaf,
            "a_cons_x_incons": a_cons_x_incons,
            "a_split_x_cons": a_split_x_cons,
            "a_split_x_leaf": a_split_x_leaf,
            "a_leaf_x_cons": a_leaf_x_cons,
            "a_leaf_x_leaf": a_leaf_x_leaf,
            "a_leaf_x_split": a_leaf_x_split,
            "a_split_x_split_diff_num_childs": a_split_x_split_diff_num_childs,
            "a_split_x_split_same_num_childs": a_split_x_split_same_num_childs,
        }

        tp = a_cons_x_cons
        tn = a_leaf_x_leaf + a_split_x_split_same_num_childs
        fp = a_cons_x_incons + a_cons_x_leaf + a_cons_x_split
        fn = (
            a_leaf_x_cons
            + a_leaf_x_split
            + a_split_x_cons
            + a_split_x_leaf
            + a_split_x_split_diff_num_childs
        )

        count_dict["tp"] = tp
        count_dict["tn"] = tn
        count_dict["fp"] = fp
        count_dict["fn"] = fn

        recognized_splits = (
            a_split_x_split_diff_num_childs + a_split_x_split_same_num_childs
        )
        not_detected_splits = a_cons_x_split + a_leaf_x_split
        wrong_split_detections = a_split_x_cons + a_split_x_leaf
        associations_break = a_cons_x_incons + a_leaf_x_cons

        count_dict["recognized_splits"] = recognized_splits
        count_dict["not_detected_splits"] = not_detected_splits
        count_dict["wrong_split_detections"] = wrong_split_detections
        count_dict["associations_break"] = associations_break

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy, count_dict

    def idf1_score(self, pred_graph, gt_graph, verbose: bool = True):
        """
        Calculate the IDF1 score for the predicted graph against the ground truth graph.
        This method is a placeholder and should be implemented based on the specific IDF1 calculation needed.
        """
        # We need to create a list of all gt_ids and all pred_ids (without the temporal index)
        # We count how often each association between a gt_id and a pred_id occurs
        gt_ids = set()
        pred_ids = {"NONE"}
        for node in gt_graph.nodes:
            gt_ids.add(gt_graph.nodes[node]["id"].split("_")[1])
        for node in pred_graph.nodes:
            pred_ids.add(pred_graph.nodes[node]["tracked_id"])
        count_matrix = pd.DataFrame(0, index=list(gt_ids), columns=list(pred_ids))
        for node in gt_graph.nodes:
            gt_id = gt_graph.nodes[node]["id"].split("_")[1]
            if "tracked_with_%" in gt_graph.nodes[node]:
                # associated tracked instance has been found
                associated_pred_node = gt_graph.nodes[node]["tracked_with_%"]
                if associated_pred_node != "NONE":
                    pred_id = pred_graph.nodes[associated_pred_node]["tracked_id"]
            else:
                pred_id = "NONE"
            count_matrix.at[gt_id, pred_id] += 1
        # Calculate the IDF1 score based on the count matrix
        # Get the Tp matching pairs between gt and pred using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(count_matrix, maximize=True)
        # Filter out the indices where the column is 'NONE'
        filtered_row_ind = []
        filtered_col_ind = []
        for r, c in zip(row_ind, col_ind):
            if count_matrix.columns[c] != "NONE":
                filtered_row_ind.append(r)
                filtered_col_ind.append(c)

        tp = count_matrix.values[filtered_row_ind, filtered_col_ind].sum()
        fp_and_fn = count_matrix.values.sum() - tp
        idf1 = tp / (tp + 0.5 * fp_and_fn) if (tp + 0.5 * fp_and_fn) > 0 else 0
        if verbose:
            logging.info("Count matrix:")
            logging.info(count_matrix)
            logging.info("Row indices:", row_ind)
            logging.info("Column indices:", col_ind)
            logging.info(
                f"Errors in tracking du to registration errors {row_ind.size - len(filtered_row_ind)}"
            )
            logging.info(f"IDF1 score: {idf1}")
        return idf1

    def calculate_metrics(self):
        """
        Calculate the metrics for the cell tracking dataset.
        This method is a placeholder and should be implemented based on the specific metrics needed.
        """
        # First calculate the temporal consistency metrics
        avg_acc = 0
        avg_idf1 = 0
        s_r_counter = 0
        for subject in self.cohort.get_subjects():
            metric_file = self.config["general"]["result_folder"]
            metric_file = os.path.join(metric_file, subject, "metrics.txt")
            logging.info(f"Saving metrics to {metric_file}")
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                s_r_counter += 1
                logging.info(f"Calculating metrics for subject {subject}, ROI {roi}")
                gt_graph = self.cohort.get_information_source_files_as_list(
                    information_source="lineage_graph",
                    channel="channel_1",
                    subject_ids=[subject],
                    rois=roi,
                    file_type="with_associated_tracked_ids",
                    label_setting="ground_truth",
                )[0]
                pred_graph = self.cohort.get_information_source_files_as_list(
                    information_source="lineage_graph",
                    channel="channel_1",
                    subject_ids=[subject],
                    rois=roi,
                    file_type="with_associated_gt_ids",
                    label_setting=self.config["Cell_Tracking_Dataset"]["label_setting"],
                )[0]
                gt_graph = nx.read_graphml(gt_graph)
                pred_graph = nx.read_graphml(pred_graph)
                idf1 = self.idf1_score(pred_graph=pred_graph, gt_graph=gt_graph)
                acc, counts = self.temporal_consistency(
                    pred_graph=pred_graph, gt_graph=gt_graph
                )
                with open(metric_file, "a") as f:
                    f.write(
                        "-----------------------------------------------------------------\n"
                    )
                    f.write(f"Metrics for ROI {roi}:\n")
                    f.write(f"IDF1 Score: {idf1}\n")
                    f.write(f"Accuracy: {acc}\n")
                    f.write("Counts:\n")
                    for key, value in counts.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                logging.info(f"Accuracy: {acc}")
                logging.info("Counts:")
                for key, value in counts.items():
                    logging.info(f"  {key}: {value}")
                avg_acc += acc
                avg_idf1 += idf1
        avg_idf1 /= s_r_counter
        logging.info(f"Average IDF1 over all subjects and ROIs: {avg_idf1}")
        avg_acc /= s_r_counter
        logging.info(f"Average accuracy over all subjects and ROIs: {avg_acc}")

    def plot_images_in_color(self):
        """
        Plot the raw images in color.
        """
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                for tp in self.cohort.get_subject_roi_weeks(
                    subject_id=subject, roi=roi
                ):
                    img_path = self.cohort.get_information_source_files_as_list(
                        information_source="nifti",
                        channel="channel_1",
                        subject_ids=[subject],
                        rois=roi,
                        file_type="channel_1",
                        weeks=[tp],
                    )[0]
                    img_data = nib.load(img_path).get_fdata()
                    # Normalize the image data to 0-255
                    img_data = (img_data - np.min(img_data)) / (
                        np.max(img_data) - np.min(img_data)
                    )
                    # VIRISDIS colormap from matplotlib
                    norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
                    colored_img = (cmap.to_rgba(img_data)[:, :, :3] * 255).astype(
                        np.uint8
                    )
                    # Save the colored image
                    save_path = os.path.join(
                        self.config["general"]["result_folder"],
                        "colored_images",
                        subject,
                        roi,
                        "channel_1_colored",
                        f"colored_tp_{extract_number_from_week_string(tp)}.png",
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    Image.fromarray(colored_img).save(save_path)
                    logging.info(f"Saved colored image to {save_path}")

    def plot_lineage_and_tracked_predictions(self):
        """
        Plot the lineage graph and tracked instances.
        This method is a placeholder and should be implemented based on the specific plotting needs.
        """
        background_color = self.config["Cell_Tracking_Dataset"]["video_setting"][
            "background_color"
        ]
        background_color = (
            background_color["r"],
            background_color["g"],
            background_color["b"],
        )
        background_color_hash = hash(background_color)
        for subject in self.cohort.get_subjects():
            for roi in self.cohort.get_subject_rois(subject_id=subject):
                tracked_to_color = {}
                seen_tracked_colors = {background_color_hash}
                lineage_to_color = {}
                seen_lineage_colors = {background_color_hash}
                for tp in self.cohort.get_subject_roi_weeks(
                    subject_id=subject, roi=roi
                ):
                    # make rgb folder if it does not exist
                    rgb_folder = os.path.join(
                        self.config["general"]["result_folder"],
                        "lineage_plots",
                        subject,
                        roi,
                    )
                    os.makedirs(rgb_folder, exist_ok=True)
                    lineage = nib.load(
                        self.cohort.get_information_source_files_as_list(
                            information_source="label",
                            channel="channel_1",
                            subject_ids=[subject],
                            rois=roi,
                            file_type="lineage",
                            label_setting="manual",
                            weeks=[tp],
                        )[0]
                    ).get_fdata()
                    tracked = nib.load(
                        self.cohort.get_information_source_files_as_list(
                            information_source="label",
                            channel="channel_1",
                            subject_ids=[subject],
                            file_type="tracked",
                            label_setting=self.config["Cell_Tracking_Dataset"][
                                "label_setting"
                            ],
                            weeks=[tp],
                        )[0]
                    ).get_fdata()
                    # plot the lineage and tracked instances
                    rgb_lineage = np.zeros(
                        (lineage.shape[0], lineage.shape[1], 3), dtype=np.uint8
                    )
                    for i in np.unique(lineage):
                        mask = lineage == i
                        if i == 0:
                            color = background_color
                        else:
                            if i in lineage_to_color.keys():
                                color = lineage_to_color[i]
                            else:
                                color = background_color
                                while hash(color) in seen_lineage_colors:
                                    color = (
                                        np.random.randint(25, 230),
                                        np.random.randint(25, 230),
                                        np.random.randint(25, 230),
                                    )
                                lineage_to_color[i] = color
                                seen_lineage_colors.add(hash(color))
                        rgb_lineage[mask] = np.asarray(color).astype(np.uint8)
                    rgb_tracked = np.zeros(
                        (tracked.shape[0], tracked.shape[1], 3), dtype=np.uint8
                    )
                    for i in np.unique(tracked):
                        mask = tracked == i
                        if i == 0:
                            color = background_color
                        else:
                            if i in tracked_to_color.keys():
                                color = tracked_to_color[i]
                            else:
                                color = background_color
                                while hash(color) in seen_tracked_colors:
                                    color = (
                                        np.random.randint(25, 230),
                                        np.random.randint(25, 230),
                                        np.random.randint(25, 230),
                                    )
                                tracked_to_color[i] = color
                                seen_tracked_colors.add(hash(color))
                        rgb_tracked[mask] = np.asarray(color).astype(np.uint8)
                    # save the images
                    lineage_path = os.path.join(
                        rgb_folder,
                        "lineage",
                        f"lineage_tp_{extract_number_from_week_string(tp)}.png",
                    )
                    tracked_path = os.path.join(
                        rgb_folder,
                        "tracked",
                        f"tracked_tp_{extract_number_from_week_string(tp)}.png",
                    )
                    os.makedirs(os.path.dirname(lineage_path), exist_ok=True)
                    os.makedirs(os.path.dirname(tracked_path), exist_ok=True)
                    Image.fromarray(rgb_lineage).save(lineage_path)
                    Image.fromarray(rgb_tracked).save(tracked_path)
                    logging.info(f"Saved lineage plot to {lineage_path}")

    @staticmethod
    def download_datasets(datasets: List[Dict]):
        os.makedirs("tmp_downloads", exist_ok=True)

        request_header = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
        }
        for idx, d in enumerate(datasets):
            d_dir = os.path.join(dl_dir, f"{d['name']}.zip")
            datasets[idx]["zip"] = d_dir
            logging.info(f"Downloading {d['name']} dataset to {d_dir}")
            response = requests.get(d["url"], stream=True, headers=request_header)

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(d_dir, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)

        # Extract the datasets to the data directory
        data_dir = os.path.join("data", "CellTracking")

        for d in datasets:
            extract_dir = os.path.join(data_dir, d["name"])
            os.makedirs(data_dir, exist_ok=True)
            logging.info(f"Extracting {d['name']} to {extract_dir}")
            with zipfile.ZipFile(d["zip"], "r") as zip_ref:
                zip_ref.extractall(data_dir)

            # Add config file to the dataset dir
            with open(config_template, "r") as template_file:
                config_content = template_file.read()
                config_content = config_content.replace("<DATASET_NAME>", d["name"])

                # write file to the config file in the data dir
                config_file_name = os.path.join(extract_dir, "config.json")
                with open(config_file_name, "w") as config_file:
                    config_file.write(config_content)

        logging.info("done.")
        logging.info(f"Removing {dl_dir}.")
        shutil.rmtree(dl_dir)

    def update(self):
        # Returning a new dataset will collect all new files that have been created since the generation of the previous
        # dataset class instance.
        return CellTrackingDataset(self.config)

    def run_evaluation(self):
        logging.info("Matching predictions with ground truth")
        self.man_track_to_graph_and_lineage()
        self.associate_gt_with_tracked_instances()
        self.associate_tracked_instances_with_gt()
        logging.info("Calculating the metrics")
        self.calculate_metrics()


if __name__ == "__main__":
    from src.utils.common import logging_to_stdout
    logging.basicConfig(level=logging.INFO)
    logging_to_stdout()
    CellTrackingDataset.setup_datasets()