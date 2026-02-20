import json

import numpy as np
from src.utils.io.nd2.load_nd2 import load_nd2, path_checks
from src.preprocessing.functional.nifti_operations import (
    auto_fit_contrast,
    overall_intensity_matching,
)

import logging
import os

from src.utils.io.data_structure.files_and_paths.nifti_path_operations import (
    get_nifti_path,
)
from src.utils.io.nifti.write_to_nifti import write_to_nifti
from skimage.transform import resize

from utils.io.data_structure.files_and_paths.nd2_path_operations import (
    collect_and_check_path_information,
)


def nd2_to_nifti(
    data_path: str,
    input_paths: str,
    auto_adjust_contrast: bool = False,
    channel_settings: dict = None,
    channel_mapping: dict = None,
    channel_selection_conditions: dict = None,
    possible_ref_images: str = None,
    channels_for_intensity_matching: list = None,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    z_scale: float = 1.0,
):

    # Check if the input paths are valid
    path_checks(input_paths)

    # Collect and check path information of the input paths
    subject_id, week, year, month, day, roi = collect_and_check_path_information(
        input_paths
    )
    subject_folder = os.path.join(data_path, subject_id)

    # create a nifti folder
    folder = os.path.join(os.path.dirname(input_paths[0]), "nifti")
    os.makedirs(folder, exist_ok=True)

    img_data = None
    metadata = None
    Z = None
    C = None
    X = None
    Y = None

    already_seen_channels = []
    combined_metadata = {
        "contents": {"channelCount": len(channel_mapping.keys()), "frameCount": None},
        "channels": [
            {"channel": {"name": k, "available": False}}
            for k in range(len(channel_mapping.keys()))
        ],
    }

    logging.info(f"Combining: [{', '.join([p for p in input_paths])}]")

    for p in input_paths:

        p_data, p_metadata, p_Z, p_C, p_Y, p_X = load_nd2(p)
        p_channels = [c["channel"]["name"] for c in p_metadata["channels"]]

        if img_data is None:
            img_data = p_data
            metadata = p_metadata
            Z = p_Z
            C = p_C
            Y = p_Y
            X = p_X
            combined_metadata["contents"]["frameCount"] = p_metadata["contents"][
                "frameCount"
            ]

        # Check all the images have the same shape
        assert (
            Z == p_Z and Y == p_Y and X == p_X
        ), f"Data shapes do not match in critical dimension for {p} and {input_paths[0]}: {(Z, Y, X)} and {(p_Z, p_Y, p_X)}"

        # Save the channel data as seperate files
        for c in range(p_C):
            channel_name = p_metadata["channels"][c]["channel"]["name"]
            if (
                channel_name in channel_mapping.keys()
                and channel_name not in already_seen_channels
            ):
                if channel_name in channel_selection_conditions.keys():
                    # check if channel should be included based on the presence or absence of other channels
                    # example: we want the mCherry channel to be taken if the Hoechst33258 channel is present and
                    # the Alexa 488 water channel is not present
                    for present_channel in channel_selection_conditions[channel_name][
                        "if_channels_present"
                    ]:
                        if present_channel not in p_channels:
                            logging.warning(
                                f"Channel {present_channel} not present. Skipping channel {channel_name}."
                            )
                            continue
                    for absent_channel in channel_selection_conditions[channel_name][
                        "if_channels_not_present"
                    ]:
                        if absent_channel in p_channels:
                            logging.warning(
                                f"Channel {absent_channel} present. Skipping channel {channel_name}."
                            )
                            continue
                channel_idx = channel_mapping[channel_name]
                combined_metadata["channels"][channel_idx] = p_metadata["channels"][c]
                combined_metadata["channels"][channel_idx]["channel"][
                    "available"
                ] = True
                if channel_name not in already_seen_channels:
                    already_seen_channels.append(channel_name)
            elif channel_name not in channel_mapping.keys():
                raise ValueError(
                    f"Channel {channel_name} not found in channel mapping."
                )
            else:
                logging.warning(
                    f"Channel {channel_name} already seen. Skipping channel."
                )
                continue

            channel_data = p_data[:, c, :, :]
            channel_data = np.moveaxis(channel_data, 0, -1)
            channel_data = channel_data[:, :, ::-1]

            if auto_adjust_contrast and possible_ref_images is None:
                try:
                    channel_name = "channel_" + str(channel_idx + 1)
                    # Perform automatic constrast enhancement using quantiles
                    low_quantile = channel_settings[channel_name]["low_quantile"]
                    high_quantile = channel_settings[channel_name]["high_quantile"]
                    channel_data = auto_fit_contrast(
                        channel_data,
                        low_quantile=low_quantile,
                        high_quantile=high_quantile,
                    )
                except KeyError:
                    logging.warning(
                        f"Channel settings for {channel_name} not found. Skipping contrast adjustment."
                    )
                    logging.warning("Nd2 File Path: " + p)
            elif not auto_adjust_contrast and possible_ref_images is not None:
                if channels_for_intensity_matching is None:
                    raise ValueError(
                        "Defined the channels, where the intensities should be matched with the ref image."
                    )
                if channel_name not in channels_for_intensity_matching:
                    # no intensity matching required for this channel
                    continue
                for ref_image in possible_ref_images:
                    re_data, ref_metadata, _, _, _, _ = load_nd2(ref_image)
                    ref_channels = [
                        c["channel"]["name"] for c in ref_metadata["channels"]
                    ]
                    if channel_name in channel_selection_conditions.keys():
                        # Perform same check on the ref images
                        for present_channel in channel_selection_conditions[
                            channel_name
                        ]["if_channels_present"]:
                            if present_channel not in ref_channels:
                                logging.warning(
                                    f"Channel {present_channel} not present. Skipping channel {channel_name}."
                                )
                                continue
                        for absent_channel in channel_selection_conditions[
                            channel_name
                        ]["if_channels_not_present"]:
                            if absent_channel in ref_channels:
                                logging.warning(
                                    f"Channel {absent_channel} present. Skipping channel {channel_name}."
                                )
                                continue

                    ref_channel_index = ref_channels.index(channel_name)
                    ref_channel_data = re_data[:, ref_channel_index, :, :]
                    channel_data, scaling_factor = overall_intensity_matching(
                        ref_img=ref_channel_data, img=channel_data
                    )

                    if scaling_factor < 0.5 or scaling_factor > 2.0:
                        logging.warning(
                            f"Scaling factor {scaling_factor} is outside the range [0.5, 2.0] for image {p}."
                        )

            else:
                raise ValueError(
                    "Either Choose auto_adjut_contrast or specifiy ref_image_instensity_matching."
                )

            ds_channel_data = resize(
                channel_data,
                (int(Y * y_scale), int(X * x_scale), int(Z * z_scale)),
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(channel_data.dtype)
            channel_data = ds_channel_data

            filepath = get_nifti_path(
                subject_folder, week, roi, str(channel_idx + 1), year, month, day
            )
            logging.info(filepath)
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            logging.info(f"Writing to nifti: {filepath}")
            write_to_nifti(channel_data, channel_data.dtype, filepath)

    metadata_path = os.path.join(os.path.dirname(input_paths[0]), "metadata.json")
    logging.info(metadata_path)
    if not os.path.exists(os.path.dirname(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # # save image also as rgb nifti
    # rgb_data = convert_channels_to_rgb(
    #     red=channel_data[1], green=np.zeros_like(channel_data[0]), blue=channel_data[0]
    # )
    # rgb_data = np.expand_dims(rgb_data, axis=3)

    # rgb_filepath = get_rgb_path(subject_folder, week, roi, year, month, day)
    # logging.info(rgb_filepath)
    # if not os.path.exists(os.path.dirname(rgb_filepath)):
    #     os.makedirs(os.path.dirname(rgb_filepath))
    # write_to_nifti(rgb_data, nd2_data.dtype, rgb_filepath)

    # nd2_file.close()
