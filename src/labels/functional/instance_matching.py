import logging

import networkx as nx
import numpy as np
from PIL import Image

from src.labels.functional.morphological_graph import (
    assign_tcui_ids,
    get_morphological_graph_frame_pair_mapping,
    get_morphological_graph_global_view_mapping,
)
from src.labels.functional.instance_labeling import label_instance_ids
from src.labels.Matches import CrossChannelMatches, Matches
from src.utils.io.nifti.write_to_nifti import write_to_nifti


def _mapping_to_per_img_mapping(
    original_to_shared_id: dict,
    num_images: int,
) -> list[dict]:
    """Convert a mapping from original instance ids to shared instance ids
    into a list of mappings, one for each image.
    Args:
        original_to_shared_id (dict): A mapping from original instance ids to shared instance ids.
        num_images (int): The number of images.
    Returns:
        list[dict]: A list of mappings, one for each image.
    """
    per_img_mapping = [{} for _ in range(num_images)]
    for original_id, shared_id in original_to_shared_id.items():
        img_idx, original_instance = original_id.split("_")
        img_idx = int(img_idx)
        original_instance = int(float(original_instance))
        per_img_mapping[img_idx][original_instance] = shared_id
    return per_img_mapping


def exclude_instances_at_mask(
    instance_image: np.ndarray,
    mask: np.ndarray,
    background_instance: int,
):
    """
    Excludes instances from the label image that are outside the mask.

    Args:
        instance_image (np.ndarray): The label image.
        mask (np.ndarray): The mask.
        background_instance (int): The id of the background instance.

    Returns:
        np.ndarray: The label image with the instances outside the mask set to 0.
        set: The set of instances that were excluded.
    """
    instances = np.unique(instance_image)
    inst_to_exclude = set()
    for inst in instances:
        if inst == background_instance:
            continue
        instance = instance_image == inst
        masked = (instance_image * mask) == inst
        if np.sum(instance) != np.sum(masked):
            # some voxels got removed by the mask
            instance_image[instance] = background_instance
            inst_to_exclude.add(inst)
    return instance_image, inst_to_exclude


def find_connected_components_in_adjacencies(
    adjacencies,
    global_view_mapping,
    iou=None,
    enforce_no_merge=False,
    enforce_no_split=False,
):
    """
    Get the connected components of the adjacencies.
    """
    G = nx.Graph()
    for key_id, value_ids in adjacencies.items():
        # treat the old instance as a node as we want to find connected components of the original instances
        for v_id in value_ids:
            G.add_edge(key_id, v_id)

    if not global_view_mapping:
        G = get_morphological_graph_frame_pair_mapping(
            G,
            iou=iou,
            enforce_no_merge=enforce_no_merge,
            enforce_no_split=enforce_no_split,
        )
    else:
        G = get_morphological_graph_global_view_mapping(
            G,
            iou=iou,
            enforce_no_merge=enforce_no_merge,
            enforce_no_split=enforce_no_split,
        )

    cc = nx.connected_components(G.to_undirected())
    return cc, G


def get_final_matching(
    adjacencies,
    global_view_mapping,
    background_instance=0,
    iou=None,
    enforce_no_merge=False,
    enforce_no_split=False,
):
    """
    Get the final matching of the instances, by identifying connected components in the adjacencies
    and assigning a new id to all the original instances in the component.
    """
    # original ids to connected component ids
    cc, G = find_connected_components_in_adjacencies(
        adjacencies=adjacencies,
        global_view_mapping=global_view_mapping,
        iou=iou,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    # get the lineage ids
    connected_components = list(cc)
    original_to_lineage_id = {}
    for c_idx, component in enumerate(connected_components):
        # for each connected component, assign a new id to all the original instances in the component
        if background_instance == c_idx:
            # do not assign the background instance to a new id
            new_idx = len(list(connected_components))
        else:
            new_idx = c_idx
        for instance_id in component:
            if type(instance_id) is not str:
                instance_id = int(instance_id)
            original_to_lineage_id[instance_id] = new_idx

    # get the temporally consistent unique instance (tcui) ids in the graph
    graph, original_to_tcui_id = assign_tcui_ids(G)

    return original_to_lineage_id, original_to_tcui_id, graph


def apply_matching(
    matches: dict,
    image: np.ndarray,
    ignore_label: int,
    background_instance: int,
    verbose=False,
) -> np.ndarray:
    """
    Applies the matching to the image.

    Args:
        matches (dict): The matching dictionary.
        image (np.ndarray): The image to apply the matching to.
        verbose (bool): If True, print the unique values in the image before and after the matching.

    Returns:
        np.ndarray: The image with the matching applied.
    """
    # Check for instances that are not matched, i.e. instances that are present in the unregistred image
    # but not in the registered image, due to interpolation or other reasons
    unique_instances = set(np.unique(image))
    matched_instances = set(matches.keys())
    unmatched_instances = unique_instances - matched_instances
    if verbose:
        logging.info("Uniques before applying match: " + str(np.unique(image)))
        logging.info("Unmatched instances: " + str(unmatched_instances))

    for unmatched_instance in unmatched_instances:
        if unmatched_instance == background_instance:
            continue
        matches[unmatched_instance] = ignore_label

    # apply the matching
    matched_image = np.zeros_like(image).astype(np.int32)
    for instance in matches.keys():
        matched_image[image == instance] = matches[instance]

    if verbose:
        logging.info("Uniques after applying match: " + str(np.unique(matched_image)))

    return matched_image


def match_instances_by_pairwise_binary_union(
    img_list,
    unreg_img_list,
    background_instance,
    verbose=False,
    enforce_no_merge=False,
    enforce_no_split=False,
):
    logging.info(f"Starting pairwise matching for {len(img_list)} images...")
    m = Matches(background_id=background_instance)

    iou_map = {"forward_in_time": {}, "backward_in_time": {}}
    # newly appearing instances will be added to the iou_map with empty key
    iou_map["forward_in_time"][""] = [f"0_{i}" for i in np.unique(img_list[0])]
    # last frame instances will be added to the iou_map with empty key
    iou_map["backward_in_time"][""] = [
        f"{len(img_list) - 1}_{i}" for i in np.unique(img_list[-1])
    ]
    for idx in range(1, len(img_list)):
        # perform binary union on previous image:
        prev_img = unreg_img_list[idx - 1]
        img = img_list[idx]

        # pad the image to avoid index errors
        # calculate the padding size
        padding = [
            (
                (img.shape[i] - prev_img.shape[i]) // 2,
                (img.shape[i] - prev_img.shape[i]) // 2,
            )
            for i in range(len(img.shape))
        ]
        # pad the previous image with zeros
        prev_img = np.pad(prev_img, padding, mode="constant", constant_values=0)

        prev_img_bin = prev_img != background_instance
        img_bin = img != background_instance
        binary_union = np.logical_or(prev_img_bin, img_bin)

        # id the overlapping components
        labels, _ = label_instance_ids(binary_union)

        for union_id in np.unique(labels):

            if union_id == 0:
                # skip the background instance union id
                continue

            formated_union_id = f"U_{idx}_{union_id}"

            # Get bounding box around the union instance to reduce memory usage
            union_mask = labels == union_id
            bbox = np.array(np.where(union_mask)).T
            if len(bbox) == 0:
                continue
            min_coords = bbox.min(axis=0)
            max_coords = bbox.max(axis=0) + 1
            
            # Crop images to bounding box (much smaller than full image)
            img_crop = img[tuple(slice(min_coords[i], max_coords[i]) for i in range(img.ndim))]
            prev_img_crop = prev_img[tuple(slice(min_coords[i], max_coords[i]) for i in range(prev_img.ndim))]

            # get the original instances that overlap with the union instance, excluding the background instance
            original_instances = np.unique(img_crop[union_mask[tuple(slice(min_coords[i], max_coords[i]) for i in range(img.ndim))]])
            original_instances = [
                i for i in original_instances if i != background_instance
            ]
            orig_ids = [
                f"{idx}_{original_instance}" for original_instance in original_instances
            ]

            prev_original_instances = np.unique(prev_img_crop[union_mask[tuple(slice(min_coords[i], max_coords[i]) for i in range(prev_img.ndim))]])
            prev_original_instances = [
                i for i in prev_original_instances if i != background_instance
            ]
            prev_orig_ids = [
                f"{idx - 1}_{prev_original_instance}"
                for prev_original_instance in prev_original_instances
            ]

            # --- calculate the IOU forward in time (IOU is symmetric, backward reuses these values)
            if prev_original_instances == []:
                iou_map["forward_in_time"][""].extend(orig_ids)
            
            # Pre-compute all instance masks to avoid repeated computations (cropped to bbox)
            prev_masks = {poi: prev_img_crop == poi for poi in prev_original_instances}
            curr_masks = {oi:    img_crop == oi for oi in original_instances}
            
            # Vectorized IOU computation: compute all pairs at once
            if prev_original_instances and original_instances:
                # Keep as bool - more memory efficient since masks are already boolean
                prev_stack = np.stack([prev_masks[poi] for poi in prev_original_instances], axis=0)
                curr_stack = np.stack([curr_masks[oi] for oi in original_instances], axis=0)
                
                # Vectorized intersection and union using broadcasting
                # Shape: (num_prev, num_curr, ...)
                intersections = np.logical_and(prev_stack[:, None, ...], curr_stack[None, :, ...])
                unions = np.logical_or(prev_stack[:, None, ...], curr_stack[None, :, ...])
                
                # Sum over spatial dimensions to get IOU values
                intersection_sums = intersections.reshape(len(prev_original_instances), len(original_instances), -1).sum(axis=2)
                union_sums = unions.reshape(len(prev_original_instances), len(original_instances), -1).sum(axis=2)
                
                # Compute IOU, handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou_matrix = intersection_sums / union_sums
                    iou_matrix = np.nan_to_num(iou_matrix, nan=0.0)
                
                # Store results in iou_map
                for p_idx, (poi, poi_id) in enumerate(zip(prev_original_instances, prev_orig_ids)):
                    if original_instances == []:
                        iou_map["forward_in_time"][poi_id] = []
                    else:
                        iou_map["forward_in_time"][poi_id] = []
                        for c_idx, (oi, oi_id) in enumerate(zip(original_instances, orig_ids)):
                            iou_map["forward_in_time"][poi_id].append((oi_id, float(iou_matrix[p_idx, c_idx])))

            # --- backward in time: reuse forward IOU values (symmetric)
            if idx == len(img_list) - 1:
                iou_map["backward_in_time"][""].extend(prev_orig_ids)
            for oi, oi_id in zip(original_instances, orig_ids):
                if prev_original_instances == []:
                    iou_map["backward_in_time"][oi_id] = []
                for poi, poi_id in zip(prev_original_instances, prev_orig_ids):
                    if oi_id not in iou_map["backward_in_time"].keys():
                        iou_map["backward_in_time"][oi_id] = []
                    # Reuse IOU value from forward calculation (IOU is symmetric)
                    forward_iou = next((iou_val for iou_val in iou_map["forward_in_time"][poi_id] if iou_val[0] == oi_id), None)
                    if forward_iou:
                        iou_map["backward_in_time"][oi_id].append((poi_id, forward_iou[1]))

            # add the matches to the Matches object
            for orig_combined_id in orig_ids:
                m.add_match(old_id=orig_combined_id, new_id=formated_union_id)

            # add the matches to the Matches object
            for prev_orig_combined_id in prev_orig_ids:
                m.add_match(old_id=prev_orig_combined_id, new_id=formated_union_id)

    # get the final matching
    logging.info("Computing final matching from connected components...")
    original_ids_to_lineage, original_ids_to_tscui, graph = get_final_matching(
        adjacencies=m.original_to_shared_id,
        global_view_mapping=False,
        background_instance=background_instance,
        iou=iou_map,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    logging.info(f"Pairwise matching complete. Lineage: {len(original_ids_to_lineage)} instances, TCUI: {len(original_ids_to_tscui)} instances.")
    lineage_mapping = _mapping_to_per_img_mapping(
        original_ids_to_lineage, num_images=len(img_list)
    )
    tcui_mapping = _mapping_to_per_img_mapping(
        original_ids_to_tscui, num_images=len(img_list)
    )

    return lineage_mapping, tcui_mapping, graph


def match_instances_by_binary_union(
    img_list,
    background_instance,
    verbose=False,
    enforce_no_merge=False,
    enforce_no_split=False,
):
    """
    Matches the images using the union over all the binarised connected-component-id images.

    Properties:
        - The instances that touch in the overlap of the images are considered the same instance.
        - The connected component in the binarised union equals the overlap of all the instances in the images that overlap across the series.

    Args:
        img_list (list[np.ndarray]): A list of images of the same size to match the instances in.
        background_instance (int): The id of the background instance.
        verbose (bool): If True, print the progress of the matching.

    Returns:
        matches (list[Matches]): A list of matches, one per image.
        labels (np.ndarray): The labels of the connected components in the binarised union.
    """
    logging.info("Matching instances by global binary union...")
    binary_union = np.zeros_like(img_list[0])

    # calculate the union of all binarised labels
    for idx, img in enumerate(img_list):
        binary_img = img != background_instance
        binary_union = np.logical_or(binary_union, binary_img)

    # run connected component analysis on the union
    labels, _ = label_instance_ids(binary_union)

    union_instances = np.unique(labels)
    # remove the background instance
    union_instances = [i for i in union_instances if i != 0]

    # save the the union view as png with pil
    union_view = np.zeros_like(img_list[0])
    for union_id in union_instances:
        union_view[labels == union_id] = union_id
    if len(union_view.shape) == 2:
        union_view = Image.fromarray(union_view.astype(np.uint8) * 255)
        union_view.save("union_view.png")
    else:
        write_to_nifti(union_view.astype(np.uint8), dtype=np.uint8, filename="union_view.nii.gz")

    m = Matches(background_id=background_instance)
    for img_idx, img in enumerate(img_list):
        # for each instance in the union, get the overlapping instances in the image
        for union_id in union_instances:
            # get the original instances that overlap with the union instance, excluding the background instance
            original_instances = np.unique(img[labels == union_id])
            original_instances = [
                i for i in original_instances if i != background_instance
            ]
            union_id = f"U_0_{union_id}"  # format the union id
            # add the matches to the Matches object
            for original_instance in original_instances:
                orig_combined_id = f"{img_idx}_{original_instance}"
                m.add_match(old_id=orig_combined_id, new_id=union_id)

    iou_map = {"forward_in_time": {}, "backward_in_time": {}}
    # newly appearing instances will be added to the iou_map with empty key
    iou_map["forward_in_time"][""] = [f"0_{i}" for i in np.unique(img_list[0])]
    # last frame instances will be added to the iou_map with empty key
    iou_map["backward_in_time"][""] = [
        f"{len(img_list) - 1}_{i}" for i in np.unique(img_list[-1])
    ]
    
    logging.info(f"Calculating IOU for {len(union_instances)} union instances across {len(img_list)} images...")

    for idx in range(1, len(img_list)):
        logging.info(f"Processing image {idx}/{len(img_list)-1} for IOU calculation...")
        prev_img = img_list[idx - 1]
        img = img_list[idx]

        for union_id in union_instances:
            # Get bounding box around the union instance to reduce memory usage
            union_mask = labels == union_id
            bbox = np.array(np.where(union_mask)).T
            if len(bbox) == 0:
                continue
            min_coords = bbox.min(axis=0)
            max_coords = bbox.max(axis=0) + 1
            
            # Crop images to bounding box (much smaller than full image)
            img_crop = img[tuple(slice(min_coords[i], max_coords[i]) for i in range(img.ndim))]
            prev_img_crop = prev_img[tuple(slice(min_coords[i], max_coords[i]) for i in range(prev_img.ndim))]

            # get the original instances that overlap with the union instance, excluding the background instance
            original_instances = np.unique(img_crop[union_mask[tuple(slice(min_coords[i], max_coords[i]) for i in range(img.ndim))]])
            original_instances = [
                i for i in original_instances if i != background_instance
            ]
            orig_ids = [
                f"{idx}_{original_instance}" for original_instance in original_instances
            ]

            prev_original_instances = np.unique(prev_img_crop[union_mask[tuple(slice(min_coords[i], max_coords[i]) for i in range(prev_img.ndim))]])
            prev_original_instances = [
                i for i in prev_original_instances if i != background_instance
            ]
            prev_orig_ids = [
                f"{idx - 1}_{prev_original_instance}"
                for prev_original_instance in prev_original_instances
            ]

            # add the matches to the Matches object
            for orig_combined_id in orig_ids:
                m.add_match(old_id=orig_combined_id, new_id=f"U_{idx}_{union_id}")

            # add the matches to the Matches object
            for prev_orig_combined_id in prev_orig_ids:
                m.add_match(old_id=prev_orig_combined_id, new_id=f"U_{idx}_{union_id}")

            # --- calculate the IOU forward in time (IOU is symmetric, backward reuses these values)
            if prev_original_instances == []:
                iou_map["forward_in_time"][""].extend(orig_ids)
            
            # Pre-compute all instance masks to avoid repeated computations (cropped to bbox)
            prev_masks = {poi: prev_img_crop == poi for poi in prev_original_instances}
            curr_masks = {oi: img_crop == oi for oi in original_instances}
            
            # Vectorized IOU computation: compute all pairs at once
            if prev_original_instances and original_instances:
                # Keep as bool - more memory efficient since masks are already boolean
                prev_stack = np.stack([prev_masks[poi] for poi in prev_original_instances], axis=0)
                curr_stack = np.stack([curr_masks[oi] for oi in original_instances], axis=0)
                
                intersections = np.logical_and(prev_stack[:, None, ...], curr_stack[None, :, ...])
                unions = np.logical_or(prev_stack[:, None, ...], curr_stack[None, :, ...])
                
                intersection = intersections.reshape(len(prev_original_instances), len(original_instances), -1).sum(axis=2)
                union = unions.reshape(len(prev_original_instances), len(original_instances), -1).sum(axis=2)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    iou_matrix = intersection / union
                    iou_matrix = np.nan_to_num(iou_matrix, nan=0.0)
                
                for p_idx, (poi, poi_id) in enumerate(zip(prev_original_instances, prev_orig_ids)):
                    if original_instances == []:
                        iou_map["forward_in_time"][poi_id] = []
                    else:
                        iou_map["forward_in_time"][poi_id] = []
                        for c_idx, (oi, oi_id) in enumerate(zip(original_instances, orig_ids)):
                            iou_map["forward_in_time"][poi_id].append((oi_id, float(iou_matrix[p_idx, c_idx])))

            # --- backward in time: reuse forward IOU values (symmetric)
            if idx == len(img_list) - 1:
                iou_map["backward_in_time"][""].extend(prev_orig_ids)
            for oi, oi_id in zip(original_instances, orig_ids):
                if prev_original_instances == []:
                    iou_map["backward_in_time"][oi_id] = []
                for poi, poi_id in zip(prev_original_instances, prev_orig_ids):
                    if oi_id not in iou_map["backward_in_time"].keys():
                        iou_map["backward_in_time"][oi_id] = []
                    # Reuse IOU value from forward calculation (IOU is symmetric)
                    forward_iou = next((iou_val for iou_val in iou_map["forward_in_time"][poi_id] if iou_val[0] == oi_id), None)
                    if forward_iou:
                        iou_map["backward_in_time"][oi_id].append((poi_id, forward_iou[1]))

    # get the final matching
    # this will create connected component analysis across time and space
    # if an instance is split due to the registration (either the connection of two parts of that instance is cut
    # by the registration or vanished due to interpolation if connection was small) it can lead to the same orignal
    # instance being matched to multiple instances in the union image
    logging.info("Computing final matching from connected components (global view)...")
    original_ids_to_lineage, original_ids_to_tcui, morphological_graph = (
        get_final_matching(
            m.original_to_shared_id,
            global_view_mapping=True,
            background_instance=background_instance,
            iou=iou_map,
            enforce_no_merge=enforce_no_merge,
            enforce_no_split=enforce_no_split,
        )
    )

    logging.info(f"Global matching complete. Lineage: {len(original_ids_to_lineage)} instances, TCUI: {len(original_ids_to_tcui)} instances.")
    lineage_mapping = _mapping_to_per_img_mapping(
        original_ids_to_lineage, num_images=len(img_list)
    )
    tcui_mapping = _mapping_to_per_img_mapping(
        original_ids_to_tcui, num_images=len(img_list)
    )

    return lineage_mapping, tcui_mapping, morphological_graph


def validate_matched_by_union_images(
    matched_images,
    union_instances,
    background_instance,
):
    # Each of the labeled instances int he matched instances should be contained fully
    # in a single instance in the union image

    for matched_img in matched_images:
        for instance in np.unique(matched_img):
            if instance == background_instance:
                continue
            union_instance = union_instances[matched_img == instance]
            unique_union_instances = np.unique(union_instance)
            if len(unique_union_instances) > 1:
                raise ValueError(
                    f"Instance {instance} in the matched image is not fully contained in a single instance in the union image."
                )
            if unique_union_instances[0] != instance:
                raise ValueError(
                    f"Instance {instance} is not matched to the same instance in the union image {unique_union_instances[0]}."
                )


def match_instances_by_intersection_of_two_images(
    image_channel_a, image_channel_b, channel_id_a, channel_id_b, background_instance
) -> CrossChannelMatches:
    """
    Matches the instances in the two images based on the binary union.
    We will return two mappings:
    - The mapping from the instances in the first image a to the instances in the second image b.
    - The mapping from the instances in the second image b to the instances in the first image a.

    Args:
        image_channel_a (np.ndarray): The first image.
        image_channel_b (np.ndarray): The second image.
        channel_id_a (int): The id of the first channel.
        channel_id_b (int): The id of the second channel.
        background_instance (int): The id of the background instance.

    Returns:
        match_a_to_b: A  with the mapping from the instances in the first image a to the instances in the second image b.
        match_b_to_a: A dictionary with the mapping from the instances in the second image b to the instances in the first image a.
    """
    logging.info(f"Matching instances by intersection: channel {channel_id_a} vs {channel_id_b}...")
    cross_channel_matches = CrossChannelMatches(
        channel_a_id=channel_id_a, channel_b_id=channel_id_b
    )

    # get the unique instances in the masked images and the original images
    instances_image_channel_a = np.unique(image_channel_a)
    instances_image_channel_b = np.unique(image_channel_b)

    # add all the instances to the mapping, initally without matches
    # now we have all the existing instances from both channels in the mapping
    for instance_a in instances_image_channel_a:
        if instance_a == background_instance:
            continue
        cross_channel_matches.add_match(
            channel_a_instance_id=instance_a, channel_b_instance_id=None
        )
    for instance_b in instances_image_channel_b:
        if instance_b == background_instance:
            continue
        cross_channel_matches.add_match(
            channel_a_instance_id=None, channel_b_instance_id=instance_b
        )

    # iterate over the instances in channel a and find the instances in channel b that overlap
    # only iterating over the image channel a and finding matches in channel b is enough
    # as all intances that do not overlap with any instance in the other channel are already in the mapping
    for instance_a in instances_image_channel_a:
        if instance_a == background_instance:
            continue
        # add the instance to the mapping
        masked_image_channel_b_by_a_instance = image_channel_b[
            image_channel_a == instance_a
        ]
        unique_instances_masked_channel_b = np.unique(
            masked_image_channel_b_by_a_instance
        )
        for instance_b in unique_instances_masked_channel_b:
            if instance_b == background_instance:
                continue
            cross_channel_matches.add_match(instance_a, instance_b)

    logging.info(f"Cross-channel matching complete. Found {len(cross_channel_matches.a_to_b)} matches from A to B.")
    return cross_channel_matches


def collect_vams_information_for_matched_ids(
    matched_images, original_images, background_instance, week_numbers=None
):
    """
    Vanish, Appear, Merge, Split (VAMS) information for matched instances.
    This function only works with the matched_by_union method.
    Over the time series the orignal instances can occupy the
    same space after the registration as other instances in the series.
    Therefore each matched instance can be the result of multiple instance in the same image as long as they overlap at
    some point with instance in the timeseries and all are mapped to the same matched instance id.

    As matched instances can be merged from mulitple unmatched instances,
    split into multiple instances or emerge at some point or vanish
    We need to keep track of the changes in the instances over the time series
    We need to know:
       when an matched instance is present in the series (we also same first and last time point where it is present explicitly)
       when the matched instance is the result of multiple instances that count as one instance per sample
          (we save it as the number of unmatched instances)
       from one sample to the next we need to know if the matched instance is the
          result of a split, merge, vanish, reappear, emerge, vanish for the rest of the series
          (we save the difference in the number of instances)

    Args:
        matched_images (list[np.ndarray]): The matched images.
        original_images (list[np.ndarray]): The original images.
        week_numbers (list[int]): The week number of the images.
    """
    instance_presence = {}
    sub_instance_count = {}
    sub_instance_original_ids = {}

    # to track if an instance consists of multiple instances we can compare
    # the instances in the matched image to the original image

    matched_orig_zip = zip(matched_images, original_images)

    for image_idx, (matched_img, orig_img) in enumerate(matched_orig_zip):
        unique_instances = np.unique(matched_img)
        for instance in unique_instances:
            if instance == background_instance:
                continue
            # track the presence of the instance
            if instance not in instance_presence.keys():
                # instance not seen yet
                instance_presence[instance] = []
                sub_instance_count[instance] = []
                sub_instance_original_ids[instance] = []

            if week_numbers is not None:
                instance_presence[instance].append(week_numbers[image_idx])
            else:
                instance_presence[instance].append(image_idx)

            # count the number of original instances that make up the matched instance
            orig_subinstances = np.unique(orig_img[matched_img == instance]).tolist()
            orig_subinstances = [
                i for i in orig_subinstances if i != background_instance
            ]
            num_orig_subinstances = len(orig_subinstances)
            sub_instance_count[instance].append(num_orig_subinstances)
            sub_instance_original_ids[instance].append(orig_subinstances)

    # Merge the information into a single dictionary
    presence_title = "present_at_week"
    if week_numbers is None:
        # we dont have the week information, so just use the index in the series
        presence_title += "_index"
    else:
        # we have the week information, so use the week number
        presence_title += "_number"

    stats_matched_instances = {}
    for instance in instance_presence.keys():
        stats_matched_instances[instance] = {
            presence_title: instance_presence[instance],
            "sub_instance_count": sub_instance_count[instance],
            "sub_instance_original_ids": sub_instance_original_ids[instance],
        }

    return stats_matched_instances
