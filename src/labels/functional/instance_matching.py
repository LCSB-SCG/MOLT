"""
Instance Matching Module

This module provides functionality for matching instances across multiple images (time points)
using binary union of connected components. It supports two modes:
1. Global Binary Union: Uses the union over ALL time points to create a single global view
2. Pairwise Binary Union: Uses pairwise unions between consecutive time points

The module produces two types of mappings:
- Lineage Mapping: Groups all connected instances (via binary union) into the same lineage
- TCUI (Temporally Consistent Unique Instance) Mapping: Assigns unique IDs that persist
  across time, handling splits and merges

Key Concepts:
- Binary Union: The union of all binarized (non-background) instances across images
- Connected Components: Groups of instances that touch/overlap in the binary union
- Union ID: An ID representing a connected component in the binary union (format: U_{time_idx}_{union_id})
- Lineage ID: Groups all instances that are connected through the binary union
- TCUI ID: A unique ID that persists across time for tracking individual instances,
           getting new IDs when splits or merges occur
"""

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


def _get_bounding_box_crop(image: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Crop an image to the bounding box of a mask for efficient processing.

    This function computes the bounding box of all True pixels in the mask
    and returns a cropped view of the image along with the slices that
    can be used to index back into the original mask.

    Args:
        image: The image to crop (3D or 2D numpy array).
        mask: Boolean mask indicating which pixels to compute bounding box for.

    Returns:
        Tuple of (cropped_image, bounding_box_slices) where:
        - cropped_image: The image cropped to the bounding box
        - bounding_box_slices: Tuple of slices that can be used to index
          the original mask/image (e.g., mask[bbox_slices])
        Returns (None, None) if mask has no True pixels.
    """
    bbox = np.array(np.where(mask)).T
    if len(bbox) == 0:
        return None, None
    min_coords = bbox.min(axis=0)
    max_coords = bbox.max(axis=0) + 1
    crop_slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(image.ndim))
    return image[crop_slices], crop_slices


def _compute_instance_masks(image: np.ndarray, instance_ids: list, bbox_slices: tuple) -> dict:
    """
    Pre-compute boolean masks for each instance, cropped to a bounding box.

    Creates a dictionary mapping each instance ID to a boolean mask of where
    that instance appears in the cropped image region.

    Args:
        image: The full image array.
        instance_ids: List of instance IDs to create masks for.
        bbox_slices: Tuple of slices defining the bounding box region.

    Returns:
        Dictionary mapping instance_id -> boolean numpy array (True where
        that instance appears in the cropped region).
    """
    cropped = image[bbox_slices]
    return {inst_id: cropped == inst_id for inst_id in instance_ids}


def _compute_iou_matrix(prev_masks: dict, curr_masks: dict, prev_ids: list, curr_ids: list) -> np.ndarray:
    """
    Compute the Intersection over Union (IOU) matrix between two sets of instance masks.

    IOU = intersection / union
    This is computed efficiently using vectorized numpy operations by stacking
    all masks and computing pairwise intersections/unions.

    Args:
        prev_masks: Dictionary mapping instance_id to boolean mask from previous image.
        curr_masks: Dictionary mapping instance_id to boolean mask from current image.
        prev_ids: List of instance IDs from previous image (order determines row order).
        curr_ids: List of instance IDs from current image (order determines column order).

    Returns:
        2D numpy array of IOU values with shape (len(prev_ids), len(curr_ids)).
        Element [i, j] represents the IOU between prev_ids[i] and curr_ids[j].
    """
    if not prev_ids or not curr_ids:
        return np.zeros((len(prev_ids), len(curr_ids)))

    # Stack all masks as separate channels for vectorized computation
    prev_stack = np.stack([prev_masks[p] for p in prev_ids], axis=0)
    curr_stack = np.stack([curr_masks[c] for c in curr_ids], axis=0)

    # Compute pairwise intersections and unions using broadcasting
    # Shape: (len(prev_ids), len(curr_ids), *spatial_dims)
    intersections = np.logical_and(prev_stack[:, None, ...], curr_stack[None, :, ...])
    unions = np.logical_or(prev_stack[:, None, ...], curr_stack[None, :, ...])

    # Sum over spatial dimensions to get total intersection/union counts
    intersection_sums = intersections.reshape(len(prev_ids), len(curr_ids), -1).sum(axis=2)
    union_sums = unions.reshape(len(prev_ids), len(curr_ids), -1).sum(axis=2)

    # Compute IOU, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = intersection_sums / union_sums
        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0)

    return iou_matrix


def _get_non_background_instances(img, mask, background_instance):
    """
    Extract unique non-background instance IDs from an image masked region.

    Args:
        img: The label image array.
        mask: Boolean mask indicating which pixels to consider.
        background_instance: The instance ID representing background (to exclude).

    Returns:
        List of instance IDs present in the masked region, excluding background.
    """
    instances = np.unique(img[mask])
    return [i for i in instances if i != background_instance]


def _add_matches_for_instances(m, combined_ids, union_id):
    """
    Add matches from multiple original instance IDs to a single union ID.

    This is a helper that iterates over combined_ids and adds each as a match
    pointing to the same union_id. Used when multiple original instances
    are connected through a union (e.g., during merge or split events).

    Args:
        m: Matches object to add matches to.
        combined_ids: List of original combined IDs (format: "{img_idx}_{instance_id}").
        union_id: The union ID to map these instances to.
    """
    for combined_id in combined_ids:
        m.add_match(old_id=combined_id, new_id=union_id)


def _populate_iou_map_wrapper(iou_map, prev_orig_ids, orig_ids, iou_matrix, idx, num_images):
    """
    Populate the IOU map for forward time connections, handling edge cases.

    This wrapper handles:
    - New instances appearing (prev_orig_ids is empty)
    - Instances disappearing (last frame: extend forward map with prev_orig_ids)
    - Normal forward IOU population

    Args:
        iou_map: Dictionary to store IOU values (modified in place).
        prev_orig_ids: List of instance IDs from previous time point.
        orig_ids: List of instance IDs from current time point.
        iou_matrix: Precomputed IOU matrix between prev and curr instances.
        idx: Current time point index.
        num_images: Total number of images in the sequence.
    """
    # Handle newly appearing instances (no previous frame match)
    if not prev_orig_ids:
        iou_map["forward_in_time"][""].extend(orig_ids)

    # Populate forward IOU mappings
    _populate_iou_map_forward(iou_map, prev_orig_ids, orig_ids, iou_matrix)

    # Handle disappearing instances (last frame)
    # Extend forward map with instances that will disappear
    if idx == num_images - 1:
        iou_map["forward_in_time"][""].extend(prev_orig_ids)


def _create_iou_map(img_list, background_instance):
    """
    Initialize the IOU map structure for tracking instance overlaps across time.

    The IOU map stores pairwise IOU values between instances across consecutive
    time points. It's used by the morphological graph building to handle
    merge and split events.

    Args:
        img_list: List of images in the time series.
        background_instance: The ID representing background (to exclude).

    Returns:
        Dictionary with structure:
        {
            "forward_in_time": {
                "": [list of instance IDs appearing in first frame],
                "img_idx_instance_id": [(match_id, iou_value), ...],
                ...
            }
        }
    """
    iou_map = {"forward_in_time": {}}
    # Initialize with instances from the first frame
    iou_map["forward_in_time"][""] = [f"0_{i}" for i in np.unique(img_list[0])]
    return iou_map


def _populate_iou_map_forward(
    iou_map: dict,
    prev_orig_ids: list,
    orig_ids: list,
    iou_matrix: np.ndarray,
) -> None:
    """
    Populate the forward-in-time IOU map from a computed IOU matrix.

    Stores IOU values for each instance in the previous frame, indicating
    how much it overlaps with each instance in the current frame.

    Args:
        iou_map: Dictionary to store IOU values (modified in place).
        prev_orig_ids: List of instance IDs from previous time point.
        orig_ids: List of instance IDs from current time point.
        iou_matrix: 2D array of IOU values.
    """
    for p_idx, poi_id in enumerate(prev_orig_ids):
        if not orig_ids:
            iou_map["forward_in_time"][poi_id] = []
        else:
            iou_map["forward_in_time"][poi_id] = [
                (oi_id, float(iou_matrix[p_idx, c_idx]))
                for c_idx, oi_id in enumerate(orig_ids)
            ]


def _mapping_to_per_img_mapping(
    original_to_shared_id: dict,
    num_images: int,
) -> list[dict]:
    """
    Convert a global mapping from original IDs to shared IDs into per-image mappings.

    The original mappings use combined IDs (format: "{img_idx}_{original_instance_id}").
    This function converts them into a list of dictionaries, one per image,
    where keys are original instance IDs and values are the shared/lineage/TCUI IDs.

    Example:
        Input: {"0_1": "lineage_0", "1_1": "lineage_0", "2_2": "lineage_1"}
        Output: [
            {1: "lineage_0"},    # frame 0: instance 1 -> lineage_0
            {1: "lineage_0"},    # frame 1: instance 1 -> lineage_0
            {2: "lineage_1"}     # frame 2: instance 2 -> lineage_1
        ]

    Args:
        original_to_shared_id: Dictionary mapping combined IDs to shared IDs.
        num_images: Number of images in the time series.

    Returns:
        List of dictionaries, one per image, mapping original instance IDs
        to their shared/lineage/TCUI IDs.
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
    Exclude instances from a label image that are outside a given mask.

    Any instance that has voxels outside the mask will be set to background.
    This is useful for restricting analysis to a specific region.

    Args:
        instance_image: The label image (3D or 2D numpy array).
        mask: Binary mask defining the region to keep.
        background_instance: The ID representing background.

    Returns:
        Tuple of (modified_image, set_of_excluded_instances).
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
    Find connected components in the adjacency graph and build morphological graph.

    This function:
    1. Builds an undirected graph from the adjacency mappings
    2. Converts to a directed morphological graph using either global or pairwise mapping
    3. Returns connected components and the morphological graph

    Args:
        adjacencies: Dictionary mapping original IDs to their union IDs.
        global_view_mapping: If True, use global binary union mapping.
                           If False, use pairwise binary union mapping.
        iou: Optional IOU map for handling merge/split in morphological graph.
        enforce_no_merge: If True, prevent merges in the graph.
        enforce_no_split: If True, prevent splits in the graph.

    Returns:
        Tuple of (connected_components, morphological_graph).
    """
    # Build undirected graph from adjacencies
    # Each edge connects an original ID to a union ID
    G = nx.Graph()
    for key_id, value_ids in adjacencies.items():
        for v_id in value_ids:
            G.add_edge(key_id, v_id)

    # Build directed morphological graph based on mode
    if not global_view_mapping:
        # Pairwise: Connect only consecutive time points
        G = get_morphological_graph_frame_pair_mapping(
            G,
            iou=iou,
            enforce_no_merge=enforce_no_merge,
            enforce_no_split=enforce_no_split,
        )
    else:
        # Global: Connect all time points through union
        G = get_morphological_graph_global_view_mapping(
            G,
            iou=iou,
            enforce_no_merge=enforce_no_merge,
            enforce_no_split=enforce_no_split,
        )

    # Get connected components in the undirected version
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
    Compute the final lineage and TCUI mappings from instance adjacencies.

    This is the core function that produces the final mappings:
    1. Finds connected components (lineage) - all instances connected through
       the binary union get the same lineage ID
    2. Assigns TCUI (Temporally Consistent Unique Instance) IDs by traversing
       the morphological graph and assigning unique IDs to each branch

    Args:
        adjacencies: Dictionary mapping original IDs to union IDs.
        global_view_mapping: If True, use global binary union approach.
        background_instance: The ID representing background.
        iou: Optional IOU map for merge/split handling.
        enforce_no_merge: If True, prevent merges.
        enforce_no_split: If True, prevent splits.

    Returns:
        Tuple of (original_to_lineage_id, original_to_tcui_id, morphological_graph).
    """
    # Find connected components and build morphological graph
    cc, G = find_connected_components_in_adjacencies(
        adjacencies=adjacencies,
        global_view_mapping=global_view_mapping,
        iou=iou,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    # === LINEAGE MAPPING ===
    # Each connected component gets one lineage ID
    # All original instances in the component map to the same lineage ID
    connected_components = list(cc)
    original_to_lineage_id = {}
    for c_idx, component in enumerate(connected_components):
        # Assign new lineage ID for this component
        if background_instance == c_idx:
            # Don't assign lineage ID to background
            new_idx = len(list(connected_components))
        else:
            new_idx = c_idx
        # Map all instances in this component to the same lineage ID
        for instance_id in component:
            if type(instance_id) is not str:
                instance_id = int(instance_id)
            original_to_lineage_id[instance_id] = new_idx

    # === TCUI MAPPING ===
    # Assign temporally consistent unique instance IDs by traversing the graph
    # Each branch gets a unique ID; splits and merges create new IDs
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
    Apply a mapping to transform instance IDs in an image.

    This function remaps instance IDs in an image according to a mapping dictionary.
    Any instances not in the mapping are assigned the ignore_label.

    Args:
        matches: Dictionary mapping original instance IDs to new IDs.
                 Keys are original IDs, values are new IDs (lineage or TCUI).
        image: The label image to transform.
        ignore_label: Value to assign to instances not in the mapping.
        background_instance: The ID representing background (kept unchanged).
        verbose: If True, print logging information about unmatched instances.

    Returns:
        New image array with instance IDs remapped according to the mapping.
    """
    # Find instances that are not in the mapping
    unique_instances = set(np.unique(image))
    matched_instances = set(matches.keys())
    unmatched_instances = unique_instances - matched_instances
    
    if verbose:
        logging.info("Uniques before applying match: " + str(np.unique(image)))
        logging.info("Unmatched instances: " + str(unmatched_instances))

    # Assign ignore_label to unmatched instances (except background)
    for unmatched_instance in unmatched_instances:
        if unmatched_instance == background_instance:
            continue
        matches[unmatched_instance] = ignore_label

    # Apply the mapping: create new image with remapped IDs
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
    """
    Match instances using pairwise binary union between consecutive time points.

    This mode computes the binary union for each pair of consecutive images,
    connecting instances that overlap in adjacent frames. This creates a
    chain of connections through time.

    Algorithm:
    1. For each pair of consecutive images (t-1, t):
       - Compute binary union of both images
       - Find connected components in the union
       - For each union component, record which instances overlap
       - Compute IOU between overlapping instances
    2. Build morphological graph connecting instances across time
    3. Compute lineage and TCUI mappings from the graph

    Args:
        img_list: List of registered/aligned label images (same size).
        unreg_img_list: List of unregistered original label images.
        background_instance: The ID representing background.
        verbose: If True, print progress information.
        enforce_no_merge: If True, prevent merges in final mapping.
        enforce_no_split: If True, prevent splits in final mapping.

    Returns:
        Tuple of (lineage_mapping, tcui_mapping, morphological_graph).
        - lineage_mapping: List of dicts, one per image, mapping original IDs to lineage IDs
        - tcui_mapping: List of dicts, one per image, mapping original IDs to TCUI IDs
        - morphological_graph: NetworkX directed graph of instance relationships
    """
    logging.info(f"Starting pairwise matching for {len(img_list)} images...")
    m = Matches(background_id=background_instance)

    # Initialize IOU map for tracking overlaps across time
    iou_map = _create_iou_map(img_list, background_instance)
    
    # Process each pair of consecutive images
    for idx in range(1, len(img_list)):
        # Get registered image for current frame and unregistered for previous
        # (unregistered images may have different sizes due to padding)
        prev_img = unreg_img_list[idx - 1]
        img = img_list[idx]

        # Pad previous image to match current image size (for alignment differences)
        padding = [
            (
                (img.shape[i] - prev_img.shape[i]) // 2,
                (img.shape[i] - prev_img.shape[i]) // 2,
            )
            for i in range(len(img.shape))
        ]
        prev_img = np.pad(prev_img, padding, mode="constant", constant_values=0)

        # Compute binary union of current and previous images
        prev_img_bin = prev_img != background_instance
        img_bin = img != background_instance
        binary_union = np.logical_or(prev_img_bin, img_bin)

        # Find connected components in the binary union
        labels, _ = label_instance_ids(binary_union)

        # Process each union component
        for union_id in np.unique(labels):
            if union_id == 0:
                # Skip background
                continue

            # Create unique ID for this union instance at this time point
            formated_union_id = f"U_{idx}_{union_id}"
            
            # Get bounding box around the union component for efficiency
            union_mask = labels == union_id
            img_crop, crop_slices = _get_bounding_box_crop(img, union_mask)
            prev_img_crop, _ = _get_bounding_box_crop(prev_img, union_mask)

            if img_crop is None or prev_img_crop is None:
                logging.warning(
                    f"Union id {union_id} has no overlapping region in image {idx}. "
                    f"Skipping IOU calculation for this union instance."
                )
                continue

            # Find which original instances overlap with this union component
            original_instances = _get_non_background_instances(
                img_crop, union_mask[crop_slices], background_instance
            )
            orig_ids = [f"{idx}_{i}" for i in original_instances]

            prev_original_instances = _get_non_background_instances(
                prev_img_crop, union_mask[crop_slices], background_instance
            )
            prev_orig_ids = [f"{idx - 1}_{i}" for i in prev_original_instances]

            # Compute IOU between overlapping instances
            prev_masks = _compute_instance_masks(prev_img_crop, prev_original_instances, crop_slices)
            curr_masks = _compute_instance_masks(img_crop, original_instances, crop_slices)
            iou_matrix = _compute_iou_matrix(
                prev_masks, curr_masks, prev_original_instances, original_instances
            )
            
            # Store IOU values for morphological graph building
            _populate_iou_map_wrapper(
                iou_map, prev_orig_ids, orig_ids, iou_matrix, idx, len(img_list)
            )

            # Record matches: connect both current and previous instances to this union
            _add_matches_for_instances(m, orig_ids, formated_union_id)
            _add_matches_for_instances(m, prev_orig_ids, formated_union_id)

    # Compute final lineage and TCUI mappings from the adjacency graph
    logging.info("Computing final matching from connected components...")
    original_ids_to_lineage, original_ids_to_tscui, graph = get_final_matching(
        adjacencies=m.original_to_shared_id,
        global_view_mapping=False,
        background_instance=background_instance,
        iou=iou_map,
        enforce_no_merge=enforce_no_merge,
        enforce_no_split=enforce_no_split,
    )

    logging.info(
        f"Pairwise matching complete. "
        f"Lineage: {len(original_ids_to_lineage)} instances, "
        f"TCUI: {len(original_ids_to_tscui)} instances."
    )
    
    # Convert global mappings to per-image mappings
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
    Match instances using global binary union across ALL time points.

    This mode computes a single binary union over all images in the series,
    then finds connected components in that global union. All instances
    that touch at any point in the series are considered connected.

    Algorithm:
    1. Compute binary union of ALL images (not pairwise)
    2. Find connected components in the global union
    3. For each image, map its instances to the union components they overlap
    4. For consecutive image pairs, compute IOU for additional tracking info
    5. Build morphological graph and compute lineage/TCUI mappings

    Properties:
    - Instances that touch in ANY overlap across the series are connected
    - The connected component in the global binary union equals the overlap
      of all instances that should be in the same lineage

    Args:
        img_list: List of label images (same size, 3D or 2D).
        background_instance: The ID representing background.
        verbose: If True, print progress information.
        enforce_no_merge: If True, prevent merges in final mapping.
        enforce_no_split: If True, prevent splits in final mapping.

    Returns:
        Tuple of (lineage_mapping, tcui_mapping, morphological_graph).
    """
    logging.info("Matching instances by global binary union...")
    
    # === STEP 1: Compute global binary union ===
    # Start with zeros, then OR in each non-background image
    binary_union = np.zeros_like(img_list[0])
    for idx, img in enumerate(img_list):
        binary_img = img != background_instance
        binary_union = np.logical_or(binary_union, binary_img)

    # === STEP 2: Find connected components in global union ===
    labels, _ = label_instance_ids(binary_union)

    union_instances = np.unique(labels)
    # Remove background (typically 0)
    union_instances = [i for i in union_instances if i != 0]

    # Save union view for debugging/visualization
    union_view = np.zeros_like(img_list[0])
    for union_id in union_instances:
        union_view[labels == union_id] = union_id
    if len(union_view.shape) == 2:
        union_view = Image.fromarray(union_view.astype(np.uint8) * 255)
        union_view.save("union_view.png")
    else:
        write_to_nifti(
            union_view.astype(np.uint8), dtype=np.uint8, filename="union_view.nii.gz"
        )

    # === STEP 3: Map instances to union components ===
    # For each image and each union component, find which instances overlap
    m = Matches(background_id=background_instance)
    for img_idx, img in enumerate(img_list):
        for union_id in union_instances:
            original_instances = _get_non_background_instances(
                img, labels == union_id, background_instance
            )
            orig_ids = [f"{img_idx}_{i}" for i in original_instances]
            # Format: U_0_{union_id} - union at initial time point
            union_id = f"U_0_{union_id}"
            _add_matches_for_instances(m, orig_ids, union_id)

    # === STEP 4: Compute IOU for consecutive pairs ===
    iou_map = _create_iou_map(img_list, background_instance)

    logging.info(
        f"Calculating IOU for {len(union_instances)} union instances "
        f"across {len(img_list)} images..."
    )

    for idx in range(1, len(img_list)):
        logging.info(
            f"Processing image {idx}/{len(img_list)-1} for IOU calculation..."
        )
        prev_img = img_list[idx - 1]
        img = img_list[idx]

        # Process each union component
        for union_id in union_instances:
            union_mask = labels == union_id
            img_crop, crop_slices = _get_bounding_box_crop(img, union_mask)
            prev_img_crop, _ = _get_bounding_box_crop(prev_img, union_mask)

            if img_crop is None or prev_img_crop is None:
                continue

            # Find instances overlapping this union component in both frames
            original_instances = _get_non_background_instances(
                img_crop, union_mask[crop_slices], background_instance
            )
            orig_ids = [f"{idx}_{i}" for i in original_instances]

            prev_original_instances = _get_non_background_instances(
                prev_img_crop, union_mask[crop_slices], background_instance
            )
            prev_orig_ids = [f"{idx - 1}_{i}" for i in prev_original_instances]

            # Record matches to union
            _add_matches_for_instances(m, orig_ids, f"U_{idx}_{union_id}")
            _add_matches_for_instances(m, prev_orig_ids, f"U_{idx}_{union_id}")

            # Compute IOU between overlapping instances
            prev_masks = _compute_instance_masks(
                prev_img_crop, prev_original_instances, crop_slices
            )
            curr_masks = _compute_instance_masks(
                img_crop, original_instances, crop_slices
            )
            iou_matrix = _compute_iou_matrix(
                prev_masks, curr_masks, prev_original_instances, original_instances
            )
            _populate_iou_map_wrapper(
                iou_map, prev_orig_ids, orig_ids, iou_matrix, idx, len(img_list)
            )

    # === STEP 5: Compute final lineage and TCUI mappings ===
    # This identifies connected components across all time points
    # and assigns unique tracking IDs handling splits and merges
    logging.info(
        "Computing final matching from connected components (global view)..."
    )
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

    logging.info(
        f"Global matching complete. "
        f"Lineage: {len(original_ids_to_lineage)} instances, "
        f"TCUI: {len(original_ids_to_tcui)} instances."
    )
    
    # Convert to per-image mappings
    lineage_mapping = _mapping_to_per_img_mapping(
        original_ids_to_lineage, num_images=len(img_list)
    )
    tcui_mapping = _mapping_to_per_img_mapping(
        original_ids_to_tcui, num_images=len(img_list)
    )

    # Validate that all instances are properly mapped
    _validate_all_instances_mapped(
        img_list=img_list,
        lineage_mapping=lineage_mapping,
        tcui_mapping=tcui_mapping,
        background_instance=background_instance,
    )

    return lineage_mapping, tcui_mapping, morphological_graph


def _validate_all_instances_mapped(
    img_list,
    lineage_mapping,
    tcui_mapping,
    background_instance,
):
    """
    Validate that every original instance is mapped to valid lineage and TCUI IDs.

    This validation ensures that:
    1. All original instances appear in the lineage mapping
    2. No instance maps to the background ID (invalid)
    3. All original instances appear in the TCUI mapping

    Args:
        img_list: List of original images.
        lineage_mapping: Mapping from original to lineage IDs.
        tcui_mapping: Mapping from original to TCUI IDs.
        background_instance: The ID representing background.

    Raises:
        ValueError: If any validation fails.
    """
    unmapped_lineage = []
    unmapped_tcui = []

    for img_idx, img in enumerate(img_list):
        original_instances = np.unique(img)
        original_instances = [
            i for i in original_instances if i != background_instance
        ]

        for instance_id in original_instances:
            # Check lineage mapping
            if instance_id not in lineage_mapping[img_idx]:
                unmapped_lineage.append(f"{img_idx}_{instance_id}")
            elif lineage_mapping[img_idx][instance_id] == background_instance:
                unmapped_lineage.append(f"{img_idx}_{instance_id}")

            # Check TCUI mapping
            if instance_id not in tcui_mapping[img_idx]:
                unmapped_tcui.append(f"{img_idx}_{instance_id}")

    if unmapped_lineage:
        raise ValueError(
            f"The following {len(unmapped_lineage)} instances are not mapped "
            f"to a valid lineage ID: "
            f"{unmapped_lineage[:10]}{'...' if len(unmapped_lineage) > 10 else ''}"
        )

    if unmapped_tcui:
        raise ValueError(
            f"The following {len(unmapped_tcui)} instances are not mapped "
            f"to a valid tracked (TCUI) ID: "
            f"{unmapped_tcui[:10]}{'...' if len(unmapped_tcui) > 10 else ''}"
        )


def validate_matched_by_union_images(
    matched_images,
    union_instances,
    background_instance,
):
    """
    Validate that matched instances are fully contained within union instances.

    Each labeled instance in a matched image should be completely contained within
    a single instance in the union image. This ensures the matching is valid.

    Args:
        matched_images: List of images with matched/renumbered instances.
        union_instances: The union label image.
        background_instance: The ID representing background.

    Raises:
        ValueError: If any instance spans multiple union instances.
    """
    for matched_img in matched_images:
        for instance in np.unique(matched_img):
            if instance == background_instance:
                continue
            union_instance = union_instances[matched_img == instance]
            unique_union_instances = np.unique(union_instance)
            if len(unique_union_instances) > 1:
                raise ValueError(
                    f"Instance {instance} in the matched image is not fully "
                    f"contained in a single instance in the union image."
                )
            if unique_union_instances[0] != instance:
                raise ValueError(
                    f"Instance {instance} is not matched to the same instance "
                    f"in the union image {unique_union_instances[0]}."
                )


def match_instances_by_intersection_of_two_images(
    image_channel_a, image_channel_b, channel_id_a, channel_id_b, background_instance
) -> CrossChannelMatches:
    """
    Match instances between two channels based on intersection/overlap.

    This is used for cross-channel matching (e.g., matching nuclei to cytoplasm).
    Instances that overlap between the two channels are considered matches.

    Args:
        image_channel_a: Label image from channel A.
        image_channel_b: Label image from channel B.
        channel_id_a: Identifier for channel A.
        channel_id_b: Identifier for channel B.
        background_instance: The ID representing background.

    Returns:
        CrossChannelMatches object containing bidirectional mappings.
    """
    logging.info(
        f"Matching instances by intersection: channel {channel_id_a} vs {channel_id_b}..."
    )
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
