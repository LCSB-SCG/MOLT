from scipy import ndimage

from src.utils.functional.masks import get_exclude_border_mask
import numpy as np


def create_binary_label(
    image: np.ndarray,
    smoothing_sigma: int = 3,
    threshold: float = None,
    keep_top_quantile: float = None,
) -> np.ndarray:
    """
    Creates binary labels from an image.

    Args:
        image (np.ndarray): The image to be labeled.
        keep_top_quantile (float): The top quantile of the image to be used for the gaussian fit.
        smoothing_sigma (float): The sigma of the gaussian filter to be used for smoothing the image.
            Set to 0 for no smoothing.
    """
    if threshold is not None and keep_top_quantile is not None:
        raise ValueError("Only one of threshold and keep_top_quantile can be used")
    elif threshold is None and keep_top_quantile is None:
        raise ValueError("Either threshold or keep_top_quantile must be used")

    if smoothing_sigma is None:
        pass
    elif smoothing_sigma > 0:
        # Smooth the image
        image = ndimage.gaussian_filter(input=image, sigma=smoothing_sigma)
    elif smoothing_sigma < 0:
        raise ValueError("The smoothing sigma must be positive")

    # Get the threshold value
    if threshold is not None:
        filter_value = threshold
    else:
        filter_value = np.quantile(image, 1 - keep_top_quantile)

    # Apply the threshold
    image[image < filter_value] = 0
    image[image >= filter_value] = 1

    image = image.astype(np.uint8)

    return image


def collect_meta_data(instances: np.ndarray) -> dict[int, dict[str, int]]:
    """
    Collects meta data about the instances.

    Args:
        instances (np.ndarray): The instances image to be used for meta data collection.

    Returns:
        dict[int: dict[str: int]]: The meta data of the instances.
    """
    label = instances > 0

    # calculate the volume as the number of voxels
    ids, volume_in_voxels = np.unique(instances, return_counts=True)

    # exclude background
    ids = ids[1:]
    volume_in_voxels = volume_in_voxels[1:]

    meta_data = {}

    # identify the center of mass of each instance
    coms = ndimage.center_of_mass(label, instances, ids)

    # get border mask with 0s in the center and 1s at the border
    border_mask = 1 - get_exclude_border_mask(label.shape)

    # identify the statistics of each instance
    for l_idx, inst_id in enumerate(ids):
        inst_id = int(inst_id)
        meta_data[inst_id] = {}
        # calculate the center of mass
        meta_data[inst_id]["center_of_mass"] = [
            coordinate for coordinate in coms[l_idx]
        ]
        # calculate the volume as the number of voxels
        meta_data[inst_id]["volume_in_voxels"] = int(volume_in_voxels[l_idx])
        # calculate if the instance touches the border
        meta_data[inst_id]["touches_border"] = str(
            np.any(instances[border_mask == 1] == inst_id)
        )
    return meta_data


def indentify_instances_and_meta(
    bin_label: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    """
    Identifies the instances in the label.

    Args:
        bin_label (np.ndarray): The binary label to be used for instance identification.
    """
    labels, _ = label_instance_ids(bin_label)

    meta_data = collect_meta_data(labels)

    return labels, meta_data


def threshold_volume_filter(
    instances: np.ndarray,
    meta_data: dict[int, dict[str, int]],
    background_instance: int,
    min_volume_threshold=0,
    max_volume_threshold=np.inf,
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    """
    Filters the instances by volume.

    Args:
        instances (np.ndarray): The instances to be filtered.
        meta_data (dict): The meta data of the instances.
        volume_threshold (int): The volume threshold to be used for filtering in voxels.
    """
    if min_volume_threshold == "inf":
        raise ValueError("The min_volume_threshold must be a number")
    elif min_volume_threshold == "-inf":
        min_volume_threshold = -np.inf
    if max_volume_threshold == "inf":
        max_volume_threshold = np.inf
    elif max_volume_threshold == "-inf":
        raise ValueError("The max_volume_threshold must be a number")

    for i in np.unique(instances):
        i = int(i)
        if i == background_instance:
            continue
        if (
            meta_data[i]["volume_in_voxels"] < min_volume_threshold
            or meta_data[i]["volume_in_voxels"] > max_volume_threshold
        ):
            instances[instances == i] = background_instance
            meta_data.pop(i)
    return instances, meta_data


def label_instance_ids(image: np.ndarray) -> np.ndarray:
    """
    Labels the instances in the image.

    Args:
        image (np.ndarray): The image to be labeled.
    """
    rank_and_connectivity = len(image.shape)
    full_neighborhood = ndimage.generate_binary_structure(
        rank=rank_and_connectivity, connectivity=rank_and_connectivity
    )
    labels, num_instances = ndimage.label(image, structure=full_neighborhood)
    return labels, num_instances
