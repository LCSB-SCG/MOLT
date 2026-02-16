import logging
import numpy as np
from skimage import exposure


def array_to_range_0_255(array: np.ndarray):
    """
    Converts an array to the range 0-255. (e.g. image to RGB image)

    Args:
        array (np.ndarray): The array to convert to the range 0-255.

    Returns:
        np.ndarray: The array in the range 0-255.
    """
    minimum = np.min(array)
    maximum = np.max(array)
    if maximum - minimum == 0:
        return np.zeros_like(array).astype(np.uint8)
    array = array - minimum
    array = array / (maximum - minimum)
    array = array * 255
    return array.astype(np.uint8)


def convert_channels_to_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray):
    """
    Converts a list of channels to a single RGB image.

    Args:
        channels (list[np.ndarray]): The channels to convert to a single RGB image.

    Returns:
        np.ndarray: The RGB image.
    """
    r = array_to_range_0_255(red)
    g = array_to_range_0_255(green)
    b = array_to_range_0_255(blue)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def auto_fit_contrast(
    image, low_quantile=0.005, high_quantile=0.995, data_range=5000, round=True
):
    """
    Automatically adjusts the contrast of an image using quantile-based rescaling.
    Args:
        image (np.ndarray): The input image to be processed.
        low_quantile (float): The lower quantile for contrast stretching (default: 0.005).
        high_quantile (float): The upper quantile for contrast stretching (default: 0.995).
        rescale_factor (int): The factor to rescale the image back to its original range (default: 5000).
    Returns:
        np.ndarray: The contrast-adjusted image.
    """
    # Convert the image to float [0, 1]
    max_val = np.max(image)
    min_val = np.min(image)
    if max_val - min_val == 0:
        return np.zeros_like(image)
    image_float = (image - min_val) / (max_val - min_val)

    # Calculate the lower and upper quantiles
    low = np.quantile(image_float, low_quantile)
    high = np.quantile(image_float, high_quantile)

    # Adjust the contrast by stretching the intensity values
    image_rescaled = exposure.rescale_intensity(image_float, in_range=(low, high))

    # Convert the image back to the original data type
    if data_range is None:
        image_rescaled = image_rescaled * max_val
    else:
        # Rescale the image to specified range
        image_rescaled = image_rescaled * data_range
    if round:
        image_rescaled = np.round(image_rescaled).astype(np.uint16)

    return image_rescaled


def overall_intensity_matching(
    ref_img: np.ndarray,
    img: np.ndarray,
    data_range=5000,
    verbose=False,
    upper_bound=0.98,
):
    # Sort the values and take lowest 30%
    ref_max = np.max(ref_img)
    ref_min = np.min(ref_img)
    img_max = np.max(img)
    img_min = np.min(img)

    ref_img = (ref_img - ref_min) / (ref_max - ref_min)
    img = (img - img_min) / (img_max - img_min)

    ref_values = np.sort(ref_img.flatten())[
        int(0.0 * ref_img.size) : int(upper_bound * ref_img.size)
    ]
    img_values = np.sort(img.flatten())[
        int(0.0 * ref_img.size) : int(upper_bound * img.size)
    ]

    mean_ref = np.mean(ref_values)
    mean_img = np.mean(img_values)

    # Calculate scaling factors and clip the values
    scaling_factor = mean_ref / mean_img
    new_img = img * scaling_factor
    new_mean = np.mean(
        np.sort(new_img.flatten())[
            int(0.0 * ref_img.size) : int(upper_bound * img.size)
        ]
    )
    # Clip the data
    new_img = np.clip(new_img, 0, 1)
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * data_range

    after_scaling = np.mean(
        np.sort(new_img.flatten())[
            int(0.0 * ref_img.size) : int(upper_bound * new_img.size)
        ]
    )
    if verbose:
        logging.info(
            f"Ref Mean: {mean_ref}, Mean before: {mean_img}, Mean after: {new_mean}, \
Rescaling: {scaling_factor}, After Rescaling: {after_scaling}"
        )

    return new_img, scaling_factor
