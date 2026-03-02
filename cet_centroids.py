import numpy as np
import nibabel as nib


def get_centroids(image, label):
    """
    Get the centroids of the connected components in the image.

    Parameters
    ----------
    image : numpy.ndarray
        The labeled image uint, where each number is a label.
    label : numpy.ndarray
        label id.

    Returns
    -------
    centroids : list of tuples
        The centroids of the connected components.
    """
    # For each time setp, x,y,z,t get the centroid of the label
    centroids = []
    for t in range(image.shape[3]):
        mask = image[:, :, :, t] == label
        # calculate the centroid of the mask
        mask = np.argwhere(mask)
        if np.sum(mask) == 0:
            centroids.append((0, 0, 0))
        else:
            centroids.append(np.mean(mask, axis=0).astype(int))
    # print the results
    for i, centroid in enumerate(centroids):
        print(f"Centroid of label {label} at time {i}: {centroid}")


if __name__ == "__main__":
    while True:
        # image loading in loop as it can chaneg duriong annotation process
        image = nib.load(
        "results/6_months/hen-i218/001/tiff_label_series_0_weeks/t_3000.0_s_2_v_0_inf_reg_lineage/lineage_4d_corrected.nii.gz"
        ).get_fdata()
        print(image.shape)
        print("Enter the path to the image and label (or 'exit' to quit):")
        label = int(input())
        if label == "exit":
            break
        get_centroids(image=image, label=label)
