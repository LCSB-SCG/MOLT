import os
from src.utils.io.nifti.write_to_nifti import write_to_nifti
import nibabel as nib
import numpy as np


def convert_to_integers(nifti_path):
    """
    Convert a 3D numpy array to a Nifti1Image object with a correct header.

    Args:
    nifti_data (np.ndarray): 3D numpy array to convert to Nifti1Image.
    dtype (np.dtype): Data type of the input array.
    """

    nifti_image = nib.load(nifti_path).get_fdata().astype(np.int8)

    filename = os.path.join(
        os.path.dirname(nifti_path), "blender_int8_" + nifti_path.split(os.sep)[-1][11:]
    )

    write_to_nifti(
        nifti_data=nifti_image,
        filename=filename,
        dtype=np.int8,
    )
