import numpy as np
import nibabel as nib


def get_header():
    header = nib.Nifti1Header()
    header["sizeof_hdr"] = 348
    header["data_type"] = b"          "
    header["db_name"] = b"                  "
    header["extents"] = 16384
    header["session_error"] = 0
    header["regular"] = b"r"
    header["dim_info"] = 32
    header["dim"] = np.array([3, 512, 512, 101, 1, 1, 1, 1])
    header["intent_p1"] = 0.0
    header["intent_p2"] = 0.0
    header["intent_p3"] = 0.0
    header["intent_code"] = 0
    header["datatype"] = 512  # 16-bit unsigned integer
    header["bitpix"] = 16
    header["slice_start"] = 0
    header["pixdim"] = np.array([1, np.nan, np.nan, np.nan, 1, 1, 1, 1])
    header["vox_offset"] = 0.0
    header["scl_slope"] = np.nan
    header["scl_inter"] = np.nan
    header["slice_end"] = 0
    header["slice_code"] = 0
    header["xyzt_units"] = 0
    header["cal_max"] = 0.0
    header["cal_min"] = 0.0
    header["slice_duration"] = 0.0
    header["toffset"] = 0.0
    header["glmax"] = 0
    header["glmin"] = 0
    header["descrip"] = (
        b"                                                                                "
    )
    header["aux_file"] = b"                        "
    header["qform_code"] = 0
    header["sform_code"] = 0
    header["quatern_b"] = 0.0
    header["quatern_c"] = 0.0
    header["quatern_d"] = 0.0
    header["qoffset_x"] = 0.0
    header["qoffset_y"] = 0.0
    header["qoffset_z"] = 0.0
    header["srow_x"] = np.array([0.0, 0.0, 0.0, 0.0])
    header["srow_y"] = np.array([0.0, 0.0, 0.0, 0.0])
    header["srow_z"] = np.array([0.0, 0.0, 0.0, 0.0])
    header["intent_name"] = b"                "
    header["magic"] = b"n+1"

    return header


def convert_to_nifti(nifti_data, dtype):
    """
    Convert a 3D numpy array to a Nifti1Image object with a correct header.

    Args:
    nifti_data (np.ndarray): 3D numpy array to convert to Nifti1Image.
    dtype (np.dtype): Data type of the input array.
    """
    header = get_header()

    nifti_image = nib.Nifti1Image(
        nifti_data, affine=np.eye(4), dtype=dtype, header=header
    )

    nifti_image.header.set_sform(np.zeros_like(np.eye(4)), code="unknown")

    return nifti_image


def write_to_nifti(nifti_data, dtype, filename):
    """
    Convert a 3D numpy array to a Nifti1Image object with a correct header and save it to a file.

    Args:
    nifti_data (np.ndarray): 3D numpy array to convert to Nifti1Image.
    dtype (np.dtype): Data type of the input array.
    filename (str): Path to the file to save the Nifti1Image to.
    """
    nifti_image = convert_to_nifti(nifti_data, dtype)
    nib.save(nifti_image, filename)
