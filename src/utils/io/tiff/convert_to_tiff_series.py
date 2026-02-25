import tifffile
from src.general.Cohort import Cohort
import os
import nibabel as nib
import numpy as np
from src.utils.cell_tracking_challenge.ctc_format import write_to_tiff
from src.utils.io.nifti.write_to_nifti import write_to_nifti
def write_4d_ome_tiff(tiff_data, filename):
    """
    Save a 4D numpy array to an OME-TIFF file.

    Args:
        tiff_data (np.ndarray): 4D numpy array to save as OME-TIFF.
        filename (str): Path to the file to save the OME-TIFF to.
    """
    # Convert to uint16 for tiff
    tiff_data = tiff_data.astype(np.uint16)
    tiff_data = np.transpose(tiff_data, (3,2,1,0)) # Transpose to (t, z, y, x) order for OME-TIFF
    tifffile.imwrite(filename, tiff_data, metadata={'axes': 'TZYX'})


def create_tiff_sequence(
    cohort: Cohort,
    results_folder: str,
    label: bool = False,
    label_settings: dict = None,
    channel: str = "channel_1",
    file_type: str = "instances",
    save_as_4d: bool = False,
    use_registered: bool = False,
    registration_to: str = "",
):
    """
    Creates a series of tiff images in the results folder to be used with the
    Cell Tracking Challenge format.
    
    For each subject and ROI, the nifti files in the channel_1 directories
    are converted to 3D tiff files and saved as rec_xxx.tif.
    
    Args:
        cohort (Cohort): Cohort object containing subject and ROI information.
        results_folder (str): Path to the results folder where TIFF files will be saved.
        label (bool): If True, include label information in the TIFF conversion process.
        label_settings (dict): Settings for label processing if label is True.
        file_type (str): The type of file to retrieve for conversion (e.g., "instances"). Defaults to "instances".
        channel (str): The channel to use for NIfTI file retrieval. Defaults to "channel_1".
        save_as_4d (bool): If True, save all weeks as a single 4D TIFF image. Defaults to False.
        use_registered (bool): If True, use registered NIfTI files instead of raw NIfTI files. Defaults to False.
        registration_to (str): If use_registered is True, specify the target for registration 
            (e.g., "0_weeks"). Defaults to "".
    """
    # Iterate over subjects and ROIs
    for subject in cohort.get_subjects():
        for roi in cohort.get_subject_rois(subject):
            # Create the output directory for the subject and ROI
            if label:
                output_dir = os.path.join(results_folder, subject, roi, f"tiff_label_series{'_' + registration_to if use_registered else ''}", label_settings)
            else:
                output_dir = os.path.join(results_folder, subject, roi, f"tiff_series{'_' + registration_to if use_registered else ''}")

            os.makedirs(output_dir, exist_ok=True)
            
            # Get the weeks for the subject and ROI
            weeks = cohort.get_subject_roi_weeks(subject_id=subject, roi=roi)
            
            if save_as_4d:
                # Collect all weeks' data into a 4D array
                tiff_data_list = []
                for idx, week in enumerate(weeks):
                    print(f"Processing {subject} - {roi} - {week} for 4D TIFF conversion...")
                    # Get the nifti file path for channel_1 or label
                    if label:
                        # Use label_settings to determine the label type
                        nifti_file = cohort.get_information_source_files_as_list(
                            information_source="label",
                            channel=channel,
                            subject_ids=[subject],
                            rois=[roi],
                            weeks=[week],
                            label_setting=label_settings,
                            file_type=file_type,
                            registered=use_registered,
                            registration_to=registration_to,
                            reg_file_type="reg_instances"
                        )[0]
                    else:
                        nifti_file = cohort.get_information_source_files_as_list(
                            information_source="nifti",
                            channel=channel,
                            subject_ids=[subject],
                            rois=[roi],
                            weeks=[week],
                            registered=use_registered,
                            registration_to=registration_to,
                            reg_file_type="reg_nifti"
                        )[0]
                
                    # Load the nifti file
                    nifti_img = nib.load(nifti_file)
                    nifti_data = nifti_img.get_fdata()
                     
                    # Convert to uint16 for tiff
                    tiff_data = nifti_data.astype(np.uint16)
                    tiff_data_list.append(tiff_data)
                
                # Stack all weeks into a 4D array (x, y, z, t)
                tiff_data_4d = np.stack(tiff_data_list, axis=-1)

                print(f"Number of weeks stacked for {subject} - {roi}: {len(weeks)}. Shape of 4D data: {tiff_data_4d.shape}")
                write_to_nifti(tiff_data_4d, dtype=np.uint16 ,filename=os.path.join(output_dir, f"{file_type}_4d.nii.gz"))
                
                # Save as a single 4D tiff
                if label:
                    tiff_filename = os.path.join(output_dir, f"{file_type}_4d.ome.tif")
                else:
                    tiff_filename = os.path.join(output_dir, "rec_4d.ome.tif")
                write_4d_ome_tiff(tiff_data_4d, tiff_filename)
                print(f"Converted all weeks to 4D TIFF: {tiff_filename}")
            else:
                # Iterate over weeks and save individually
                for idx, week in enumerate(weeks):
                    # Get the nifti file path for channel_1 or label
                    if label:
                        # Use label_settings to determine the label type
                        nifti_file = cohort.get_information_source_files_as_list(
                            information_source="label",
                            channel=channel,
                            subject_ids=[subject],
                            rois=[roi],
                            weeks=[week],
                            label_setting=label_settings,
                            file_type=file_type,
                            registered=use_registered,
                            registration_to=registration_to,
                            reg_file_type="reg_instances"
                        )[0]
                    else:
                        nifti_file = cohort.get_information_source_files_as_list(
                            information_source="nifti",
                            channel=channel,
                            subject_ids=[subject],
                            rois=[roi],
                            weeks=[week],
                            registered=use_registered,
                            registration_to=registration_to,
                            reg_file_type="reg_nifti"
                        )[0]
                    
                    # Load the nifti file
                    nifti_img = nib.load(nifti_file)
                    nifti_data = nifti_img.get_fdata()
                     
                    # Convert to uint16 for tiff
                    tiff_data = nifti_data.astype(np.uint16)
                
                    # Save as tiff with enumerated filename
                    if label:
                        tiff_filename = os.path.join(output_dir, f"{file_type}_{idx:03d}.tif")
                    else:
                        tiff_filename = os.path.join(output_dir, f"nifti_{idx:03d}.tif")
                    write_to_tiff(tiff_data, tiff_filename)
                    print(f"Converted {nifti_file} to {tiff_filename}")

                print(f"Converted {nifti_file} to {tiff_filename}")
