from cellSAM import segment_cellular_image, get_model
# import nibabel as nib
import time
import numpy as np
from src.utils.io.nifti.write_to_nifti import write_to_nifti
import time
import tifffile


# img = nib.load("/workspaces/MOLT/data/CellTracking/PhC-C2DL-PSC/Cells_01/245_weeks/nifti/001/channel_1/1901_01_01_001_channel_1.nii.gz").get_fdata()
img = tifffile.imread("/workspaces/MOLT/data/CellTracking/PhC-C2DL-PSC/01/t245.tif")
model = get_model()

start_time = time.time()  # Start measuring execution time
mask, _, _ = segment_cellular_image(img, model=model, device='mps', bbox_threshold=0.4, fast=True)
end_time = time.time()    # End measuring execution time
execution_time = end_time - start_time
print("Execution Time: ", execution_time)  # Print the measured execution time

# save the image
write_to_nifti(mask, np.uint16, "test.nii")
