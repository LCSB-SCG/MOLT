ß# import nibabel as nib
import tifffile
from src.utils.io.nifti.write_to_nifti import write_to_nifti
from cellpose import models, io
import numpy as np

io.logger_setup() # run this to get printing of progress


model = models.CellposeModel(gpu=False)

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

img = tifffile.imread("/workspaces/MOLT/data/CellTracking/PhC-C2DL-PSC/01/t245.tif")

masks, flows, styles = model.eval(img, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize={"tile_norm_blocksize": tile_norm_blocksize})

# save the image
write_to_nifti(masks, np.uint16, "test.nii")