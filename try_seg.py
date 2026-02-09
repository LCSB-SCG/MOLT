import numpy as np
from cellSAM import cellsam_pipeline
from PIL import Image

img = Image.open("/workspaces/MOLT/data/CellTracking/Fluo-N2DL-HeLa/01/t007.tif")
img = np.asarray(img)
mask = cellsam_pipeline(
    img, use_wsi=False, low_contrast_enhancement=False, gauge_cell_size=False
)
