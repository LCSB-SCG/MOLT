# to  run tis script you need to install cellpose and natsort. in this container only cpu version of cellpose is 
# available, but you can install the gpu version in your local environment.

import os
import numpy as np
from cellpose import models, core
from natsort import natsorted
import tifffile
from tqdm import tqdm

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

folders = ["data/CellTracking/Fluo-N2DH-GOWT1/01",
           "data/CellTracking/Fluo-N2DH-GOWT1/02",
           "data/CellTracking/Fluo-N2DL-HeLa/01",
           "data/CellTracking/Fluo-N2DL-HeLa/02",
           "data/CellTracking/PhC-C2DL-PSC/01",
           "data/CellTracking/PhC-C2DL-PSC/02"]

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0
bs = 18

for datafolder in folders:
    print(f"Processing folder: {datafolder}")
    os.makedirs(f"{datafolder}_Cellpose", exist_ok=True)
    os.makedirs(f"{datafolder}_Cellpose_RGB", exist_ok=True)


    images = natsorted([f for f in os.listdir(datafolder) if f.endswith(".tif")])
    images_paths = [os.path.join(datafolder, f) for f in images]
    # read the images and convert them to 3 channels
    images = [tifffile.imread(f) for f in images_paths]
    images = [np.stack([img, np.zeros_like(img), np.zeros_like(img)], axis=-1) for img in images]
    # split into batches of bs
    images = [images[i:i+bs] for i in range(0, len(images), bs)]
    images_paths = [images_paths[i:i+bs] for i in range(0, len(images_paths), bs)]

    for b_imgs, b_paths in tqdm(zip(images, images_paths)):
        masks, flows, styles = model.eval(b_imgs, batch_size=bs, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, normalize=True)

        # save the masks
        for i, (mask, image_path) in enumerate(zip(masks, b_paths)):
            filename = os.path.basename(image_path)
            filename = "mask" + filename.replace(".tif", "").replace("t", "") + ".tif"
            output_path = os.path.join(datafolder + "_Cellpose", filename)
            tifffile.imwrite(output_path, mask.astype(np.uint16))

            # color the mask with random colors
            output_path_rgb = os.path.join(datafolder + "_Cellpose_RGB", filename)
            mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for j in range(1, mask.max() + 1):
                mask_colored[mask == j] = np.random.randint(0, 255, size=3)            
            tifffile.imwrite(output_path_rgb, mask_colored)