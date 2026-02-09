import matplotlib.pyplot as plt
import tifffile
from PIL import Image


def visualize_mask(image_path, mask_path):
    """
    Load a mask TIFF image and visualize it.

    Parameters:
    image_path (str): Path to the mask.tiff image
    output_path (str): Path where the visualization will be saved
    """
    mask = tifffile.imread(mask_path)
    img = tifffile.imread(image_path)

    # Overlay the mask and the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("mask.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def image_to_png(image_path):
    image = tifffile.imread(image_path)
    image = Image.fromarray(image)

    # increase dynamic range
    image = image.point(lambda x: x * 10)

    image.save("image.png")

visualize_mask("/workspaces/MOLT/data/Fluo-N2DL-HeLa 2/02/t083.tif", "/workspaces/MOLT/mask.tiff")
