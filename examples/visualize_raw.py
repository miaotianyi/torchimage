import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors, cm


from src.cisp.modules.demosaic.bilinear.module import InterpolateDemosaic



BAYER_PATTERN = "RGGB"


def visualize_raw(image: np.ndarray, title: str = "Image"):
    """
    Visualize a 4-channel CHW raw image as 4 separate images
    """
    image = np.array(image)
    fig, axes = plt.subplots(2, 2)
    bayer_pattern = BAYER_PATTERN
    for i in range(4):
        coord = divmod(i, 2)
        axes[coord].imshow(image[i], cmap="cividis", vmin=0.0, vmax=1.0)
        axes[coord].set_title(bayer_pattern[i])
    mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="cividis")
    fig.colorbar(mappable, ax=axes.flat)
    plt.suptitle(title)
    plt.show()


def load_raw(file_path: str):
    """
    Load a .npy file to CHW format
    """
    return np.squeeze(np.load(file_path))


def visualize_raw_as_rgb(image):
    # image has CHW format
    image = torch.from_numpy(image).unsqueeze(0)  # convert to 1CHW format
    rgb = InterpolateDemosaic().process(image, bayer_pattern=BAYER_PATTERN)
    # move channel to the last for matplotlib display
    rgb = rgb.squeeze()  # remove batch dimension
    rgb_display = np.moveaxis(rgb.numpy(), 0, -1)
    plt.imshow(rgb_display)
    plt.show()


def make_float_image(image, n_bit=16):
    # image has CHW format, but is stored as an int array
    return image / float(2 ** n_bit - 1)

