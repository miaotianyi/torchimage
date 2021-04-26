import torch
from torch.nn import functional as F

sobel_3_kernel_1 = (1, 2, 1)
prewitt_3_kernel_1 = (1, 1, 1)
scharr_3_kernel_1 = (3, 10, 3)
general_3_kernel_2 = (-1, 0, 1)
# general_3_kernel_2 = [-1, 0, 1]

# These filter weights can be found in Farid & Simoncelli (2004),
# Table 1 (3rd and 4th row). Additional decimal places were computed
# using the code found at https://www.cs.dartmouth.edu/farid/
p = [0.0376593171958126, 0.249153396177344, 0.426374573253687, 0.249153396177344, 0.0376593171958126]
d1 = [0.109603762960254, 0.276690988455557, 0, -0.276690988455557, -0.109603762960254]
# HFARID_WEIGHTS = d1.T * p


# horizontal derivative, G_x, detect vertical edges (corresponds to skimage's sobel_v)

def sobel_x(image: torch.Tensor, axis_0: int = -2, axis_1: int = -1, stride: int = 1):
    # return F.conv2d(image.unsqueeze(0).unsqueeze(0), weight=torch.tensor([[[[1, 0, -1], [2,  0, -2], [1, 0, -1]]]], dtype=image.dtype)).squeeze()
    image = image.unfold(dimension=axis_0, size=3, step=stride) @ torch.tensor(sobel_3_kernel_1, device=image.device, dtype=image.dtype)
    image = image.unfold(dimension=axis_1, size=3, step=stride) @ torch.tensor(general_3_kernel_2, device=image.device, dtype=image.dtype)
    return image


def sobel_y(image: torch.Tensor, axis_0: int = -2, axis_1: int = -1, stride: int = 1):
    return sobel_x(image, axis_0=axis_1, axis_1=axis_0, stride=stride)

