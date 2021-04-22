import torch


def mse(image_true, image_test):
    # input image must be 3D (c, h, w) or (n, c, h, w)
    # the last 3 dimensions will be reduced
    mse = torch.mean((image_true - image_test) ** 2, dim=[-1, -2, -3])
    return mse
