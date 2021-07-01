import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from torchimage.utils import NdSpec


def f1():
    # single kernel
    a = NdSpec([1, 2, 3], item_shape=())
    print(a)
    print(a[0])
    print(a[1])
    print(a[-1])
    print(a[-2])

    print("=" * 80)
    # list of 2 kernels
    a = NdSpec([[1, 2, 3], [4, 5, 6]], item_shape=())
    print(a)
    print(a[0])
    print(a[1])
    print(a[-1])
    print(a[-2])


def run_sharpen():
    from torchimage.filtering import UnsharpMask, pool_to_filter
    from torchimage.pooling import AveragePoolNd, GaussianPoolNd
    from torchimage.padding import Padder

    um1 = UnsharpMask(
        pool_to_filter(GaussianPoolNd)(kernel_size=7, sigma=1.5),
        amount=0.5,
        threshold=0.0
    )
    original = plt.imread("D:/Users/miaotianyi/Downloads/images/cat_1.jpg") / 255.
    image1 = um1(torch.tensor(original).permute(2, 0, 1),
                 axes=-1,
                 padder=Padder(mode="symmetric")
                 )

    print(image1)

    plt.imshow(image1.permute(1, 2, 0).numpy())
    plt.show()


def run_inplace_test():
    a = torch.rand(1, 1, 10, 7, requires_grad=True)
    w1 = torch.rand(1, 1, 3, 3)
    w2 = torch.rand(1, 1, 3, 3)

    b = F.conv2d(a, w1)
    b.add_(F.conv2d(a, w2))
    c = b.sum()
    c.backward()


if __name__ == '__main__':
    run_inplace_test()
