import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from torchimage.contrib.edge_loss import GaussianEdgeLoss
from torchimage.random import random_crop, add_gauss_noise
from torchimage.pooling import GaussianPoolNd, AvgPoolNd

nchw_axes = (2, 3)


def gks(sigma):
    # fast gaussian kernel size
    return sigma * 8 + 1


image_path = "D:/Users/miaotianyi/Downloads/images/street.png"
image = torch.tensor(plt.imread(image_path))

image = image.movedim(-1, 0).unsqueeze(0)
image = random_crop(image, axes=nchw_axes, size=512)

# kernel_size_list = list(range(3, 30, 2))
sigma_list = np.arange(1, 10, 0.5)
edge_scores = []
l1_scores = []
# for ks in kernel_size_list:
#     f = AvgPoolNd(kernel_size=ks).to_filter("reflect").forward

sigma = 4
my_loss = GaussianEdgeLoss(kernel_size=gks(sigma), sigma=sigma, order=2)

for sigma in sigma_list:
    ks = gks(sigma)
    f = GaussianPoolNd(kernel_size=ks, sigma=sigma).to_filter("reflect").forward
    changed = f(image, axes=nchw_axes)
    edge_scores.append(my_loss(image, changed))
    l1_scores.append(nn.L1Loss()(image, changed))

for sigma in np.arange(0.01, 0.1, 0.01):
    changed = add_gauss_noise(image, sigma=sigma)
    edge_scores.append(my_loss(image, changed))
    l1_scores.append(nn.L1Loss()(image, changed))

line_edge, = plt.plot(np.array(edge_scores) * 10, label="edge scores")
line_l1, = plt.plot(l1_scores, label="l1 scores")
plt.legend(handles=[line_edge, line_l1])
plt.show()
