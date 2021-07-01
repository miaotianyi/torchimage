import numpy as np

import torch
from torch.nn import functional as F

from torchimage.random import random_crop, add_gauss_noise, add_poisson_gauss_noise
from torchimage.filtering import edges
from torchimage.padding import Padder
from torchimage.pooling import GaussianPoolNd, AvgPoolNd
from torchimage.misc.stats import describe


from matplotlib import pyplot as plt


def visualize_transformations(x: torch.Tensor, funcs, names, batch_metric, main_title):
    # x: h, w, c image tensor
    # funcs: list of functions (x -> y) that blurs/adds noise to the image
    # names: list of str, name for each transformation
    # batch metric: takes in a list of images, evaluates each one of them and return a list
    n_cols = len(funcs) + 1
    fig, ax = plt.subplots(nrows=2, ncols=n_cols)
    data = [[None] * n_cols for _ in range(2)]
    titles = [[""] * n_cols for _ in range(2)]

    titles[0][0] = "Original"
    data[0][0] = x

    for i, (func, name) in enumerate(zip(funcs, names)):
        y = func(x)
        print(f"{name}: {describe(y)}")
        data[0][i + 1] = y
        titles[0][i + 1] = name

    data[1] = batch_metric(data[0])

    for r in range(2):
        for c in range(n_cols):
            if data[r][c].ndim == 3:
                ax[r, c].imshow(data[r][c])
            else:
                ax[r, c].imshow(data[r][c], cmap="cividis", vmin=0, vmax=2)
            ax[r, c].set_title(titles[r][c])

    plt.title(main_title)
    plt.show()


def quantify_edge_loss(x: torch.Tensor, edge_extractor):
    list(range(3, 26, 2))




def main():
    axes = [0, 1]  # image format: h, w, c

    x = plt.imread("D:/Users/miaotianyi/Downloads/images/street.png")
    x = torch.tensor(x, dtype=torch.float64)
    x = random_crop(x, axes=axes, size=1024)

    def edge_loss_1(image_list):
        # hwc images
        detector = edges.Sobel(normalize=True)
        ret = [detector.magnitude(image, axes=[0, 1], p=2) for image in image_list]
        return ret

    def edge_loss_2(image_list):
        detector = edges.LaplacianOfGaussian(kernel_size=11, sigma=1.5)
        ret = [detector.forward(image, axes=[0, 1]) for image in image_list]
        for im in ret:
            im *= 10
            print(describe(im))
        return ret

    def edge_loss_3(image_list):
        detector = edges.Laplace()
        ret = [detector.forward(image, axes=[0, 1]) for image in image_list]

        ret[1:] = [(ret[0] - im).abs().max(dim=-1)[0] for im in ret[1:]]
        # ret[1:] = [(ret[0] - im).abs().mean() for im in ret[1:]]

        for im in ret:
            print(describe(im))
        return ret

    batch_metric = edge_loss_3

    visualize_transformations(
        x, funcs=[
            lambda x: add_gauss_noise(x, sigma=0.1).clamp(0, 1),
            lambda x: add_poisson_gauss_noise(x, sigma=0.1, k=1e-5).clamp(0, 1)],
        names=["gaussian sigma=0.1", "pg sigma=0.1, k=1"],
        batch_metric=batch_metric,
        main_title="full edge map"
    )

    visualize_transformations(
        x, funcs=[
            lambda x: GaussianPoolNd(kernel_size=11, sigma=1.5).to_filter(padder=Padder(mode="reflect")).forward(x, axes=axes),
            lambda x: AvgPoolNd(kernel_size=11).to_filter(padder=Padder(mode="reflect")).forward(x, axes=axes)],
        names=["gaussian pool", "average pool"],
        batch_metric=batch_metric,
        main_title="full edge map"
    )


    # b1 =

    # b2 = AvgPoolNd(kernel_size=11).to_filter(padder=Padder(mode="reflect")).forward(x, axes=axes)

    # plt.show()


if __name__ == '__main__':
    main()