import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from scipy import ndimage
from skimage import data, filters
from matplotlib import pyplot as plt
from torchimage.pooling.sobel import sobel_x


def sobel_x_y(image: torch.Tensor, grayscale=True, mode="reflect"):
    """
    Calculate the edge gradients along x and y axis using a Sobel filter.

    Note that sqrt(0) is a common cause for NaN gradients,
    so we defer the ``G=sqrt(G_x^2 + G_y^2)`` calculation
    to the outer scope.

    Parameters
    ----------
    image : torch.Tensor
        Input image batch in ``N, C, H, W`` format.

    grayscale : bool
        If True, averages the color channels of the image before
        applying the Sobel filter.

        This parameter doesn't matter when the input image itself is grayscale.

    mode : str
        Padding mode to use (see ``F.pad`` for all possible parameters).

        Since Sobel filter is 3x3 and thus only requires 1 padding width,
        padding mode should have little influence over final outcome.
        Don't need to change this parameter.

    Returns
    -------
    gx, gy : torch.Tensor
        Normalized edge gradients in x and y dimensions (possible value ranges
        from -1 to 1).

        If ``grayscale=True``, they will have shape ``n1hw``. Otherwise, they
        will have shape ``nchw``.
    """
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype, device=image.device, requires_grad=False
    ).view(1, 1, 3, 3)
    kernel_y = torch.transpose(kernel_x, dim0=-2, dim1=-1)  # transpose last 2 dimensions (h and w)

    if grayscale:
        # first convert the image batch to grayscale
        image = image.mean(dim=-3, keepdim=True)  # average across color channels; n, 1, h, w
    else:
        # no grayscale, output and input tensor have the same number of channels
        out_channels = image.shape[-3]  # n, c, h, w
        kernel_x = kernel_x.repeat(out_channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(out_channels, 1, 1, 1)

    image = F.pad(image, pad=(1, 1, 1, 1), mode=mode)

    groups = 1 if grayscale else image.shape[-3]  # n, c, h, w
    gx = F.conv2d(image, weight=kernel_x, groups=groups) / 4
    gy = F.conv2d(image, weight=kernel_y, groups=groups) / 4
    return gx, gy


def sobel_weight(image: torch.Tensor, grayscale=True, gamma_1=3, gamma_2=0.1, eps=1e-3):
    """
    Given a ground truth image, calculate a weight tensor of the same shape that emphasizes
    the edges. The underlying algorithm is a Sobel filter.

    The essential formula for this algorithm is as follows:
    ``(gx ** gamma_1 + gy ** gamma_1).clamp(eps, 1) ** gamma_2``
    where ``gx`` and ``gy`` are the gradients calculated with a Sobel filter.

    Parameters
    ----------
    image : torch.Tensor
        Input image of shape ``nchw``.

    grayscale : bool
        Average the image across color channels before applying the Sobel filter.
        Will *not* affect the shape of output weight tensor.

    gamma_1, gamma_2, eps : float
        Useful parameters in calculation. See the formula above for more information.

    Returns
    -------
    g : torch.Tensor
        Output weight tensor of shape ``nchw``.
    """
    gx, gy = sobel_x_y(image, grayscale=grayscale)
    g = gx.abs() ** gamma_1 + gy.abs() ** gamma_1
    g = g.clamp(eps, 1.0) ** gamma_2
    if grayscale and image.shape[1] != 1:  # if the image itself is not grayscale, but the weight is
        g = g.repeat(1, image.shape[1], 1, 1)
    return g


class EdgeLoss(nn.Module):
    def __init__(self, loss: nn.Module, alpha=0.3, grayscale=True, gamma_1=3, gamma_2=0.1, eps=1e-3):
        """
        Edge loss is a convex combination between the original loss and a new loss weighted
        by a tensor (range 0~1) that emphasizes edges.

        Specifically, ``alpha * loss(test * weight, true * weight) + (1 - alpha) * loss(test, true)``

        Be aware that this weighting scheme may cause some more complex visual metrics
        (beyond L1 loss and MSE loss) to exhibit undefined behaviors.

        Parameters
        ----------
        loss : nn.Module
            A PyTorch loss function to be modified.

        alpha : float
            Weight added to the edge-weighted loss

        grayscale : bool
        gamma_1, gamma_2, eps : float
            Sobel weight parameters
        """
        super(EdgeLoss, self).__init__()
        self.loss = loss
        self.alpha = alpha
        self.grayscale = grayscale
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.eps = eps

    def forward(self, image_test, image_true):
        weight = sobel_weight(image=image_true, grayscale=self.grayscale, gamma_1=self.gamma_1, gamma_2=self.gamma_2, eps=self.eps)
        return self.alpha * self.loss(image_test * weight, image_true * weight) + (1 - self.alpha) * self.loss(image_test, image_true)


def calibrate_sobel():
    a = data.checkerboard().astype(float)
    # print(a.shape)
    # print(a)

    plt.imshow(a, cmap="gray")
    plt.show()

    edges = ndimage.sobel(a, axis=1)
    print(edges)
    print(edges.min())
    print(edges.max())
    im = plt.imshow(edges, cmap="gray")
    plt.colorbar(im)
    plt.show()

    # G_x
    edges = sobel_x(torch.from_numpy(a), axis_0=0, axis_1=1).numpy()
    print(edges)
    print(edges.min())
    print(edges.max())
    im = plt.imshow(edges, cmap="gray")
    plt.colorbar(im)
    plt.show()


    edges = filters.sobel_v(a)
    print(edges)
    print(edges.min())
    print(edges.max())
    im = plt.imshow(edges, cmap="gray")
    plt.colorbar(im)
    plt.show()


def run_edge_detect():
    a = data.camera()
    b = ndimage.gaussian_filter(a, sigma=1.5)
    c = np.abs(filters.sobel(a))
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(a, cmap="gray")
    ax[0, 1].imshow(np.abs(a - b), cmap="gray")
    ax[1, 0].imshow(c, cmap="gray")
    d = ndimage.maximum_filter(c, size=3)
    ax[1, 1].imshow(d, cmap="gray")
    plt.show()


def run_torch_edge_detect():
    a = data.brick()
    # a = data.camera()
    a = a.astype(float) / 256
    if a.ndim == 2:
        b = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    else:
        b = torch.from_numpy(np.moveaxis(a, -1, 0)).unsqueeze(0)

    graysgale = False
    gx, gy = sobel_x_y(b, grayscale=graysgale)
    # g = gx ** 2 + gy ** 2
    g = gx.abs() ** 3 + gy.abs() ** 3

    if a.ndim == 2 or graysgale:  # g, gx, gy have only 1 channel
        f = lambda x: x.squeeze().numpy()
    else:
        f = lambda x: np.moveaxis(x.squeeze().numpy(), source=0, destination=-1)

    g, gx, gy = f(g), f(gx), f(gy)
    stat = lambda x: [x.min(), x.mean(), np.median(x), x.max()]

    print(stat(g), stat(gx), stat(gy))
    # plt.hist(g_sq.ravel(), bins=50)
    # plt.show()

    g = g.clip(1e-3, None)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(a, cmap="gray", vmin=0, vmax=1)
    ax[0, 1].imshow(g ** 0.2, cmap="gray", vmin=0, vmax=1)
    ax[1, 0].imshow(np.abs(gx), cmap="gray", vmin=0, vmax=1)
    ax[1, 1].imshow(np.abs(gy), cmap="gray", vmin=0, vmax=1)
    plt.show()


run_torch_edge_detect()

'''
def get_canny(image, grayscale=True, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False):
    """
    Parameters
    ----------
    image : np.ndarray
        Ground truth image batch with format NCHW

    grayscale : bool
        If True, the ground truth image will be averaged into a N1HW grayscale image batch
        before running through the edge detection algorithm
    """
    if grayscale:
        image = np.mean(image, axis=1, keepdims=True)
        weight = np.empty(shape=image.shape, dtype=float)
        for i in range(image.shape[0]):
            weight[i, 0] = feature.canny(image[i, 0], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, mask=mask, use_quantiles=use_quantiles)
    else:
        weight = np.empty(shape=image.shape, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                weight[i, j] = feature.canny(image[i, 0], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold, mask=mask, use_quantiles=use_quantiles)
    return weight


a = torch.rand(10, 3, 8, 7, requires_grad=True)
b = torch.rand(10, 3, 8, 7)
loss = 100
optimizer = torch.optim.Adam([a])
e1 = EdgeLoss(loss=nn.L1Loss(), alpha=0.3)
while loss > 0.01:
    optimizer.zero_grad(set_to_none=True)
    loss = e1(a, b)
    loss.backward()
    optimizer.step()
    print(loss)
'''
