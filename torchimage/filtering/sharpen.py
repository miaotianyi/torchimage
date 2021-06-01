import torch
from torch import nn


class UnsharpMask(nn.Module):
    """
    Sharpen an image with unsharp masking.

    The basic formula of unsharp masking is: y = x + amount * (x - blur(x))

    Attributes
    ----------
    filter_layer : any pooling/filtering module
        Smoothing (blurring) filter for unsharp masking.

        A filter should output a tensor whose shape is completely the same as the input tensor.

    amount : float
        The "amount" of sharpening applied to the image.

        See the formula for unsharp masking for more detailed explanation.

    threshold : float
        Ignore differences between x and blur(x) that are below threshold. Default: ``0.0``
    """
    def __init__(self, filter_layer, amount=0.5, threshold=0):
        super(UnsharpMask2d, self).__init__()

        self.filter_layer = filter_layer
        # examples:
        # pool_to_filter(GaussianPoolNd)(kernel_size=7, sigma=1.5)
        # pool_to_filter(AveragePoolNd)(kernel_size=5)
        self.amount = amount
        self.threshold = threshold

    def forward(self, x: torch.Tensor, axes, padder):
        blurred = self.filter_layer(x, axes=axes, padder=padder)
        y = x - blurred
        if self.threshold > 0:
            y[y.abs() < self.threshold] = 0
        return x + self.amount * y
