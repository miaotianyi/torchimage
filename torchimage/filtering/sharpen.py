import torch

from ..utils.validation import check_axes
from ..pooling import BasePoolNd
from ..padding import Padder


class UnsharpMask:
    """
    Sharpen an image with unsharp masking.

    The basic formula of unsharp masking is: y = x + amount * (x - blur(x))

    Attributes
    ----------
    blur : BasePoolNd
        Smoothing (blurring) filter for unsharp masking.

        A filter should output a tensor whose shape is completely the same as the input tensor.
        The `to_filter` method from base pooling will be automatically called here, so the user
        only needs to specify essential parameters, such as
        ``GaussianPoolNd(kernel_size=7, sigma=1.5)`` or
        ``AvgPoolNd(kernel_size=5)``


    amount : float
        The "amount" of sharpening applied to the image.

        See the formula for unsharp masking for more detailed explanation.

    threshold : float
        Ignore differences between x and blur(x) that are below threshold. Default: ``0.0``
    """
    def __init__(self, blur: BasePoolNd, amount=0.5, threshold=0, *, padder: Padder = None):
        super().__init__()

        self.blur = blur
        self.blur.to_filter(padder=padder)

        self.amount = amount
        self.threshold = threshold

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
        axes = check_axes(x, axes)
        blurred = self.blur.forward(x, axes=axes)
        y = x - blurred
        if self.threshold > 0:
            y[y.abs() < self.threshold] = 0
        return x + self.amount * y
