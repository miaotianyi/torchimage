"""
In torchimage, filtering is a special subset of pooling
that has ``stride=1`` and (usually) same padding.
(Considering that torch has not implemented a general
method to perform dilated unfold on a tensor, dilation=1
is the default.)
In same padding, the input and output shapes are the same.

When ``same=True``, the ``pad_width`` argument in padder will
be overridden (so you may leave it to default when constructing
the padder in the first place).

If False, padder will be used as-is. So if you wish to use
valid padding (in image filtering terminology, that means
no padding), simply put ``padder=None``. For full padding
(return the entire processed image, especially when customized
padding makes the image shape larger), use any padder that
you want.

"""
import torch
from torch import nn
from inspect import signature
from functools import wraps

from ..padding import GenericPadNd
from ..pooling import SeparablePoolNd
from .utils import _same_padding_pair


def pool_to_filter(cls=SeparablePoolNd, same=True):
    class Wrapper(cls):
        @wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            super(Wrapper, self).__init__(stride=1, *args, **kwargs)

        @wraps(cls.forward)
        def forward(self, *args, **kwargs):
            ba = signature(super(Wrapper, self).forward).bind(*args, **kwargs)
            if same:  # make same padder
                if "padder" in ba.arguments and ba.arguments["padder"] is not None:
                    same_pad_width = self.kernel_size.map(_same_padding_pair)
                    padder = ba.arguments["padder"]  # old padder
                    padder = GenericPadNd(pad_width=same_pad_width,
                                          mode=padder.mode.data,
                                          constant_values=padder.constant_values.data,
                                          end_values=padder.end_values.data,
                                          stat_length=padder.stat_length.data)
                    ba.arguments["padder"] = padder
            return super(Wrapper, self).forward(*ba.args, **ba.kwargs)

    # override init signature
    sig = signature(Wrapper.__init__)
    sig = sig.replace(parameters=[val for key, val in sig.parameters.items() if key != "stride"])
    Wrapper.__init__.__signature__ = sig

    return Wrapper
