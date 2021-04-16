import torch


def empty_pad_nd(x, padding):
    """
    Parameters
    ----------
    x : torch.Tensor
        Original tensor to be padded

    padding : tuple of int
        Number of padding like ``(pad_left, pad_right, pad_top, pad_bottom)``.

        Essentially, it's (pad beg axis -1, pad end axis -1, pad beg axis -2,
        pad end axis -2, ...)
    """
    assert isinstance(padding, tuple)
    assert len(padding) % 2 == 0
    ndim_padded = len(padding) // 2  # the number of dimensions that are padded
    assert ndim_padded <= x.ndim
    pad_beg = (0,) * (x.ndim - ndim_padded) + padding[::2][::-1]
    pad_end = (0,) * (x.ndim - ndim_padded) + padding[1::2][::-1]
    new_shape = tuple((torch.tensor(x.shape) + torch.tensor(pad_beg) + torch.tensor(pad_end)).tolist())
    y = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    y[tuple([slice(beg, side-end) for beg, end, side in zip(pad_beg, pad_end, y.shape)])] = x
    return y
