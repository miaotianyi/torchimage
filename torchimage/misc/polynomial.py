import torch


def poly1d(x: torch.Tensor, p: list or tuple):
    """
    Calculate single-variable (one-dimensional) polynomial
    ``y = a_n * x^n + ... + a_2 * x^2 + a_1 * x + a_0``

    The extra memory required by this function is at most
    2 times the size of x, one to store the output tensor
    and the other to keep a running ith power of x.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor to be substituted for the variable x.

    p : sequence of float
        List of weights in the order of descending exponents, namely
        [a_n, a_{n-1}, ..., a_2, a_1, a_0].

    Returns
    -------
    y : torch.Tensor
        Output tensor with the same shape as x
    """
    p = tuple(p)
    bias = p[-1]
    weight = p[:-1][::-1]  # change to ascending order
    y = None
    xx = x  # running x^i
    for i, w in enumerate(weight):
        if i == 0:
            y = w * xx
        else:
            y.add_(xx, alpha=w)
        if i < len(weight) - 1:
            xx = xx * x
    y += bias
    return y

