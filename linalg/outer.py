import torch


def outer(u, v):
    """
    Compute the outer product of two tensors.

    For two tensors **u** of shape :math:`(k_1, k_2, ..., k_m)` and **v** of shape :math:`(l_1, l_2, ..., l_n)`,
    their outer product :math:`\mathbf{w} = \mathbf{u} \otimes \mathbf{v}` is defined such that
    :math:`\mathbf{w}[i_1, i_2, ..., i_m, j_1, j_2, ..., j_n] =
    \mathbf{u}[i_1, i_2, ..., i_m] \mathbf{v}[j_1, j_2, ..., j_n]`.
    **w** will have :math:`m+n` dimensions and the shape of :math:`(k_1, k_2, ..., k_m, l_1, l_2, ..., l_n)`

    Parameters
    ----------
    u, v : torch.Tensor
        Input tensor with arbitrary shape.

    Returns
    -------
    w : torch.Tensor
        Outer product of u and v.

        ``w.shape`` is the same as ``u.shape + v.shape``.

    """
    assert isinstance(u, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    return u.view(u.shape + (1, ) * v.ndim) * v
