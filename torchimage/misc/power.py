import torch


def safe_power(x, exponent, *, epsilon=1e-6):
    """
    Takes the power of each element in input with exponent and returns a tensor with the result.

    This is a safer version of ``torch.pow`` (``out = x ** exponent``), which avoids:

    1. NaN/imaginary output when ``x < 0`` and exponent has a fractional part
        In this case, the function returns the signed (negative) magnitude of the complex number.

    2. NaN/infinite gradient at ``x = 0`` when exponent has a fractional part
        In this case, the positions of 0 are added by ``epsilon``,
        so the gradient is back-propagated as if ``x = epsilon``.

    However, this function doesn't deal with float overflow, such as 1e10000.

    Parameters
    ----------
    x : torch.Tensor or float
        The input base value.

    exponent : torch.Tensor or float
        The exponent value.

        (At least one of ``x`` and ``exponent`` must be a torch.Tensor)

    epsilon : float
        A small floating point value to avoid infinite gradient. Default: 1e-6

    Returns
    -------
    out : torch.Tensor
        The output tensor.
    """
    # convert float to scalar torch.Tensor
    if not torch.is_tensor(x):
        if not torch.is_tensor(exponent):
            # both non-tensor scalars
            x = torch.tensor(x)
            exponent = torch.tensor(exponent)
        else:
            x = torch.tensor(x, dtype=exponent.dtype, device=exponent.device)
    else:  # x is tensor
        if not torch.is_tensor(exponent):
            exponent = torch.tensor(exponent, dtype=x.dtype, device=x.device)

    exp_fractional = torch.floor(exponent) != exponent
    if not exp_fractional.any():  # no exponent has a fractional part
        return torch.pow(x, exponent)

    x, x_lt_0, x_eq_0, exponent, exp_fractional = torch.broadcast_tensors(
        x, x < 0, x == 0, exponent, exp_fractional)

    # deal with x = 0
    if epsilon != 0:
        mask = x_eq_0 & exp_fractional
        if mask.any():  # has zero value
            x = x.clone()
            x[mask] += epsilon

    # deal with x < 0
    mask = x_lt_0 & exp_fractional
    if mask.any():
        x = x.masked_scatter(mask, -x[mask])
        out = torch.pow(x, exponent)
        out = out.masked_scatter(mask, -out[mask])
    else:
        out = torch.pow(x, exponent)
    return out


