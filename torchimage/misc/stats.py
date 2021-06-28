import torch


def describe(x: torch.Tensor):
    """
    Describe the distribution of numbers in the input tensor.

    This function mimics the behavior of DataFrame and Series
    describe method in pandas.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be described

    Returns
    -------
    desc : dict
        A dictionary from keywords (such as mean, std) to float values
    """
    desc = {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "25%": x.quantile(0.25),
        "50%": x.quantile(0.5),
        "75%": x.quantile(0.75),
        "max": x.max()
    }
    desc = {key: val.item() for key, val in desc.items()}
    return desc
