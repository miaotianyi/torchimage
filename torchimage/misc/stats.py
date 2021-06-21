import torch


def describe(x: torch.Tensor):
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
