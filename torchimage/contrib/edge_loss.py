import torch
from torch import nn
from ..filtering.edges import GaussianGrad


class GaussianEdgeLoss(nn.Module):
    def __init__(self, kernel_size, sigma, order):
        super(GaussianEdgeLoss, self).__init__()
        self.gg = GaussianGrad(kernel_size=kernel_size, sigma=sigma, edge_order=order, same_padder="reflect")
        self.loss = nn.L1Loss()

    # y1, y2 must have nchw format
    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        edge_1 = torch.cat(self.gg.all_components(y1, axes=(2, 3)), dim=1)
        edge_2 = torch.cat(self.gg.all_components(y2, axes=(2, 3)), dim=1)
        return self.loss(edge_1, edge_2)


