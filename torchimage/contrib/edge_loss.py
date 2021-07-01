import torch
from torch import nn
from ..filtering.edges import GaussianGrad


class GaussianEdgeLoss(nn.Module):
    def __init__(self, kernel_size, sigma, order):
        super(GaussianEdgeLoss, self).__init__()
        self.gg = GaussianGrad(kernel_size=kernel_size, sigma=sigma, edge_order=order, same_padder="reflect")
        self.loss = nn.L1Loss()

    def _get_edge_tensor(self, y):
        return torch.cat(self.gg.all_components(y, axes=(2, 3)), dim=1)

    # y1, y2 must have nchw format
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        edge_pred = self._get_edge_tensor(y_pred)
        edge_true = self._get_edge_tensor(y_true)
        return self.loss(edge_pred, edge_true)


class WeightedMul(GaussianEdgeLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        edge_pred = self._get_edge_tensor(y_pred)
        edge_true = self._get_edge_tensor(y_true)
        return torch.mean(torch.abs(edge_true - edge_pred) * edge_true)


class WeightedDiv(GaussianEdgeLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        edge_pred = self._get_edge_tensor(y_pred)
        edge_true = self._get_edge_tensor(y_true)
        return torch.mean(torch.abs(edge_true - edge_pred) / edge_true)


class WeightedSum(GaussianEdgeLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        edge_pred = self._get_edge_tensor(y_pred)
        edge_true = self._get_edge_tensor(y_true)
        return (edge_true.abs().mean(dim=(2, 3)) - edge_pred.abs().mean(dim=(2, 3))).abs().mean()
