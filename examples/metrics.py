import torch
from torch import nn
from torch.nn import functional as F

from math import exp


def create_ssim_window(window_size, channel):
    def gaussian(_window_size, sigma):
        gauss = torch.Tensor([exp(-(x - _window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(_window_size)])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def ssim(img1, img2, window_size=11, size_average=True, full=False):
    channel = img1.size()[1]

    window = create_ssim_window(window_size, channel)
    window = window.to(img1.device)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast * sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs
        ret = ssim_map

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        ssims.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# SSIM_LOSS
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2, retain_grad=True):
        ssim_value = ssim(img1, img2, self.window_size, self.size_average)
        ssim_loss = 1 - ssim_value
        if retain_grad:
            ssim_loss.retain_grad()
        return ssim_loss


# Multi_Scale SSIM Loss
class MS_SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(MS_SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2, retain_grad=True):
        msssim_value = msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        msssim_loss = 1 - msssim_value
        if retain_grad:
            msssim_loss.retain_grad()
        return msssim_loss


# L1 and multi_ssim fusion loss function
class L1_MSSSIM(nn.Module):
    def __init__(self, alpha=0.84):
        super(L1_MSSSIM, self).__init__()
        self.alpha = alpha
        self.msssim_loss = MS_SSIMLoss()
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, img1, img2, retain_grad=True, debug=False):
        channel = img1.size()[1]

        msssim_loss_res = self.msssim_loss(img1, img2, retain_grad=True)
        l1_loss_res = self.l1_loss(img1, img2)

        window = create_ssim_window(self.msssim_loss.window_size, channel).to(img1.device)
        l1_loss_gaussian = F.conv2d(l1_loss_res, window, padding=self.msssim_loss.window_size // 2, groups=channel)
        l1_loss_gaussian = l1_loss_gaussian.mean()
        l1_msssim_loss = self.alpha * msssim_loss_res + (1 - self.alpha) * l1_loss_gaussian

        if retain_grad:
            l1_msssim_loss.retain_grad()
        if debug:
            return l1_msssim_loss, l1_loss_res.mean(), msssim_loss_res
        else:
            # return l1_msssim_loss
            return l1_loss_gaussian, msssim_loss_res


# test code, please ignore it
if __name__ == "__main__":
    a = torch.rand(size=(10,4,512,960), requires_grad=True, device="cuda:0")
    b = torch.rand(size=(10,4,512,960), requires_grad=False, device="cuda:0")

    loss = L1_MSSSIM()
    optimizer = torch.optim.Adam([a], lr=0.001)
    l1_value, msssim_value = loss(a,b)
    l1_w = l1_value.clone().detach()
    msssim_w = msssim_value.clone().detach()
    loss_value = l1_value * msssim_w / (l1_w + msssim_w) + msssim_value * l1_w / (l1_w + msssim_w)
    while loss_value > 0.05:
        optimizer.zero_grad()
        l1_value, msssim_value = loss(a,b)
        l1_w = l1_value.clone().detach()
        msssim_w = msssim_value.clone().detach()
        loss_value = l1_value * msssim_w / (l1_w + msssim_w) + msssim_value * l1_w / (l1_w + msssim_w)
        loss_value.backward()
        optimizer.step()
        print(loss_value)
        print(loss_value.item())
