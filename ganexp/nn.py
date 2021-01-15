import torch.nn as nn

from ganexp.functional import (
    make_ssim_kernel,
    _ssim_per_channel,
    _msssim,
    MSSSIM_POWER_FACTORS,
)


class SSIM(nn.Module):
    def __init__(
        self, channel, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03
    ):
        super().__init__()
        self.kernel = make_ssim_kernel(filter_size, filter_sigma, channel)
        self.max_val = max_val
        self.k1 = k1
        self.k2 = k2

    def forward(self, img1, img2):
        kernel = self.kernel.to(img1.device)
        ssim_val, _ = _ssim_per_channel(
            img1, img2, kernel, max_val=self.max_val, k1=self.k1, k2=self.k2
        )
        return ssim_val.mean(-1).mean()


class MSSSIM(nn.Module):
    def __init__(
        self,
        channel,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        power_factors=MSSSIM_POWER_FACTORS,
    ):
        super().__init__()
        self.kernel = make_ssim_kernel(filter_size, filter_sigma, channel)
        self.max_val = max_val
        self.k1 = k1
        self.k2 = k2
        self.power_factors = power_factors

    def forward(self, img1, img2):
        kernel = self.kernel.to(img1.device)
        return _msssim(
            img1, img2, kernel, self.max_val, self.k1, self.k2, self.power_factors
        ).mean()