import torch
from torch.nn.functional import pad
from qmodules.QConv import Qconv

class DWBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=3, upscaling_factor = 2):
        super().__init__()

        # Compute mid channels and ensure compatibility with PixelShuffle(2)
        expanded_channels = in_channels * expand_ratio
        upscaled_channels = expanded_channels// (upscaling_factor * upscaling_factor)


        self.expand = torch.nn.Sequential(
            Qconv(in_channels, expanded_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            #`torch.nn.BatchNorm2d(expanded_channels),
            torch.nn.GELU(),
        )

        # self.learned_upscaling = torch.nn.ConvTranspose2d(expanded_channels,upscaled_channels,kernel_size=kernel_size,padding=kernel_size//2, stride=upscaling_factor)

        self.shuffle_upscaling = torch.nn.PixelShuffle(upscale_factor=upscaling_factor)

        self.depthwise = torch.nn.Sequential(
            Qconv(upscaled_channels, upscaled_channels, kernel_size=kernel_size, stride=kernel_size//2,
                      padding=1, groups=upscaled_channels, bias=False),
            # torch.nn.BatchNorm2d(upscaled_channels),
            torch.nn.ELU(),
        )

        self.project = torch.nn.Sequential(
            Qconv(upscaled_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )

        #self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        identity = x
        expand = self.expand(x)
        # upscaled = self.learned_upscaling(expand)
        upscaled = self.shuffle_upscaling(expand)
        dw = self.depthwise(upscaled)
        out = self.project(dw)
        #out = out + (self.use_residual)*identity
        return out

