import torch
from qmodules.QConv import Qconv

class DWBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=5, stride=1, pixel_shuffle = 2):
        super().__init__()

        # Compute mid channels and ensure compatibility with PixelShuffle(2)
        mid_channels = in_channels * expand_ratio
        assert mid_channels % (pixel_shuffle * pixel_shuffle) == 0, "expanded_channel must be div by pixshuf^2"

        shuffled_channels = mid_channels // (pixel_shuffle * pixel_shuffle)   # because PixelShuffle(2) reduces channels by 4

        self.expand = torch.nn.Sequential(
            Qconv(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU6(inplace=True),
        )

        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=pixel_shuffle)

        self.depthwise = torch.nn.Sequential(
            Qconv(shuffled_channels, shuffled_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=shuffled_channels, bias=False),
            torch.nn.BatchNorm2d(shuffled_channels),
            torch.nn.ReLU6(inplace=True),
        )

        self.project = torch.nn.Sequential(
            Qconv(shuffled_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )

        #self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        identity = x
        expand = self.expand(x)                # [B, mid_channels, H, W]
        upscaled = self.pixel_shuffle(expand)         # [B, mid_channels//4, 2H, 2W]
        dw = self.depthwise(upscaled)             # [B, mid_channels//4, 2H, 2W] or lower if stride > 1
        out = self.project(dw)               # [B, out_channels, ...]
        #out = out + (self.use_residual)*identity
        return out
