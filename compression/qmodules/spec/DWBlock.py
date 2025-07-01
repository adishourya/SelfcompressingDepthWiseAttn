import torch
from qmodules.QConv import Qconv

class DWBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, stride=1):
        super().__init__()

        mid_channels = in_channels * expand_ratio

        # in_channels -> in_channels * expand ratio
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False, padding=0),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU6(inplace=True),
        )

        # 2. Depthwise convolution
        self.depthwise = torch.nn.Sequential(
            Qconv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_channels, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU6(inplace=True),
        )

        # 3. Projection (1x1 Conv)
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False,padding=0),
            torch.nn.BatchNorm2d(out_channels),
        )

        # Residual connection if possible
        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        identity = x
        expand = self.expand(x)
        dw = self.depthwise(expand)
        out =  self.project(dw)
        out = out + (self.use_residual) * identity
        return out

