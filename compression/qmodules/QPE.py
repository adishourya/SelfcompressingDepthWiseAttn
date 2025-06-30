import torch
from qmodules.QConv import Qconv

class ProjectionEmbedding(torch.nn.Module):
    def __init__(self,img_channels :int =4 , out_channels = 128, patch_size =3):
        super().__init__()
        # todo :: probably keep bit depth high... at init
        self.projection = Qconv(in_channels=img_channels,
                                out_channels=out_channels,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding = patch_size//2,
                                b=4,e=-8)
    def forward(self,x):
        return self.projection(x)
