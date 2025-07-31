import torch
from compression.utils import next_nearest32
from qmodules.QConv import Qconv, QconvT
from qmodules.QLinear import QlinearMLP
from qmodules.spec.DWBlock import DWBlock
from qmodules.QEagerLinAttn import QEagerLinearAttention,MultiHeadLinearAttention

# from qmodules.spec.eagerLinAttn import SimpleLinearAttention2D , MultiHeadLinearAttention
from qmodules.QPE import ProjectionEmbedding
import matplotlib.pyplot as plt
import einops


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, num_heads, head_dim):
        super().__init__()
        self.dwblock = DWBlock(in_channels=in_channels, out_channels=in_channels)
        # self.attn = SimpleLinearAttention2D(in_channels=in_channels, heads=num_heads, dim_per_head=head_dim)
        self.attn = MultiHeadLinearAttention(in_channels=in_channels, heads=num_heads, dim_per_head=head_dim)
        # self.bn = torch.nn.BatchNorm2d(in_channels)
        self.ln = torch.nn.GroupNorm(1,in_channels)

    def forward(self, x):
        identity = x
        x = self.dwblock(x)
        x = self.attn(x)
        return identity+self.ln(x)


class EagerDWLin(torch.nn.Module):
    def __init__(self,img_channels=3, projection_out=16,num_heads=16, head_dim=32,repeat_transformer=4,num_classes=10):
        super().__init__()
        self.pe_out = projection_out
        self.num_heads = num_heads
        self.head_dim = head_dim

        # self.pe = ProjectionEmbedding(img_channels=1,out_channels=self.pe_out,patch_size=2)
        self.pe = ProjectionEmbedding(img_channels=img_channels,out_channels=self.pe_out,patch_size=2)
        self.transformer = torch.nn.Sequential(*[TransformerBlock(self.pe_out, self.num_heads, self.head_dim) for _ in range(repeat_transformer)])

        self.mlp_block = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((8,8)),
            torch.nn.Flatten(),
            QlinearMLP(self.pe_out*8*8,next_nearest32(num_classes)*2),
            torch.nn.GELU(),
            QlinearMLP(next_nearest32(num_classes)*2,num_classes,bias=False)
        )

    @staticmethod
    def stop_for_inspection(out,path:str):
        print("======stopping for inspection========")
        print(f"{out.shape=}")

        out = out[0].detach().cpu().float()
        plt.figure(figsize=(15,8))
        plt.imshow(einops.rearrange(out,"n h w -> h (n w)"))
        plt.show()
        plt.savefig(path)
        raise Exception("Stop for inspection")

    def forward(self, x):
        out = self.pe(x)
        #self.stop_for_inspection(out,"after_pe")

        for blocks in self.transformer:
            out = blocks(out) 

        #self.stop_for_inspection(out,"after_1_transformer.png")

        # no flattening...
        # out = torch.flatten(out,start_dim=1)
        # instead i saw:
        # out = out.mean(dim=(2, 3))  # Global average pool on height and width
        # or better maybe use AdaptiveAvgPooling
        out = self.mlp_block(out)
        return out

    def _targetModules(self):
        return (Qconv,QconvT, QlinearMLP,QEagerLinearAttention)

