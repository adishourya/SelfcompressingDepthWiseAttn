import torch
from qmodules.QConv import Qconv
from qmodules.QLinear import QlinearMLP
from qmodules.spec.DWBlock import DWBlock
from qmodules.spec.eagerLinAttn import SimpleLinearAttention2D
from qmodules.QPE import ProjectionEmbedding

class EagerDWLin(torch.nn.Module):
    def __init__(self, projection_out=64,num_heads=16, head_dim=32,repeat_transformer=4):
        super().__init__()
        self.pe_out = projection_out
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.pe = ProjectionEmbedding(img_channels=1,out_channels=self.pe_out,patch_size=2)
        self.transformer_block = torch.nn.Sequential(
                DWBlock(in_channels=self.pe_out, out_channels=self.pe_out),
                SimpleLinearAttention2D(in_channels=self.pe_out,heads=self.num_heads,dim_per_head=self.head_dim),
                )
        self.transformer = torch.nn.Sequential(*(self.transformer_block for _ in range(repeat_transformer)))
        # the output of the shape would be B, heads , h,w (h,w might be 15,15 for mnsit images) 
        self.mlp_block = torch.nn.Sequential(
                QlinearMLP(self.pe_out * 15 * 15,256),
                QlinearMLP(256,64),
                QlinearMLP(64,10)
                )
        ...

    def forward(self, x):
        out = self.pe(x)
        for blocks in self.transformer:
            out = blocks(out) 
        out = torch.flatten(out,start_dim=1)
        out = self.mlp_block(out)
        return out

    def _targetModules(self):
        return (Qconv, QlinearMLP)

