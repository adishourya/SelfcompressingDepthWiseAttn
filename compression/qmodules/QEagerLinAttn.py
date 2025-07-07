import torch
import einops
from .quant_func import STERound
steRound = STERound.apply

class QEagerLinearAttention(torch.nn.Module):
    def __init__(self, dim_in, heads=16, dim_per_head=32, b=2.0 ,e=-8.0):
        super().__init__()
        b = torch.as_tensor(b)
        e = torch.as_tensor(e)
        self.heads = torch.as_tensor(heads)
        self.dim_per_head = torch.as_tensor(dim_per_head)
        self.total_dim = heads * dim_per_head
        self.eps = 1e-6
        self.kernel = torch.nn.functional.relu6

        # Use fused QlinearHead to project into QKVO: (B, T, 4*total_dim)
        self.qkvo_head = torch.nn.Linear(dim_in, 4 * self.total_dim)

        # receives 4, b, t, h ,d
        self.depth_bit = torch.ones(1,1,1,self.heads,1) * b
        self.exp_bit= torch.ones(1,1,1,self.heads,1) * e


        # Output projection
        self.out_proj = torch.nn.Linear(self.total_dim, dim_in, bias=False)

    def _quantized_weight(self,x):
        b = torch.relu(self.depth_bit.to(x.device))
        e = self.exp_bit.to(x.device)

        x_upscaled = x/torch.exp2(e)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = steRound(x_clipped)
        return torch.exp2(e) * x_round

    def size_layer(self):
        return torch.sum(torch.relu(self.depth_bit) * self.dim_per_head)

    def _fakebits(self):
        return torch.sum(torch.exp2(torch.relu(self.depth_bit)) * 4* self.dim_per_head * self.dim_per_head)

    def forward(self, x):
        # x: (B, T, D_in)
        B, T, _ = x.shape

        qkvo = self.qkvo_head(x)  # (B,T,D) -> (B, T, 4 * (h * d_h)) -> B,T,D_total
        qkvo = einops.rearrange(qkvo, 'b t (four h d) -> four b t h d', four=4, h=self.heads)
        qkvo = self._quantized_weight(qkvo)

        # indexing squeezes out dimension 4,b,t,h,d -> b,t,h,d
        q = self.kernel(qkvo[0])
        k = self.kernel(qkvo[1])
        v = qkvo[2]
        o = qkvo[3]

        # sum across the token dimension
        k_sum = k.sum(dim=1, keepdim=True) + self.eps  # (B, 1, H, D_h)
        # the contraction here would be a dot porduct of: bthd_q , bthd_k
        # and then we add a fake dimension for later.
        # here o is 1. torch einsum does not allow "1".. smh
        D_inv = 1.0 / (torch.einsum('bthq, bohk -> bth', q, k_sum).unsqueeze(-1) + self.eps)  # (B, T, H, 1)

        # across tokens quadratic on head size
        context = torch.einsum('bthk,bthv->bhkv', k, v)  # (B, H, D_h, D_h)
        out = torch.einsum('bthq,bhkv->bthv', q, context) * D_inv  # (B, T, H, D_h)

        out = out + o  # Residual connection

        # Merge heads
        out = einops.rearrange(out, 'b t h d -> b t (h d)')  # (B, T, total_dim)

        # ffn
        return self.out_proj(out)  # (B, T, D_in)


class MultiHeadLinearAttention(torch.nn.Module):
    def __init__(self, in_channels, heads=16, dim_per_head=32):
        super().__init__()
        self.attn = QEagerLinearAttention(in_channels, heads, dim_per_head)
        self.bn = torch.nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims
        x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C)

        # Batched attention/ we will use flash here later.
        out = self.attn(x_flat)

        # out proj.
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return self.bn(out)
