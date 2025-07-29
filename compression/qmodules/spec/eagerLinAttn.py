import torch
import einops
from qmodules.QConv import Qconv
from qmodules.QLinear import QlinearHead, QlinearMLP


class SingleHeadLinearAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eps = 1e-6

        # Fused QKVO projection: (B, T, 4 * D_out)
        # This would quantize each head seperately
        self.qkvo_proj = QlinearHead(dim_in, 4 * dim_out)

        # Activation kernel
        # self.kernel = torch.nn.functional.sigmoid
        self.kernel = torch.nn.functional.relu6

    def forward(self, x):
        # x: (B, T, D_in)
        B, T, _ = x.shape

        # Compute fused projection
        qkvo = self.qkvo_proj(x)  # (B, T, 4*D)
        # Rearrange to (4, B, T, D)
        qkv = einops.rearrange(qkvo, "b t (four d) -> four b t d", four=4)

        q = self.kernel(qkv[0])  # (B, T, D)
        k = self.kernel(qkv[1])  # (B, T, D)
        v = qkv[2]               # (B, T, D)
        o = qkv[3]               # (B, T, D)

        # Compute attention
        k_sum = k.sum(dim=1, keepdim=True) + self.eps         # (B, 1, D)
        D_inv = 1.0 / (torch.einsum("btd,btd->bt", q, k_sum) + self.eps)  # (B, T)

        # Context: (B, D, D)
        context = torch.einsum("btd,bte->bde", k, v)
        out = torch.einsum("btd,bde->bte", q, context)         # (B, T, D)

        return out + o


class MultiHeadLinearAttention(torch.nn.Module):
    def __init__(self, in_channels, heads=16, dim_per_head=32):
        super().__init__()
        self.heads = heads
        self.dim_per_head = dim_per_head
        self.total_dim = heads * dim_per_head

        # We will project input channels into heads separately
        # no point of using QlinearMLP... doesnt prune much
        self.input_proj = torch.nn.Linear(in_channels, self.total_dim, bias=False)

        # Create a list of single head attention modules, one per head
        self.heads_attn = torch.nn.ModuleList([
            SingleHeadLinearAttention(dim_per_head, dim_per_head) for _ in range(heads)
        ])

        # Final output projection
        # similarly no point of QlinearMLP
        self.out_proj = torch.nn.Linear(self.total_dim, in_channels, bias=False)

        # self.bn = torch.nn.BatchNorm2d(in_channels)
        self.ln = torch.nn.GroupNorm(1,in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims and project input to total_dim
        x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')
        x_proj = self.input_proj(x_flat)  # (B, N, total_dim)

        # Split into heads: (B, N, heads, dim_per_head) → (heads, B, N, dim_per_head)
        x_heads = einops.rearrange(x_proj, 'b n (h d) -> h b n d', h=self.heads)

        # we will do this later parallely... dont know if compile does this sequentially
        out_heads = [self.heads_attn[i](x_heads[i]) for i in range(self.heads)]  # each (B, N, dim_per_head)

        # Concatenate heads: list of (B, N, dim_per_head) → (B, N, total_dim)
        out = torch.cat(out_heads, dim=2)

        # Final output projection
        out = self.out_proj(out)  # (B, N, C)

        # Reshape back to spatial
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        # BatchNorm expects (B,C,H,W)
        out = self.ln(out)

        return out

