import torch
import einops
from qmodules.QConv import Qconv

class SimpleLinearAttention2D(torch.nn.Module):
    def __init__(self, in_channels, heads=4, dim_per_head=32):
        super().__init__()
        self.heads = heads
        self.dim = dim_per_head
        self.total_dim = heads * dim_per_head
        self.eps = 1e-6

        # 1 output channel each for q,k,v
        self.qkv_proj = Qconv(in_channels, 3 * self.total_dim, kernel_size=1)
        self.out_proj = Qconv(self.total_dim, in_channels, kernel_size=1)


        # self.kernel = lambda x : torch.nn.functional.elu(x) + 1
        # self.kernel = lambda x : 1 - x + x**2/2
        # self.kernel = torch.sin
        # self.kernel = torch.nn.functional.relu6
        self.kernel = lambda x : torch.exp(x - x.amax(dim=-1, keepdim=True))


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # Sequence length

        # 1. Project to Q, K, V: shape → (B, 3*total_dim, H, W)
        qkv = self.qkv_proj(x)  # (B, 3D, H, W)

        # 2. Split and flatten to (B, heads, dim, N)
        qkv = einops.rearrange(qkv, 'b (three h d) h1 w1 -> three b h d (h1 w1)',three=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, heads, dim, N)

        # 3. Apply kernel activation
        q = self.kernel(q)
        k = self.kernel(k)

        # 4. Add normalizer row to V
        v = torch.nn.functional.pad(v, (0, 0, 0, 1), value=1.0)  # (B, heads, dim+1, N)

        # 5. Pre-aggregate: Kᵀ @ V → (B, heads, dim+1, dim)
        kv = torch.matmul(v, k.transpose(-1, -2))  # (B, heads, dim+1, dim)

        # 6. Attention output: (B, heads, dim+1, N)
        out = torch.matmul(kv, q)  # (B, heads, dim+1, N)

        # 7. Normalize using last row (normalizer trick)
        norm = out[:, :, -1:, :] + self.eps
        out = out[:, :, :-1, :] / norm  # (B, heads, dim, N)

        # 8. Reshape: (B, total_dim, H, W)
        # out = out.reshape(B, self.total_dim, H, W)
        out = einops.rearrange(out, 'b h d (h1 w1) -> b (h d) h1 w1', h1=H, w1=W)


        # 9. Final projection back to in_channels
        out = self.out_proj(out)  # (B, C, H, W)
        return out
