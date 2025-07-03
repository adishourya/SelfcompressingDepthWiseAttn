import torch
import einops
from qmodules.QConv import Qconv
from qmodules.QLinear import QlinearHead

class SimpleLinearAttention2D(torch.nn.Module):
    def __init__(self, in_channels, heads=4, dim_per_head=32):
        super().__init__()
        self.heads = heads
        self.dim = dim_per_head
        self.total_dim = heads * dim_per_head
        self.eps = 1e-6

        # 1 output channel each for q,k,v
        self.qkv_proj = torch.nn.Conv2d(in_channels, 3 * self.total_dim, kernel_size=1)
        self.out_proj = torch.nn.Conv2d(self.total_dim, in_channels, kernel_size=1)


        # self.kernel = lambda x : torch.nn.functional.elu(x) + 1
        # self.kernel = lambda x : 1 - x + x**2/2
        # self.kernel = torch.sin
        #self.kernel = torch.nn.functional.relu
        self.kernel = torch.nn.functional.relu6
        # self.kernel = lambda x : torch.exp(x - x.amax(dim=-1, keepdim=True))


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






#=========== a more std implementation
class SingleHeadLinearAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eps = 1e-6

        # Vanilla linear layers for Q,K,V and output projection
        self.q_proj = QlinearHead(dim_in, dim_out)
        self.k_proj = QlinearHead(dim_in, dim_out)
        self.v_proj = QlinearHead(dim_in, dim_out)
        self.out_proj = QlinearHead(dim_out, dim_out)

        self.kernel = torch.nn.functional.sigmoid
        # self.kernel = torch.nn.functional.gelu
        # self.kernel = torch.nn.functional.relu6
        # self.kernel = torch.nn.functional.relu


    def forward(self, x):
        # x: (B, N, dim_in)
        B, N, _ = x.shape

        # Project Q,K,V
        q = self.kernel(self.q_proj(x))  # (B, N, dim_out)
        k = self.kernel(self.k_proj(x))  # (B, N, dim_out)
        v = self.v_proj(x)               # (B, N, dim_out)

        # Add normalizer row to V along feature dim
        # We need to pad dim_out → dim_out + 1 for the trick
        v = torch.cat([v, torch.ones(B, N, 1, device=x.device)], dim=2)  # (B, N, dim_out+1)

        # Compute K^T V: 
        # k: (B, N, dim_out), v: (B, N, dim_out+1)
        # We want: (B, dim_out+1, dim_out)
        kv = torch.matmul(v.transpose(1, 2), k)  # (B, dim_out+1, dim_out)

        # Compute output: (B, N, dim_out+1)
        out = torch.matmul(kv, q.transpose(1, 2))  # (B, dim_out+1, N)
        out = out.transpose(1, 2)                   # (B, N, dim_out+1)

        # Normalize by last feature
        norm = out[:, :, -1:] + self.eps            # (B, N, 1)
        out = out[:, :, :-1] / norm                  # (B, N, dim_out)

        # Output projection
        out = self.out_proj(out)                     # (B, N, dim_out)

        return out


class MultiHeadLinearAttention(torch.nn.Module):
    def __init__(self, in_channels, heads=16, dim_per_head=32):
        super().__init__()
        self.heads = heads
        self.dim_per_head = dim_per_head
        self.total_dim = heads * dim_per_head

        # We will project input channels into heads separately
        self.input_proj = torch.nn.Linear(in_channels, self.total_dim, bias=False)

        # Create a list of single head attention modules, one per head
        self.heads_attn = torch.nn.ModuleList([
            SingleHeadLinearAttention(dim_per_head, dim_per_head) for _ in range(heads)
        ])

        # Final output projection
        self.out_proj = torch.nn.Linear(self.total_dim, in_channels, bias=False)

        self.bn = torch.nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dims and project input to total_dim
        x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')
        x_proj = self.input_proj(x_flat)  # (B, N, total_dim)

        # Split into heads: (B, N, heads, dim_per_head) → (heads, B, N, dim_per_head)
        x_heads = einops.rearrange(x_proj, 'b n (h d) -> h b n d', h=self.heads)

        # Apply each head's attention independently
        out_heads = [self.heads_attn[i](x_heads[i]) for i in range(self.heads)]  # each (B, N, dim_per_head)

        # Concatenate heads: list of (B, N, dim_per_head) → (B, N, total_dim)
        out = torch.cat(out_heads, dim=2)

        # Final output projection
        out = self.out_proj(out)  # (B, N, C)

        # Reshape back to spatial
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        # BatchNorm expects (B,C,H,W)
        out = self.bn(out)

        return out

