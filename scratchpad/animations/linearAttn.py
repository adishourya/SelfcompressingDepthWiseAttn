import os
import manim
import numpy as np
import torch
import torch.nn as nn
import einops
import matplotlib.pyplot as plt

font_style = dict(font_size=22, font="Simple Nerd Font")

# --- Helper to convert tensor to RGB image ---
def tensor_to_rgb(x):
    """Convert 2D tensor (H,W) to 3-channel uint8 RGB image"""
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return np.stack([x, x, x], axis=-1)

# --- QEagerLinearAttention with einsum fix ---
class QEagerLinearAttention(nn.Module):
    def __init__(self, dim_in, heads=4, dim_per_head=16):
        super().__init__()
        self.heads = heads
        self.dim_per_head = dim_per_head
        self.total_dim = heads * dim_per_head
        self.eps = 1e-6
        self.kernel = nn.ReLU()

        self.qkvo_head = nn.Linear(dim_in, 4 * self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, dim_in, bias=False)

    def forward(self, x):
        # x: (B, T, D_in)
        B, T, _ = x.shape
        qkvo = self.qkvo_head(x)  # (B, T, 4*D_total)
        qkvo = einops.rearrange(qkvo, "b t (four h d) -> four b t h d", four=4, h=self.heads)
        q = self.kernel(qkvo[0])
        k = self.kernel(qkvo[1])
        v = qkvo[2]
        o = qkvo[3]

        k_sum = k.sum(dim=1, keepdim=True)  # (B,1,H,D_h)
        # Fixed einsum: no squeezing, shapes match
        D_inv = 1.0 / (torch.einsum("bthq,bthq->bth", q, k_sum.expand_as(q)) + self.eps).unsqueeze(-1)

        context = torch.einsum("bthk,bthv->bhkv", k, v)
        out = torch.einsum("bthq,bhkv->bthv", q, context) * D_inv
        out = out + o
        out = einops.rearrange(out, "b t h d -> b t (h d)")
        return self.out_proj(out)

# --- Multihead wrapper ---
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, in_channels, heads=4, dim_per_head=16):
        super().__init__()
        self.attn = QEagerLinearAttention(in_channels, heads, dim_per_head)
        self.ln = nn.GroupNorm(1, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = einops.rearrange(x, "b c h w -> b (h w) c")
        out = self.attn(x_flat)
        out = einops.rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return self.ln(out)

# --- Manim Scene ---
class LinearAttentionScene(manim.Scene):
    def construct(self):
        # Load image
        img_path = "./dolphin.jpg"
        np_img = plt.imread(img_path)
        if np_img.ndim == 3:
            if np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]
            np_img = 0.2 * np_img[:, :, 0] + 0.7 * np_img[:, :, 1] + 0.1 * np_img[:, :, 2]
        np_img = np.asarray(np_img, dtype=np.float32)
        H, W = np_img.shape

        # Input image in Manim
        img_m = manim.ImageMobject(tensor_to_rgb(np_img))
        img_m.height = 3.0
        img_label = manim.Text("Input Image", **font_style).scale(0.6)
        img_group = manim.Group(img_m, img_label).arrange(manim.DOWN)
        img_group.to_edge(manim.LEFT)
        self.play(manim.FadeIn(img_group))

        # Commentary
        commentary = manim.Text("", **font_style).to_edge(manim.DOWN)
        self.add(commentary)

        # Convert to tensor for attention
        x = torch.from_numpy(np_img[None, None, :, :]).float()  # (1,1,H,W)
        commentary.set_text("Applying MultiHead Linear Attention")
        self.wait(0.5)

        attn = MultiHeadLinearAttention(1, heads=4, dim_per_head=8)
        out = attn(x)  # (1,1,H,W)
        out_img = out[0,0].detach().numpy()
        commentary.set_text("Computed Q/K/V and attention output")

        # Show output image
        out_m = manim.ImageMobject(tensor_to_rgb(out_img))
        out_m.height = 3.0
        out_label = manim.Text("Attention Output", **font_style).scale(0.6)
        out_group = manim.Group(out_m, out_label).arrange(manim.DOWN)
        out_group.next_to(img_group, manim.RIGHT, buff=1.0)
        self.play(manim.FadeIn(out_group))
        self.wait(1)

        # Optionally, show flattening of spatial tokens
        x_flat = einops.rearrange(x, "b c h w -> b (h w) c")
        T = x_flat.shape[1]
        token_imgs = []
        max_tokens = min(16, T)
        for i in range(max_tokens):
            tok_img = tensor_to_rgb(x_flat[0,i,0].unsqueeze(0).repeat(H,W))
            tok_m = manim.ImageMobject(tok_img)
            tok_m.height = 0.6
            tok_m.move_to(img_group.get_center() + manim.RIGHT*1.5 + manim.UP*(1.5-i*0.4))
            token_imgs.append(tok_m)
        token_group = manim.Group(*token_imgs)
        commentary.set_text("Flatten spatial dims into tokens")
        self.play(manim.FadeIn(token_group))
        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -ql --resolution 1920,1080 linearAttn.py LinearAttentionScene")
