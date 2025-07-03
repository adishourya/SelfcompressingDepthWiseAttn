import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import einops
    return einops, plt, torch, torchvision


@app.cell
def _(einops, plt, torch, torchvision):
    sample_img = torchvision.io.decode_image("./dolphin.jpg")
    print(f"original size:{sample_img.shape=}")
    # Add batch dimension before interpolate
    sample_img = einops.rearrange(sample_img, "c h w -> 1 c h w")

    sample_img = torch.nn.functional.interpolate(sample_img, size=(2160, 3180))

    # sample_img = einops.rearrange(sample_img,"c h w -> 1 c h w") # add batch dim
    print(sample_img.shape)
    plt.imshow(einops.rearrange(sample_img, "1 c h w -> h w c"))
    return (sample_img,)


@app.cell
def _(torch):
    class ProjectionEmbedding(torch.nn.Module):
        def __init__(self,img_channels :int =3 , out_channels = 4, patch_size =4):
            super().__init__()
            self.projection = torch.nn.Conv2d(in_channels=img_channels,
                                             out_channels=out_channels,
                                             kernel_size=patch_size,
                                             stride=patch_size)
        def forward(self,x):
            return self.projection(x)
    pe = ProjectionEmbedding()
    return ProjectionEmbedding, pe


@app.cell
def _(einops, pe, plt, sample_img):
    sample_img_fp = sample_img.float()
    out_pe = pe(sample_img_fp)
    print(out_pe.shape)
    plt.figure(figsize=(15,8))
    plt.imshow(einops.rearrange(out_pe.detach(),"1 c h w -> h (c w)"))
    plt.show()
    return out_pe, sample_img_fp


@app.cell
def _(torch):
    class DWBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=3, stride=1, upscaling_factor = 2,):
            super().__init__()

            # Compute mid channels and ensure compatibility with PixelShuffle(2)
            expanded_channels = in_channels * expand_ratio
            upscaled_channels = expanded_channels// (upscaling_factor * upscaling_factor)


            self.expand = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, expanded_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
                # torch.nn.BatchNorm2d(expanded_channels),
                torch.nn.GELU(),
            )

            # self.learned_upscaling = torch.nn.ConvTranspose2d(expanded_channels,upscaled_channels,kernel_size=kernel_size,padding=kernel_size//2, stride=upscaling_factor)

            self.shuffle_upscaling = torch.nn.PixelShuffle(upscale_factor=upscaling_factor)
            self.transpose_upscaling = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=expanded_channels, out_channels= upscaled_channels,kernel_size=kernel_size, padding=kernel_size//2, stride=2),
                torch.nn.GELU()
            )

            self.depthwise = torch.nn.Sequential(
                torch.nn.Conv2d(upscaled_channels, upscaled_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=upscaled_channels, bias=False),
                # torch.nn.BatchNorm2d(upscaled_channels),
                torch.nn.GELU(),
            )

            self.project = torch.nn.Sequential(
                torch.nn.Conv2d(upscaled_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,stride=2, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )

            #self.use_residual = (in_channels == out_channels and stride == 1)

        def forward(self, x):
            identity = x
            expand = self.expand(x)
            # upscaled = self.learned_upscaling(expand)
            # upscaled = self.shuffle_upscaling(expand)
            upscaled = self.transpose_upscaling(expand)
            dw = self.depthwise(upscaled)
            out = self.project(dw)
            #out = out + (self.use_residual)*identity
            return out, expand, upscaled, dw

    dwconv = DWBlock(in_channels=4,out_channels=4)
    return DWBlock, dwconv


@app.cell
def _(dwconv, einops, out_pe, plt):
    out_mbconv, out_expand, out_upscaled, out_dw = dwconv(out_pe)

    print(f"{out_expand.shape=}")
    print(f"{out_dw.shape=}")
    print(f"{out_mbconv.shape=}")

    plt.figure(figsize=(15,16))
    plt.imshow(einops.rearrange(out_expand.detach() , "1 p h w -> h (p w)"),cmap="Oranges")
    plt.show()

    plt.figure(figsize=(30,8))
    plt.imshow(einops.rearrange(out_upscaled.detach() , "1 p h w -> h (p w)"),cmap="grey")
    plt.show()

    plt.figure(figsize=(30,8))
    plt.imshow(einops.rearrange(out_dw.detach() , "1 p h w -> h (p w)"),cmap="grey")
    plt.show()

    plt.figure(figsize=(15,8))
    plt.imshow(einops.rearrange(out_mbconv.detach() , "1 p h w -> h (p w)"),cmap="grey")
    plt.show()
    return out_dw, out_expand, out_mbconv, out_upscaled


@app.cell
def _(einops, torch):
    class SimpleLinearAttention2D(torch.nn.Module):
        def __init__(self, in_channels, heads=8, dim_per_head=16):
            super().__init__()
            self.heads = heads
            self.dim = dim_per_head
            self.total_dim = heads * dim_per_head
            self.eps = 1e-6

            # 1 output channel each for q,k,v
            self.qkv_proj = torch.nn.Conv2d(in_channels, 3 * self.total_dim, kernel_size=1,padding=0)
            self.out_proj = torch.nn.Conv2d(self.total_dim, in_channels, kernel_size=1,padding=0)


            # self.kernel = lambda x : torch.nn.functional.elu(x) + 1
            # self.kernel = lambda x : 1 - x + x**2/2
            # self.kernel = torch.sin
            self.kernel = torch.nn.functional.relu6
            # self.kernel = torch.nn.functional.gelu6
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
    sa = SimpleLinearAttention2D(in_channels=4)
    return SimpleLinearAttention2D, sa


@app.cell
def _(einops, out_mbconv, plt, sa):
    sa_out = sa(out_mbconv)
    print(f"{sa_out.shape=}")

    plt.figure(figsize=(15,8))
    plt.imshow(einops.rearrange(sa_out.detach(),"1 c h w -> h (c w)"),cmap="grey")
    return (sa_out,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
