import marimo

__generated_with = "0.11.21"
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
                # torch.nn.Sigmoid(),
            )

            # self.learned_upscaling = torch.nn.ConvTranspose2d(expanded_channels,upscaled_channels,kernel_size=kernel_size,padding=kernel_size//2, stride=upscaling_factor)

            self.shuffle_upscaling = torch.nn.PixelShuffle(upscale_factor=upscaling_factor)
            self.transpose_upscaling = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=expanded_channels, out_channels= upscaled_channels,kernel_size=kernel_size,padding =1,output_padding=1 ,stride=2),
                torch.nn.GELU()
            )

            self.depthwise = torch.nn.Sequential(
                torch.nn.Conv2d(upscaled_channels, upscaled_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=upscaled_channels, bias=False),
                # torch.nn.BatchNorm2d(upscaled_channels),
                torch.nn.GELU(),
            )
            self.shuffle_downscaling =  torch.nn.PixelUnshuffle(2)

            self.project = torch.nn.Sequential(
                torch.nn.Conv2d(upscaled_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,stride=2, bias=False),
                # torch.nn.BatchNorm2d(out_channels),
                torch.nn.GELU(),
            )

            #self.use_residual = (in_channels == out_channels and stride == 1)

        def forward(self, x):
            identity = x
            print(x.shape)
            expand = self.expand(x)
            print(expand.shape)
            # upscaled = self.learned_upscaling(expand)
            # upscaled = self.shuffle_upscaling(expand)
            upscaled = self.transpose_upscaling(expand)
            print(upscaled.shape)
            dw = self.depthwise(upscaled)
            # out = self.shuffle_downscaling(dw)
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
    class ConvLinearAttention2D(torch.nn.Module):
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
            # self.kernel = torch.nn.functional.sigmoid
            # self.kernel = torch.nn.functional.gelu6
            # self.kernel = lambda x : torch.exp(x - x.amax(dim=-1, keepdim=True))

            #
            self.bn = torch.nn.BatchNorm2d(in_channels)


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
            return self.bn(out)
    ca = ConvLinearAttention2D(in_channels=4)
    return ConvLinearAttention2D, ca


@app.cell
def _(ca, einops, out_mbconv, plt):
    sa_out = ca(out_mbconv)
    print(f"{sa_out.shape=}")

    plt.figure(figsize=(15,8))
    plt.imshow(einops.rearrange(sa_out.detach(),"1 c h w -> h (c w)"),cmap="grey")
    return (sa_out,)


@app.cell
def _():
    # class SimpleLinearAttention2D_Vanilla(torch.nn.Module):
    #     def __init__(self, in_channels, heads=8, dim_per_head=16):
    #         super().__init__()
    #         self.heads = heads
    #         self.dim = dim_per_head
    #         self.total_dim = heads * dim_per_head
    #         self.eps = 1e-6

    #         # Vanilla linear layers for Q, K, V projection and output projection
    #         self.qkv_proj = torch.nn.Linear(in_channels, 3 * self.total_dim, bias=False)
    #         self.out_proj = torch.nn.Linear(self.total_dim, in_channels, bias=False)

    #         # Kernel activation function
    #         # self.kernel = torch.nn.functional.relu6
    #         self.kernel = torch.nn.functional.sigmoid

    #         #bn
    #         self.bn = torch.nn.BatchNorm2d(4)

    #     def forward(self, x):
    #         B, C, H, W = x.shape
    #         N = H * W  # sequence length

    #         # Flatten spatial dims and rearrange to (B, N, C)
    #         x_flat = einops.rearrange(x, 'b c h w -> b (h w) c')

    #         # Project Q, K, V: (B, N, 3 * total_dim)
    #         qkv = self.qkv_proj(x_flat)

    #         # Split Q, K, V: (B, N, 3, heads, dim) → rearranged to (3, B, heads, dim, N)
    #         qkv = einops.rearrange(
    #             qkv,
    #             'b n (three h d) -> three b h d n',
    #             three=3, h=self.heads, d=self.dim, n=N
    #         )
    #         q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, dim, N)

    #         # Apply kernel activation
    #         q = self.kernel(q)
    #         k = self.kernel(k)

    #         # Pad V with normalizer row (dim + 1)
    #         v = torch.nn.functional.pad(v, (0, 0, 0, 1), value=1.0)  # (B, heads, dim+1, N)

    #         # Compute K^T @ V → (B, heads, dim+1, dim)
    #         kv = torch.matmul(v, k.transpose(-1, -2))

    #         # Attention output → (B, heads, dim+1, N)
    #         out = torch.matmul(kv, q)

    #         # Normalize output by last row
    #         norm = out[:, :, -1:, :] + self.eps
    #         out = out[:, :, :-1, :] / norm  # (B, heads, dim, N)

    #         # Reshape back to (B, N, heads * dim)
    #         out = einops.rearrange(out, 'b h d n -> b n (h d)')

    #         # Project back to input channels
    #         out = self.out_proj(out)  # (B, N, C)

    #         # Reshape back to (B, C, H, W)
    #         out = einops.rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

    #         return self.bn(out)

    # vanilla_la = SimpleLinearAttention2D_Vanilla(in_channels=4)
    return


@app.cell
def _():
    # vanilla_out= vanilla_la(out_mbconv)
    # print(f"{vanilla_out.shape=}")

    # plt.figure(figsize=(15,8))
    # plt.imshow(einops.rearrange(vanilla_out.detach(),"1 c h w -> h (c w)"),cmap="grey")
    return


@app.cell
def _(einops, torch):
    # class SingleHeadLinearAttention(torch.nn.Module):
    #     def __init__(self, dim_in, dim_out):
    #         super().__init__()
    #         self.dim_in = dim_in
    #         self.dim_out = dim_out
    #         self.eps = 1e-6

    #         # Vanilla linear layers for Q,K,V and output projection
    #         self.qkvo_proj = torch.nn.Linear(dim_in, 4 * dim_out)


    #         self.q_proj = torch.nn.Linear(dim_in, dim_out, bias=False)
    #         self.k_proj = torch.nn.Linear(dim_in, dim_out, bias=False)
    #         self.v_proj = torch.nn.Linear(dim_in, dim_out, bias=False)
    #         self.out_proj = torch.nn.Linear(dim_out, dim_out, bias=False)

    #         self.kernel = torch.nn.functional.sigmoid
    #         # self.kernel = torch.nn.functional.gelu
    #         # self.kernel = torch.nn.functional.relu6
    #         # self.kernel = torch.nn.functional.relu


    #     def forward(self, x):
    #         # x: (B, N, dim_in)
    #         B, N, _ = x.shape

    #         qkvo = self.qkvo_proj(x)
    #         # Project Q,K,V
    #         q = self.kernel(qkvo[0])  # (B, N, dim_out)
    #         k = self.kernel(qkvo[1])  # (B, N, dim_out)
    #         v = self.v_proj(qkvo[2])               # (B, N, dim_out)

    #         # Add normalizer row to V along feature dim
    #         # We need to pad dim_out → dim_out + 1 for the trick
    #         v = torch.cat([v, torch.ones(B, N, 1, device=x.device)], dim=2)  # (B, N, dim_out+1)

    #         # Compute K^T V: 
    #         # k: (B, N, dim_out), v: (B, N, dim_out+1)
    #         # We want: (B, dim_out+1, dim_out)
    #         kv = torch.matmul(v.transpose(1, 2), k)  # (B, dim_out+1, dim_out)

    #         # Compute output: (B, N, dim_out+1)
    #         out = torch.matmul(kv, q.transpose(1, 2))  # (B, dim_out+1, N)
    #         out = out.transpose(1, 2)                   # (B, N, dim_out+1)

    #         # Normalize by last feature
    #         norm = out[:, :, -1:] + self.eps            # (B, N, 1)
    #         out = out[:, :, :-1] / norm                  # (B, N, dim_out)

    #         # Output projection
    #         out = self.out_proj(out)                     # (B, N, dim_out)

    #         return out



    class SingleHeadLinearAttention(torch.nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.eps = 1e-6

            # Fused QKVO projection: (B, T, 4 * D_out)
            self.qkvo_proj = torch.nn.Linear(dim_in, 4 * dim_out)

            # Activation kernel
            self.kernel = torch.nn.functional.sigmoid

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
        def __init__(self, in_channels, heads=16, dim_per_head=16):
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
            self.ln = torch.nn.GroupNorm(1,in_channels)

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
            out = self.ln(out)
            # out = self.bn(out)

            return out

    mha = MultiHeadLinearAttention(in_channels=4)
    return MultiHeadLinearAttention, SingleHeadLinearAttention, mha


@app.cell
def _(einops, mha, out_mbconv, plt):
    mha_out= mha(out_mbconv)
    print(f"{mha_out.shape=}")

    plt.figure(figsize=(15,8))
    plt.imshow(einops.rearrange(mha_out.detach(),"1 c h w -> h (c w)"),cmap="grey")
    return (mha_out,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
