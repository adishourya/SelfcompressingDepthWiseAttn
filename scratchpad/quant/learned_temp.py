import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import marimo
    import matplotlib.pyplot as plt
    torch.manual_seed(10)
    return marimo, plt, torch


@app.cell
def _(torch):
    def quantize_weight(x,b=2,e=-8):
        e = torch.as_tensor(e)
        b = torch.as_tensor(b)
        b = torch.relu(b)
        x_scaled = x/torch.exp2(e)
        x_clipped = torch.clip(x_scaled,min=-1*torch.exp2(b-1), max = torch.exp2(b-1)-1)
        x_round = torch.round(x_clipped)
        x_scaled_back = torch.exp2(e) * x_round
        return x_scaled_back
    return (quantize_weight,)


@app.cell
def _(torch):
    batch ,s, h = 1,10, 5 #[ i.e we have 2 attn head]
    q = torch.randn((batch,s,h))
    k = torch.randn((batch,s,h))
    v = torch.randn((batch,s,h))
    return batch, h, k, q, s, v


@app.cell
def _(quantize_weight, torch):
    # Efficient implementation equivalent to the following:

    def scaled_dot_product_attention_impl1(query, key, value,b=2,e=8, attn_mask=None,other_head=0.2) -> torch.Tensor:
        # single head of attn (for now we will also assume b =1)
        # q,j,v would be of shape b,seq_len,head_size

        q = quantize_weight(query,b,e)
        k = quantize_weight(key,b,e)
        v = quantize_weight(value,b,e)
        temp = torch.as_tensor(b/(b+other_head + 1e-4))
        print(f"{temp=}")



        head_size = torch.as_tensor(q.size(-1))

        scale_factor = 1 / torch.sqrt(head_size)

        # b,s,h @ b,h,s -> b, s^2 [quadratic in seq len]
        score = q @ k.transpose(-2, -1) * scale_factor * temp
        attn_weight = torch.softmax(score, dim=-1)
        out = attn_weight @ (v)
        # out = attn_weight @ v
        print("out_var",torch.var(out))
        return out , score
    return (scaled_dot_product_attention_impl1,)


@app.cell(hide_code=True)
def _(marimo):
    other_head = marimo.ui.slider(start=0,stop=10,step=1)
    bit_slider = marimo.ui.slider(start=0,stop=8,step=0.5)
    exp_slider = marimo.ui.slider(start=-8,stop=-3,step=1)
    marimo.md(f"""
    other_head:{other_head}
    depth_bit:{bit_slider}
    exp_bit:{exp_slider}
    """)
    return bit_slider, exp_slider, other_head


@app.cell(hide_code=True)
def _(
    bit_slider,
    exp_slider,
    k,
    other_head,
    plt,
    q,
    scaled_dot_product_attention_impl1,
    v,
):
    sdp, sdp_score = scaled_dot_product_attention_impl1(
        q,k,v,
        b = bit_slider.value,
        e = exp_slider.value,
        other_head=other_head.value)
    sdp , sdp_score = sdp.squeeze(0) , sdp_score.squeeze(0) # squeeze out the batchdim


    fig,ax = plt.subplots(nrows=1,ncols=2)
    score_img = ax[0].imshow(sdp_score,cmap="Blues")
    sdp_img = ax[1].imshow(sdp,cmap="Blues")
    plt.show()
    plt.tight_layout()
    return ax, fig, score_img, sdp, sdp_img, sdp_score


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
