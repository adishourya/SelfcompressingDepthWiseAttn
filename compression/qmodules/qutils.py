import torch
import matplotlib.pyplot as plt
from qmodules.QConv import Qconv
import einops

@torch.no_grad
def inspect_weights(out,layer):
    assert isinstance(layer,Qconv), "Hein? should be Qconv layer"
    out = out.to("cpu")
    kernel = layer._quantized_weight().to("cpu")
    _,in_channels,k,_ = kernel.shape
    if in_channels>1:
        kernel = kernel[:,0,:,:][:,None,:,:]
    
    kernel_plot = einops.rearrange(kernel,"out_ch in_ch k1 k2 ->  (in_ch k1) (out_ch k2)")
    out_plot = einops.rearrange(out,"b c h w -> (b h) (c w)")


    plt.figure(figsize=(15,8))
    plt.imshow(kernel_plot.float(),cmap="gray")
    plt.show()
    plt.figure(figsize=(15,8))
    plt.imshow(out_plot.float(),cmap="Blues")
    plt.show()
    return out_plot,kernel_plot


def example_inspect_weight(loader,count=5):
    loader = iter(loader)
    img , _ = next(loader) 
    input_img = torch.as_tensor(img[:count]) # 1,28,28
    qconv_layer1 = Qconv(1,32)
    qconv_layer2 = Qconv(32,48)

    out1 = qconv_layer1(input_img)
    out2 = qconv_layer2(out1)

    inspect_weights(out1,qconv_layer1)
    inspect_weights(out2,qconv_layer2)

