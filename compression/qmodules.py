import torch
import matplotlib.pyplot as plt
import einops

from quant_func import STERound
from tqdm import tqdm
import sympy as sp

# aliases
torch.steRound = STERound.apply
torch.gelu = torch.nn.functional.gelu
torch.mul_reduce = torch.prod

torch.manual_seed(10)

class Qconv(torch.nn.Module):
    def  __init__ (self,in_channels,out_channels,kernel_size=3,b=2.0,e=-8.0,):
        """
        # weight tying exp_bits and depth_bits
        # note number  of output  channels  is number of filterkernels  launched
        # we will try to not just compress precision but take out entire filter kernels if needed.
        """
        super().__init__()
       
       # these asserts them to be pytorch tensors
        in_channels = torch.as_tensor(in_channels)
        out_channels = torch.as_tensor(out_channels)
        kernel_size = torch.as_tensor(kernel_size)
        b = torch.as_tensor(b)
        e= torch.as_tensor(e)

        # fan_in is just in_channels
        weight_scale = 1/ torch.sqrt(in_channels*out_channels*out_channels)
        self.weight = torch.ones(out_channels,in_channels,kernel_size,kernel_size)
        self.weight = self.weight.uniform_(-weight_scale,weight_scale)

        # 1 for each kernel (out_channel).. to perform safe broadcasting we fill the rest of them with 1
        self.exp_bit = torch.ones(out_channels,1,1,1)*e
        self.depth_bit = torch.ones(out_channels,1,1,1)*b

        # exp and depth also as trainables
        self.weight = torch.nn.Parameter(self.weight)
        self.exp_bit = torch.nn.Parameter(self.exp_bit)
        self.depth_bit = torch.nn.Parameter(self.depth_bit)
        ...

    def size_layer(self):
        """
        given by equation 4 : I*H*W * sum(b(i,l)
        Where O , I , H and W are the output, input, height, and
        width dimensions (so shape) of the weight tensor of layer l respec-
        tively, and b(i,l) is the bit depth of output channel i of layer l.
        """
        prods = torch.as_tensor(self.weight.shape[1:])
        size = torch.mul_reduce(prods) *  torch.sum(torch.relu(self.depth_bit))
        return size

    def _quantized_weight(self):
        b = torch.relu(self.depth_bit)
        x_upscaled = self.weight/torch.exp2(self.exp_bit)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = torch.steRound(x_clipped)
        return torch.exp2(self.exp_bit) * x_round


    def __call__(self,x):
        # quantize every forward pass
        W = self._quantized_weight()
        # assert self.weight.shape==W.shape
        # valid padding or should we do same..
        return torch.nn.functional.conv2d(x,W,padding=1)
        
        
