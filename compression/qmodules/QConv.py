import torch
from .quant_func import STERound
steRound = STERound.apply


class Qconv(torch.nn.Module):
    """
    weight tying exp_bits and depth_bits
    note number  of output  channels  is number of filterkernels  launched
    we will try to not just compress  but take out entire filter kernels...
    these asserts them to be pytorch tensors
    """
    def  __init__ (self,in_channels,
                   out_channels,
                   kernel_size=3,
                   b=2.0,e=-8.0,
                   bias = True,
                   padding=0,
                   stride=1,
                   groups=1,
                   dillation = 1,
                   padding_mode='zeros'):
        super().__init__()
        # doing this makes fake tensor
        in_channels = torch.as_tensor(in_channels)
        out_channels = torch.as_tensor(out_channels)
        kernel_size = torch.as_tensor(kernel_size)

        # in_channels = torch.tensor(in_channels)
        # out_channels = torch.tensor(out_channels)
        # kernel_size = torch.tensor(kernel_size)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.dilation = dillation
        self.padding_mode = padding_mode

        b = torch.as_tensor(b)
        e= torch.as_tensor(e)

        # fan_in is just in_channels
        weight_scale = 1/ torch.sqrt(in_channels*out_channels*out_channels)
        self.weight = torch.ones(out_channels,in_channels//groups,kernel_size,kernel_size)
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
        # [Remember] torch.prod is a reduction on product operation.. not a scan op.
        size = torch.prod(prods) *  torch.sum(torch.relu(self.depth_bit))
        return size

    def _fakebits(self):
        return torch.sum(
            torch.exp2(torch.relu(torch.ceil(self.depth_bit))) * torch.prod(torch.as_tensor(self.weight.shape[1:]))
            ) 

    def _quantized_weight(self):
        b = torch.relu(self.depth_bit)
        x_upscaled = self.weight/torch.exp2(self.exp_bit)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = steRound(x_clipped)
        return torch.exp2(self.exp_bit) * x_round



    def __call__(self,x):
        # quantize every forward pass
        W = self._quantized_weight()
        # assert self.weight.shape==W.shape
        # valid padding or should we do same.. paper does not say
        return torch.nn.functional.conv2d(x,W,
                                          bias=self.bias,
                                          padding=self.padding,
                                          stride=self.stride,
                                          groups=self.groups,
                                          dilation=self.dilation
                                          )

if __name__ == '__main__':
    m = Qconv(3,8)
    print(m.size_layer())
