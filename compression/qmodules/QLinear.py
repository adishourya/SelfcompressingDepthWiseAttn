import torch
from .quant_func import STERound
steRound = STERound.apply

class QlinearMLP(torch.nn.Module):
    """
    Linear layer .. we will only use this to make mlp head and not attention weights
    tie each column to exp and depth bit. or else (m*n) weight would make (m*n)*3 trainable weights.
    i.e we will try to prune column weights
    """

    def __init__(self,m:int,n:int,b=2.0,e=-8.0,bias=True):
        self.m , self.n = m,n
        self.bias = bias
        super().__init__()
        b = torch.as_tensor(b)
        e = torch.as_tensor(e)
        # note that torch.nn.Linear(12,8).weight has a shape of 8,12
        # this is already a nn.Parameter.. no need to wrap
        self.linear = torch.nn.Linear(m,n)
        # print(self.linear)
        self.depth_bit = torch.ones(1,m) * b
        self.exp_bit = torch.ones(1,m) * e

        self.depth_bit = torch.nn.Parameter(self.depth_bit)
        self.exp_bit = torch.nn.Parameter(self.exp_bit)

    def size_layer(self):
        # sum of depth bits in rows.
        return torch.sum(torch.relu(self.depth_bit)) / (self.m* self.n)

    def _fakebits(self):
        return torch.sum(torch.exp2(torch.relu(torch.ceil(self.depth_bit))) * self.n)

    def _quantized_weight(self):
        b = torch.relu(self.depth_bit)
        # print(self.linear.weight.shape , self.exp_bit.shape) # should be (m,n) * (1,n)
        x_upscaled = self.linear.weight/torch.exp2(self.exp_bit)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = steRound(x_clipped)
        return torch.exp2(self.exp_bit) * x_round

    def __call__(self,x):
        # quantize weight every forward pass
        W = self._quantized_weight()
        bias = self.linear.bias * (self.bias * 1)
        return torch.nn.functional.linear(x,W,bias=bias)
        
        
class QlinearHead(torch.nn.Module):
    """
    use this only for attention head.
    tie one bit depth per head
    """

    def __init__(self,m:int,n:int,b=2.0,e=-8.0):
        self.m , self.n = m,n
        super().__init__()
        b = torch.as_tensor(b)
        e = torch.as_tensor(e)
        # note that torch.nn.Linear(12,8).weight has a shape of 8,12
        # this is already a nn.Parameter.. no need to wrap
        self.linear = torch.nn.Linear(m,n)
        # print(self.linear)
        self.depth_bit = b
        self.exp_bit = e

        self.depth_bit = torch.nn.Parameter(self.depth_bit)
        self.exp_bit = torch.nn.Parameter(self.exp_bit)

    def size_layer(self):
        # sum of depth bits in rows.
        return torch.relu(self.depth_bit) * self.m * self.n

    def _fakebits(self):
        return torch.sum(torch.exp2(torch.relu(torch.ceil(self.depth_bit))) *self.m *self.n)

    def _quantized_weight(self):
        b = torch.relu(self.depth_bit)
        # print(self.linear.weight.shape , self.exp_bit.shape) # should be (m,n) * (1,n)
        x_upscaled = self.linear.weight/torch.exp2(self.exp_bit)
        half = torch.exp2(b -1)
        x_clipped = torch.clip(x_upscaled,-1*half,half-1)
        x_round = steRound(x_clipped)
        return torch.exp2(self.exp_bit) * x_round

    def __call__(self,x):
        # quantize weight every forward pass
        W = self._quantized_weight()
        return torch.nn.functional.linear(x,W)
