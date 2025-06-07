import torch

class STERound(torch.autograd.Function):
    """
    same gradient as parent node. Also called as pass through gradient.
    """
    @staticmethod
    def forward(ctx,x):
        return torch.round(x)

    @staticmethod
    def backward(ctx,upstream):
        """
        local jacobian is 1.
        """
        return upstream

