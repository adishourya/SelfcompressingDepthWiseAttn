import torch
from qmodules.QConv import Qconv
from qmodules.QLinear import QlinearMLP

class QconvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Qconv(1, 16,b=2)
        self.conv2 = Qconv(16,16,b=2)
        self.conv3 = Qconv(16,32,b=1.5,e=-7)
        self.conv4 = Qconv(32,32,b=1.3,e=-6)
        
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.L1 = QlinearMLP(32*7*7, 32)
        # L2 will never be pruned...
        self.L2 = QlinearMLP(32, 10)


    def forward(self, x):
        # (2 convolutions -> bn -> pool) * 2 
        conv1_out = torch.nn.functional.gelu(self.conv1(x))
        conv2_out = torch.nn.functional.gelu(self.conv2(conv1_out))
        conv2_out = self.bn1(conv2_out)
        pool1 = self.pool(conv2_out)

        conv3_out = torch.nn.functional.gelu(self.conv3(pool1))
        conv4_out = torch.nn.functional.gelu(self.conv4(conv3_out))
        conv4_out = self.bn2(conv4_out)
        pool2 = self.pool(conv4_out)

        # then linear layers
        flat = torch.flatten(pool2, 1) # flatten except batch
        l1 = torch.nn.functional.gelu(self.L1(flat))
        l2 = torch.nn.functional.gelu(self.L2(l1))
        logits = l2
        return logits

