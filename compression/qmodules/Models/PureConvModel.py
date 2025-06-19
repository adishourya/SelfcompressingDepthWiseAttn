import torch
from qmodules.QConv import Qconv

class QPureconvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Qconv(1, 32,b=2)
        self.conv2 = Qconv(32,64,b=2)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv3 = Qconv(64,32,b=1.5,e=-7)
        self.conv4 = Qconv(32,16,b=1.3,e=-6)
        self.bn2 = torch.nn.BatchNorm2d(16)
        
        self.pool = torch.nn.MaxPool2d(2, 2)

        # none of these would be pruned
        self.conv_lin = Qconv(16, 10, kernel_size=7,padding=0)



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
        # torch.Size([256, 10, 7, 7]) torch.Size([256, 490])

        flat = self.conv_lin(pool2) #B, 10, 1,1
        logits = flat.view(flat.size(0),-1) # B,10
        # print(pool2.shape, flat.shape, logits.shape)
        return logits

    def _targetModules(self):
        return (Qconv,)

