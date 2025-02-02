import torch
import torch.nn as nn
from .base import BaseArch

class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv=2, kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, is_unet_decoder=False):
        super(MultiConv, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device=self.device, dtype=torch.bfloat16)
        
        in_channels = in_channels * 3 if is_unet_decoder else in_channels
        
        self.convolutions = nn.Sequential()

        for i in range(num_conv):
            self.convolutions.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, 
                         kernel_size=kernel_size,
                         stride=stride,
                         padding="same",
                         bias=bias),
            )
            self.convolutions.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.convolutions(x)
