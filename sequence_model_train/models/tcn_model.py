import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_channels,
                 kernel_size,
                 dropout):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.tcn = nn.Sequential(
            *[weight_norm(nn.Conv1d(in_channels=num_channels[i-1],
                                    out_channels=num_channels[i],
                                    kernel_size=kernel_size,
                                    dilation=2**(i-1)))
              for i in range(1, len(num_channels))])
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: [batch_size, input_size, input_steps]
        x = x.permute(0, 2, 1)
        # x: [batch_size, input_steps, input_size]
        y = self.tcn(x)
        # y: [batch_size, output_channels[-1], input_steps - kernel_size + 1]
        y = y[:, :, -1]
        # y: [batch_size, output_channels[-1]]
        y = self.linear(y)
        # y: [batch_size, output_size]
        return y
