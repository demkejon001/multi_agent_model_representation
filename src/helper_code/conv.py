from typing import Union, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, batch_first=False):
        super(ConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_first = batch_first
        self.output_order = (0, 1, 2, 3, 4)
        if self.batch_first:
            self.output_order = (1, 0, 2, 3, 4)
        self.convlstmcell = ConvLSTMCell(self.in_channels, self.hidden_channels, self.kernel_size)

    def forward(self, seq, hidden=None):
        if self.batch_first:
            seq = seq.permute(self.output_order)
        # if hx is None:
            # batch_size, _, rows, cols = seq[0].shape
            # hx = torch.zeros(batch_size, self.hidden_channels, rows, cols).to(device), \
            #      torch.zeros(batch_size, self.hidden_channels, rows, cols).to(device)
        for step in seq:
            output, hidden = self.convlstmcell(step, hidden)
            # print('seq size', seq.shape)
            # print('seq step', step.shape)
            # print('hn after step', hidden[0].shape)
            # print('cn after step', hidden[1].shape)
            # print('------------')
        output = output.unsqueeze(0).permute(self.output_order)
        # print('final output', output.shape)
        # print('final hidden', hidden[0].shape)
        # print('final context', hidden[1].shape)
        return output, hidden


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, unused_stride=1,
                 unused_padding=1):
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.Wxh_ifco = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels * 4, self.kernel_size,
                                  padding=padding)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def init_hidden(self, input):
        batch_size, _, rows, cols = input.shape
        return torch.zeros(batch_size, self.hidden_channels, rows, cols, dtype=input.dtype, device=input.device), \
               torch.zeros(batch_size, self.hidden_channels, rows, cols, dtype=input.dtype, device=input.device)

    def forward(self, x, hc=None):
        if hc is None:
            hc = self.init_hidden(x)

        h, c = hc[0], hc[1]

        if self.Wci is None:
            batch_size, rows, cols = c.shape[0], c.shape[-2], c.shape[-1]
            self.Wci = Parameter(torch.zeros(1, self.hidden_channels, rows, cols))
            self.Wcf = Parameter(torch.zeros(1, self.hidden_channels, rows, cols))
            self.Wco = Parameter(torch.zeros(1, self.hidden_channels, rows, cols))

        Wxh_i, Wxh_f, Wxh_c, Wxh_o = torch.chunk(self.Wxh_ifco(torch.cat((x, h), dim=1)), chunks=4, dim=1)

        context = torch.sigmoid(Wxh_f + self.Wcf * c) * c + torch.sigmoid(Wxh_i + self.Wci * c) * torch.tanh(Wxh_c)
        hidden = torch.sigmoid(Wxh_o + self.Wco * context) * torch.tanh(context)

        return hidden.contiguous(), (hidden, context)


class Conv2DRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Conv2DRes, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        output = self.bn2(self.conv2(self.bn1(self.relu1(self.conv1(x)))))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return self.relu2(output + identity)


def make_resnet(in_channels: int, num_resnet_blocks: int, resnet_channels: Union[int, List[int]]) -> nn.Module:
    if num_resnet_blocks <= 0:
        raise ValueError('num_resnet_blocks needs to be >0')

    if isinstance(resnet_channels, int):
        resnet_channels = [resnet_channels for _ in range(num_resnet_blocks)]

    if 0 < num_resnet_blocks < len(resnet_channels):
        raise ValueError('length of hidden_channels (%d) must be <= num_resnet_blocks (%d)' %
                         (len(resnet_channels), num_resnet_blocks))
    if len(resnet_channels) < num_resnet_blocks:
        num_additional_channels = num_resnet_blocks - len(resnet_channels)
        resnet_channels = resnet_channels + [resnet_channels[-1] for _ in range(num_additional_channels)]

    resnet_channels = [in_channels] + resnet_channels
    residual_layers = [Conv2DRes(resnet_channels[channel_idx], resnet_channels[channel_idx + 1], 3, 1, 1)
                       for channel_idx in range(num_resnet_blocks)]
    resnet = nn.Sequential(
        *residual_layers
    )
    return resnet
