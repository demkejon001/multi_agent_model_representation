from typing import List, Union

import torch
import torch.nn as nn

from tommas.helper_code.conv import make_resnet


class PredictionNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_resnet_blocks: int,
                 resnet_channels: Union[int, List[int]]):
        super(PredictionNet, self).__init__()
        self.resnet = make_resnet(in_channels, num_resnet_blocks, resnet_channels)

    def forward(self, query, embeddings: Union[tuple, list]):
        return self.resnet(torch.cat((query, *embeddings), dim=1))


class ActionPredictionNet(nn.Module):
    def __init__(self, in_channels, out_channels, world_dim, num_actions):
        super(ActionPredictionNet, self).__init__()
        self.conv_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.AvgPool2d(2)
                                      )

        _, channels, row, col = self.conv_net(torch.zeros((1, in_channels, world_dim[0], world_dim[1]))).shape
        self.fc_net = nn.Linear(channels * row * col, num_actions)

    def forward(self, mental_ouput):
        return self.fc_net(torch.flatten(self.conv_net(mental_ouput), start_dim=1))


class GoalPredictionNet(nn.Module):
    def __init__(self, in_channels, out_channels, world_dim, num_goals):
        super(GoalPredictionNet, self).__init__()
        self.conv_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.AvgPool2d(2)
                                      )

        _, channels, row, col = self.conv_net(torch.zeros((1, in_channels, world_dim[0], world_dim[1]))).shape
        self.fc_net = nn.Linear(channels * row * col, num_goals)

    def forward(self, mental_ouput):
        return self.fc_net(torch.flatten(self.conv_net(mental_ouput), start_dim=1))


class SRPredictionNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SRPredictionNet, self).__init__()
        self.conv_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, 3, 3, 1, 1)
                                      )

    def forward(self, mental_ouput):
        return self.conv_net(mental_ouput)
