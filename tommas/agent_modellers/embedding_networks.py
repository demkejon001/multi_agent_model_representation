import numpy as np
from typing import Union, Tuple, List

import torch
import torch.nn as nn

from tommas.helper_code.conv import ConvLSTM, make_resnet


lstm_types = ['conv_lstm', 'lstm', 'downsample_lstm']
pool_types = [None, 'avg', 'max']

base_agent_embedding_specs = {
    "embedding_size": 4,
    "flatten_embedding": False,
    "num_resnet_blocks": 2,
    "resnet_channels": 16,
    "recurrent_hidden_size": 4,
    "lstm_type": "conv_lstm",
    "pooling": "avg",
    "pre_resnet_layer": None,
}


class IterativeActionEmbeddingNetwork(nn.Module):
    def __init__(self, in_features: int, layer_features: Union[List[int], None], embedding_size: int,
                 batch_first=False, n_layer=1):
        super().__init__()
        self.embedding_size = embedding_size
        if layer_features is None:
            layer_features = []
        all_layers_features = [in_features] + layer_features
        layers = []
        for i in range(len(all_layers_features)-1):
            layers.append(nn.Linear(all_layers_features[i], all_layers_features[i+1]))
            if i < len(all_layers_features) - 1:
                layers.append(nn.ReLU())
        self.fc_net = nn.Sequential(*layers)
        self.lstm = nn.LSTM(all_layers_features[-1], embedding_size, batch_first=batch_first, num_layers=n_layer)

    def forward(self, trajectory, return_all_layers=False):
        if return_all_layers:
            return self.lstm(self.fc_net(trajectory))[0].reshape((-1, self.embedding_size))
        else:
            return self.lstm(self.fc_net(trajectory))[0][-1]  # return last hidden state

    def get_empty_agent_embedding(self, batch_size):
        return torch.zeros((batch_size, self.embedding_size))


class AgentEmbeddingNetwork(nn.Module):
    def __init__(self, in_channels: int, world_dim: Tuple[int, int], embedding_size: int, flatten_embedding: bool,
                 num_resnet_blocks: int, resnet_channels: Union[int, List[int]], recurrent_hidden_size: int,
                 lstm_type: str = 'conv_lstm', pooling: Union[str, None] = None,
                 pre_resnet_layer: Union[Tuple[int, int, int, int], None] = None, batch_first: bool = False):
        # world_dim should be (row, col)
        super(AgentEmbeddingNetwork, self).__init__()
        if isinstance(resnet_channels, list):
            self.recurrent_input_size = resnet_channels[-1]
        else:
            self.recurrent_input_size = resnet_channels
        self.recurrent_hidden_size = recurrent_hidden_size
        self.embedding_size = embedding_size
        self.batch_first = batch_first

        self.gridworld_net = GridworldNetwork(in_channels, resnet_channels, num_resnet_blocks, pooling,
                                              pre_resnet_layer)

        zeros = torch.zeros((1, in_channels, world_dim[0], world_dim[1]))
        gridworld_features = self.gridworld_net(zeros).unsqueeze(0)
        _, _, _, hidden_row, hidden_col = gridworld_features.shape
        self.hidden_input_dim = (hidden_row, hidden_col)
        if lstm_type == "conv_lstm":
            self.recurrent_net = ConvLSTMGridworldTrajectoryNetwork(self.recurrent_input_size, self.hidden_input_dim,
                                                                    embedding_size, flatten_embedding,
                                                                    recurrent_hidden_size)
        elif lstm_type == "lstm":
            self.recurrent_net = LSTMGridworldTrajectoryNetwork(self.recurrent_input_size, self.hidden_input_dim,
                                                                embedding_size, flatten_embedding,
                                                                recurrent_hidden_size, world_dim)
        elif lstm_type == "downsample_lstm":
            self.recurrent_net = DownSampledLSTMGridworldTrajectoryNetwork(self.recurrent_input_size,
                                                                           self.hidden_input_dim,
                                                                           embedding_size, flatten_embedding,
                                                                           recurrent_hidden_size, world_dim)
        else:
            raise ValueError('lstm_type=%s does not exist. Choose from options %s' % (lstm_type, lstm_types))
        self._embedding_shape = tuple(self.recurrent_net(gridworld_features).shape[1:])

    def get_empty_agent_embedding(self, batch_size):
        return torch.zeros((batch_size, *self._embedding_shape))

    def forward(self, trajectories):
        if self.batch_first:
            trajectories = trajectories.permute(1, 0, 2, 3, 4)
        seq_len, batch_size, features, row, col = trajectories.shape
        trajectories = trajectories.reshape(seq_len * batch_size, features, row, col)
        preprocessed_traj = self.gridworld_net(trajectories)
        preprocessed_traj = preprocessed_traj.reshape(seq_len, batch_size, self.recurrent_input_size,
                                                      self.hidden_input_dim[0], self.hidden_input_dim[1])
        return self.recurrent_net(preprocessed_traj)


class AgentEmbeddingTransformerNetwork(AgentEmbeddingNetwork):
    def __init__(self, in_channels: int, world_dim: Tuple[int, int], embedding_size: int, flatten_embedding: bool,
                 num_resnet_blocks: int, resnet_channels: Union[int, List[int]], recurrent_hidden_size: int,
                 lstm_type: str = 'conv_lstm', pooling: Union[str, None] = None,
                 pre_resnet_layer: Union[Tuple[int, int, int, int], None] = None, batch_first: bool = False):
        # world_dim should be (row, col)
        super(AgentEmbeddingNetwork, self).__init__()
        if isinstance(resnet_channels, list):
            self.recurrent_input_size = resnet_channels[-1]
        else:
            self.recurrent_input_size = resnet_channels
        self.recurrent_hidden_size = recurrent_hidden_size
        self.embedding_size = embedding_size
        self.batch_first = batch_first

        self.gridworld_net = GridworldNetwork(in_channels, resnet_channels, num_resnet_blocks, pooling,
                                              pre_resnet_layer)

        zeros = torch.zeros((1, in_channels, world_dim[0], world_dim[1]))
        gridworld_features = self.gridworld_net(zeros).unsqueeze(0)
        _, _, features, row, col = gridworld_features.shape
        d_model = features * row * col
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=2, dim_feedforward=128)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2, norm=None)
        self.embedding_layer = nn.Linear(d_model, self.embedding_size)

        #self._embedding_shape = tuple(self.recurrent_net(gridworld_features).shape[1:])
        self._embedding_shape = (self.embedding_size,)

    def forward(self, trajectories):
        if not self.batch_first:
            trajectories = trajectories.permute(1, 0, 2, 3, 4)
        batch_size, seq_len, features, row, col = trajectories.shape
        trajectories = trajectories.reshape(seq_len * batch_size, features, row, col)
        preprocessed_traj = self.gridworld_net(trajectories)
        preprocessed_traj = preprocessed_traj.reshape(batch_size, seq_len, -1)
        return self.embedding_layer(self.decoder(preprocessed_traj, preprocessed_traj)[:, -1])


class GridworldNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 resnet_channels: Union[int, List[int]],
                 num_resnet_blocks: int,
                 pooling: Union[str, None] = None,
                 pre_resnet_layer: Union[Tuple[int, int, int, int], None] = None):
        # world_dim should be (row, col)
        super(GridworldNetwork, self).__init__()
        resnet_in_channels = in_channels
        if pre_resnet_layer is not None:
            resnet_in_channels, kernel, stride, padding = pre_resnet_layer
            self.pre_resnet = nn.Conv2d(in_channels, resnet_in_channels, kernel, stride=stride, padding=padding)
        else:
            self.pre_resnet = nn.Sequential()
        self.resnet = make_resnet(resnet_in_channels, num_resnet_blocks=num_resnet_blocks,
                                  resnet_channels=resnet_channels)
        if pooling is None:
            self.pool = nn.Sequential()
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(2)
        elif pooling == 'max':
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError('pooling=%s does not exist. Choose from options %s' % (pooling, pool_types))

    def forward(self, gridworld_states):
        return self.pool(self.resnet(self.pre_resnet(gridworld_states)))


class RecurrentGridworldTrajectoryNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: Tuple[int, int],
                 embedding_size: int,
                 flatten_embedding: bool,
                 hidden_output_size: int):
        super(RecurrentGridworldTrajectoryNetwork, self).__init__()

    def forward(self, preprocessed_trajectories):
        raise NotImplementedError


class ConvLSTMGridworldTrajectoryNetwork(RecurrentGridworldTrajectoryNetwork):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: Tuple[int, int],
                 embedding_size: int,
                 flatten_embedding: bool,
                 hidden_output_size: int):
        super(ConvLSTMGridworldTrajectoryNetwork, self).__init__(in_channels, hidden_dim, embedding_size,
                                                                 flatten_embedding, hidden_output_size)
        if flatten_embedding:
            self.lstm_net = ConvLSTM(in_channels, hidden_output_size)
            self.final_net = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(int(np.prod((*hidden_dim, hidden_output_size))), embedding_size)
            )
        else:
            self.lstm_net = ConvLSTM(in_channels, embedding_size)
            self.final_net = None

    def forward(self, preprocessed_trajectories):
        output, _ = self.lstm_net(preprocessed_trajectories)
        output.squeeze_(0)
        if self.final_net is None:
            return output
        else:
            return self.final_net(output)


class LSTMGridworldTrajectoryNetwork(RecurrentGridworldTrajectoryNetwork):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: Tuple[int, int],
                 embedding_size: int,
                 flatten_embedding: bool,
                 hidden_output_size: int,
                 embedding_dim: Union[Tuple[int, int], None] = None):
        super(LSTMGridworldTrajectoryNetwork, self).__init__(in_channels, hidden_dim, embedding_size, flatten_embedding,
                                                             hidden_output_size)
        if flatten_embedding:
            self.lstm_net = nn.Sequential(
                nn.Flatten(2),
                nn.LSTM(np.prod((in_channels, *hidden_dim)), embedding_size),
            )
            self.final_net = None
        else:
            self.lstm_net = nn.Sequential(
                nn.Flatten(2),
                nn.LSTM(np.prod((in_channels, *hidden_dim)), hidden_output_size),
            )
            if embedding_dim is not None:
                self.final_net = nn.ConvTranspose2d(hidden_output_size, embedding_size, kernel_size=embedding_dim)
            else:
                self.final_net = nn.ConvTranspose2d(hidden_output_size, embedding_size, kernel_size=hidden_dim)

    def forward(self, preprocessed_trajectories):
        output, _ = self.lstm_net(preprocessed_trajectories)
        output = output[-1]
        if self.final_net is None:
            return output
        else:
            output.unsqueeze_(-1).unsqueeze_(-1)
            return self.final_net(output)


class DownSampledLSTMGridworldTrajectoryNetwork(LSTMGridworldTrajectoryNetwork):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: Tuple[int, int],
                 embedding_size: int,
                 flatten_embedding: bool,
                 hidden_output_size: int,
                 embedding_dim: Union[Tuple[int, int], None] = None):
        super(DownSampledLSTMGridworldTrajectoryNetwork, self).__init__(in_channels, hidden_dim, embedding_size,
                                                                        flatten_embedding, hidden_output_size,
                                                                        embedding_dim)
        if flatten_embedding:
            self.lstm_net = nn.Sequential(
                nn.Flatten(2),
                nn.Linear(int(np.prod((in_channels, *hidden_dim))), in_channels),
                nn.LSTM(in_channels, embedding_size),
            )
        else:
            self.lstm_net = nn.Sequential(
                nn.Flatten(2),
                nn.Linear(int(np.prod((in_channels, *hidden_dim))), in_channels),
                nn.LSTM(in_channels, hidden_output_size),
            )




