from argparse import ArgumentParser

import torch
import torch.nn as nn

from tommas.agent_modellers.modeller_inputs import GridworldToMnetInput
from tommas.agent_modellers.modeller_outputs import TOMMASPredictions
from tommas.helper_code.conv import Conv2DRes
from tommas.agent_modellers.tomnet_base import AgentModeller, LSTMCharacterNet, LSTMMentalNet, TTXCharacterNet, \
    TTXMentalNet


class GridworldNetwork(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        super(GridworldNetwork, self).__init__()
        self.resnet = nn.Sequential(
            Conv2DRes(in_channels, 64, padding=1),
            Conv2DRes(64, 64, padding=1),
            nn.MaxPool2d((3, 3), 2),
            Conv2DRes(64, 32, padding=1),
            Conv2DRes(32, 3, padding=1),
        )
        resnet_output = torch.flatten(self.resnet(torch.zeros((1, in_channels, 21, 21))), start_dim=1).shape[1]
        self.fc1 = nn.Linear(resnet_output, out_features)

    def forward(self, gridworld_states):
        batch, seq_len, features, row, col = gridworld_states.shape
        resnet_output = torch.flatten(self.resnet(gridworld_states.reshape(batch*seq_len, features, row, col)), start_dim=1)
        return self.fc1(resnet_output).reshape(batch, seq_len, -1)


class GridworldToMNet(AgentModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_gridworld_features: int,
                 gridworld_embedding_size: int,
                 action_embedding_size: int,
                 lstm_char: bool,
                 lstm_mental: bool,
                 char_hidden_layer_features: int,
                 char_embedding_size: int,
                 char_n_head: int,
                 char_n_layer: int,
                 mental_hidden_layer_features: int,
                 mental_embedding_size: int,
                 mental_n_head: int,
                 mental_n_layer: int,
                 pred_net_features: int,
                 num_agents: int,
                 num_actions: int,
                 num_goals: int,
                 no_action_loss=False,
                 no_goal_loss=False,
                 no_sr_loss=False,
                 optimizer_type: str = "adam",
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         optimizer_type=optimizer_type,
                         no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        self.save_hyperparameters()

        self.gridworld_net = GridworldNetwork(num_gridworld_features, gridworld_embedding_size)
        self.action_net = nn.Linear(num_actions * num_agents, action_embedding_size)
        state_joint_action_size = gridworld_embedding_size + action_embedding_size
        if lstm_char:
            self.character_net = LSTMCharacterNet(state_joint_action_size, char_hidden_layer_features, char_embedding_size, char_n_layer)
        else:
            self.character_net = TTXCharacterNet(state_joint_action_size, char_hidden_layer_features, char_embedding_size, char_n_layer, char_n_head)

        num_mental_features = state_joint_action_size + char_embedding_size
        if lstm_mental:
            self.mental_net = LSTMMentalNet(num_mental_features, mental_hidden_layer_features, mental_embedding_size, mental_n_layer)
        else:
            self.mental_net = TTXMentalNet(num_mental_features, mental_hidden_layer_features, mental_embedding_size, mental_n_layer, mental_n_head)

        self.pred_net = nn.Sequential(nn.Linear(gridworld_embedding_size + mental_embedding_size + char_embedding_size, pred_net_features),
                                      nn.ReLU(),
                                      nn.Linear(pred_net_features, pred_net_features))
        self.action_pred_net = nn.Linear(pred_net_features, num_actions)
        self.goal_pred_net = nn.Linear(pred_net_features, num_goals)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("GridworldToMnet")
        parser = AgentModeller.add_model_specific_args(parser)
        parser.add_argument("--gridworld_embedding_size", type=int, default=4)
        parser.add_argument("--action_embedding_size", type=int, default=2)
        parser.add_argument("--lstm_char", action="store_true", default=False)
        parser.add_argument("--lstm_mental", action="store_true", default=False)
        parser.add_argument("--char_hidden_layer_features", nargs='+', type=int, default=[],
                            help="the number of features for the embedding network's hidden layers (default: None)")
        parser.add_argument("--char_embedding_size", type=int, default=2)
        parser.add_argument("--char_n_layer", type=int, default=2)
        parser.add_argument("--char_n_head", type=int, default=2)
        parser.add_argument("--mental_hidden_layer_features", nargs='+', type=int, default=[],
                            help="the number of features for the embedding network's hidden layers (default: None)")
        parser.add_argument("--mental_embedding_size", type=int, default=2)
        parser.add_argument("--mental_n_layer", type=int, default=2)
        parser.add_argument("--mental_n_head", type=int, default=2)
        parser.add_argument("--pred_net_features", type=int, default=2)
        return parent_parser

    def forward(self, x: GridworldToMnetInput, return_embeddings=False):
        past_trajectories = x.past_trajectories
        past_actions = x.past_actions
        current_trajectory = x.current_trajectory
        current_actions = x.current_actions
        hidden_state_indices = x.hidden_state_indices

        char_net_input = torch.cat((self.gridworld_net(past_trajectories), self.action_net(past_actions)), dim=2)
        char_embeddings = self.character_net(char_net_input, hidden_state_indices=hidden_state_indices)

        num_trajectories = char_embeddings.shape[1]
        single_seq_len = current_trajectory.shape[1] - 1
        interleaved_embedding = torch.repeat_interleave(char_embeddings, single_seq_len, dim=1)

        current_trajectory = self.gridworld_net(current_trajectory)

        # don't include last state-action because it isn't used for the mental embeddings
        current_traj = current_trajectory[:, :-1]
        current_actions = self.action_net(current_actions[:, :-1])
        if num_trajectories == 1:
            mental_net_input = torch.cat((current_traj, current_actions, interleaved_embedding), dim=2)
        else:
            current_state_action = torch.cat((current_traj, current_actions), dim=2).repeat((1, num_trajectories, 1))
            mental_net_input = torch.cat((current_state_action, interleaved_embedding), dim=2)

        mental_embeddings = self.mental_net(mental_net_input, num_trajectories)
        pred_shape = (mental_embeddings.shape[0], -1)

        if num_trajectories == 1:
            states = current_trajectory[:, 1:].reshape(pred_shape)
        else:
            states = current_trajectory[:, 1:].repeat((1, num_trajectories, 1)).reshape(pred_shape)
        pred_output = self.pred_net(torch.cat((states, mental_embeddings, interleaved_embedding.reshape(pred_shape)), dim=1))
        pred_action = self.action_pred_net(pred_output)
        pred_goal = self.goal_pred_net(pred_output)

        predictions = TOMMASPredictions(pred_action, pred_goal, None)

        if return_embeddings:
            predictions.embeddings = (char_embeddings, mental_embeddings)
        return predictions
