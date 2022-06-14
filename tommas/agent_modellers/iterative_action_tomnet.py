from argparse import ArgumentParser

import torch
import torch.nn as nn

from tommas.agent_modellers.tomnet_base import AgentModeller, LSTMCharacterNet, LSTMMentalNet, TTXCharacterNet, \
    TTXMentalNet
from tommas.agent_modellers.modeller_inputs import IterativeActionToMnetInput
from tommas.agent_modellers.modeller_outputs import IterativeTOMMASPredictions


class IterativeActionPastCurrentNet(AgentModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_joint_agent_features: int,
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
                 num_actions: int,
                 optimizer_type: str = "adam",
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         optimizer_type=optimizer_type, no_action_loss=False, no_goal_loss=True, no_sr_loss=True)
        self.save_hyperparameters()

        if lstm_char:
            self.character_net = LSTMCharacterNet(num_joint_agent_features, char_hidden_layer_features, char_embedding_size, char_n_layer)
        else:
            self.character_net = TTXCharacterNet(num_joint_agent_features, char_hidden_layer_features, char_embedding_size, char_n_layer, char_n_head)

        num_mental_features = char_embedding_size + num_joint_agent_features
        if lstm_mental:
            self.mental_net = LSTMMentalNet(num_mental_features, mental_hidden_layer_features, mental_embedding_size, mental_n_layer)
        else:
            self.mental_net = TTXMentalNet(num_mental_features, mental_hidden_layer_features, mental_embedding_size, mental_n_layer, mental_n_head)

        self.action_pred_net = nn.Linear(mental_embedding_size, num_actions)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IterativeActionToMnet")
        parser = AgentModeller.add_model_specific_args(parser)
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
        return parent_parser

    def forward(self, x: IterativeActionToMnetInput, return_embeddings=False):
        past_trajectories = x.past_trajectories
        current_trajectory = x.current_trajectory
        hidden_state_indices = x.hidden_state_indices
        batch_size, multi_seq_len, features = current_trajectory.shape
        char_embeddings = self.character_net(past_trajectories, hidden_state_indices=hidden_state_indices)

        num_trajectories = char_embeddings.shape[1]
        single_seq_len = multi_seq_len // num_trajectories
        tiled_embedding = torch.repeat_interleave(char_embeddings, single_seq_len, dim=1)
        input_embeddings = torch.cat((current_trajectory, tiled_embedding), dim=2)

        mental_embeddings = self.mental_net(input_embeddings, num_trajectories)
        predictions = IterativeTOMMASPredictions(self.action_pred_net(mental_embeddings))
        if return_embeddings:
            predictions.embeddings = (char_embeddings, mental_embeddings)
        return predictions
