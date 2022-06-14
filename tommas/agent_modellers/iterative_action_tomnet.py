from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

from tommas.agent_modellers.iterative_action_tommas import IterativeActionModeller, add_embeddings_to_trajectory
from tommas.agent_modellers.modeller_inputs import IterativeActionToMnetInput
from tommas.agent_modellers.modeller_outputs import IterativeTOMMASPredictions


class LSTMCharacterNet(nn.Module):
    def __init__(self, num_joint_agent_features, hidden_layer_features, embedding_size, num_layers=1):
        super().__init__()
        self.embedding_size = embedding_size
        if hidden_layer_features is None:
            hidden_layer_features = []
        all_layers_features = [num_joint_agent_features] + hidden_layer_features + [embedding_size]
        layers = []
        for i in range(len(all_layers_features)-1):
            layers.append(nn.Linear(all_layers_features[i], all_layers_features[i+1]))
            if i < len(all_layers_features) - 2:
                layers.append(nn.ReLU())
        self.embedding_net = nn.Sequential(*layers)
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers=num_layers, batch_first=True)

    def forward(self, trajectory, hidden_state_indices=None):
        return self.lstm(self.embedding_net(trajectory))[0][:, hidden_state_indices]

    def get_empty_agent_embedding(self, batch_size):
        return torch.zeros((batch_size, self.embedding_size))


class TTXCharacterNet(nn.Module):
    def __init__(self, num_joint_agent_features, hidden_layer_features, embedding_size, n_layer, n_head):
        super().__init__()
        self.embedding_size = embedding_size
        if hidden_layer_features is None:
            hidden_layer_features = []
        all_layers_features = [num_joint_agent_features] + hidden_layer_features + [embedding_size]
        layers = []
        for i in range(len(all_layers_features) - 1):
            layers.append(nn.Linear(all_layers_features[i], all_layers_features[i + 1]))
            if i < len(all_layers_features) - 2:
                layers.append(nn.ReLU())
        self.embedding_net = nn.Sequential(*layers)
        gpt2config = GPT2Config(vocab_size=2, n_positions=1024, n_ctx=1024, n_embd=embedding_size, n_layer=n_layer,
                                n_head=n_head, n_inner=None, activation_function='gelu')
        decoder = GPT2Model(gpt2config)
        self.decoder = decoder

    def forward(self, trajectory, hidden_state_indices=None):
        input_embeddings = self.embedding_net(trajectory)
        decoder_output = self.decoder(inputs_embeds=input_embeddings, attention_mask=None)
        return decoder_output.last_hidden_state[:, hidden_state_indices]

    def get_empty_agent_embedding(self, batch_size):
        return torch.zeros((batch_size, self.embedding_size))


# class LSTMMentalNet(LSTMCharacterNet):
#     def forward(self, trajectory):
#         traj_embedding = self.embedding_net(trajectory)
#         return self.lstm(traj_embedding)[0].reshape((-1, self.embedding_size))
#
#
# class TTXMentalNet(TTXCharacterNet):
#     def forward(self, trajectory):
#         input_embeddings = self.embedding_net(trajectory)
#         batch_size, seq_len, _ = trajectory.shape
#         decoder_output = self.decoder(inputs_embeds=input_embeddings, attention_mask=None)
#         return decoder_output.last_hidden_state.reshape((seq_len * batch_size, self.embedding_size))
class LSTMMentalNet(LSTMCharacterNet):
    def forward(self, trajectory, num_trajectories):  # num_trajectories = (n_past + 1)
        if num_trajectories > 1:
            batch_size, multi_seq_len, _ = trajectory.shape
            single_seq_len = multi_seq_len // num_trajectories
            trajectory = trajectory.reshape((batch_size * num_trajectories, single_seq_len, -1))
        traj_embedding = self.embedding_net(trajectory)
        return self.lstm(traj_embedding)[0].reshape((-1, self.embedding_size))


class TTXMentalNet(TTXCharacterNet):
    def forward(self, trajectory, num_trajectories):  # num_trajectories = (n_past + 1)
        if num_trajectories > 1:
            batch_size, multi_seq_len, _ = trajectory.shape
            single_seq_len = multi_seq_len // num_trajectories
            trajectory = trajectory.reshape((batch_size * num_trajectories, single_seq_len, -1))
        input_embeddings = self.embedding_net(trajectory)
        batch_size, seq_len, _ = trajectory.shape
        decoder_output = self.decoder(inputs_embeds=input_embeddings, attention_mask=None)
        return decoder_output.last_hidden_state.reshape((seq_len * batch_size, self.embedding_size))


class IterativeActionPastCurrentNet(IterativeActionModeller):
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
                         optimizer_type=optimizer_type)
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
        # self.action_pred_net = nn.Sequential(
        #     nn.Linear(embedding_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, num_actions),
        # )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IterativeActionToMnet")
        parser = IterativeActionModeller.add_model_specific_args(parser)
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
        # tiled_embedding = tiled_embedding.reshape((batch_size * num_trajectories, single_seq_len, -1))
        # current_trajectory = current_trajectory.reshape((batch_size * num_trajectories, single_seq_len, -1))
        input_embeddings = torch.cat((current_trajectory, tiled_embedding), dim=2)

        mental_embeddings = self.mental_net(input_embeddings, num_trajectories)
        # print(mental_embeddings)
        # raise ValueError
        predictions = IterativeTOMMASPredictions(self.action_pred_net(mental_embeddings))
        if return_embeddings:
            predictions.embeddings = (char_embeddings, mental_embeddings)
        return predictions
