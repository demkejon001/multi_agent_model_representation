from typing import Union, List
from argparse import ArgumentParser

import torch
import torch.nn as nn

from tommas.agent_modellers.embedding_networks import IterativeActionEmbeddingNetwork
from tommas.agent_modellers.modeller_inputs import IterativeActionTOMMASInput, IterativeActionTransformerInput
from tommas.agent_modellers.modeller_outputs import IterativeTOMMASPredictions
from tommas.agent_modellers.tommas import AgentModeller


def add_embeddings_to_trajectory(trajectory: torch.Tensor, embeddings: Union[list, tuple, torch.Tensor],
                                 is_past_traj: bool, batch_size: int = 0):
    if type(embeddings) is torch.Tensor:
        tensor_embedding = embeddings
    else:
        # Each embedding should be shape (batch, embedding_size)
        tensor_embedding = torch.cat(embeddings, dim=1)

    if is_past_traj:
        seq_len = trajectory.shape[0]
        num_seq = trajectory.shape[1] // batch_size
        tiled_embedding = torch.repeat_interleave(tensor_embedding, num_seq, dim=0).unsqueeze(0)
        tiled_embedding = tiled_embedding.repeat((seq_len, 1, 1))
        return torch.cat((trajectory, tiled_embedding), dim=2)
    else:
        seq_len = trajectory.shape[0]
        tiled_embedding = tensor_embedding.unsqueeze(0)
        tiled_embedding = tiled_embedding.repeat((seq_len, 1, 1,))
        return torch.cat((trajectory, tiled_embedding), dim=2)


class IterativeActionModeller(AgentModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 optimizer_type: str = "adam",
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         optimizer_type=optimizer_type, no_action_loss=False, no_goal_loss=True, no_sr_loss=True)

    # @staticmethod
    # def add_model_specific_args(parser):
    #     parser = AgentModeller.add_model_specific_args(parser)
    #     # parser.add_argument("--learning_rate", "--lr", type=float, default=1e-2)
    #     return parser


class IterativeActionTOMMAS(IterativeActionModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_independent_agent_features: int,
                 num_joint_agent_features: int,
                 ia_char_hidden_layer_features: Union[List[int], None],
                 ia_char_embedding_size: int,
                 ia_mental_hidden_layer_features: Union[List[int], None],
                 ia_mental_embedding_size: int,
                 ja_char_hidden_layer_features: Union[List[int], None],
                 ja_char_embedding_size: int,
                 ja_mental_hidden_layer_features: Union[List[int], None],
                 ja_mental_embedding_size: int,
                 num_actions: int,
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate)
        self.save_hyperparameters()

        self.ia_char_net = IterativeActionEmbeddingNetwork(in_features=num_independent_agent_features,
                                                           layer_features=ia_char_hidden_layer_features,
                                                           embedding_size=ia_char_embedding_size)
        ia_mental_input = num_independent_agent_features + ia_char_embedding_size
        self.ia_mental_net = IterativeActionEmbeddingNetwork(in_features=ia_mental_input,
                                                             layer_features=ia_mental_hidden_layer_features,
                                                             embedding_size=ia_mental_embedding_size)
        ja_char_input = num_joint_agent_features + ia_char_embedding_size
        self.ja_char_net = IterativeActionEmbeddingNetwork(in_features=ja_char_input,
                                                           layer_features=ja_char_hidden_layer_features,
                                                           embedding_size=ja_char_embedding_size)
        ja_mental_input = num_joint_agent_features + ia_char_embedding_size + ia_mental_embedding_size + ja_char_embedding_size
        self.ja_mental_net = IterativeActionEmbeddingNetwork(in_features=ja_mental_input,
                                                             layer_features=ja_mental_hidden_layer_features,
                                                             embedding_size=ja_mental_embedding_size)
        action_pred_net_input = ia_char_embedding_size + ja_char_embedding_size + ia_mental_embedding_size + \
                                ja_mental_embedding_size
        self.action_pred_net = nn.Linear(action_pred_net_input, num_actions)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser, model_name: str = "IterativeActionTOMMAS"):
        parser = parent_parser.add_argument_group(model_name)
        parser = IterativeActionModeller.add_model_specific_args(parser)
        for network_type in ['ia_char', 'ia_mental', 'ja_char', 'ja_mental']:
            parser.add_argument('--' + network_type + '_embedding_size', type=int, default=2,
                                help="the network's embedding size (default: 2)")
            parser.add_argument('--' + network_type + '_hidden_layer_features', nargs='+', type=int, default=[],
                                help="the number of features for the network's hidden resnet layers (default: None")
        return parent_parser

    def get_ia_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        if self.ia_char_net is None:
            return None
        self.ia_char_net.get_empty_agent_embedding(batch_size)

    def get_ia_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        if self.ia_mental_net is None:
            return None
        self.ia_mental_net.get_empty_agent_embedding(batch_size)

    def get_ja_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        if self.ja_char_net is None:
            return None
        self.ja_char_net.get_empty_agent_embedding(batch_size)

    def get_ja_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        if self.ja_mental_net is None:
            return None
        self.ja_mental_net.get_empty_agent_embedding(batch_size)

    def _get_num_seq_per_agent(self, past_trajectories, batch_size):
        return past_trajectories.shape[1] // batch_size

    def _postprocess_past_seq_embeddings(self, past_seq_embeddings, batch_size, num_seq_per_agent):
        char_embeddings = []
        for i in range(batch_size):
            char_embeddings.append(
                torch.sum(past_seq_embeddings[num_seq_per_agent * i:num_seq_per_agent * (i + 1)], dim=0).unsqueeze(0))
        return torch.cat(char_embeddings, dim=0)

    def _forward_prediction_networks(self, query_state, embeddings):
        pred_output = self.pred_net(query_state, embeddings)
        pred_action = self.action_pred_net(pred_output)
        pred_goal = self.goal_pred_net(pred_output)
        pred_sr = self.sr_pred_net(pred_output)
        return pred_action, pred_goal, pred_sr

    def forward(self, x: IterativeActionTOMMASInput, return_embeddings=False):
        # num_features = 1 (wall) + num goals (goal positions) + num agents (agent positions) +
        #           (num actions * num agents)
        # num_features_without_actions = 1 (wall) + num goals (goal positions) + num agents (agent positions)
        # past_trajectories: [seq length, num sequences * batch, num_features, gridworld rows, gridworld columns]
        #           the (num sequences * batch) dimension is a flat 1d version of the 2d array[batch][num seq] so
        #           a single agent, i.e. 1 batch, has its different sequence next to each other.
        # current_trajectory: [seq length, batch, num_features, gridworld rows, gridworld columns]
        past_trajectories = x.past_trajectories
        current_trajectory = x.current_trajectory
        ia_features = x.independent_agent_features

        _, batch_size, num_features = current_trajectory.shape

        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, batch_size)

            past_seq_embeddings = self.ia_char_net(past_trajectories[:, :, ia_features])
            e_char_ia = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            past_traj_and_e_char_ia = add_embeddings_to_trajectory(past_trajectories, e_char_ia,
                                                                   is_past_traj=True, batch_size=batch_size)
            past_seq_embeddings = self.ja_char_net(past_traj_and_e_char_ia)
            e_char_ja = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
        else:
            e_char_ia = self.ia_char_net.get_empty_agent_embedding(batch_size).to(current_trajectory.device)
            e_char_ja = self.ja_char_net.get_empty_agent_embedding(batch_size).to(current_trajectory.device)

        # Get mental embeddings
        current_trajectory = current_trajectory
        current_traj_and_e_char_ia = add_embeddings_to_trajectory(current_trajectory[:, :, ia_features],
                                                                  e_char_ia, is_past_traj=False)
        e_mental_ia = self.ia_mental_net(current_traj_and_e_char_ia)
        current_traj_and_multi_embeddings = add_embeddings_to_trajectory(current_trajectory,
                                                                         (e_char_ia,
                                                                          e_char_ja,
                                                                          e_mental_ia),
                                                                         is_past_traj=False)
        e_mental_ja = self.ja_mental_net(current_traj_and_multi_embeddings)


        # Make predictions
        embeddings = (e_char_ia, e_mental_ia, e_char_ja, e_mental_ja)
        action_prediction = IterativeTOMMASPredictions(self.action_pred_net(torch.cat(embeddings, dim=1)))
        if return_embeddings:
            action_prediction.embeddings = embeddings
        return action_prediction

    @torch.no_grad()
    def forward_given_embeddings(self, past_trajectories, current_trajectory, query_state, user_embeddings,
                                 return_embeddings=False):
        pass


class IterativeActionToMnet(IterativeActionTOMMAS):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_independent_agent_features: int,
                 num_joint_agent_features: int,
                 ia_char_hidden_layer_features: Union[List[int], None],
                 ia_char_embedding_size: int,
                 ia_mental_hidden_layer_features: Union[List[int], None],
                 ia_mental_embedding_size: int,
                 ja_char_hidden_layer_features: Union[List[int], None],
                 ja_char_embedding_size: int,
                 ja_mental_hidden_layer_features: Union[List[int], None],
                 ja_mental_embedding_size: int,
                 num_actions: int,
                 ):
        super(IterativeActionTOMMAS, self).__init__(model_type=model_type, model_name=model_name,
                                                    learning_rate=learning_rate)
        self.save_hyperparameters()
        self.ia_char_net = None
        self.ia_mental_net = None
        self.ja_char_net = None
        self.ja_mental_net = None
        if ia_char_embedding_size > 0:
            self.ia_char_net = IterativeActionEmbeddingNetwork(in_features=num_independent_agent_features,
                                                               layer_features=ia_char_hidden_layer_features,
                                                               embedding_size=ia_char_embedding_size)
            ia_mental_input = num_independent_agent_features + ia_char_embedding_size
            self.ia_mental_net = IterativeActionEmbeddingNetwork(in_features=ia_mental_input,
                                                                 layer_features=ia_mental_hidden_layer_features,
                                                                 embedding_size=ia_mental_embedding_size)
            action_pred_net_input = ia_char_embedding_size + ia_mental_embedding_size
        else:
            self.ja_char_net = IterativeActionEmbeddingNetwork(in_features=num_joint_agent_features,
                                                               layer_features=ja_char_hidden_layer_features,
                                                               embedding_size=ja_char_embedding_size)
            ja_mental_input = num_joint_agent_features + ja_char_embedding_size
            self.ja_mental_net = IterativeActionEmbeddingNetwork(in_features=ja_mental_input,
                                                                 layer_features=ja_mental_hidden_layer_features,
                                                                 embedding_size=ja_mental_embedding_size)
            action_pred_net_input = ja_char_embedding_size + ja_mental_embedding_size

        self.action_pred_net = nn.Sequential(
            nn.Linear(action_pred_net_input, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions))
        # self.action_pred_net = nn.Linear(action_pred_net_input, num_actions)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return IterativeActionTOMMAS.add_model_specific_args(parent_parser, model_name="ToMnet")

    def _tomnet_forward(self, char_net: IterativeActionEmbeddingNetwork, mental_net: IterativeActionEmbeddingNetwork,
                        past_trajectories, current_trajectory, return_embeddings=False):
        _, batch_size, feature_no_actions = current_trajectory.shape

        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, batch_size)
            past_seq_embeddings = char_net(past_trajectories)
            e_char = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
        else:
            e_char = char_net.get_empty_agent_embedding(batch_size).to(current_trajectory.device)

        # Get mental embeddings
        current_trajectory = current_trajectory
        current_traj_and_e_char = add_embeddings_to_trajectory(current_trajectory, e_char,
                                                               is_past_traj=False)
        e_mental = mental_net(current_traj_and_e_char)


        # Make predictions
        embeddings = (e_char, e_mental)
        action_prediction = IterativeTOMMASPredictions(self.action_pred_net(torch.cat(embeddings, dim=1)))
        if return_embeddings:
            action_prediction.embeddings = embeddings
        return action_prediction

    def forward(self, x: IterativeActionTOMMASInput, return_embeddings=False):
        # num_features = 1 (wall) + num goals (goal positions) + num agents (agent positions) +
        #           (num actions * num agents)
        # num_features_without_actions = 1 (wall) + num goals (goal positions) + num agents (agent positions)
        # past_trajectories: [seq length, num sequences * batch, num_features, gridworld rows, gridworld columns]
        #           the (num sequences * batch) dimension is a flat 1d version of the 2d array[batch][num seq] so
        #           a single agent, i.e. 1 batch, has its different sequence next to each other.
        # current_trajectory: [seq length, batch, num_features, gridworld rows, gridworld columns]
        # query_state: [batch, num_features_without_actions, gridworld rows, gridworld cols]
        past_trajectories = x.past_trajectories
        current_trajectory = x.current_trajectory
        ia_features = x.independent_agent_features

        if self.ia_char_net is not None:
            past_trajectories = past_trajectories[:, :, ia_features] if past_trajectories is not None else None
            current_trajectory = current_trajectory[:, :, ia_features]
            return self._tomnet_forward(self.ia_char_net, self.ia_mental_net, past_trajectories, current_trajectory,
                                        return_embeddings)
        elif self.ja_char_net is not None:
            return self._tomnet_forward(self.ja_char_net, self.ja_mental_net, past_trajectories, current_trajectory,
                                        return_embeddings)

    def _tomnet_forward_given_embeddings(self, char_net: IterativeActionEmbeddingNetwork,
                                         mental_net: IterativeActionEmbeddingNetwork,
                                         past_trajectories, current_trajectory, query_state, user_embeddings,
                                         return_embeddings=False, ia_modelling=False):
        pass

    def forward_given_embeddings(self, past_trajectories, current_trajectory, query_state, user_embeddings,
                                 return_embeddings=False):
        # num_features = 1 (wall) + num goals (goal positions) + num agents (agent positions) +
        #           (num actions * num agents)
        # num_features_without_actions = 1 (wall) + num goals (goal positions) + num agents (agent positions)
        # past_trajectories: [seq length, num sequences * batch, num_features, gridworld rows, gridworld columns]
        #           the (num sequences * batch) dimension is a flat 1d version of the 2d array[batch][num seq] so
        #           a single agent, i.e. 1 batch, has its different sequence next to each other.
        # current_trajectory: [seq length, batch, num_features, gridworld rows, gridworld columns]
        # query_state: [batch, num_features_without_actions, gridworld rows, gridworld cols]
        # user_embeddings: (e_char_ia, e_mental_ia, e_char_ja, e_mental_ja)

        if self.ia_char_net is not None:
            ia_features = self._get_ia_features()
            past_trajectories = past_trajectories[:, :, ia_features] if past_trajectories is not None else None
            current_trajectory = current_trajectory[:, :, ia_features]
            return self._tomnet_forward_given_embeddings(self.ia_char_net, self.ia_mental_net, past_trajectories,
                                                         current_trajectory, query_state, user_embeddings,
                                                         return_embeddings, ia_modelling=True)
        elif self.ja_char_net is not None:
            return self._tomnet_forward_given_embeddings(self.ja_char_net, self.ja_mental_net, past_trajectories,
                                                         current_trajectory, query_state, user_embeddings,
                                                         return_embeddings, ia_modelling=False)


class IterativeActionLSTM(IterativeActionModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_joint_agent_features: int,
                 hidden_layer_features: Union[List[int], None],
                 embedding_size: int,
                 num_actions: int,
                 n_layer: int = 1,
                 ):
        super().__init__(model_type, model_name, learning_rate)
        self.save_hyperparameters()
        if hidden_layer_features is None:
            hidden_layer_features = []
        all_layers_features = [num_joint_agent_features] + hidden_layer_features + [embedding_size]
        layers = []
        for i in range(len(all_layers_features)-1):
            layers.append(nn.Linear(all_layers_features[i], all_layers_features[i+1]))
            if i < len(all_layers_features) - 2:
                layers.append(nn.ReLU())
        self.embedding_net = IterativeActionEmbeddingNetwork(in_features=num_joint_agent_features,
                                                             layer_features=hidden_layer_features,
                                                             embedding_size=embedding_size,
                                                             batch_first=True)
        self.embedding_size = embedding_size
        self.action_pred_net = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IterativeActionLSTM")
        parser = IterativeActionModeller.add_model_specific_args(parser)
        parser.add_argument("--hidden_layer_features", nargs='+', type=int, default=[],
                            help="the number of features for the embedding network's hidden layers (default: None)")
        parser.add_argument("--embedding_size", type=int, default=2)
        parser.add_argument("--n_layer", type=int, default=1,
                            help="the number of lstm layers")
        return parent_parser

    def forward(self, x: IterativeActionTransformerInput, return_embeddings=False):
        trajectory = x.trajectory
        embeddings = self.embedding_net(trajectory, return_all_layers=True)
        predictions = IterativeTOMMASPredictions(self.action_pred_net(embeddings))
        if return_embeddings:
            predictions.embeddings = embeddings
        return predictions
