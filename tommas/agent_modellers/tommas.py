from typing import Tuple, Union, List
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from pytorch_lightning.core.lightning import LightningModule

from tommas.agent_modellers.embedding_networks import AgentEmbeddingNetwork
from tommas.helper_code.metrics import action_loss, goal_consumption_loss, successor_representation_loss, \
    action_transformer_loss, goal_consumption_transformer_loss, successor_representation_transformer_loss, \
    action_acc, goal_acc, successor_representation_acc
from tommas.agent_modellers.embedding_networks import lstm_types, pool_types
from tommas.agent_modellers.prediction_networks import PredictionNet, ActionPredictionNet, GoalPredictionNet, \
    SRPredictionNet
from tommas.agent_modellers.modeller_inputs import TOMMASInput
from tommas.agent_modellers.modeller_outputs import AgentObjectives, AgentWeightedObjectives, TOMMASPredictions


def add_embeddings_to_trajectory(trajectory: torch.Tensor, embeddings: Union[list, tuple, torch.Tensor],
                                 is_past_traj: bool, batch_size: int = 0):
    if type(embeddings) is torch.Tensor:
        tensor_embedding = embeddings
    else:
        # Each embedding should be shape (batch, embedding_size, row, col)
        tensor_embedding = torch.cat(embeddings, dim=1)

    if is_past_traj:
        seq_len = trajectory.shape[0]
        num_seq = trajectory.shape[1] // batch_size
        tiled_embedding = torch.repeat_interleave(tensor_embedding, num_seq, dim=0).unsqueeze(0)
        tiled_embedding = tiled_embedding.repeat((seq_len, 1, 1, 1, 1))
        return torch.cat((trajectory, tiled_embedding), dim=2)
    else:
        seq_len = trajectory.shape[0]
        tiled_embedding = tensor_embedding.unsqueeze(0)
        tiled_embedding = tiled_embedding.repeat((seq_len, 1, 1, 1, 1))
        return torch.cat((trajectory, tiled_embedding), dim=2)


def spatialise(tensor, row, col):
    if len(tensor.shape) == 4:
        if tensor.shape[2:] == (row, col):
            return tensor
        else:
            batch_size, e_features, e_row, e_col = tensor.shape
            num_pad_rows = row - e_row
            num_pad_cols = col - e_col
            return nn.ConstantPad2d((0, num_pad_cols, 0, num_pad_rows), 0)(tensor)
    elif len(tensor.shape) == 2:
        batch_size, embedding_size = tensor.shape
        tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        return tensor.expand(batch_size, embedding_size, row, col)
    else:
        raise ValueError("tensor.shape must be size 2 or 4, given tensor shape %d" % (len(tensor.shape)))


class AgentModeller(LightningModule):
    def __init__(self, model_type: str, model_name: str, learning_rate: float, optimizer_type="adam",
                 no_action_loss=False, no_goal_loss=False, no_sr_loss=False):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.no_action_loss = no_action_loss
        self.no_goal_loss = no_goal_loss
        self.no_sr_loss = no_sr_loss

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
        parser.add_argument("--optimizer", "--optim", type=str, default="adam")
        parser.add_argument("--no_action_loss", action="store_true", default=False)
        parser.add_argument("--no_sr_loss", action="store_true", default=False)
        parser.add_argument("--no_goal_loss", action="store_true", default=False)
        return parser

    def forward(self, x):
        raise NotImplementedError

    def forward_step(self, batch: Tuple[TOMMASInput, AgentObjectives], batch_idx, forward_type):
        modeller_input, true_agent_objectives = batch
        pred_agent_objectives = self(modeller_input)
        a_loss = self._action_loss(pred_agent_objectives, true_agent_objectives)
        g_loss = self._goal_loss(pred_agent_objectives, true_agent_objectives)
        sr_loss = self._srs_loss(pred_agent_objectives, true_agent_objectives)
        overall_loss = a_loss + g_loss + sr_loss
        self.log('overall_' + forward_type + '_loss', overall_loss)
        if forward_type == "eval":
            self._log_acc(pred_agent_objectives, true_agent_objectives, forward_type)
        return overall_loss

    def _action_loss(self, pred_agent_objectives, true_agent_objectives):
        if self.no_action_loss:
            return 0
        if isinstance(true_agent_objectives, AgentWeightedObjectives):
            loss = action_transformer_loss(pred_agent_objectives.action,
                                           (true_agent_objectives.action, true_agent_objectives.action_weight))
        else:
            loss = action_loss(pred_agent_objectives.action, true_agent_objectives.action)
        return loss

    def _goal_loss(self, pred_agent_objectives, true_agent_objectives):
        if self.no_goal_loss:
            return 0
        if isinstance(true_agent_objectives, AgentWeightedObjectives):
            loss = goal_consumption_transformer_loss(pred_agent_objectives.goal_consumption,
                                                     (true_agent_objectives.goal_consumption,
                                                      true_agent_objectives.goal_consumption_weight))
        else:
            loss = goal_consumption_loss(pred_agent_objectives.goal_consumption,
                                         true_agent_objectives.goal_consumption)
        return loss

    def _srs_loss(self, pred_agent_objectives, true_agent_objectives):
        if self.no_sr_loss:
            return 0
        if isinstance(true_agent_objectives, AgentWeightedObjectives):
            loss = successor_representation_transformer_loss(pred_agent_objectives.successor_representation,
                                                             (true_agent_objectives.successor_representation,
                                                              true_agent_objectives.successor_representation_weight))
        else:
            loss = successor_representation_loss(pred_agent_objectives.successor_representation,
                                                 true_agent_objectives.successor_representation)
        return loss

    def _log_acc(self, pred_agent_objectives, true_agent_objectives, forward_type):
        s_acc = 0
        g_acc = 0
        a_acc = 0
        if not self.no_action_loss:
            a_acc = action_acc(pred_agent_objectives.action, true_agent_objectives.action)
        if not self.no_goal_loss:
            g_acc = goal_acc(pred_agent_objectives.goal_consumption, true_agent_objectives.goal_consumption)
        if not self.no_sr_loss:
            s_acc = successor_representation_acc(pred_agent_objectives.successor_representation,
                                                 true_agent_objectives.successor_representation)
        self.log(forward_type + '_action_acc', a_acc)
        self.log(forward_type + '_goal_acc', g_acc)
        self.log(forward_type + '_sr_acc', s_acc)

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "training")

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "eval")

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            return Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            return SGD(self.parameters(), lr=self.learning_rate, momentum=0.0)
        else:
            raise ValueError(f"Don't recognize optimizer: {self.optimizer_type}")


class SharedPredictionNetModeller(AgentModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 pred_net_in_channels: int,
                 pred_num_resnet_blocks: int,
                 pred_resnet_channels: Union[int, List[int]],
                 action_pred_hidden_channels: int,
                 goal_pred_hidden_channels: int,
                 sr_pred_hidden_channels: int,
                 world_dim: Tuple[int, int],
                 num_actions: int,
                 num_goals: int,
                 no_action_loss=False,
                 no_goal_loss=False,
                 no_sr_loss=False,
                 ):

        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        pred_net = PredictionNet(pred_net_in_channels, pred_num_resnet_blocks, pred_resnet_channels)
        if isinstance(pred_resnet_channels, list):
            pred_out_channels = pred_resnet_channels[-1]
        else:
            pred_out_channels = pred_resnet_channels
        action_pred_net = ActionPredictionNet(pred_out_channels, action_pred_hidden_channels, world_dim, num_actions)
        goal_pred_net = GoalPredictionNet(pred_out_channels, goal_pred_hidden_channels, world_dim, num_goals)
        sr_pred_net = SRPredictionNet(pred_out_channels, sr_pred_hidden_channels)
        self.pred_net = pred_net
        self.action_pred_net = action_pred_net
        self.goal_pred_net = goal_pred_net
        self.sr_pred_net = sr_pred_net

    @staticmethod
    def add_model_specific_args(parser):
        parser = AgentModeller.add_model_specific_args(parser)
        parser.add_argument('--pred_num_resnet_blocks', type=int, default=2,
                            help="the number of resnet blocks in the prediction network's resnet layer (default: 2)")
        parser.add_argument('--pred_resnet_channels', nargs='+', type=int, default=[16],
                            help="the number of features for the prediction network's hidden resnet layers (default: 16")
        parser.add_argument('--action_pred_hidden_channels', type=int, default=8,
                            help="the number of features in the action prediction network (default: 8)")
        parser.add_argument('--goal_pred_hidden_channels', type=int, default=8,
                            help="the number of features in the goal consumption prediction network (default: 8)")
        parser.add_argument('--sr_pred_hidden_channels', type=int, default=8,
                            help="the number of features in the successor representation prediction network (default: 8)")
        return parser

    def _forward_prediction_networks(self, query_state, embeddings):
        pred_output = self.pred_net(query_state, embeddings)
        pred_action = self.action_pred_net(pred_output)
        pred_goal = self.goal_pred_net(pred_output)
        pred_sr = self.sr_pred_net(pred_output)
        return pred_action, pred_goal, pred_sr


class TOMMAS(SharedPredictionNetModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_independent_agent_features: int,
                 num_joint_agent_features: int,
                 ia_char_embedding_specs: Union[dict, None],
                 ia_mental_embedding_specs: Union[dict, None],
                 ja_char_embedding_specs: Union[dict, None],
                 ja_mental_embedding_specs: Union[dict, None],
                 pred_net_in_channels: int,
                 pred_num_resnet_blocks: int,
                 pred_resnet_channels: Union[int, List[int]],
                 action_pred_hidden_channels: int,
                 goal_pred_hidden_channels: int,
                 sr_pred_hidden_channels: int,
                 world_dim: Tuple[int, int],
                 num_actions: int,
                 num_goals: int,
                 no_action_loss=False,
                 no_goal_loss=False,
                 no_sr_loss=False,
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         pred_net_in_channels=pred_net_in_channels, pred_num_resnet_blocks=pred_num_resnet_blocks,
                         pred_resnet_channels=pred_resnet_channels,
                         action_pred_hidden_channels=action_pred_hidden_channels,
                         goal_pred_hidden_channels=goal_pred_hidden_channels,
                         sr_pred_hidden_channels=sr_pred_hidden_channels, world_dim=world_dim,
                         num_actions=num_actions, num_goals=num_goals,
                         no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        self.save_hyperparameters()
        self.batch_first = False

        self.ia_char_net = AgentEmbeddingNetwork(in_channels=num_independent_agent_features, world_dim=world_dim,
                                                 **ia_char_embedding_specs)
        ia_mental_in_channels = num_independent_agent_features + self.ia_char_net.embedding_size
        self.ia_mental_net = AgentEmbeddingNetwork(in_channels=ia_mental_in_channels,
                                                   world_dim=world_dim, **ia_mental_embedding_specs)
        ja_char_in_channels = num_joint_agent_features + self.ia_char_net.embedding_size
        self.ja_char_net = AgentEmbeddingNetwork(in_channels=ja_char_in_channels,
                                                 world_dim=world_dim, **ja_char_embedding_specs)
        ja_mental_in_channels = num_joint_agent_features + self.ia_char_net.embedding_size + \
                                self.ia_mental_net.embedding_size + self.ja_char_net.embedding_size
        self.ja_mental_net = AgentEmbeddingNetwork(in_channels=ja_mental_in_channels, world_dim=world_dim,
                                                   **ja_mental_embedding_specs)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser, model_name: str = "TOMMAS"):
        parser = parent_parser.add_argument_group(model_name)
        parser = SharedPredictionNetModeller.add_model_specific_args(parser)
        for network_type in ['ia_char', 'ia_mental', 'ja_char', 'ja_mental']:
            parser.add_argument('--' + network_type + '_embedding_size', type=int, default=2,
                                help="the network's embedding size (default: 2)")
            parser.add_argument('--' + network_type + '_flatten_embedding', action='store_true', default=False,
                                help="whether to flatten the network's embedding (default: False)")
            parser.add_argument('--' + network_type + '_num_resnet_blocks', type=int, default=2,
                                help="the number of resnet blocks in the network's resnet layer (default: 2)")
            parser.add_argument('--' + network_type + '_resnet_channels', nargs='+', type=int, default=[16],
                                help="the number of features for the network's hidden resnet layers (default: 16")
            parser.add_argument('--' + network_type + '_recurrent_hidden_size', type=int, default=8,
                                help="the number of features in the network's recurrent layer (default: 8)")
            parser.add_argument('--' + network_type + '_lstm_type', type=str, default="conv_lstm",
                                help="the lstm type used by the network. Potential values %s (default: conv_lstm)" % lstm_types)
            parser.add_argument('--' + network_type + '_pooling', type=str, default=None,
                                help="the pooling type used after the network's resnet. Potential values %s. (default: None)" % pool_types)
            parser.add_argument('--' + network_type + '_pre_resnet_layer', nargs='+', type=int, default=[],
                                help="give the out_channels, kernel_size, stride, padding for the Conv2d before the network's resnet (default: no Conv2d before the resnet)")
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

    def _get_num_seq_per_agent(self, past_trajectories, query_state):
        return past_trajectories.shape[1] // query_state.shape[0]

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

    def _get_ia_features(self):
        action_offset = 1 + self.num_goals + self.num_agents
        return list(range(1 + self.num_goals + 1)) + list(range(action_offset, action_offset + self.num_actions))

    def forward(self, x: TOMMASInput, return_embeddings=False):
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
        query_state = x.query_state
        ia_features = x.independent_agent_features

        batch_size, num_feature_no_actions, row, col = query_state.shape

        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, query_state)

            past_seq_embeddings = self.ia_char_net(past_trajectories[:, :, ia_features, :, :])
            e_char_ia = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            spatialised_e_char_ia = spatialise(e_char_ia, row, col)
            past_traj_and_e_char_ia = add_embeddings_to_trajectory(past_trajectories, spatialised_e_char_ia,
                                                                   is_past_traj=True, batch_size=batch_size)
            past_seq_embeddings = self.ja_char_net(past_traj_and_e_char_ia)
            e_char_ja = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            spatialised_e_char_ja = spatialise(e_char_ja, row, col)
        else:
            e_char_ia = self.ia_char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_char_ia = spatialise(e_char_ia, row, col)
            e_char_ja = self.ja_char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_char_ja = spatialise(e_char_ja, row, col)

        # Get mental embeddings
        if current_trajectory is not None:
            current_trajectory = current_trajectory
            current_traj_and_e_char_ia = add_embeddings_to_trajectory(current_trajectory[:, :, ia_features, :, :],
                                                                      spatialised_e_char_ia, is_past_traj=False)
            e_mental_ia = self.ia_mental_net(current_traj_and_e_char_ia)
            spatialised_e_mental_ia = spatialise(e_mental_ia, row, col)
            current_traj_and_multi_embeddings = add_embeddings_to_trajectory(current_trajectory,
                                                                             (spatialised_e_char_ia,
                                                                              spatialised_e_char_ja,
                                                                              spatialised_e_mental_ia),
                                                                             is_past_traj=False)
            e_mental_ja = self.ja_mental_net(current_traj_and_multi_embeddings)
            spatialised_e_mental_ja = spatialise(e_mental_ja, row, col)
        else:
            e_mental_ia = self.ia_mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_mental_ia = spatialise(e_mental_ia, row, col)
            e_mental_ja = self.ja_mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_mental_ja = spatialise(e_mental_ja, row, col)

        # Make predictions
        spatialised_embeddings = (
            spatialised_e_char_ia, spatialised_e_mental_ia, spatialised_e_char_ja, spatialised_e_mental_ja)
        if return_embeddings:
            return TOMMASPredictions(*self._forward_prediction_networks(query_state, (spatialised_embeddings)),
                                     (e_char_ia, e_mental_ia, e_char_ja, e_mental_ja))
        return TOMMASPredictions(*self._forward_prediction_networks(query_state, (spatialised_embeddings)))

    @torch.no_grad()
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
        def convert_embedding(embedding, user_embedding):
            if user_embedding is not None:
                embedding[:] = torch.from_numpy(user_embedding)

        user_e_char_ia, user_e_mental_ia, user_e_char_ja, user_e_mental_ja = user_embeddings
        batch_size, num_feature_no_actions, row, col = query_state.shape

        # ia: independent action, ja: joint action
        ia_features = self._get_ia_features()
        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, query_state)

            past_seq_embeddings = self.ia_char_net(past_trajectories[:, :, ia_features, :, :])
            e_char_ia = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            convert_embedding(e_char_ia, user_e_char_ia)
            spatialised_e_char_ia = spatialise(e_char_ia, row, col)
            past_traj_and_e_char_ia = add_embeddings_to_trajectory(past_trajectories, spatialised_e_char_ia,
                                                                   is_past_traj=True, batch_size=batch_size)
            past_seq_embeddings = self.ja_char_net(past_traj_and_e_char_ia)
            e_char_ja = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            convert_embedding(e_char_ja, user_e_char_ja)
            spatialised_e_char_ja = spatialise(e_char_ja, row, col)
        else:
            e_char_ia = self.ia_char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_char_ia, user_e_char_ia)
            spatialised_e_char_ia = spatialise(e_char_ia, row, col)
            e_char_ja = self.ja_char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_char_ja, user_e_char_ja)
            spatialised_e_char_ja = spatialise(e_char_ja, row, col)

        # Get mental embeddings
        if current_trajectory is not None:
            current_trajectory = current_trajectory
            current_traj_and_e_char_ia = add_embeddings_to_trajectory(current_trajectory[:, :, ia_features, :, :],
                                                                      spatialised_e_char_ia, is_past_traj=False)
            e_mental_ia = self.ia_mental_net(current_traj_and_e_char_ia)
            convert_embedding(e_mental_ia, user_e_mental_ia)
            spatialised_e_mental_ia = spatialise(e_mental_ia, row, col)
            current_traj_and_multi_embeddings = add_embeddings_to_trajectory(current_trajectory,
                                                                             (spatialised_e_char_ia,
                                                                              spatialised_e_char_ja,
                                                                              spatialised_e_mental_ia),
                                                                             is_past_traj=False)
            e_mental_ja = self.ja_mental_net(current_traj_and_multi_embeddings)
            convert_embedding(e_mental_ja, user_e_mental_ja)
            spatialised_e_mental_ja = spatialise(e_mental_ja, row, col)
        else:
            e_mental_ia = self.ia_mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_mental_ia, user_e_mental_ia)
            spatialised_e_mental_ia = spatialise(e_mental_ia, row, col)
            e_mental_ja = self.ia_mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_mental_ja, user_e_mental_ja)
            spatialised_e_mental_ja = spatialise(e_mental_ja, row, col)

        # Make predictions
        spatialised_embeddings = (
            spatialised_e_char_ia, spatialised_e_mental_ia, spatialised_e_char_ja, spatialised_e_mental_ja)
        if return_embeddings:
            return self._forward_prediction_networks(query_state, (spatialised_embeddings)), \
                   (e_char_ia, e_mental_ia, e_char_ja, e_mental_ja)
        return self._forward_prediction_networks(query_state, (spatialised_embeddings))


class ToMnet(TOMMAS):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_independent_agent_features: int,
                 num_joint_agent_features: int,
                 ia_char_embedding_specs: Union[dict, None],
                 ia_mental_embedding_specs: Union[dict, None],
                 ja_char_embedding_specs: Union[dict, None],
                 ja_mental_embedding_specs: Union[dict, None],
                 pred_net_in_channels: int,
                 pred_num_resnet_blocks: int,
                 pred_resnet_channels: Union[int, List[int]],
                 action_pred_hidden_channels: int,
                 goal_pred_hidden_channels: int,
                 sr_pred_hidden_channels: int,
                 world_dim: Tuple[int, int],
                 num_actions: int,
                 num_goals: int,
                 no_action_loss=False,
                 no_goal_loss=False,
                 no_sr_loss=False,
                 ):
        super(TOMMAS, self).__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                                     pred_net_in_channels=pred_net_in_channels,
                                     pred_num_resnet_blocks=pred_num_resnet_blocks,
                                     pred_resnet_channels=pred_resnet_channels,
                                     action_pred_hidden_channels=action_pred_hidden_channels,
                                     goal_pred_hidden_channels=goal_pred_hidden_channels,
                                     sr_pred_hidden_channels=sr_pred_hidden_channels, world_dim=world_dim,
                                     num_actions=num_actions, num_goals=num_goals,
                                     no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        self.save_hyperparameters()
        self.ia_char_net = None
        self.ia_mental_net = None
        self.ja_char_net = None
        self.ja_mental_net = None
        if ia_char_embedding_specs is not None:
            self.ia_char_net = AgentEmbeddingNetwork(in_channels=num_independent_agent_features, world_dim=world_dim,
                                                     **ia_char_embedding_specs)
            ia_mental_in_channels = num_independent_agent_features + self.ia_char_net.embedding_size
            self.ia_mental_net = AgentEmbeddingNetwork(in_channels=ia_mental_in_channels,
                                                       world_dim=world_dim, **ia_mental_embedding_specs)
        if ja_char_embedding_specs is not None:
            self.ja_char_net = AgentEmbeddingNetwork(in_channels=num_joint_agent_features,
                                                     world_dim=world_dim, **ja_char_embedding_specs)
            ja_mental_in_channels = num_joint_agent_features + self.ja_char_net.embedding_size
            self.ja_mental_net = AgentEmbeddingNetwork(in_channels=ja_mental_in_channels, world_dim=world_dim,
                                                       **ja_mental_embedding_specs)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        return TOMMAS.add_model_specific_args(parent_parser, model_name="ToMnet")

    def _tomnet_forward(self, char_net: AgentEmbeddingNetwork, mental_net: AgentEmbeddingNetwork,
                        past_trajectories, current_trajectory, query_state, return_embeddings=False):
        batch_size, feature_no_actions, row, col = query_state.shape

        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, query_state)

            past_seq_embeddings = char_net(past_trajectories)
            e_char = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            spatialised_e_char = spatialise(e_char, row, col)
        else:
            e_char = char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_char = spatialise(e_char, row, col)

        # Get mental embeddings
        if current_trajectory is not None:
            current_trajectory = current_trajectory
            current_traj_and_e_char = add_embeddings_to_trajectory(current_trajectory, spatialised_e_char,
                                                                   is_past_traj=False)
            e_mental = mental_net(current_traj_and_e_char)
            spatialised_e_mental = spatialise(e_mental, row, col)
        else:
            e_mental = mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            spatialised_e_mental = spatialise(e_mental, row, col)

        # Make predictions
        spatialised_embeddings = (spatialised_e_char, spatialised_e_mental)
        if return_embeddings:
            return TOMMASPredictions(*self._forward_prediction_networks(query_state, spatialised_embeddings),
                                     (e_char, e_mental))
        return TOMMASPredictions(*self._forward_prediction_networks(query_state, spatialised_embeddings))

    def forward(self, x: TOMMASInput, return_embeddings=False):
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
        query_state = x.query_state
        ia_features = x.independent_agent_features

        if self.ia_char_net is not None:
            past_trajectories = past_trajectories[:, :, ia_features, :, :] if past_trajectories is not None else None
            current_trajectory = current_trajectory[:, :, ia_features, :, :] if current_trajectory is not None else None
            return self._tomnet_forward(self.ia_char_net, self.ia_mental_net, past_trajectories, current_trajectory,
                                        query_state, return_embeddings)
        elif self.ja_char_net is not None:
            return self._tomnet_forward(self.ja_char_net, self.ja_mental_net, past_trajectories, current_trajectory,
                                        query_state, return_embeddings)

    def _tomnet_forward_given_embeddings(self, char_net: AgentEmbeddingNetwork, mental_net: AgentEmbeddingNetwork,
                                         past_trajectories, current_trajectory, query_state, user_embeddings,
                                         return_embeddings=False, ia_modelling=False):
        def convert_embedding(embedding, user_embedding):
            if user_embedding is not None:
                embedding[:] = torch.from_numpy(user_embedding)

        batch_size, feature_no_actions, row, col = query_state.shape
        user_e_char, user_e_mental = user_embeddings

        # Get character embeddings
        if past_trajectories is not None:
            num_seq_per_agent = self._get_num_seq_per_agent(past_trajectories, query_state)

            past_seq_embeddings = char_net(past_trajectories)
            e_char = self._postprocess_past_seq_embeddings(past_seq_embeddings, batch_size, num_seq_per_agent)
            convert_embedding(e_char, user_e_char)
            spatialised_e_char = spatialise(e_char, row, col)
        else:
            e_char = char_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_char, user_e_char)
            spatialised_e_char = spatialise(e_char, row, col)

        # Get mental embeddings
        if current_trajectory is not None:
            current_trajectory = current_trajectory
            current_traj_and_e_char = add_embeddings_to_trajectory(current_trajectory, spatialised_e_char,
                                                                   is_past_traj=False)
            e_mental = mental_net(current_traj_and_e_char)
            convert_embedding(e_mental, user_e_mental)
            spatialised_e_mental = spatialise(e_mental, row, col)
        else:
            e_mental = mental_net.get_empty_agent_embedding(batch_size).to(query_state.device)
            convert_embedding(e_mental, user_e_mental)
            spatialised_e_mental = spatialise(e_mental, row, col)

        # Make predictions
        spatialised_embeddings = (spatialised_e_char, spatialised_e_mental)
        if return_embeddings:
            if ia_modelling:
                embeddings = (e_char, e_mental, None, None)
            else:
                embeddings = (None, None, e_char, e_mental)
            return self._forward_prediction_networks(query_state, spatialised_embeddings), embeddings
        return self._forward_prediction_networks(query_state, spatialised_embeddings)

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
            past_trajectories = past_trajectories[:, :, ia_features, :, :] if past_trajectories is not None else None
            current_trajectory = current_trajectory[:, :, ia_features, :, :] if current_trajectory is not None else None
            return self._tomnet_forward_given_embeddings(self.ia_char_net, self.ia_mental_net, past_trajectories,
                                                         current_trajectory, query_state, user_embeddings,
                                                         return_embeddings, ia_modelling=True)
        elif self.ja_char_net is not None:
            return self._tomnet_forward_given_embeddings(self.ja_char_net, self.ja_mental_net, past_trajectories,
                                                         current_trajectory, query_state, user_embeddings,
                                                         return_embeddings, ia_modelling=False)


class RandomPolicyToMnet(AgentModeller):
    def __init__(self, model_type: str, model_name: str, learning_rate: float, no_action_loss=False, no_goal_loss=True,
                 no_sr_loss=True):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        from tommas.helper_code.conv import ConvLSTM
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Conv2d(11, 8, 3, 1, 1),
                                 nn.ReLU(),
                                 )
        self.lstm = ConvLSTM(8, 8)
        self.lstm_out_net = nn.Sequential(nn.AvgPool2d(2), nn.Flatten(1))
        zeros = torch.zeros((2, 11, 11, 11))
        lstm_output = self.lstm_out_net(self.lstm((self.net(zeros).unsqueeze(0)))[0].squeeze(0))
        self.fc1 = nn.Linear(lstm_output.size(1), 2)
        self.pred_net = nn.Sequential(nn.Conv2d(8, 32, 3),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, 3),
                                      nn.AvgPool2d(2),
                                      nn.Flatten(1)
                                      )
        in_channels = self.pred_net(torch.zeros((2, 8, 11, 11))).size(1)
        self.fc2 = nn.Linear(in_channels, 5)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("RandomPolicyToMnet")
        AgentModeller.add_model_specific_args(parser)
        return parent_parser

    def get_ia_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return torch.zeros((batch_size, 2))

    def get_ia_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def get_ja_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def get_ja_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def forward(self, x: TOMMASInput, return_embeddings=False):
        past_trajectories = x.past_trajectories
        query_state = x.query_state

        batch_size = query_state.size(0)
        if past_trajectories is not None:
            seq_len, batch_size, features, row, col = past_trajectories.shape
            trajectories = past_trajectories.reshape(seq_len * batch_size, features, row, col)
            preprocessed_traj = self.net(trajectories)
            preprocessed_traj = preprocessed_traj.reshape(seq_len, batch_size, 8, 11, 11)
            e_char = self.fc1(self.lstm_out_net(self.lstm(preprocessed_traj)[0].squeeze(0)))
            spatialised_e_char = spatialise(e_char, 11, 11)
        else:
            e_char = torch.zeros((batch_size, 2))
            spatialised_e_char = spatialise(e_char, 11, 11).to(query_state.device)

        action_pred = self.fc2(self.pred_net(torch.cat((query_state, spatialised_e_char), dim=1)))
        if return_embeddings:
            return TOMMASPredictions(action_pred, None, None, (e_char, None, None, None))
        return TOMMASPredictions(action_pred, None, None)


def forward_shuffle_embeddings(agent_modeller: TOMMAS, past_trajectories, current_trajectory, query_state,
                               embeddings_to_shuffle):
    with torch.no_grad():
        _, _, _, embeddings = agent_modeller(past_trajectories, current_trajectory, query_state, True)
        for embedding, shuffle_embedding in zip(embeddings, embeddings_to_shuffle):
            if shuffle_embedding:
                embedding = embedding[:, torch.randperm(embedding.size(1))]
        return agent_modeller.forward_given_embeddings(past_trajectories, current_trajectory, query_state, embeddings)
