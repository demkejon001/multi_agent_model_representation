from typing import Union, Tuple, List
from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

from tommas.agent_modellers.tommas import SharedPredictionNetModeller, add_embeddings_to_trajectory, spatialise
from tommas.agent_modellers.modeller_inputs import TOMMASTransformerInput
from tommas.agent_modellers.modeller_outputs import TOMMASTransformerPredictions
from tommas.agent_modellers.embedding_networks import GridworldNetwork, pool_types


class TOMMASTransformer(SharedPredictionNetModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 num_joint_agent_features: int,
                 gridworld_embedding_specs: dict,
                 embedding_size: int,
                 n_layer: int,
                 n_head: int,
                 pred_net_in_channels: int,
                 pred_num_resnet_blocks: int,
                 pred_resnet_channels: Union[int, List[int]],
                 action_pred_hidden_channels: int,
                 goal_pred_hidden_channels: int,
                 sr_pred_hidden_channels: int,
                 world_dim: Tuple[int, int],
                 num_actions: int,
                 num_goals: int,
                 no_action_loss=True,
                 no_goal_loss=True,
                 no_sr_loss=True
                 ):

        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         pred_net_in_channels=pred_net_in_channels, pred_num_resnet_blocks=pred_num_resnet_blocks,
                         pred_resnet_channels=pred_resnet_channels,
                         action_pred_hidden_channels=action_pred_hidden_channels,
                         goal_pred_hidden_channels=goal_pred_hidden_channels,
                         sr_pred_hidden_channels=sr_pred_hidden_channels,
                         world_dim=world_dim, num_actions=num_actions, num_goals=num_goals,
                         no_action_loss=no_action_loss, no_goal_loss=no_goal_loss, no_sr_loss=no_sr_loss)
        self.save_hyperparameters()
        gridworld_net = GridworldNetwork(num_joint_agent_features,
                                         gridworld_embedding_specs['resnet_channels'],
                                         gridworld_embedding_specs['num_resnet_blocks'],
                                         gridworld_embedding_specs['pooling'],
                                         gridworld_embedding_specs['pre_resnet_layer'])

        gridworld_embedding_size = torch.flatten(gridworld_net(torch.zeros((1, num_joint_agent_features, *world_dim))),
                                                 start_dim=1).size(1)
        gridworld_embedding_net = nn.Sequential(
            gridworld_net,
            nn.Flatten(start_dim=1),
            nn.Linear(gridworld_embedding_size, embedding_size)
        )

        gpt2config = GPT2Config(vocab_size=2, n_positions=1024, n_ctx=1024, n_embd=embedding_size, n_layer=n_layer,
                                n_head=n_head, n_inner=None, activation_function='gelu')
        decoder = GPT2Model(gpt2config)

        self.gridworld_embedding_net = gridworld_embedding_net
        self.decoder = decoder
        self.embedding_size = embedding_size

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TOMMASTransformer")
        parser = SharedPredictionNetModeller.add_model_specific_args(parser)
        parser.add_argument("--num_resnet_blocks", type=int, default=2,
                            help="the number of resnet blocks in the network's resnet layer (default: 2)")
        parser.add_argument('--resnet_channels', nargs='+', type=int, default=[16],
                            help="the number of features for the network's hidden resnet layers (default: 16")
        parser.add_argument('--pooling', type=str, default=None,
                            help="the pooling type used after the network's resnet. Potential values %s. (default: None)" % pool_types)
        parser.add_argument('--pre_resnet_layer', nargs='+', type=int, default=[],
                            help="give the out_channels, kernel_size, stride, padding for the Conv2d before the network's resnet (default: no Conv2d before the resnet)")
        parser.add_argument("--embedding_size", type=int, default=2)
        parser.add_argument("--n_layer", type=int, default=2)
        parser.add_argument("--n_head", type=int, default=2)
        return parent_parser

    def get_ia_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def get_ia_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def get_ja_char_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def get_ja_mental_zero_embedding(self, batch_size=1) -> Union[None, torch.Tensor]:
        return None

    def forward(self, x: TOMMASTransformerInput, return_embeddings=False):
        # num_features = 1 (wall) + num goals (goal positions) + num agents (agent positions) +
        #           (num actions * num agents)
        # num_features_without_actions = 1 (wall) + num goals (goal positions) + num agents (agent positions)
        trajectory = x.trajectory
        query_state = x.query_state
        attention_mask = x.attention_mask
        embedding_positions = x.embedding_positions
        batch_size, seq_len, features, rows, cols = trajectory.shape
        gridworld_embeddings = self.gridworld_embedding_net(
            trajectory.reshape((batch_size * seq_len, features, rows, cols)))
        gridworld_embeddings = gridworld_embeddings.reshape(batch_size, seq_len, self.embedding_size)
        output = self.decoder(inputs_embeds=gridworld_embeddings, attention_mask=attention_mask).last_hidden_state
        output = output.reshape((seq_len * batch_size, self.embedding_size))
        pred_indices = torch.flatten(embedding_positions)
        spatialised_embeddings = spatialise(output[pred_indices], rows, cols)
        pred_action, pred_goal, pred_sr = self._forward_prediction_networks(query_state, (spatialised_embeddings,))

        if return_embeddings:
            return TOMMASTransformerPredictions(pred_action, pred_goal, pred_sr, (None, None, None, None))
        return TOMMASTransformerPredictions(pred_action, pred_goal, pred_sr)

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
        ia_features = range(1 + self.num_goals + self.num_agents + self.num_actions)
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
