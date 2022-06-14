from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import GPT2Model, GPT2Config
from pytorch_lightning.core.lightning import LightningModule

from src.helper_code.metrics import action_loss, goal_consumption_loss, successor_representation_loss, \
    action_acc, goal_acc, successor_representation_acc
from src.agent_modellers.modeller_inputs import TensorClass
from src.agent_modellers.modeller_outputs import AgentObjectives


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

    def forward_step(self, batch: Tuple[TensorClass, AgentObjectives], batch_idx, forward_type):
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
        return action_loss(pred_agent_objectives.action, true_agent_objectives.action)

    def _goal_loss(self, pred_agent_objectives, true_agent_objectives):
        if self.no_goal_loss:
            return 0
        return goal_consumption_loss(pred_agent_objectives.goal_consumption,
                                     true_agent_objectives.goal_consumption)

    def _srs_loss(self, pred_agent_objectives, true_agent_objectives):
        if self.no_sr_loss:
            return 0
        return successor_representation_loss(pred_agent_objectives.successor_representation,
                                             true_agent_objectives.successor_representation)

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
