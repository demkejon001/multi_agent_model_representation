from argparse import ArgumentParser

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

from tommas.agent_modellers.iterative_action_tommas import IterativeActionModeller
from tommas.agent_modellers.modeller_inputs import IterativeActionTransformerInput
from tommas.agent_modellers.modeller_outputs import IterativeTOMMASPredictions


class IterativeActionTOMMASTransformer(IterativeActionModeller):
    def __init__(self,
                 model_type: str,
                 model_name: str,
                 learning_rate: float,
                 optimizer_type: str,
                 num_joint_agent_features: int,
                 hidden_layer_features: int,
                 embedding_size: int,
                 n_head: int,
                 n_layer: int,
                 num_actions: int,
                 ):
        super().__init__(model_type=model_type, model_name=model_name, learning_rate=learning_rate,
                         optimizer_type=optimizer_type)
        self.save_hyperparameters()

        if hidden_layer_features is None:
            hidden_layer_features = []
        all_layers_features = [num_joint_agent_features] + hidden_layer_features + [embedding_size]
        layers = []
        for i in range(len(all_layers_features)-1):
            layers.append(nn.Linear(all_layers_features[i], all_layers_features[i+1]))
            if i < len(all_layers_features) - 2:
                layers.append(nn.ReLU())
        self.embedding_net = nn.Sequential(*layers)

        gpt2config = GPT2Config(vocab_size=2, n_positions=1024, n_ctx=1024, n_embd=embedding_size, n_layer=n_layer,
                                n_head=n_head, n_inner=None, activation_function='gelu')
        decoder = GPT2Model(gpt2config)
        self.decoder = decoder
        self.embedding_size = embedding_size
        self.action_pred_net = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )
        # self.action_pred_net = nn.Linear(embedding_size, num_actions)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("IterativeActionTOMMASTransformer")
        parser = IterativeActionModeller.add_model_specific_args(parser)
        parser.add_argument("--hidden_layer_features", nargs='+', type=int, default=[],
                            help="the number of features for the embedding network's hidden layers (default: None)")
        parser.add_argument("--embedding_size", type=int, default=2)
        parser.add_argument("--n_layer", type=int, default=2)
        parser.add_argument("--n_head", type=int, default=2)
        return parent_parser

    def forward(self, x: IterativeActionTransformerInput, return_embeddings=False, output_attentions=False,
                head_mask=None, output_hidden_states=False):
        trajectory = x.trajectory
        batch_size, seq_len, features = trajectory.shape
        input_embeddings = self.embedding_net(trajectory)
        decoder_output = self.decoder(inputs_embeds=input_embeddings, attention_mask=None,
                                      output_attentions=output_attentions, head_mask=head_mask,
                                      output_hidden_states=output_hidden_states)
        output = decoder_output.last_hidden_state
        embeddings = output.reshape((seq_len * batch_size, self.embedding_size))
        predictions = IterativeTOMMASPredictions(self.action_pred_net(embeddings))
        if return_embeddings:
            predictions.embeddings = embeddings
        if output_attentions:
            predictions.attentions = decoder_output.attentions
        if output_hidden_states:
            predictions.hidden_states = decoder_output.hidden_states
        return predictions



