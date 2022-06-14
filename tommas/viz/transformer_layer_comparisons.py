import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
import ecco
from ecco import LM, pack_tokenizer_config
import ecco.analysis as analysis

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import ipywidgets as widgets
from IPython.display import display

from tommas.agent_modellers.iterative_action_tommas_transformer import IterativeActionTOMMASTransformer
from tommas.agents.create_iterative_action_agents import get_random_iterative_action_agent, RandomStrategySampler
from tommas.data.datamodule_factory import make_datamodule
from experiments.experiment_base import load_modeller

from tommas.agent_modellers.iterative_action_tommas_transformer import IterativeActionTOMMASTransformer
from tommas.data.gridworld_transforms import IterativeActionFullTrajectory, IterativeActionTrajectory


def get_ttx_output(model: IterativeActionTOMMASTransformer, trajectories, head_mask):
    for trajectory in trajectories:
        model()


def get_ttx_output(model, agents_trajs, head_mask=None, timestep=-1, n_past=0, attention_layer=-1,
                   return_embeddings=False, output_attentions=False, output_hidden_states=False):
    transform = IterativeActionFullTrajectory()

    batch = []
    for trajs in agents_trajs:
        batch.append(transform([IterativeActionTrajectory([0, 1], traj) for traj in trajs], 0, n_past, np.inf))
    collate_fn = transform.get_collate_fn()
    input, output = collate_fn(batch)
    seq_len = input.trajectory.shape[1]

    if head_mask is None:
        head_mask = [1 for _ in range(model.hparams["n_head"])]

    ttx_output = dict()
    with torch.no_grad():
        predictions = model(input, return_embeddings=True, head_mask=torch.tensor(head_mask, dtype=float),
                            output_attentions=True, output_hidden_states=True)

        losses = F.cross_entropy(predictions.action, output.action, reduction="none").numpy()
        ttx_output["loss"] = losses

        pred_max_indices = torch.max(predictions.action, dim=1)[1].unsqueeze(1)
        accs = output.action.unsqueeze(1).eq(pred_max_indices).numpy()
        ttx_output["acc"] = accs

        if return_embeddings:
            embeddings = predictions.embeddings
            embeddings = embeddings.reshape((len(batch), seq_len, -1))
            # embeddings = embeddings[:, -1, :].numpy()
            embeddings = embeddings[:, timestep, :].numpy()
            ttx_output["embedding"] = embeddings

        if output_attentions:
            #             attentions = predictions.attentions[attention_layer][0, :, timestep, :].numpy()
            attentions = predictions.attentions[attention_layer][0, :, :, :].numpy()
            ttx_output["attention"] = attentions

        if output_hidden_states:
            hidden_states = np.array([hidden_state.numpy() for hidden_state in predictions.hidden_states])
            ttx_output["hidden_state"] = hidden_states

    return ttx_output



def get_head_embeddings(strategy, n_layer, n_head, trajectories):
    num_embeddings = 400
    ones_head_mask = np.ones((n_layer, n_head))
    head_embeddings = [[[] for _ in range(n_head)] for _ in range(n_layer)]
    for i in range(ones_head_mask.shape[0]):
        for j in range(ones_head_mask.shape[1]):
            head_mask = ones_head_mask.copy()
            head_mask[i, j] = 0
            np.random.seed(i*j+j)
            head_embeddings[i][j] = get_final_embeddings(trajectories, head_mask)

    np.random.seed(100)
    embeddings_all = get_final_embeddings(traj, [1 for _ in range(n_head)])
    return np.array(embeddings_all), np.array(head_embeddings)


def plot_head_mask_embeddings(model):

    pass