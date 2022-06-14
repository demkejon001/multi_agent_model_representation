import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
from tqdm import tqdm

import torch
import ecco.analysis as analysis

# from tommas.agents.create_iterative_action_agents import get_random_iterative_action_agent, RandomStrategySampler
from tommas.data.multi_strategy_gridworld_dataset import play_episode, get_random_iter_agent, RandomStrategySampler, get_random_env
from tommas.data.gridworld_transforms import CoopGridworldTrajectory
# from tommas.data.iterative_action_dataset import IterativeActionTrajectory, play_episode
from tommas.helper_code.metrics import action_loss, action_acc
from experiments.experiment_base import load_modeller
from tommas.viz.embedding_responsibility import cosine_loss, get_responsible_embeddings_silhouette_score
from tommas.analysis.disentanglement import dci


data_filepath = "data/results/iterative_action/"
basename = "MultiStrategyRepresentationData"

agent_specific_param_metrics = ['starting_action_score', 'starting_action_dim',
                                'trigger_score', 'trigger_dim',
                                'action_pattern_score', 'action_pattern_dim',
                                'opponent_idx_score', 'opponent_idx_dim',
                                'starting_action_vs_rest_score', 'starting_action_vs_rest_dim',
                                'trigger_vs_rest_score', 'trigger_vs_rest_dim',
                                'action_pattern_vs_rest_score', 'action_pattern_vs_rest_dim',
                                'opponent_idx_vs_rest_score', 'opponent_idx_vs_rest_dim',
                                ]

representation_metrics = ["cluster_silhouette_score", "final_embedding_similarity", "final_embedding_distance",
                          "pca_50%", "pca_75%", "pca_80%", "pca_90%",
                          *agent_specific_param_metrics,
                          'shared_starting_action_score', 'shared_starting_action_dim',
                          'shared_trigger_score', 'shared_trigger_dim',
                          'shared_action_pattern_score', 'shared_action_pattern_dim',
                          'shared_opponent_idx_score', 'shared_opponent_idx_dim',
                          'agent_type_score', 'agent_type_dim',
                          'mirror_vs_rest_score', 'mirror_vs_rest_dim',
                          'grim_trigger_vs_rest_score', 'grim_trigger_vs_rest_dim',
                          'wsls_vs_rest_score', 'wsls_vs_rest_dim',
                          'mixed_trigger_pattern_vs_rest_score',
                          'mixed_trigger_pattern_vs_rest_dim', "disentanglement", "completeness", "informativeness"]


def get_all_potential_clusters(df, metadata, n_past=5, current_traj_len=19):
    all_agent_info = df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "embeddings", "opponent_idx"]]
    embeddings = []
    labels = []
    unique_labels = set()
    for agent_info in all_agent_info.iterrows():
        agent_id = agent_info[1].agent_id
        embeddings.append(agent_info[1].embeddings[0])
        agent_params = metadata[agent_id].copy()
        agent_params["opponent_idx"] = agent_info[1].opponent_idx
        for key in agent_params.keys():
            if isinstance(agent_params[key], list):
                agent_params[key] = ''.join([str(a) for a in agent_params[key]])

        label = frozenset(agent_params.items())
        labels.append(label)
        unique_labels.add(label)
    label_to_idx = dict()
    for i, label in enumerate(unique_labels):
        label_to_idx[label] = i

    true_labels = []
    for label in labels:
        true_labels.append(label_to_idx[label])
    return embeddings, true_labels


def create_all_potential_clusters(models, num_agents_per_cluster=20, seed=42, n_past=5, remove_action_patterns=False):
    starting_actions = [0, 1]
    mixed_strategies = [[1., 0], [0, 1.]]
    trigger_actions = [0, 1]
    rss = RandomStrategySampler(0, True)
    trigger_action_patterns = rss.trigger_action_patterns
    opponent_indices = [1, 2, 3]

    agent_infos = dict()
    agent_infos["mirror"] = [{"starting_action": 0}, {"starting_action": 1}]
    agent_clusters = []
    for starting_action, opponent_idx in itertools.product(starting_actions, opponent_indices):
        agent_type = "mirror"
        agent_info = {"starting_action": starting_action}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions, opponent_indices):
        agent_type = "grim_trigger"
        agent_info = {"starting_action": starting_action, "trigger_action": [trigger_action]}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions, opponent_indices):
        agent_type = "wsls"
        agent_info = {"starting_action": starting_action, "win_trigger": [trigger_action]}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    if remove_action_patterns:
        for mixed_strategy, trigger_action, opponent_idx in itertools.product(mixed_strategies, trigger_actions, opponent_indices):
            agent_type = "mixed_trigger_pattern"
            agent_info = {"mixed_strategy": mixed_strategy, "trigger_action": [trigger_action], "action_pattern": [0, 1, 0]}
            agent_clusters.append((agent_type, agent_info, opponent_idx))
    else:
        for mixed_strategy, trigger_action, trigger_action_pattern, opponent_idx in itertools.product(mixed_strategies, trigger_actions, trigger_action_patterns, opponent_indices):
            agent_type = "mixed_trigger_pattern"
            agent_info = {"mixed_strategy": mixed_strategy, "trigger_action": [trigger_action], "action_pattern": trigger_action_pattern}
            agent_clusters.append((agent_type, agent_info, opponent_idx))

    labels = []
    models_embeddings = [[] for _ in models]
    models_losses = [[] for _ in models]
    models_accs = [[] for _ in models]
    for agent_type, agent_info, opponent_idx in tqdm(agent_clusters):
        trajs = []
        for _ in range(num_agents_per_cluster):
            trajs.append(generate_random_traj(agent_type, agent_info, seed=seed, n_past=n_past))
            seed += 1
        for model, m_embeddings, m_losses, m_accs in zip(models, models_embeddings, models_losses, models_accs):
            char_embeddings, losses, accs = _get_model_prediction_data(model, trajs)
            m_embeddings.append(char_embeddings)
            m_losses.append(losses)
            m_accs.append(accs)
        agent_info_repr = ''
        for param in agent_info:
            if isinstance(agent_info[param], list):
                agent_info_repr += f"{param}:{''.join([str(a) for a in agent_info[param]])}, "
            else:
                agent_info_repr += f"{param}:{agent_info[param]}, "
        label = f"{agent_type}, {agent_info_repr}opponent_idx:{opponent_idx}"
        labels.extend([label for _ in range(num_agents_per_cluster)])

    for i, (embeddings, losses, accs) in enumerate(zip(models_embeddings, models_losses, models_accs)):
        models_embeddings[i] = np.concatenate(embeddings, axis=0)
        models_losses[i] = np.concatenate(losses, axis=0)
        models_accs[i] = np.concatenate(accs, axis=0)

    return models_embeddings, models_losses, models_accs, np.array(labels)


def get_multi_strat_agent_cluster_info(limited=True):
    starting_actions = [0, 1]
    mixed_strategies = [[1., 0], [0, 1.]]
    trigger_actions = [0, 1]
    rss = RandomStrategySampler(0, True)
    if not limited:
        trigger_action_patterns = rss.trigger_action_patterns
    else:
        three_action_pattern = rss.trigger_action_patterns[0]
        four_action_pattern = rss.trigger_action_patterns[-1]
        rand_action_patterns = [rss.trigger_action_patterns[i] for i in range(1, len(rss.trigger_action_patterns) - 1)]
        trigger_action_patterns = [three_action_pattern, four_action_pattern, rand_action_patterns]
    opponent_indices = [1, 2, 3]

    agent_infos = dict()
    agent_infos["mirror"] = [{"starting_action": 0}, {"starting_action": 1}]
    agent_clusters = []
    for starting_action, opponent_idx in itertools.product(starting_actions, opponent_indices):
        agent_type = "mirror"
        agent_info = {"starting_action": starting_action}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions,
                                                                           opponent_indices):
        agent_type = "grim_trigger"
        agent_info = {"starting_action": starting_action, "trigger_action": [trigger_action]}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions,
                                                                           opponent_indices):
        agent_type = "wsls"
        agent_info = {"starting_action": starting_action, "win_trigger": [trigger_action]}
        agent_clusters.append((agent_type, agent_info, opponent_idx))

    for mixed_strategy, trigger_action, trigger_action_pattern, opponent_idx in itertools.product(mixed_strategies,
                                                                                                  trigger_actions,
                                                                                                  trigger_action_patterns,
                                                                                                  opponent_indices):
        agent_type = "mixed_trigger_pattern"
        agent_info = {"mixed_strategy": mixed_strategy, "trigger_action": [trigger_action],
                      "action_pattern": trigger_action_pattern}
        agent_clusters.append((agent_type, agent_info, opponent_idx))
    return agent_clusters


def create_models_output_df(models_dict, num_agents_per_cluster=100, limited=True, seed=42, n_past=5):
    def get_agent_label():
        agent_info_repr = ''
        for param in agent_info:
            if param == "action_pattern":
                if isinstance(agent_info[param][0], list):
                    agent_info_repr += f"{param}:random, "
                    continue
            if isinstance(agent_info[param], list):
                agent_info_repr += f"{param}:{''.join([str(a) for a in agent_info[param]])}, "
            else:
                agent_info_repr += f"{param}:{agent_info[param]}, "
        return f"{agent_type}, {agent_info_repr}opponent_idx:{opponent_idx}"

    def fill_agent_info():
        starting_action = agent_info.get("starting_action", None)
        if starting_action is None:
            starting_action = agent_info.get("mixed_strategy", None)
            starting_action = starting_action.index(1.)

        trigger = agent_info.get("trigger_action", None)
        if trigger is None:
            trigger = agent_info.get("win_trigger", None)
        if type(trigger) == list:
            trigger = trigger[0]

        action_pattern = agent_info.get("action_pattern", None)
        if action_pattern is not None:
            if isinstance(action_pattern[0], list):
                action_pattern = "random"
            else:
                action_pattern = ''.join([str(action) for action in action_pattern])

        filled_agent_info = {"starting_action": starting_action,
                             "trigger": trigger,
                             "action_pattern": action_pattern}
        return filled_agent_info

    def extract_model_name_info():
        def get_params(param_str):
            param_str = param_str.split("_")[0]
            param_str = param_str.replace("[", "")
            param_str = param_str.replace("]", "")
            params = param_str.split(",")
            embed_size = int(params[0])
            n_layer = int(params[1])
            n_head = (1 if len(params) == 2 else int(params[2]))
            return embed_size, n_layer, n_head

        model_info = []
        for model_name in models_names:
            if "ttx" in model_name:
                split_model_name = model_name.split("ttx")
                if len(model_name.split("ttx")) == 2:
                    raise ValueError("Cant handle ttx_lstm or lstm_ttx models yet")
                else:
                    lstm_char = False
                    lstm_mental = False
            else:
                split_model_name = model_name.split("lstm")
                lstm_char = True
                lstm_mental = True

            char_embed, char_n_layer, char_n_head = get_params(split_model_name[1])
            mental_embed, mental_n_layer, mental_n_head = get_params(split_model_name[2])
            model_seed = 314
            if "seed" in model_name:
                model_seed = int(model_name.split("seed")[1].replace(".ckpt", ""))
            model_family = model_name.split("_seed")[0]
            model_info.append({"model_family": model_family, "model_seed": model_seed, "lstm_char": lstm_char,
                "lstm_mental": lstm_mental, "char_embedding_size": char_embed, "char_n_layer": char_n_layer,
                "char_n_head": char_n_head, "mental_embedding_size": mental_embed,
                "mental_n_layer": mental_n_layer, "mental_n_head": mental_n_head})
        return model_info

    def get_cluster_trajs():
        nonlocal seed
        cluster_trajs = []
        for i in range(num_agents_per_cluster):
            if limited:
                action_pattern = agent_info.get("action_pattern")
                if action_pattern is not None and type(action_pattern[0]) == list:
                    new_agent_info = agent_info.copy()
                    new_agent_info["action_pattern"] = action_pattern[i % len(action_pattern)]
                    cluster_trajs.append(generate_random_traj(agent_type, new_agent_info, seed=seed, n_past=n_past))
                    seed += 1
                else:
                    cluster_trajs.append(generate_random_traj(agent_type, agent_info, seed=seed, n_past=n_past))
                    seed += 1
            else:
                cluster_trajs.append(generate_random_traj(agent_type, agent_info, seed=seed, n_past=n_past))
                seed += 1
        return cluster_trajs

    agent_clusters = get_multi_strat_agent_cluster_info(limited=limited)
    initial_representation_data = []
    models_names = list(models_dict.keys())
    models_info = extract_model_name_info()

    for agent_type, agent_info, opponent_idx in tqdm(agent_clusters):
        agent_label = get_agent_label()
        filled_agent_info = fill_agent_info()
        trajs = get_cluster_trajs()
        for model_name, model_info in zip(models_names, models_info):
            char_embeddings, losses, accs = _get_model_prediction_data(models_dict[model_name], trajs)
            for n_p in range(n_past + 1):
                char_embedding = char_embeddings[:, n_p]
                loss = losses[n_p]
                acc = accs[n_p]
                initial_representation_data.append(
                    {"model_name": model_name, **model_info, "agent_label": agent_label, "agent_type": agent_type,
                     **filled_agent_info, "opponent_idx": opponent_idx, "n_past": n_p,
                     "char_embedding": char_embedding, "loss": loss, "acc": acc}
                )
    return pd.DataFrame(initial_representation_data)


def append_silhouette_scores_to_df(agent_specific_df):
    data = []
    models_names = agent_specific_df["model_name"].unique()
    for model_name in models_names:
        all_labels_and_embeddings = agent_specific_df[agent_specific_df["model_name"] == model_name][["agent_label", "n_past", "char_embedding"]]
        for n_past in range(6):
            labels_embeddings = all_labels_and_embeddings[all_labels_and_embeddings["n_past"] == n_past][
                ["agent_label", "char_embedding"]]
            labels = labels_embeddings["agent_label"].tolist()
            embeddings = labels_embeddings["char_embedding"].tolist()
            num_embed_per_cluster = embeddings[0].shape[0]
            repeat_labels = np.repeat(labels, num_embed_per_cluster)
            embeddings = np.concatenate(embeddings, axis=0)
            samples = silhouette_samples(embeddings, repeat_labels)
            for i, label in enumerate(labels):
                idx = num_embed_per_cluster * i
                label_average = np.average(samples[idx:idx + num_embed_per_cluster])
                data.append({"model_name": model_name,
                             "n_past": n_past,
                             "agent_label": label,
                             "cluster_silhouette_score": label_average})
    return agent_specific_df.merge(pd.DataFrame(data), how="inner", on=["model_name", "n_past", "agent_label"])


def _df_add_level(df):
    df.columns = pd.MultiIndex.from_tuples([(col, "") for col in df.columns], names=[None, None])
    return df


def append_pca_dims_to_df(agent_specific_df, model_specific_df):
    data = []
    var_scores = [.5, .75, .8, .9]
    models_names = agent_specific_df["model_name"].unique()
    for model_name in models_names:
        embedding_df = agent_specific_df[agent_specific_df["model_name"] == model_name][["n_past", "char_embedding"]]
        x = embedding_df.groupby(["n_past"])["char_embedding"].apply(lambda a: np.vstack(a))
        n_past = len(x)
        for n_p in range(n_past):
            if n_p > 0:
                n_embed_per_var = []
                sc = StandardScaler()
                embeddings = sc.fit_transform(x[n_p])
                cumsum_var = np.cumsum(PCA().fit(embeddings).explained_variance_ratio_)

                cumsum_var_idx = 0
                var_idx = 0
                while cumsum_var_idx < len(cumsum_var):
                    if cumsum_var[cumsum_var_idx] > var_scores[var_idx]:
                        n_embed = cumsum_var_idx + 1
                        n_embed_per_var.append(n_embed)
                        var_idx += 1
                        if var_idx >= len(var_scores):
                            break
                    else:
                        cumsum_var_idx += 1
            else:
                n_embed_per_var = [0, 0, 0, 0]
            data.append({"model_name": model_name, "n_past": n_p, "pca_50%": n_embed_per_var[0],
                         "pca_75%": n_embed_per_var[1], "pca_80%": n_embed_per_var[2], "pca_90%": n_embed_per_var[3]})
    df = pd.DataFrame(data)
    return model_specific_df.merge(_df_add_level(df), how="left", on=["model_name", "n_past"])


def append_final_embedding_similarity(agent_specific_df, model_specific_df):
    data = []
    models_names = agent_specific_df["model_name"].unique()
    for model_name in models_names:
        embedding_df = agent_specific_df[agent_specific_df["model_name"] == model_name][["n_past", "char_embedding"]]
        x = embedding_df.groupby(["n_past"])["char_embedding"].apply(lambda a: np.vstack(a))
        n_past = len(x)
        for n_p in range(n_past):
            # x[n_p].shape -> (data_points, embedding_size)
            sim_score = get_representation_similarity_score([x[n_p], x[n_past-1]], similarity="cka")[0]
            distance_score = np.average(np.linalg.norm(x[n_p] - x[n_past - 1], axis=1))
            data.append({"model_name": model_name, "n_past": n_p, "final_embedding_similarity": sim_score,
                         "final_embedding_distance": distance_score})
    df = pd.DataFrame(data)
    return model_specific_df.merge(_df_add_level(df), how="left", on=["model_name", "n_past"])


def append_dci_scores(agent_specific_df, model_specific_df):
    data = []
    n_past = 5
    for model_name, model_df in tqdm(agent_specific_df[agent_specific_df["n_past"] == n_past].groupby("model_name")):
        factors = []
        codes = []
        agent_type_cls_idx = {"grim_trigger": 0, "wsls": 1, "mixed_trigger_pattern": 2, "mirror": 3}
        for i in model_df[["agent_type", "starting_action", "opponent_idx", "char_embedding"]].iterrows():
            agent_type, start_action, opponent_idx, char_embed = i[1]
            factor = [[1 if j == agent_type_cls_idx[agent_type] else 0 for j in range(4)], [start_action],
                      [1. if j == opponent_idx else 0 for j in range(1, 4)]]
            factors += [factor for _ in range(len(char_embed))]
            codes.append(char_embed)

        codes = np.concatenate(codes)
        factors = np.array(factors, dtype=object)
        disentanglement, completeness, informativeness = dci(factors, codes)
        data.append({"model_name": model_name, "n_past": n_past,
                     "disentanglement": disentanglement,
                     "completeness": completeness,
                     "informativeness": informativeness})
    df = pd.DataFrame(data)
    return model_specific_df.merge(_df_add_level(df), how="left", on=["model_name", "n_past"])


def append_model_specific_param_scores(agent_specific_df, model_specific_df, mask_threshold=.8):
    def get_shared_param_scores():
        param_scores = {"model_name": model_name, "n_past": n_past}
        for agent_param in agent_params:
            embedding_df = agent_df[[agent_param, "char_embedding"]]
            labels = []
            embeddings = []
            for i, group_embeddings in enumerate(
                    embedding_df.groupby(agent_param)["char_embedding"].apply(lambda a: np.vstack(a))):
                labels = labels + [i for _ in range(group_embeddings.shape[0])]
                embeddings.append(group_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.array(labels)
            silhouette, feature_dim = get_param_silhouette_and_dim(embeddings, labels, mask_threshold)
            param_scores.update({f"shared_{agent_param}_score": silhouette, f"shared_{agent_param}_dim": feature_dim})
        return [param_scores]

    def get_agent_type_scores():
        labels = []
        embeddings = []
        for i, group_embeddings in enumerate(
                agent_df.groupby("agent_type")["char_embedding"].apply(lambda a: np.vstack(a))):
            labels = labels + [i for _ in range(group_embeddings.shape[0])]
            embeddings.append(group_embeddings)
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(labels)
        silhouette, feature_dim = get_param_silhouette_and_dim(embeddings, labels, mask_threshold)
        param_scores = {"model_name": model_name, "n_past": n_past,
                        "agent_type_score": silhouette, "agent_type_dim": feature_dim}
        return [param_scores]

    def get_agent_type_vs_rest_scores():
        param_scores = {"model_name": model_name, "n_past": n_past}
        for agent_type in agent_types:
            criteria = (agent_df["agent_type"] == agent_type)
            agent_type_embeddings = np.concatenate(
                agent_df[criteria]["char_embedding"].apply(lambda a: np.vstack(a)).tolist(), axis=0)
            not_agent_type_embeddings = np.concatenate(
                agent_df[~criteria]["char_embedding"].apply(lambda a: np.vstack(a)).tolist(), axis=0)

            embeddings = np.concatenate((agent_type_embeddings, not_agent_type_embeddings), axis=0)
            labels = np.array([0 for _ in range(len(agent_type_embeddings))] + [1 for _ in range(len(not_agent_type_embeddings))])
            silhouette, feature_dim = get_param_silhouette_and_dim(embeddings, labels, mask_threshold)
            param_scores.update({"agent_type": agent_type, f"{agent_type}_vs_rest_score": silhouette,
                                 f"{agent_type}_vs_rest_dim": feature_dim})
        return [param_scores]

    agent_params = ['starting_action', 'trigger', 'action_pattern', 'opponent_idx']
    agent_types = agent_specific_df["agent_type"].unique()
    dfs = dict(tuple(agent_specific_df[agent_specific_df["n_past"].isin([1, 2, 3, 4, 5])][["model_name", 'agent_type', *agent_params, "n_past", "char_embedding"]].groupby(["model_name", "n_past"])))
    shared_agent_param_data = []
    agent_type_data = []
    agent_type_vs_rest_data = []
    for (model_name, n_past), agent_df in tqdm(dfs.items()):
        shared_agent_param_data.extend(get_shared_param_scores())
        agent_type_data.extend(get_agent_type_scores())
        agent_type_vs_rest_data.extend(get_agent_type_vs_rest_scores())

    model_specific_df = model_specific_df.merge(_df_add_level(pd.DataFrame(shared_agent_param_data)), how="left",
                                                on=["model_name", "n_past"])
    model_specific_df = model_specific_df.merge(_df_add_level(pd.DataFrame(agent_type_data)), how="left",
                                                on=["model_name", "n_past"])
    return model_specific_df.merge(_df_add_level(pd.DataFrame(agent_type_vs_rest_data)), how="left",
                                   on=["model_name", "n_past"])


def get_param_silhouette_and_dim(embeddings, labels, mask_threshold):
    embedding_responsibility = cosine_loss(embeddings, labels, mask_threshold=mask_threshold)
    silhouette = -1
    feature_dim = sum(embedding_responsibility)
    if any(embedding_responsibility):
        silhouette = get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility)
    return silhouette, feature_dim


def append_agent_specific_param_scores(df, mask_threshold=.85):
    # Looking at how well we can tell an agent's parameter values from each other as well as from non-existent values, i.e. grim trigger's start action = 0 vs 1 (as well as vs non grim trigger agents)
    def get_agent_param_scores():
        data = []
        for agent_type in agent_types:
            param_scores = {"model_name": model_name, "n_past": n_past, "agent_type": agent_type}
            for agent_param in agent_params:
                criteria = (agent_df["agent_type"] == agent_type)
                embedding_df = agent_df[criteria][[agent_param, "char_embedding"]]
                if len(embedding_df[agent_param].unique()) > 1:
                    labels = []
                    embeddings = []
                    for i, group_embeddings in enumerate(
                            embedding_df.groupby(agent_param)["char_embedding"].apply(lambda a: np.vstack(a))):
                        labels = labels + [i for _ in range(group_embeddings.shape[0])]
                        embeddings.append(group_embeddings)
                    param_embeddings = np.concatenate(embeddings, axis=0)
                    param_labels = np.array(labels)
                    silhouette, feature_dim = get_param_silhouette_and_dim(param_embeddings, param_labels, mask_threshold)

                    not_agent_param_embeddings = np.concatenate(
                        agent_df[~criteria]["char_embedding"].apply(lambda a: np.vstack(a)).tolist(), axis=0)
                    labels = labels + [-1 for _ in range(not_agent_param_embeddings.shape[0])]
                    embeddings.append(not_agent_param_embeddings)
                    rest_embeddings = np.concatenate(embeddings, axis=0)
                    rest_labels = np.array(labels)
                    rest_silhouette, rest_feature_dim = get_param_silhouette_and_dim(rest_embeddings, rest_labels, mask_threshold)
                else:
                    silhouette = np.nan
                    feature_dim = np.nan
                    rest_silhouette = np.nan
                    rest_feature_dim = np.nan
                param_scores.update({f"{agent_param}_score": silhouette, f"{agent_param}_dim": feature_dim,
                                     f"{agent_param}_vs_rest_score": rest_silhouette,
                                     f"{agent_param}_vs_rest_dim": rest_feature_dim})
            data.append(param_scores)
        return data

    agent_params = ['starting_action', 'trigger', 'action_pattern', 'opponent_idx']
    agent_types = df["agent_type"].unique()
    dfs = dict(tuple(df[df["n_past"].isin([1, 2, 3, 4, 5])][["model_name", 'agent_type', *agent_params, "n_past", "char_embedding"]].groupby(["model_name", "n_past"])))
    agent_param_data = []
    for (model_name, n_past), agent_df in tqdm(dfs.items()):
        agent_param_data.extend(get_agent_param_scores())

    return df.merge(pd.DataFrame(agent_param_data), how="left", on=["model_name", "n_past", "agent_type"])


def get_dfs_representation_silhouette_score(dfs, metadata, metric="euclidean", **kwargs):
    silhouette_scores = dict()
    for df_name, df in dfs.items():
        embeddings, labels = get_all_potential_clusters(df["combined"], metadata, **kwargs)
        silhouette_scores[df_name] = silhouette_score(embeddings, labels, metric=metric)
    return silhouette_scores


def get_models_representation_silhouette_score(models, metric="euclidean", **kwargs):
    models_embeddings, _, _, labels = create_all_potential_clusters(models, **kwargs)
    return calculate_models_embedding_representation_silhouette_score(models_embeddings, labels, metric)


def calculate_models_embedding_representation_silhouette_score(models_embeddings, labels, metric="euclidean"):
    models_silhouette_scores = []
    for embeddings in models_embeddings:
        n_past_silhouette_scores = []
        for n_past in range(embeddings.shape[-1]):
            n_past_silhouette_scores.append(silhouette_score(embeddings[:, n_past], labels, metric=metric))
        models_silhouette_scores.append(n_past_silhouette_scores)
    return np.array(models_silhouette_scores)


def get_models_representation_silhouette_score(models, metric="euclidean", **kwargs):
    silhouette_scores = []
    models_embeddings, labels = create_all_potential_clusters(models, **kwargs)
    for embeddings in models_embeddings:
        silhouette_scores.append(silhouette_score(embeddings, labels, metric=metric))
    return np.array(silhouette_scores)


def plot_representation_silhouette_score_over_n_past(dfs, metadata, metric="euclidean"):
    n_past_silhouette_scores = {df_name: [] for df_name in dfs}
    n_pasts = list(range(6))
    for n_past in n_pasts:
        silhouette_scores = get_dfs_representation_silhouette_score(dfs, metadata, metric, n_past=n_past)
        for df_name, score in silhouette_scores.items():
            n_past_silhouette_scores[df_name].append(score)
    df_names = list(n_past_silhouette_scores.keys())
    for df_name in df_names:
        plt.plot(n_pasts, n_past_silhouette_scores[df_name])
    plt.legend(df_names)
    plt.show()


def _get_model_prediction_data(model, trajs):
    transform = CoopGridworldTrajectory()

    n_past = len(trajs[0]) - 1
    current_gridworld_len = len(trajs[0][0])

    batch = []
    for agents_trajs in trajs:
        batch.append(transform(agents_trajs, 0, n_past, np.inf))
    collate_fn = transform.get_collate_fn()
    input, output = collate_fn(batch)
    input.to(model.device)
    output.to(model.device)

    with torch.no_grad():
        predictions = model(input, return_embeddings=True)
        char_embeddings = predictions.embeddings[0].cpu().numpy()
        action_losses = []
        action_accs = []
        goal_losses = []
        goal_accs = []
        agent_idx_offset = (n_past + 1) * current_gridworld_len
        for n_p in range(n_past + 1):
            n_p_offset = current_gridworld_len * n_p
            n_p_indices = []
            for agent_idx in range(len(batch)):
                start_idx = agent_idx_offset * agent_idx + n_p_offset
                n_p_indices += list(range(start_idx, start_idx + current_gridworld_len))

            pred_action = predictions.action[n_p_indices]
            true_action = output.action[n_p_indices]
            action_losses.append(action_loss(pred_action, true_action).item())
            action_accs.append(action_acc(pred_action, true_action))

            # pred_goal = predictions.goal_consumption[n_p_indices]
            # true_goal = output.goal_consumption[n_p_indices]
            # goal_losses.append(goal_consumption_loss(pred_goal, true_goal).item())
            # goal_accs.append(goal_acc(pred_goal, true_goal))
        # return char_embeddings, np.array(action_losses), np.array(action_accs), np.array(goal_losses), \
        #        np.array(goal_accs)
        return char_embeddings, np.array(action_losses), np.array(action_accs)
# def _get_model_prediction_data(model, trajs):
#     transform = IterativeActionFullPastCurrentSplit()
#
#     n_past = len(trajs[0]) - 1
#     current_gridworld_len = len(trajs[0][0]) - 1
#
#     batch = []
#     for agents_trajs in trajs:
#         batch.append(transform([IterativeActionTrajectory([0, 1], traj) for traj in agents_trajs], 0, n_past, np.inf,
#                                shuffled_agent_indices=[0, 1, 2, 3]))
#     collate_fn = transform.get_collate_fn()
#     input, output = collate_fn(batch)
#     input.to(model.device)
#     output.to(model.device)
#
#     with torch.no_grad():
#         predictions = model(input, return_embeddings=True)
#         char_embeddings = predictions.embeddings[0].cpu().numpy()
#         losses = []
#         accs = []
#         agent_idx_offset = (n_past + 1) * current_gridworld_len
#         for n_p in range(n_past+1):
#             n_p_offset = current_gridworld_len * n_p
#             n_p_indices = []
#             for agent_idx in range(len(batch)):
#                 start_idx = agent_idx_offset * agent_idx + n_p_offset
#                 n_p_indices += list(range(start_idx, start_idx + current_gridworld_len))
#
#             pred_action = predictions.action[n_p_indices]
#             true_action = output.action[n_p_indices]
#
#             losses.append(action_loss(pred_action, true_action).item())
#             accs.append(action_acc(pred_action, true_action))
#
#         return char_embeddings, np.array(losses), np.array(accs)


# def generate_random_traj(agent_type, agent_kwargs, seed=42, n_past=5, num_agents=2, opponent_idx=1):
#     if opponent_idx < 1:
#         raise ValueError(f"opponent_idx needs to be greater than 1")
#     rss = RandomStrategySampler(seed, True)
#     np.random.seed(seed)
#     agents = [get_random_iterative_action_agent(agent_type, rss, -1, 0, **agent_kwargs)[0]]
#     for i in range(1, num_agents):
#         agents.append(get_random_iterative_action_agent("mixed_strategy", rss, -1, i)[0])
#     traj = np.array([play_episode(agents) for _ in range(n_past + 1)], dtype=np.int64)
#
#     agent_indices = list(range(num_agents))
#     agent_indices[1], agent_indices[opponent_idx] = agent_indices[opponent_idx], agent_indices[1]
#     return traj[:, :, agent_indices]

def generate_random_traj(agent_type, agent_kwargs, seed=42, n_past=5):
    rss = RandomStrategySampler(seed, True)
    agents = get_random_iter_agent(agent_type, [-1, -1, -1, -1], rss, **agent_kwargs)[0]
    env = get_random_env(reward_funcs=[agent.reward_function for agent in agents])
    return [play_episode(env, agents) for _ in range(n_past+1)]


def get_models_similarity_comparison(models_char_embeddings, similarity="cka"):
    similarity_fn = _get_similarity_fn(similarity)
    similarity_scores = np.zeros((len(models_char_embeddings), len(models_char_embeddings)))
    for i in range(len(models_char_embeddings)):
        for j in range(len(models_char_embeddings)):
            similarity_scores[i, j] = similarity_fn(models_char_embeddings[i].T, models_char_embeddings[j].T)
    return similarity_scores


def _get_similarity_fn(similarity_fn_name):
    if similarity_fn_name == "cka":
        similarity_fn = analysis.cka
    elif similarity_fn_name == "cca":
        similarity_fn = analysis.cca
    elif similarity_fn_name == "pwcca":
        similarity_fn = analysis.pwcca
    else:
        raise ValueError(f"Unrecognized similarity {similarity_fn_name}")
    return similarity_fn


def get_representation_similarity_score(models_char_embeddings, similarity="cka"):
    similarity_fn = _get_similarity_fn(similarity)
    similarity_scores = []
    for i in range(len(models_char_embeddings)):
        for j in range(len(models_char_embeddings)):
            if i != j:
                similarity_scores.append(similarity_fn(models_char_embeddings[i].T, models_char_embeddings[j].T))
    return np.average(similarity_scores), similarity_scores


def calculate_df_corr(df):
    rho = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1] if x <= t]))
    return rho.round(2).astype(str) + p


def lstm_ttx_boxplot_comparison(df, metric, n_past=5, x="char_embedding_size", agg_metric="mean", ax=None):
    past_df = df.copy()
    past_df['Architecture'] = np.where(past_df['lstm_char'] == True, "lstm", "transformer")
    if isinstance(n_past, int):
        n_past = [n_past]
    past_df = past_df[(past_df["n_past"].isin(n_past))]

    if (metric, agg_metric) in past_df:
        metric = (metric, agg_metric)

    sns.boxplot(x=past_df[x],
                y=past_df[metric],
                hue=past_df["Architecture"],
                ax=ax)


def lstm_ttx_agent_param_boxplot_comparison(df, param, n_past=5, x="char_embedding_size", compare=None):
    if compare is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        lstm_ttx_boxplot_comparison(df, metric=param+"_score", n_past=n_past, x=x, ax=axes[0])
        lstm_ttx_boxplot_comparison(df, metric=param+"_dim", n_past=n_past, x=x, ax=axes[1])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        lstm_ttx_boxplot_comparison(df, metric=param+"_score", n_past=n_past, x=x, ax=axes[0, 0])
        lstm_ttx_boxplot_comparison(df, metric=param+"_dim", n_past=n_past, x=x, ax=axes[0, 1])
        if compare == "shared":
            compare_param = "shared_" + param
        elif compare == "rest":
            compare_param = param + "_vs_rest"
        else:
            raise ValueError(f"Don't recognize compare={compare}")

        lstm_ttx_boxplot_comparison(df, metric=compare_param + "_score", n_past=n_past, x=x, ax=axes[1, 0])
        lstm_ttx_boxplot_comparison(df, metric=compare_param + "_dim", n_past=n_past, x=x, ax=axes[1, 1])
        axes[0, 0].set_ylim((-1, 1))
        axes[1, 0].set_ylim((-1, 1))


def calculate_model_specific_corr(df, agg_metric="mean"):
    # df = df[["char_embedding_size", "loss", "acc", *representation_metrics]]
    df = df[[
        ("char_embedding_size", ""),
        ('loss', agg_metric),
        ('acc', agg_metric),
        *[(metric, agg_metric) if ((metric, agg_metric) in df) else (metric, "") for metric in representation_metrics]]]
    return calculate_df_corr(df)


def calculate_model_param_corr(df, agg_metric="mean"):
    # df = df[[
    #     ("char_embedding_size", ""),
    #     ('loss', agg_metric),
    #     ('acc', agg_metric),
    #     ('diff_init_similarity_score', agg_metric),
    #     *[(metric, agg_metric) if ((metric, agg_metric) in df) else (metric, "") for metric in representation_metrics]]]
    df = df[[
        ("char_embedding_size", ""),
        ('diff_init_similarity_score', agg_metric)]]
    return calculate_df_corr(df)


def calculate_and_combine_model_corr(df, corr_fn, n_past=5, **kwargs):
    return pd.DataFrame(corr_fn(df[(df["n_past"] == n_past) & (df["lstm_char"] == True)], **kwargs)["char_embedding_size"]).join(corr_fn(df[(df["n_past"] == n_past) & (df["lstm_char"] == False)], **kwargs)["char_embedding_size"], lsuffix='_lstm', rsuffix='_ttx')


def aggregate_model_param_data(model_specific_df):
    model_params = ["model_family", 'lstm_char', 'lstm_mental', 'char_embedding_size', 'char_n_layer', 'char_n_head',
                    'mental_embedding_size', 'mental_n_layer', 'mental_n_head']
    model_param_df = model_specific_df[[*model_params, "n_past", "loss", "acc", *representation_metrics, ]]
    model_param_df = model_param_df.groupby([*model_params, "n_past"]).agg(['mean']).reset_index()
    model_param_df.columns = model_param_df.columns.droplevel(2)
    return model_param_df


def append_representation_similarity_over_diff_init(agent_specific_df, model_param_df):
    data = []
    multi_init_embedding_df = agent_specific_df[["model_family", "model_seed", "n_past", "char_embedding"]]
    models_families = multi_init_embedding_df["model_family"].unique()
    for n_p in range(6):
        past_df = multi_init_embedding_df[multi_init_embedding_df["n_past"] == n_p]
        dfs = dict(tuple(past_df.groupby(["model_family"])))
        for model_family in models_families:
            embeddings = dfs[model_family].groupby(["model_seed"])["char_embedding"].apply(
                lambda a: np.vstack(a)).tolist()
            avg_sim_score, indiv_sim_scores = get_representation_similarity_score(embeddings, similarity="cka")
            data.append({("model_family", ""): model_family,
                         ("n_past", ""): n_p,
                         ("diff_init_similarity_score", "count"): len(indiv_sim_scores),
                         ("diff_init_similarity_score", "mean"): avg_sim_score,
                         ("diff_init_similarity_score", "std"): np.std(indiv_sim_scores),
                         ("diff_init_similarity_score", "min"): np.min(indiv_sim_scores),
                         ("diff_init_similarity_score", "25%"): np.percentile(indiv_sim_scores, .25),
                         ("diff_init_similarity_score", "50%"): np.percentile(indiv_sim_scores, .5),
                         ("diff_init_similarity_score", "75%"): np.percentile(indiv_sim_scores, .75),
                         ("diff_init_similarity_score", "max"): np.max(indiv_sim_scores),
                         })

    return model_param_df.merge(pd.DataFrame(data), how="left", on=[("model_family", ""), ("n_past", "")])


def aggregate_architecture_data(model_specific_df):
    architecture_df = model_specific_df[["lstm_char", "lstm_mental", "n_past", "loss", "acc", *representation_metrics]]
    architecture_df = architecture_df.groupby(["lstm_char", "lstm_mental", "n_past"]).agg(['mean']).reset_index()
    architecture_df.columns = architecture_df.columns.droplevel(2)
    return architecture_df


def append_representation_similarity_over_architecture(agent_specific_df, architecture_df):
    data = []
    for n_p in range(6):
        for lstm_char, lstm_mental in [(True, True), (False, False)]:
            criteria = (agent_specific_df["lstm_char"] == lstm_char) & (agent_specific_df["lstm_mental"] == lstm_mental) & (agent_specific_df["n_past"] == n_p)
            embedding_df = agent_specific_df[criteria][["model_name", "model_seed", "char_embedding"]]
            architecture_seed_scores = []
            for seed in range(1, 6):
                embeddings = embedding_df[["model_name", "char_embedding"]]
                char_embeddings = embeddings.groupby("model_name")["char_embedding"].apply(lambda a: np.vstack(a)).tolist()
                architecture_seed_scores.extend(get_representation_similarity_score(char_embeddings, similarity="cka")[1])
            data.append({("lstm_char", ""): lstm_char,
                         ("lstm_mental", ""): lstm_mental,
                         ("n_past", ""): n_p,
                         ("architecture_similarity_score", "count"): len(architecture_seed_scores),
                         ("architecture_similarity_score", "mean"): np.average(architecture_seed_scores),
                         ("architecture_similarity_score", "std"): np.std(architecture_seed_scores),
                         ("architecture_similarity_score", "min"): np.min(architecture_seed_scores),
                         ("architecture_similarity_score", "25%"): np.percentile(architecture_seed_scores, .25),
                         ("architecture_similarity_score", "50%"): np.percentile(architecture_seed_scores, .5),
                         ("architecture_similarity_score", "75%"): np.percentile(architecture_seed_scores, .75),
                         ("architecture_similarity_score", "max"): np.max(architecture_seed_scores),
                         })
    return architecture_df.merge(pd.DataFrame(data), how="left", on=[("lstm_char", ""), ("lstm_mental", ""), ("n_past", "")])


def save_representation_data(df, representation_type):
    if representation_type == "model_specific":
        save_type = "ModelSpecific"
    elif representation_type == "model_param":
        save_type = "ModelParam"
    elif representation_type == "architecture":
        save_type = "Architecture"
    elif representation_type == "agent_specific":
        save_type = "AgentSpecific"
    else:
        raise ValueError(f"Don't recognize representation_type {representation_type}")
    if not os.path.isdir(data_filepath):
        os.makedirs(data_filepath)
    df.to_pickle(data_filepath + basename + "_" + save_type + ".pickle")


def load_representation_data(representation_type):
    if representation_type == "model_specific":
        save_type = "ModelSpecific"
    elif representation_type == "model_param":
        save_type = "ModelParam"
    elif representation_type == "architecture":
        save_type = "Architecture"
    elif representation_type == "agent_specific":
        save_type = "AgentSpecific"
    else:
        raise ValueError(f"Don't recognize representation_type {representation_type}")
    return pd.read_pickle(data_filepath + basename + "_" + save_type + ".pickle")


def get_agent_specific_data(model_dict, save=False, num_agents_per_cluster=100):
    print("Getting Model Specific Data")
    print("\tFetching models output DF")
    df = create_models_output_df(model_dict, num_agents_per_cluster=num_agents_per_cluster)
    print("\tFetching agent specific param scores")
    df = append_agent_specific_param_scores(df)
    print("\tFetching silhouette scores")
    df = append_silhouette_scores_to_df(df)
    if save:
        save_representation_data(df, representation_type="agent_specific")
    return df


def aggregate_model_specific_data(agent_specific_df):
    model_params = ["model_name", "model_seed", "model_family", 'lstm_char', 'lstm_mental', 'char_embedding_size',
                    'char_n_layer', 'char_n_head', 'mental_embedding_size', 'mental_n_layer', 'mental_n_head']

    model_specific_df = agent_specific_df[[*model_params, "n_past", "loss", "acc", "cluster_silhouette_score",
                                           *agent_specific_param_metrics]]
    model_specific_df = model_specific_df.groupby([*model_params, "n_past"]).agg(['describe']).reset_index()
    model_specific_df.columns = model_specific_df.columns.droplevel(1)
    return model_specific_df


def extract_model_specific_data(agent_specific_df, save=False):
    print("Getting Model Specific Data")
    print("\tAggregate models specific data")
    model_specific_df = aggregate_model_specific_data(agent_specific_df)
    print("\tFetching last layer similarity")
    model_specific_df = append_final_embedding_similarity(agent_specific_df, model_specific_df)
    print("\tFetching PCA dims")
    model_specific_df = append_pca_dims_to_df(agent_specific_df, model_specific_df)
    print("\tFetching agent param scores")
    model_specific_df = append_model_specific_param_scores(agent_specific_df, model_specific_df)
    print("\tFetching DCI scores")
    model_specific_df = append_dci_scores(agent_specific_df, model_specific_df)
    if save:
        save_representation_data(model_specific_df, representation_type="model_specific")
    return model_specific_df


def extract_model_param_data(agent_specific_df, model_specific_df, save=False):
    print("Getting Model Parameter Data")
    print("\tAggregating model parameter data")
    model_param_df = aggregate_model_param_data(model_specific_df)
    print("\tFetching representation similarity over different initalizations")
    model_param_df = append_representation_similarity_over_diff_init(agent_specific_df, model_param_df)
    if save:
        save_representation_data(model_param_df, representation_type="model_param")
    return model_param_df


def extract_architecture_data(agent_specific_df, model_specific_df, save=False):
    print("Getting Architecture Data")
    print("\tAggregating architecture data")
    architecture_df = aggregate_architecture_data(model_specific_df)
    print("\tFetching representation similarity over architecture")
    architecture_df = append_representation_similarity_over_architecture(agent_specific_df, architecture_df)
    if save:
        save_representation_data(architecture_df, representation_type="architecture")
    return architecture_df


def load_all_multi_strat_models():
    # trained_on = "JAGrim4,JAMirror4,JAMixTgrPtrn4,JAWSLS4"
    #
    # multi_seeds = list(range(1, 6))
    # params = [("[64,1]", "[128,1]", multi_seeds),
    #           ("[64,1]", "[64,2]", multi_seeds),
    #           ("[80,1]", "[64,1]", multi_seeds),
    #           ("[80,1]", "[48,2]", multi_seeds),
    #           ("[128,1]", "[48,2]", multi_seeds),
    #           ("[256,1]", "[48,2]", multi_seeds),
    #
    #           ("[48,6,2]", "[48,8,4]", multi_seeds),
    #           ("[64,8,4]", "[48,8,4]", multi_seeds),
    #           ("[128,2,4]", "[48,8,4]", multi_seeds),
    #           ("[80,4,4]", "[64,4,4]", multi_seeds),
    #           ("[48,8,4]", "[64,4,4]", multi_seeds),
    #           ("[128,4,8]", "[64,4,4]", multi_seeds),
    #           ("[64,4,4]", "[64,4,4]", multi_seeds),
    #
    #           ("[64,1]", "[48,2]", None),
    #           ("[80,2]", "[64,1]", None),
    #           ("[128,1]", "[48,1]", None),
    #           ("[256,1]", "[48,2]", None),
    #           ("[128,2]", "[48,2]", None),
    #           ("[512,1]", "[48,2]", None),
    #
    #           ("[32,4,2]", "[48,8,4]", None),
    #           ("[128,8,2]", "[48,8,4]", None),
    #           ("[64,8,4]", "[128,2,8]", None),
    #           ]


    model_dict = dict()
    model_dirpath = "data/models/iterative_action/"
    for filename in np.sort(os.listdir(model_dirpath)):

        modeller_params = filename.split("JAWSLS4]_")[1].replace(".ckpt", "")

        modeller_dirpath = model_dirpath + filename
        model_dict[modeller_params] = load_modeller(modeller_dirpath).to("cuda")
    #
    #
    #     model_dict
    # for char_params, mental_params, seeds in params:
    #     if char_params.count(',') == 2:
    #         char_net = "ttx"
    #     else:
    #         char_net = "lstm"
    #     if mental_params.count(',') == 2:
    #         mental_net = "ttx"
    #     else:
    #         mental_net = "lstm"
    #     if seeds != None:
    #         for seed in seeds:
    #             modeller_dirpath = f"data/models/IterPastCur[{trained_on}]_{char_net}{char_params}_{mental_net}{mental_params}_seed{seed}.ckpt"
    #             model_dict[f"{char_net}{char_params}_{mental_net}{mental_params}_seed{seed}"] = load_modeller(modeller_dirpath).to("cuda")
    #     else:
    #         modeller_dirpath = f"data/models/IterPastCur[{trained_on}]_{char_net}{char_params}_{mental_net}{mental_params}.ckpt"
    #         model_dict[f"{char_net}{char_params}_{mental_net}{mental_params}"] = load_modeller(modeller_dirpath).to("cuda")

    return model_dict


def collate_and_save_all_representation_data():
    model_dict = load_all_multi_strat_models()
    print("Getting results for", list(model_dict.keys()))
    agent_specific_df = get_agent_specific_data(model_dict, save=True)
    model_specific_df = extract_model_specific_data(agent_specific_df, save=True)
    del model_dict
    extract_model_param_data(agent_specific_df, model_specific_df, save=True)
    extract_architecture_data(agent_specific_df, model_specific_df, save=True)
