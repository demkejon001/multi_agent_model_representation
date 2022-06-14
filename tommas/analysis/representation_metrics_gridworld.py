import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, f1_score, recall_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
from tqdm import tqdm

import torch
import ecco.analysis as analysis
from pytorch_lightning import seed_everything

from tommas.agents.create_gridworld_agents import get_random_gridworld_agent, RandomAgentParamSampler
from tommas.data.gridworld_transforms import CoopGridworldTrajectory
from tommas.data.cooperative_gridworld_dataset import play_episode, get_random_env
from tommas.helper_code.metrics import action_loss, action_acc, goal_consumption_loss, goal_acc
from experiments.experiment_base import load_modeller
from tommas.data.dataset_base import save_dataset, load_dataset
from tommas.viz.embedding_responsibility import cosine_loss, get_responsible_embeddings_silhouette_score
from tommas.analysis.disentanglement import dci


data_filepath = "data/results/gridworld/"
basename = "GridworldRepresentationData"

agent_params = ["goal_ranker_type", "goal_rewards"]
shared_agent_params = ["shared_" + param for param in agent_params]

agent_specific_param_metrics = [*[param + "_score" for param in agent_params],
                                *[param + "_dim" for param in agent_params]]

representation_metrics = ["cluster_silhouette_score", "modified_cluster_silhouette_score",
                          "final_embedding_similarity", "final_embedding_distance",
                          "final_embedding_euclidean_similarity", "final_embedding_euclidean_similarity_norm",
                          "final_embedding_cosine_similarity",
                          "pca_50%", "pca_75%", "pca_80%", "pca_90%",
                          *agent_specific_param_metrics,
                          "disentanglement", "completeness", "informativeness"]


def calculate_model_specific_corr(df, agg_metric="mean"):
    df = df[[
        ("char_embedding_size", ""),
        ('pca_75%', ''),
        ('pca_50%', ''),
        ('pca_80%', ''),
        ('pca_90%', ''),
        ('action_loss', agg_metric),
        ('action_acc', agg_metric),
        ('goal_loss', agg_metric),
        ('goal_acc', agg_metric),
        ("cluster_silhouette_score", agg_metric),
        # ("modified_cluster_silhouette_score", agg_metric),
        # *[(metric, agg_metric) for metric in agent_specific_param_metrics]
        *[(metric + "_score", "") for metric in shared_agent_params],
        *[(metric + "_dim", "") for metric in shared_agent_params],
        ("agent_type_score", ""),
        ("agent_type_dim", ""),
        ("disentanglement", ""),
        ("completeness", ""),
        ("informativeness", ""),
    ]]
    return calculate_df_corr(df)


def calculate_model_param_corr(df, agg_metric="mean"):
    df = df[[
        ("char_embedding_size", ""),
        ('diff_init_similarity_score', agg_metric)]]
    return calculate_df_corr(df)


def get_gridworld_agent_cluster_info():
    agent_clusters = []
    goal_ranker_types = ["highest", "closest", "discount"]
    all_goal_rewards = RandomAgentParamSampler(1, False).goal_rewards

    for goal_ranker_type in goal_ranker_types:
        for goal_rewards in all_goal_rewards:
            agent_info = {"goal_ranker_type": goal_ranker_type, "goal_rewards": goal_rewards}
            agent_clusters.append(("collaborative", agent_info.copy()))
            agent_clusters.append(("independent", agent_info.copy()))

    return agent_clusters


def create_models_output_df(models_dict, num_agents_per_cluster=300, seed=42, n_past=5):
    def get_agent_label():
        agent_info_repr = ''
        for param in agent_info:
            agent_info_repr += f"{param}:{agent_info[param]}, "
        return f"{agent_type}, {agent_info_repr}"

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

    seed_everything(seed)
    agent_clusters = get_gridworld_agent_cluster_info()
    initial_representation_data = []
    models_names = list(models_dict.keys())
    models_info = extract_model_name_info()

    step_size = 20
    for agent_type, agent_info in tqdm(agent_clusters):
        agent_label = get_agent_label()
        agent_type, agent_label, filled_agent_info, trajs = load_analysis_data(agent_label)
        for model_name, model_info in zip(models_names, models_info):
            char_embeddings = []
            action_losses = np.zeros((n_past + 1, ))
            action_accs = np.zeros((n_past + 1, ))
            goal_losses = np.zeros((n_past + 1, ))
            goal_accs = np.zeros((n_past + 1, ))
            goal_recalls = np.zeros((n_past + 1,))
            goal_f1s = np.zeros((n_past + 1,))
            for i in range(0, min(num_agents_per_cluster, len(trajs)), step_size):
                c_e, a_l, a_a, g_l, g_a, g_r, g_f1 = _get_model_prediction_data(models_dict[model_name], trajs[i:i+step_size])
                char_embeddings.append(c_e)
                action_losses += a_l
                action_accs += a_a
                goal_losses += g_l
                goal_accs += g_a
            num_steps = len(char_embeddings)
            char_embeddings = np.concatenate(char_embeddings, axis=0)
            action_losses = action_losses / num_steps
            action_accs = action_accs / num_steps
            goal_losses = goal_losses / num_steps
            goal_accs = goal_accs / num_steps
            goal_recalls = goal_recalls / num_steps
            goal_f1s = goal_f1s / num_steps
            for n_p in range(n_past + 1):
                char_embedding = char_embeddings[:, n_p]
                action_loss = action_losses[n_p]
                action_acc = action_accs[n_p]
                goal_loss = goal_losses[n_p]
                goal_acc = goal_accs[n_p]
                goal_recall = goal_recalls[n_p]
                goal_f1 = goal_f1s[n_p]
                initial_representation_data.append(
                    {"model_name": model_name, **model_info, "agent_label": agent_label, "agent_type": agent_type,
                     **filled_agent_info, "n_past": n_p, "char_embedding": char_embedding, "action_loss": action_loss,
                     "action_acc": action_acc, "goal_loss": goal_loss, "goal_acc": goal_acc, "goal_recall": goal_recall,
                     "goal_f1": goal_f1}
                )
    return pd.DataFrame(initial_representation_data)


def append_dci_scores(agent_specific_df, model_specific_df):
    data = []
    n_past = 5
    for model_name, model_df in tqdm(agent_specific_df[agent_specific_df["n_past"] == n_past].groupby("model_name")):
        factors = []
        codes = []
        agent_type_cls_idx = {"independent": 0, "collaborative": 1}
        goal_ranker_type_cls_idx = {"discount": 0, "highest": 1, "closest": 2}
        all_goal_rewards = RandomAgentParamSampler(1, False).goal_rewards
        goal_rewards_cls_idx = dict()
        for idx, goal_rewards in enumerate(all_goal_rewards):
            goal_rewards_label = "".join([str(r) for r in goal_rewards])
            goal_rewards_cls_idx[goal_rewards_label] = idx
        for i in model_df[["agent_type", "goal_ranker_type", "goal_rewards", "char_embedding"]].iterrows():
            agent_type, goal_ranker_type, goal_rewards, char_embed = i[1]
            factor = []
            factor.append([agent_type_cls_idx[agent_type]])
            goal_ranker_one_hot = [0 for _ in range(len(goal_rewards_cls_idx))]
            goal_ranker_one_hot[goal_ranker_type_cls_idx[goal_ranker_type]] = 1.
            factor.append(goal_ranker_one_hot)
            goal_rewards_one_hot = [0 for _ in range(len(all_goal_rewards))]
            goal_rewards_one_hot[goal_rewards_cls_idx[goal_rewards]] = 1.
            factor.append(goal_rewards_one_hot)
            factors += [factor for _ in range(len(char_embed))]

            codes.append(char_embed)

        codes = np.concatenate(codes)
        factors = np.array(factors, dtype=object)
        disentanglement, completeness, informativeness = dci(factors, codes)
        data.append({"model_name": model_name, "n_past": n_past,
                     "disentanglement": disentanglement,
                     "completeness": completeness,
                     "informativeness": np.mean(informativeness),
                     "informativeness_agent_type": informativeness[0],
                     "informativeness_goal_ranker_type": informativeness[1],
                     "informativeness_goal_rewards": informativeness[2],
                     })
    df = pd.DataFrame(data)
    return model_specific_df.merge(_df_add_level(df), how="left", on=["model_name", "n_past"])


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


def append_modified_silhouette_scores_to_df(agent_specific_df):
    data = []
    models_names = agent_specific_df["model_name"].unique()
    for model_name in models_names:
        all_labels_and_embeddings = agent_specific_df[agent_specific_df["model_name"] == model_name][
            ["agent_type", "goal_ranker_type", "n_past", "char_embedding"]]
        for n_past in range(6):
            all_embeddings = []
            labels = []
            group_labels = []
            for (agent_type, goal_ranker_type), df in all_labels_and_embeddings.groupby(
                    ["agent_type", "goal_ranker_type"]):
                modified_label = agent_type + goal_ranker_type
                group_labels.append((agent_type, goal_ranker_type))
                embeddings = np.concatenate(df[df["n_past"] == n_past]["char_embedding"].tolist())
                all_embeddings.append(embeddings)
                num_embed_per_cluster = embeddings.shape[0]
                labels.append(np.repeat(modified_label, num_embed_per_cluster))
            embeddings = np.concatenate(all_embeddings, axis=0)
            labels = np.concatenate(labels)
            samples = silhouette_samples(embeddings, labels)
            for i, (agent_type, goal_ranker_type) in enumerate(group_labels):
                idx = num_embed_per_cluster * i
                label_average = np.average(samples[idx:idx + num_embed_per_cluster])
                data.append({"model_name": model_name,
                             "n_past": n_past,
                             "agent_type": agent_type,
                             "goal_ranker_type": goal_ranker_type,
                             "modified_cluster_silhouette_score": label_average})

    return agent_specific_df.merge(pd.DataFrame(data), how="left",
                                   on=["model_name", "n_past", "agent_type", "goal_ranker_type"])


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
    def get_l2_dim_norm(embedding_size):
        return np.linalg.norm(np.ones(embedding_size))

    data = []
    models_names = agent_specific_df["model_name"].unique()
    for model_name in models_names:
        embedding_df = agent_specific_df[agent_specific_df["model_name"] == model_name][["n_past", "char_embedding"]]
        x = embedding_df.groupby(["n_past"])["char_embedding"].apply(lambda a: np.vstack(a))
        n_past = len(x)
        l2_dim_norm = get_l2_dim_norm(embedding_size=x[n_past-1].shape[-1])
        for n_p in range(n_past):
            # x[n_p].shape -> (data_points, embedding_size)
            sim_score = get_representation_similarity_score([x[n_p], x[n_past-1]], similarity="cka")[0]
            distance_score = np.average(np.linalg.norm(x[n_p] - x[n_past - 1], axis=1)) / l2_dim_norm
            euclidean_sim = 1 / (1 + np.average(np.linalg.norm(x[n_p] - x[n_past - 1], axis=1)))
            euclidean_sim_norm = 1 / (1 + (np.average(np.linalg.norm(x[n_p] - x[n_past - 1], axis=1) / l2_dim_norm)))
            cosine_sim = np.average(1 - np.sum(x[n_p] * x[n_past - 1], axis=1) / (np.linalg.norm(x[n_p], axis=1) * np.linalg.norm(x[n_past - 1], axis=1)))
            data.append({"model_name": model_name, "n_past": n_p, "final_embedding_similarity": sim_score,
                         "final_embedding_distance": distance_score,
                         "final_embedding_euclidean_similarity": euclidean_sim,
                         "final_embedding_euclidean_similarity_norm": euclidean_sim_norm,
                         "final_embedding_cosine_similarity": cosine_sim})
    df = pd.DataFrame(data)
    return model_specific_df.merge(_df_add_level(df), how="left", on=["model_name", "n_past"])


def append_model_specific_param_scores(agent_specific_df, model_specific_df, mask_threshold=.75):
    def get_shared_param_scores():
        param_scores = {"model_name": model_name, "n_past": n_past}
        for agent_param in agent_params:
            embedding_df = agent_df[[agent_param, "char_embedding"]]
            labels = []
            embeddings = []
            for i, group_embeddings in enumerate(
                    embedding_df.groupby(agent_param, dropna=False)["char_embedding"].apply(lambda a: np.vstack(a))):
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

    dfs = dict(tuple(agent_specific_df[agent_specific_df["n_past"].isin([1, 2, 3, 4, 5])][["model_name", 'agent_type', *agent_params, "n_past", "char_embedding"]].groupby(["model_name", "n_past"])))
    shared_agent_param_data = []
    agent_type_data = []
    for (model_name, n_past), agent_df in tqdm(dfs.items()):
        shared_agent_param_data.extend(get_shared_param_scores())
        agent_type_data.extend(get_agent_type_scores())

    model_specific_df = model_specific_df.merge(_df_add_level(pd.DataFrame(shared_agent_param_data)), how="left",
                                                on=["model_name", "n_past"])
    model_specific_df = model_specific_df.merge(_df_add_level(pd.DataFrame(agent_type_data)), how="left",
                                                on=["model_name", "n_past"])
    return model_specific_df


def get_param_silhouette_and_dim(embeddings, labels, mask_threshold):
    embedding_responsibility = cosine_loss(embeddings, labels, mask_threshold=mask_threshold)
    silhouette = -1
    feature_dim = sum(embedding_responsibility)
    if any(embedding_responsibility):
        silhouette = get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility)
    return silhouette, feature_dim


def append_agent_specific_param_scores(df, mask_threshold=.75):
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
                            embedding_df.groupby(agent_param, dropna=False)["char_embedding"].apply(lambda a: np.vstack(a))):
                        labels = labels + [i for _ in range(group_embeddings.shape[0])]
                        embeddings.append(group_embeddings)
                    param_embeddings = np.concatenate(embeddings, axis=0)
                    param_labels = np.array(labels)
                    silhouette, feature_dim = get_param_silhouette_and_dim(param_embeddings, param_labels, mask_threshold)

                else:
                    silhouette = np.nan
                    feature_dim = np.nan
                param_scores.update({f"{agent_param}_score": silhouette, f"{agent_param}_dim": feature_dim})
            data.append(param_scores)
        return data

    agent_types = df["agent_type"].unique()
    dfs = dict(tuple(df[df["n_past"].isin([5])][["model_name", 'agent_type', *agent_params, "n_past", "char_embedding"]].groupby(["model_name", "n_past"])))
    agent_param_data = []
    for (model_name, n_past), agent_df in tqdm(dfs.items()):
        agent_param_data.extend(get_agent_param_scores())

    return df.merge(pd.DataFrame(agent_param_data), how="left", on=["model_name", "n_past", "agent_type"])


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
        goal_recalls = []
        goal_f1s = []
        agent_idx_offset = (n_past + 1) * current_gridworld_len
        for n_p in range(n_past+1):
            n_p_offset = current_gridworld_len * n_p
            n_p_indices = []
            for agent_idx in range(len(batch)):
                start_idx = agent_idx_offset * agent_idx + n_p_offset
                n_p_indices += list(range(start_idx, start_idx + current_gridworld_len))

            pred_action = predictions.action[n_p_indices]
            true_action = output.action[n_p_indices]
            action_losses.append(action_loss(pred_action, true_action).item())
            action_accs.append(action_acc(pred_action, true_action))

            pred_goal = predictions.goal_consumption[n_p_indices]
            true_goal = output.goal_consumption[n_p_indices]
            goal_losses.append(goal_consumption_loss(pred_goal, true_goal).item())
            goal_accs.append(goal_acc(pred_goal, true_goal))
            goal_recalls.append(recall_score(true_goal, pred_goal > .5, "samples"))
            goal_f1s.append(f1_score(true_goal, pred_goal > .5, "samples"))
        return char_embeddings, np.array(action_losses), np.array(action_accs), np.array(goal_losses), \
            np.array(goal_accs), np.array(goal_recalls), np.array(goal_f1s)


def generate_random_traj(agent_type, agent_kwargs, seed=42, n_past=5):
    raps = RandomAgentParamSampler(seed, True)
    np.random.seed(seed)
    agents = get_random_gridworld_agent(agent_type, raps, [-1, -1, -1, -1], **agent_kwargs)[0]
    env = get_random_env(reward_funcs=[agent.reward_function for agent in agents])
    return [play_episode(env, agents) for _ in range(n_past + 1)]


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


def lstm_ttx_boxplot_comparison(df, metric, n_past=5, x="char_embedding_size", agg_metric="mean", agent_type=None,
                                ax=None, verbose=False, showmeans=False):
    def lstm_ttx_metric_str(metrics):
        metrics = [f"{m:.3f}" for m in metrics]
        return f"({metrics[0]}, {metrics[1]}), ({metrics[2]}, {metrics[3]}), ({metrics[4]}, {metrics[5]})"

    past_df = df.copy()
    if agent_type is not None:
        past_df = past_df[past_df["agent_type"] == agent_type]
    past_df['Architecture'] = np.where(past_df['lstm_char'] == True, "lstm", "transformer")
    if isinstance(n_past, int):
        n_past = [n_past]
    past_df = past_df[(past_df["n_past"].isin(n_past))]

    if (metric, agg_metric) in past_df:
        metric = (metric, agg_metric)

    sns.boxplot(x=past_df[x],
                y=past_df[metric],
                hue=past_df["Architecture"],
                ax=ax,
                showmeans=showmeans)
    if verbose:
        medians = []
        avgs = []
        for e_size in [64, 128, 512]:
            for arch in ["lstm", "transformer"]:
                column = past_df[(past_df["Architecture"] == arch) &
                                 (past_df["char_embedding_size"] == e_size)][metric]
                medians.append(column.median())
                avgs.append(column.mean())
        print("Med (LSTM, TTX): " + lstm_ttx_metric_str(medians))
        print("Avg (LSTM, TTX): " + lstm_ttx_metric_str(avgs))


def lstm_ttx_agent_param_boxplot_comparison(df, param, n_past=5, x="char_embedding_size", compare=None):
    if compare is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        lstm_ttx_boxplot_comparison(df, metric=param+"_score", n_past=n_past, x=x, ax=axes[0])
        lstm_ttx_boxplot_comparison(df, metric=param+"_dim", n_past=n_past, x=x, ax=axes[1])
        axes[0].set_ylim((-1, 1))
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


def calculate_and_combine_model_corr(df, corr_fn, n_past=5, **kwargs):
    return pd.DataFrame(corr_fn(df[(df["n_past"] == n_past) & (df["lstm_char"] == True)], **kwargs)["char_embedding_size"]).join(corr_fn(df[(df["n_past"] == n_past) & (df["lstm_char"] == False)], **kwargs)["char_embedding_size"], lsuffix='_lstm', rsuffix='_ttx')


def aggregate_model_param_data(model_specific_df):
    model_params = ["model_family", 'lstm_char', 'lstm_mental', 'char_embedding_size', 'char_n_layer', 'char_n_head',
                    'mental_embedding_size', 'mental_n_layer', 'mental_n_head']
    model_param_df = model_specific_df[[*model_params, "n_past", "action_loss", "action_acc", "goal_loss", "goal_acc", *representation_metrics, ]]
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
    architecture_df = model_specific_df[["lstm_char", "lstm_mental", "n_past", "action_loss", "action_acc", "goal_loss", "goal_acc", *representation_metrics]]
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


def get_agent_specific_data(model_dict, save=False, num_agents_per_cluster=300):
    print("Getting Model Specific Data")
    print("\tFetching models output DF")
    df = create_models_output_df(model_dict, num_agents_per_cluster=num_agents_per_cluster)
    print("\tFetching agent specific param scores")
    df = append_agent_specific_param_scores(df)
    print("\tFetching silhouette scores")
    df = append_silhouette_scores_to_df(df)
    print("\tFetching modified silhouette scores")
    df = append_modified_silhouette_scores_to_df(df)
    if save:
        save_representation_data(df, representation_type="agent_specific")
    return df


def aggregate_model_specific_data(agent_specific_df):
    model_params = ["model_name", "model_seed", "model_family", 'lstm_char', 'lstm_mental', 'char_embedding_size',
                    'char_n_layer', 'char_n_head', 'mental_embedding_size', 'mental_n_layer', 'mental_n_head']

    model_specific_df = agent_specific_df[[*model_params, "n_past", "action_loss", "action_acc", "goal_loss",
                                           "goal_acc", "goal_recall", "goal_f1"
                                           "cluster_silhouette_score", "modified_cluster_silhouette_score",
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


def create_analysis_dataset(num_agents_per_cluster=300, seed=58, n_past=5):
    def get_agent_label():
        agent_info_repr = ''
        for param in agent_info:
            agent_info_repr += f"{param}:{agent_info[param]}, "
        return f"{agent_type}, {agent_info_repr}"

    def fill_agent_info():
        goal_ranker_type = agent_info.get("goal_ranker_type")
        goal_rewards = agent_info.get("goal_rewards")
        goal_rewards = "".join([str(reward) for reward in goal_rewards])
        filled_agent_info = {
                             "goal_ranker_type": goal_ranker_type,
                             "goal_rewards": goal_rewards,
                             }
        return filled_agent_info

    def get_cluster_trajs():
        nonlocal seed
        cluster_trajs = []
        for i in range(num_agents_per_cluster):
            cluster_trajs.append(generate_random_traj(agent_type, agent_info, seed=seed, n_past=n_past))
            seed += 1
        return cluster_trajs

    agent_clusters = get_gridworld_agent_cluster_info()

    for agent_type, agent_info in tqdm(agent_clusters):
        agent_label = get_agent_label()
        filled_agent_info = fill_agent_info()
        trajs = get_cluster_trajs()
        data = {"agent_type": agent_type, "agent_label": agent_label, "agent_info": filled_agent_info, "trajs": trajs}
        save_dataset(data, "data/datasets/gridworld_analysis/", agent_label)


def load_analysis_data(agent_label):
    dataset = load_dataset("data/datasets/gridworld_analysis/" + agent_label + ".pickle")
    agent_type = dataset["agent_type"]
    agent_info = dataset["agent_info"]
    trajs = dataset["trajs"]
    return agent_type, agent_label, agent_info, trajs


def load_all_gridworld_models():
    model_dict = dict()
    model_dirpath = "data/models/gridworld/"
    for filename in np.sort(os.listdir(model_dirpath)):
        modeller_params = filename.split("Grid]_")[1].replace(".ckpt", "")
        modeller_dirpath = model_dirpath + filename
        model_dict[modeller_params] = load_modeller(modeller_dirpath).to("cuda")
    return model_dict


def collate_and_save_all_representation_data():
    model_dict = load_all_gridworld_models()
    print("Getting results for", list(model_dict.keys()))
    agent_specific_df = get_agent_specific_data(model_dict, save=True)
    del model_dict
    model_specific_df = extract_model_specific_data(agent_specific_df, save=True)
    extract_model_param_data(agent_specific_df, model_specific_df, save=True)
    extract_architecture_data(agent_specific_df, model_specific_df, save=True)
