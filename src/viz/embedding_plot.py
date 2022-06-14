import hypertools as hyp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict


def group_policies_in_ranges(agent_policy_iter, n_groups=4):
    likelihood_ranges = np.linspace(1/n_groups, 1, n_groups)
    policy_group = dict()
    for agent_id, policy in agent_policy_iter:
        p0 = policy[0] / sum(policy)
        likelihood_range_idx = len(likelihood_ranges)
        for i, likelihood in enumerate(likelihood_ranges):
            if p0 <= likelihood:
                likelihood_range_idx = i
                break
        step_size = 1 / n_groups
        p0_label = (step_size * likelihood_range_idx) + step_size/2
        policy_group[agent_id] = f"[{p0_label:.2f}, {1-p0_label:.2f}]"
    return policy_group


def label_by_true_action(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    true_actions = dict()
    for _, agent_true_action in df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "true_action"]].iterrows():
        true_actions[agent_true_action.agent_id] = agent_true_action.true_action
    return true_actions


def label_by_empirical_policy(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19, n_groups=4):
    def unique_actions(x): return np.unique(x, return_counts=True)[1]
    agent_true_action_count = df[(df["n_past"] == n_past) & (df["current_traj_len"] < current_traj_len)][["agent_id", "true_action"]].groupby("agent_id")["true_action"].apply(unique_actions)
    return group_policies_in_ranges(agent_true_action_count.iteritems(), n_groups)


def _label_by_metadata_get(df, metadata, remove_missing_labels, label):
    agent_ids = df.agent_id.unique()
    labels = dict()
    for agent_id in agent_ids:
        val = metadata[agent_id].get(label, None)
        if val is None and remove_missing_labels:
            continue
        labels[agent_id] = val
    return labels


def label_by_agent_type(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    return _label_by_metadata_get(df, metadata, remove_missing_labels, "agent_type")


def label_by_trigger_action(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    labels = _label_by_metadata_get(df, metadata, remove_missing_labels, "trigger_action")
    for agent_id in labels:
        if type(labels[agent_id]) == list:
            if len(labels[agent_id]) == 1:
                labels[agent_id] = labels[agent_id][0]
    return labels


def label_by_trigger_patience(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    return _label_by_metadata_get(df, metadata, remove_missing_labels, "trigger_patience")


def label_by_action_pattern(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    return _label_by_metadata_get(df, metadata, remove_missing_labels, "action_pattern")


def label_by_action_pattern_len(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    labels = _label_by_metadata_get(df, metadata, remove_missing_labels, "action_pattern")
    for agent_id in labels:
        if labels[agent_id]:
            labels[agent_id] = len(labels[agent_id])
    return labels


def label_by_mixed_trigger_strategy(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    agent_ids = df.agent_id.unique()
    labels = dict()
    for agent_id in agent_ids:
        if metadata[agent_id].get("agent_type", "") == "mixed_trigger_pattern":
            labels[agent_id] = metadata[agent_id]["mixed_strategy"]
        else:
            if not remove_missing_labels:
                labels[agent_id] = None
    return labels


def label_by_win_trigger(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    return _label_by_metadata_get(df, metadata, remove_missing_labels, "win_trigger")


def label_by_previous_action(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    previous_traj_step = min(0, current_traj_len - 1)
    previous_actions = label_by_true_action(df, metadata, remove_missing_labels,
                                            n_past=n_past, current_traj_len=previous_traj_step)
    if current_traj_len == 0:
        for agent_id in previous_actions:
            previous_actions[agent_id] = -1
    return previous_actions


def label_by_predicted_policy(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19, n_groups=4):
    agents_predicted_policies = df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "pred_action_dist"]]
    agents_predicted_policies = agents_predicted_policies.set_index("agent_id")
    return group_policies_in_ranges(agents_predicted_policies.pred_action_dist.iteritems(), n_groups)


def label_by_starting_action(df, metadata, remove_missing_labels, n_past=5, current_traj_len=None):
    return label_by_true_action(df, metadata, remove_missing_labels, n_past, current_traj_len=0)


def label_by_opponent_idx(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    opponent_idx = dict()
    for _, agent_opponent_idx in df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "opponent_idx"]].iterrows():
        opponent_idx[agent_opponent_idx.agent_id] = agent_opponent_idx.opponent_idx
    return opponent_idx


def label_by_mixed_and_opponent_idx(df, metadata, remove_missing_labels, n_past=5, current_traj_len=19):
    opponent_idx = dict()
    for _, agent_opponent_idx in df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "opponent_idx"]].iterrows():
        if metadata[agent_opponent_idx.agent_id]["agent_type"] == "mixed_strategy":
            opponent_idx[agent_opponent_idx.agent_id] = -1
        else:
            opponent_idx[agent_opponent_idx.agent_id] = agent_opponent_idx.opponent_idx
    return opponent_idx


def extract_labels(df, metadata, label_by="", remove_missing_labels=False, n_past=5, current_traj_len=19):
    def join_dictionaries(dictionaries):
        joined_dict = defaultdict(list)
        for d in dictionaries:
            for key, value in d.items():
                joined_dict[key].append(value)
        return joined_dict

    label_by_funcs = {
        "true_action": label_by_true_action,
        "empirical_policy": label_by_empirical_policy,
        "agent_type": label_by_agent_type,
        "previous_action": label_by_previous_action,
        "predicted_policy": label_by_predicted_policy,
        "starting_action": label_by_starting_action,
        "opponent_idx": label_by_opponent_idx,
        "mixed_and_opponent_idx": label_by_mixed_and_opponent_idx,
        "trigger_action": label_by_trigger_action,
        "trigger_patience": label_by_trigger_patience,
        "action_pattern": label_by_action_pattern,
        "mixed_trigger_strategy": label_by_mixed_trigger_strategy,
        "win_trigger": label_by_win_trigger,
        "action_pattern_len": label_by_action_pattern_len,
        "action_pattern_length": label_by_action_pattern_len,
    }

    if isinstance(label_by, str):
        label_by = [label_by]

    separate_labels = []
    for label in label_by:
        separate_labels.append(label_by_funcs[label](df, metadata, remove_missing_labels,
                                                     n_past=n_past, current_traj_len=current_traj_len))
    joined_labels = join_dictionaries(separate_labels)
    for agent_id, agent_labels in joined_labels.items():
        joined_labels[agent_id] = str(agent_labels)
    return joined_labels


def extract_embeddings_and_labels(df, metadata, label_by="", remove_missing_labels=False, n_past=5,
                                  current_traj_len=19, get_char_embedding=True):
    embeddings_df = df[(df["n_past"] == n_past) & (df["current_traj_len"] == current_traj_len)][["agent_id", "embeddings"]]

    labels = extract_labels(df, metadata, label_by, remove_missing_labels, n_past, current_traj_len)
    labels_df = pd.Series(labels, name="label")
    embeddings_df = embeddings_df.join(labels_df, on="agent_id")

    if remove_missing_labels:
        embeddings_df = embeddings_df.dropna()

    embeddings_list = embeddings_df.embeddings.values.tolist()
    if isinstance(embeddings_list[0][0], np.ndarray):
        embeddings = np.array([embedding[0] if get_char_embedding else embedding[1] for embedding in embeddings_list])
    else:
        embeddings = np.array(embeddings_list)

    labels = np.array(embeddings_df.label.values.tolist())
    return embeddings, labels


def plot_embeddings(df, metadata, label_by="", remove_missing_labels=False, n_past=5, current_traj_len=19,
                    plot_char_embedding=True, agent_type=None, short_title=False, **kwargs):

    filtered_df = df
    if agent_type is not None:
        if type(agent_type) == str:
            agent_type = [agent_type]
        acceptable_agent_ids = []
        for agent_id, agent_data in metadata.items():
            if agent_data["agent_type"] in agent_type:
                acceptable_agent_ids.append(agent_id)
        filtered_df = df[df["agent_id"].isin(acceptable_agent_ids)]

    embeddings, labels = extract_embeddings_and_labels(filtered_df, metadata, label_by, remove_missing_labels, n_past,
                                                       current_traj_len, plot_char_embedding)

    title = f"Embeddings grouped by: {label_by}"

    if len(embeddings.shape) == 3:
        embedding_idx = 1
        embedding_name = "Mental"
        if plot_char_embedding:
            embedding_idx = 0
            embedding_name = "Character"
        embeddings = embeddings[:, embedding_idx]
        embeddings = np.stack(embeddings, axis=0).astype(float)
        title = f"{embedding_name} embeddings grouped by: {label_by}"

    if short_title:
        title = f"{label_by}"

    hyp_plot(embeddings, labels, title=title, **kwargs)


def hyp_plot(embeddings, labels, ndims=2, reduce="PCA", title=None, legend=False, ax=None):
    if embeddings.shape[1] == 1:
        ndims = 1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        if ndims >= 3:
            hyp.plot(embeddings, fmt=".", ndims=ndims, hue=labels, reduce=reduce, animate="spin", rotations=1,
                     duration=5, title=title, legend=legend, ax=ax)
        elif ndims == 2:
            hyp.plot(embeddings, fmt=".", ndims=ndims, hue=labels, reduce=reduce, title=title, legend=legend, ax=ax)
        else:
            if ax is None:
                fig, ax = plt.subplots()
            for label in np.unique(labels):
                ax.hist(embeddings[labels == label, 0], label=label, alpha=.65)
            ax.set_xticks([])
            ax.set_yticks([])
            if title is not None:
                ax.set_title(title)
            if legend:
                ax.legend()
