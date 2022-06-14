import os
import shutil
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.analysis.representation_metrics import load_representation_data, get_representation_similarity_score

plot_dirpath = "data/imgs/iterative_action/"

img_width = 6.4
img_height = 4.8


def sns_char_boxplot(df, metric, ax):
    if (metric, "mean") in df:
        metric = (metric, "mean")
    sns.boxplot(x=df["char_embedding_size"], y=df[metric], hue=df["Architecture"], ax=ax)


def get_n_past_df(df, n_past=5):
    df5 = df[df["n_past"] == n_past].copy()
    df5['Architecture'] = np.where(df5['lstm_char'] == True, "lstm", "transformer")
    return df5


def save_performance_plot(df, n_past=5):
    filename = "performance.png"
    df5 = get_n_past_df(df, n_past=n_past)

    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height))

    sns_char_boxplot(df5, "loss", axes[0])
    axes[0].set(xlabel="embedding size", ylabel="action loss")

    sns_char_boxplot(df5, "acc", axes[1])
    axes[1].set(xlabel="embedding size", ylabel="action accuracy")

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_disentanglement(df, n_past=5):
    filename = "disentanglement.png"
    df5 = get_n_past_df(df, n_past=n_past)

    fig, axes = plt.subplots(1, 3, figsize=(img_width * 3, img_height))

    sns_char_boxplot(df5, "disentanglement", axes[0])
    axes[0].set(xlabel="embedding size", ylabel="modularity", ylim=(0, 1))

    sns_char_boxplot(df5, "completeness", axes[1])
    axes[1].set(xlabel="embedding size", ylabel="compactness", ylim=(0, 1))

    sns_char_boxplot(df5, "informativeness", axes[2])
    axes[2].set(xlabel="embedding size", ylabel="explicitness", ylim=(0, 1))

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_cluster(df, n_past=5):
    filename = "cluster.png"
    df5 = get_n_past_df(df, n_past=n_past)

    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height))

    sns_char_boxplot(df5, "cluster_silhouette_score", axes[0])
    axes[0].set(xlabel="embedding size", ylabel="silhouette score", ylim=(-1, 1))

    sns_char_boxplot(df5, "modified_cluster_silhouette_score", axes[1])
    axes[1].set(xlabel="embedding size", ylabel="silhouette score", ylim=(-1, 1))

    plt.savefig(plot_dirpath + "iter_" + filename)
    

def save_pca(df, n_past=5):
    filename = "pca.png"
    df5 = get_n_past_df(df, n_past=n_past)

    fig, ax = plt.subplots(1, 1, figsize=(img_width, img_height))

    sns_char_boxplot(df5, "pca_80%", ax)
    ax.set(xlabel="embedding size", ylabel="num dimensions")

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_convergence(df):
    def riemann_sum(n_past_values):
        dx = 1
        midpoints = (n_past_values[:-1] + n_past_values[1:]) / 2
        return np.sum(midpoints * dx) / len(n_past_values)

    filename = "convergence.png"
    df1 = get_n_past_df(df, n_past=1)

    fig, axes = plt.subplots(2, 2, figsize=(img_width * 2, img_height * 2))
    sns_char_boxplot(df1, "final_embedding_similarity", axes[0, 0])
    axes[0, 0].set(xlabel="embedding size", ylabel="CKA similarity", ylim=(0, 1))

    sns_char_boxplot(df1, "final_embedding_distance", axes[0, 1])
    axes[0, 1].set(xlabel="embedding size", ylabel="normalized euclidean distance")


    data = []
    for (model_name, model_df) in df[
        ["model_name", "char_embedding_size", "lstm_char", "n_past", "final_embedding_similarity",
         "final_embedding_distance"]].groupby(["model_name"]):
        char_embedding_size = model_df.char_embedding_size.iloc[0]
        lstm_char = model_df.lstm_char.iloc[0]
        similarities = np.array(model_df["final_embedding_similarity"].tolist())
        distances = np.array(model_df["final_embedding_distance"].tolist())
        data.append({"char_embedding_size": char_embedding_size,
                     "Architecture": ("lstm" if lstm_char else "transformer"),
                     "npast_1_sim": similarities[1],
                     "npast_1_dist": distances[1],
                     "similarity_convergence": riemann_sum(similarities),
                     "distance_convergence": riemann_sum(distances) / max(distances), })

    convg_df = pd.DataFrame(data)

    sns_char_boxplot(convg_df, "similarity_convergence", axes[1, 0])
    axes[1, 0].set(xlabel="embedding size", ylabel="CKA similarity convergence")

    sns_char_boxplot(convg_df, "distance_convergence", axes[1, 1])
    axes[1, 1].set(xlabel="embedding size", ylabel="euclidean distance convergence")

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_stability(df):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def get_avg_sim_scores(embeddings_a, embeddings_b):
        sim_scores = []
        for embeddings in itertools.product(embeddings_a, embeddings_b):
            _, indiv_sim_scores = get_representation_similarity_score(embeddings, similarity="cka")
            sim_scores.append(indiv_sim_scores)
        return np.average(sim_scores)

    def get_cross_embedding_sim_matrix(df_a, df_b, families_a, families_b):
        sim_matrix = np.zeros((len(families_a), len(families_b)))
        for i, family_a in enumerate(families_a):
            for j, family_b in enumerate(families_b):
                embeddings_a = df_a[family_a].groupby(["model_seed"])["char_embedding"].apply(
                    lambda a: np.vstack(a)).tolist()
                embeddings_b = df_b[family_b].groupby(["model_seed"])["char_embedding"].apply(
                    lambda a: np.vstack(a)).tolist()
                sim_matrix[i, j] = get_avg_sim_scores(embeddings_a, embeddings_b)
        return sim_matrix

    def plot_sim_matrix(sim_matrix, ax, row_families, col_families):
        ax.imshow(sim_matrix, cmap="hot", interpolation="nearest", vmin=0., vmax=1.0)
        ttx_names = ["transformer-64", "transformer-128", "transformer-512"]
        lstm_names = ["lstm-64", "lstm-128", "lstm-512"]
        if "ttx" in col_families[0]:
            col_names = ttx_names
            row_names = lstm_names
        else:
            col_names = lstm_names
            row_names = ttx_names
        ax.set_xticks(range(len(col_families)))
        ax.set_xticklabels(col_names, rotation=90)
        ax.set_yticks(range(len(row_families)))
        ax.set_yticklabels(row_names)

    filename = "stability.png"
    fig, axes = plt.subplots(1, 3, figsize=(img_width * 3, img_height * 1.5))

    lstm_embedding_df = df[(df["lstm_char"] == True)][["model_family", "model_seed", "n_past", "char_embedding"]]
    ttx_embedding_df = df[(df["lstm_char"] == False)][["model_family", "model_seed", "n_past", "char_embedding"]]
    lstm_families = lstm_embedding_df["model_family"].unique().tolist()
    lstm_families.sort(key=natural_keys)
    ttx_families = ttx_embedding_df["model_family"].unique().tolist()
    ttx_families.sort(key=natural_keys)

    lstm_past_df = lstm_embedding_df[lstm_embedding_df["n_past"] == 5]
    ttx_past_df = ttx_embedding_df[ttx_embedding_df["n_past"] == 5]
    lstm_dfs = dict(tuple(lstm_past_df.groupby(["model_family"])))
    ttx_dfs = dict(tuple(ttx_past_df.groupby(["model_family"])))

    sim_matrix = get_cross_embedding_sim_matrix(lstm_dfs, lstm_dfs, lstm_families, lstm_families)
    plot_sim_matrix(sim_matrix, axes[0], lstm_families, lstm_families)

    sim_matrix = get_cross_embedding_sim_matrix(ttx_dfs, ttx_dfs, ttx_families, ttx_families)
    plot_sim_matrix(sim_matrix, axes[1], ttx_families, ttx_families)

    sim_matrix = get_cross_embedding_sim_matrix(lstm_dfs, ttx_dfs, lstm_families, ttx_families)
    plot_sim_matrix(sim_matrix, axes[2], lstm_families, ttx_families)

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_dist(df):
    def plot_embedding_space_as_boxplot(model_name, ax):
        percentile_order = 2
        embeddings = np.vstack(df[(df["model_name"] == model_name) & (df["n_past"] == 5)].char_embedding.tolist())
        percentiles = np.percentile(embeddings, [0, 25, 50, 75, 100], axis=0)
        embeddings = embeddings[:, np.argsort(percentiles[percentile_order, :])]
        sns.boxplot(data=embeddings, ax=ax, fliersize=.5)
        ax.set(xticks=[], ylim=(-1.5, 1.5))

    filename = "dist.png"
    fig, axes = plt.subplots(2, 2, figsize=(4 * img_width, 2 * img_height))

    plot_embedding_space_as_boxplot(model_name="lstm[64,1]_lstm[64,1]_seed1", ax=axes[0, 0])
    axes[0, 0].set(title="LSTM", ylabel="Embedding Size 64")
    plot_embedding_space_as_boxplot(model_name="ttx[64,4,4]_ttx[64,4,4]_seed1", ax=axes[0, 1])
    axes[0, 1].set(title="Transformer")
    plot_embedding_space_as_boxplot(model_name="lstm[128,1]_lstm[64,1]_seed1", ax=axes[1, 0])
    axes[1, 0].set(ylabel="Embedding Size 128")
    plot_embedding_space_as_boxplot(model_name="ttx[128,4,4]_ttx[64,4,4]_seed1", ax=axes[1, 1])

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_all_results():
    # if os.path.isdir(plot_dirpath):
    #     shutil.rmtree(plot_dirpath)
    # os.makedirs(plot_dirpath)

    agent_specific_df = load_representation_data("agent_specific")
    model_specific_df = load_representation_data("model_specific")
    # model_param_df = load_representation_data("model_param")
    # architecture_df = load_representation_data("architecture")

    # save_performance_plot(model_specific_df)
    # save_disentanglement(model_specific_df)
    # save_cluster(model_specific_df)
    # save_pca(model_specific_df)
    # save_convergence(model_specific_df)
    # save_stability(agent_specific_df)
    save_dist(agent_specific_df)
