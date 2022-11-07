import os
import shutil
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from ecco.analysis import cka

from tommas.analysis.iterative_action_representation_metrics import load_representation_data as load_iter_data
from tommas.analysis.gridworld_representation_metrics import load_representation_data as load_grid_data

plot_dirpath = "data/imgs/"

img_width = 6.4
img_height = 4.8


def sns_char_boxplot(df, metric, ax):
    if (metric, "mean") in df:
        metric = (metric, "mean")
    sns.stripplot(x=df["char_embedding_size"], y=df[metric], hue=df["Architecture"], dodge=True, jitter=False, ax=ax, marker="X", s=12.)

    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x=df["char_embedding_size"],
            y=df[metric],
            hue=df["Architecture"],
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    handles[0] = Line2D([], [], marker="X", ms=12, linestyle='', color=handles[0].get_facecolor())
    handles[1] = Line2D([], [], marker="X", ms=12, linestyle='', color=handles[1].get_facecolor())
    ax.legend(handles[:2], labels[:2])

def get_n_past_df(df, n_past=5):
    df5 = df[df["n_past"] == n_past].copy()
    df5['Architecture'] = np.where(df5['lstm_char'] == True, "lstm", "transformer")
    return df5


def set_labels(ax, ylabel, fontsize=14, **kwargs):
    ax.set(**kwargs)
    ax.set_xlabel(xlabel="embedding size", fontsize=fontsize)
    ax.set_ylabel(ylabel=ylabel, fontsize=fontsize)


def save_iter_performance_plot(df, n_past=5):
    filename = "performance.png"
    df5 = get_n_past_df(df, n_past=n_past)

    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height))

    sns_char_boxplot(df5, "loss", axes[0])
    set_labels(axes[0], "action loss")

    sns_char_boxplot(df5, "acc", axes[1])
    set_labels(axes[1], "action accuracy")

    plt.savefig(plot_dirpath + "iter_" + filename)


def save_grid_performance_plot(df, n_past=5):
    filename = "performance.png"
    df5 = get_n_past_df(df, n_past=n_past)

    num_rows = 2
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(img_width * num_cols, img_height * num_rows))

    sns_char_boxplot(df5, "action_loss", axes[0, 0])
    set_labels(axes[0, 0], "action loss")
    sns_char_boxplot(df5, "action_acc", axes[0, 1])
    set_labels(axes[0, 1], "action accuracy")

    sns_char_boxplot(df5, "goal_loss", axes[1, 0])
    set_labels(axes[1, 0], "goal loss")
    sns_char_boxplot(df5, "goal_recall", axes[1, 1])
    set_labels(axes[1, 1], "goal recall")

    plt.savefig(plot_dirpath + "grid_" + filename)


def save_disentanglement(iter_df, grid_df, n_past=5):
    filename = "disentanglement.png"
    iter_df5 = get_n_past_df(iter_df, n_past=n_past)
    grid_df5 = get_n_past_df(grid_df, n_past=n_past)

    num_rows = 3
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(img_width * num_cols, img_height * num_rows))

    sns_char_boxplot(iter_df5, "disentanglement", axes[0, 0])
    set_labels(axes[0, 0], "modularity", ylim=(0, 1))

    sns_char_boxplot(iter_df5, "completeness", axes[1, 0])
    set_labels(axes[1, 0], "compactness", ylim=(0, 1))

    sns_char_boxplot(iter_df5, "informativeness", axes[2, 0])
    set_labels(axes[2, 0], "explicitness", ylim=(0, 1))

    sns_char_boxplot(grid_df5, "disentanglement", axes[0, 1])
    set_labels(axes[0, 1], "modularity", ylim=(0, 1))

    sns_char_boxplot(grid_df5, "completeness", axes[1, 1])
    set_labels(axes[1, 1], "compactness", ylim=(0, 1))

    sns_char_boxplot(grid_df5, "informativeness", axes[2, 1])
    set_labels(axes[2, 1], "explicitness", ylim=(0, 1))

    plt.savefig(plot_dirpath + filename)


def save_cluster(iter_df, grid_df, n_past=5):
    filename = "cluster.png"
    iter_df5 = get_n_past_df(iter_df, n_past=n_past)
    grid_df5 = get_n_past_df(grid_df, n_past=n_past)

    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height))

    sns_char_boxplot(iter_df5, "cluster_silhouette_score", axes[0])
    set_labels(axes[0], "silhouette score", ylim=(-1, 1))

    sns_char_boxplot(grid_df5, "modified_cluster_silhouette_score", axes[1])
    set_labels(axes[1], "silhouette score", ylim=(-1, 1))

    plt.savefig(plot_dirpath + filename)
    

def save_pca(iter_df, grid_df, n_past=5):
    filename = "pca.png"
    iter_df5 = get_n_past_df(iter_df, n_past=n_past)
    grid_df5 = get_n_past_df(grid_df, n_past=n_past)

    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height))

    sns_char_boxplot(iter_df5, "pca_80%", axes[0])
    set_labels(axes[0], "num dimensions")

    sns_char_boxplot(grid_df5, "pca_80%", axes[1])
    set_labels(axes[1], "num dimensions")

    plt.savefig(plot_dirpath + filename)


def save_appendix_pca(iter_df, grid_df, n_past=5):
    filename = "appendix_pca.png"
    iter_df5 = get_n_past_df(iter_df, n_past=n_past)
    grid_df5 = get_n_past_df(grid_df, n_past=n_past)

    fig, axes = plt.subplots(3, 2, figsize=(img_width * 2, img_height * 3))

    sns_char_boxplot(iter_df5, "pca_50%", axes[0, 0])
    set_labels(axes[0, 0], "num dimensions (50%)")
    sns_char_boxplot(iter_df5, "pca_75%", axes[1, 0])
    set_labels(axes[1, 0], "num dimensions (75%)")
    sns_char_boxplot(iter_df5, "pca_90%", axes[2, 0])
    set_labels(axes[2, 0], "num dimensions (90%)")

    sns_char_boxplot(grid_df5, "pca_50%", axes[0, 1])
    set_labels(axes[0, 1], "num dimensions (50%)")
    sns_char_boxplot(grid_df5, "pca_75%", axes[1, 1])
    set_labels(axes[1, 1], "num dimensions (75%)")
    sns_char_boxplot(grid_df5, "pca_90%", axes[2, 1])
    set_labels(axes[2, 1], "num dimensions (90%)")

    plt.savefig(plot_dirpath + filename)


def save_convergence(iter_df, grid_df):
    def plot_dataset_convergence(df, ax):
        data = []
        for (model_name, model_df) in df[
            ["model_name", "char_embedding_size", "lstm_char", "n_past", "final_embedding_similarity",
             "final_embedding_distance"]].groupby(["model_name"]):
            char_embedding_size = model_df.char_embedding_size.iloc[0]
            lstm_char = model_df.lstm_char.iloc[0]
            similarities = np.array(model_df["final_embedding_similarity"].tolist())
            data.append({"char_embedding_size": char_embedding_size,
                         "Architecture": ("lstm" if lstm_char else "transformer"),
                         "convergence": np.average(similarities[1:-1]),})

        convg_df = pd.DataFrame(data)

        sns_char_boxplot(convg_df, "convergence", ax)
        set_labels(ax, "convergence")

    filename = "convergence.png"
    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height * 1))

    plot_dataset_convergence(iter_df, axes[0])
    plot_dataset_convergence(grid_df, axes[1])
    plt.savefig(plot_dirpath + filename)


def save_stability(iter_df, grid_df):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def get_avg_sim_scores(embeddings_a, embeddings_b, compare_diagonal):
        similarity_scores = []
        for i, embedding_a in enumerate(embeddings_a):
            for j, embedding_b in enumerate(embeddings_b):
                if i == j and not compare_diagonal:
                    continue
                similarity_scores.append(cka(embedding_a.T, embedding_b.T))
        return np.average(similarity_scores)

    def get_cross_embedding_sim_matrix(df_a, df_b, families_a, families_b):
        sim_matrix = np.zeros((len(families_a), len(families_b)))
        for i, family_a in enumerate(families_a):
            for j, family_b in enumerate(families_b):
                embeddings_a = df_a[family_a].groupby(["model_seed"])["char_embedding"].apply(
                    lambda a: np.vstack(a)).tolist()
                embeddings_b = df_b[family_b].groupby(["model_seed"])["char_embedding"].apply(
                    lambda a: np.vstack(a)).tolist()
                compare_digaonal = not (family_a == family_b)
                sim_matrix[i, j] = get_avg_sim_scores(embeddings_a, embeddings_b, compare_digaonal)
        return sim_matrix

    def plot_sim_matrix(sim_matrix, ax, row_families, col_families):
        ax.imshow(sim_matrix, cmap="hot", interpolation="nearest", vmin=0., vmax=1.0)
        ttx_names = ["transformer-64", "transformer-128", "transformer-512"]
        lstm_names = ["lstm-64", "lstm-128", "lstm-512"]
        if "ttx" in col_families[0]:
            col_names = ttx_names
        else:
            col_names = lstm_names
        if "ttx" in row_families[0]:
            row_names = ttx_names
        else:
            row_names = lstm_names

        ax.set_xticks(range(len(col_families)))
        ax.set_xticklabels(col_names, rotation=45, fontsize=14)
        ax.set_yticks(range(len(row_families)))
        ax.set_yticklabels(row_names, fontsize=14)

    def get_arch_stability_plots(df, axes):
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

    def get_cross_arch_stability_plots(df, ax):
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

        sim_matrix = get_cross_embedding_sim_matrix(lstm_dfs, ttx_dfs, lstm_families, ttx_families)
        plot_sim_matrix(sim_matrix, ax, lstm_families, ttx_families)


    filename = "stability.png"
    # fig, axes = plt.subplots(2, 2, figsize=(img_width * 2, img_height * 2), constrained_layout = True)
    fig, axes = plt.subplots(2, 2, figsize=(img_width * 2, img_height * 2))

    get_arch_stability_plots(iter_df, axes[:, 0])
    get_arch_stability_plots(grid_df, axes[:, 1])

    plt.subplots_adjust(left=.15, right=.95, bottom=.15, top=.95, hspace=.25)

    plt.savefig(plot_dirpath + filename)

    # fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height * 1), constrained_layout = True)
    fig, axes = plt.subplots(1, 2, figsize=(img_width * 2, img_height * 1.35))
    get_cross_arch_stability_plots(iter_df, axes[0])
    get_cross_arch_stability_plots(grid_df, axes[1])
    plt.subplots_adjust(bottom=.25)
    plt.savefig(plot_dirpath + "cross_" + filename)


def save_dist(iter_df, grid_df):
    def plot_embedding_space_as_boxplot(df, model_name, ax):
        percentile_order = 2
        embeddings = np.vstack(df[(df["model_name"] == model_name) & (df["n_past"] == 5)].char_embedding.tolist())
        percentiles = np.percentile(embeddings, [0, 25, 50, 75, 100], axis=0)
        embeddings = embeddings[:, np.argsort(percentiles[percentile_order, :])]
        sns.boxplot(data=embeddings, ax=ax, fliersize=.5)
        ax.set(xticks=[], ylim=(-1.5, 1.5))

    filename = "dist.png"
    fig, axes = plt.subplots(4, 2, figsize=(4 * img_width, 4 * img_height))

    fontsize = 18
    title_font = 18
    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[64,1]_lstm[64,1]_seed1", ax=axes[0, 0])
    axes[0, 0].set_ylabel(ylabel="lstm-64", fontsize=fontsize)
    axes[0, 0].set_title("Iterative Action",  fontdict={'fontsize': title_font, 'fontweight': 'medium'})
    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[128,1]_lstm[64,1]_seed1", ax=axes[1, 0])
    axes[1, 0].set_ylabel(ylabel="lstm-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[64,4,4]_ttx[64,4,4]_seed1", ax=axes[2, 0])
    axes[2, 0].set_ylabel(ylabel="transformer-64", fontsize=fontsize)
    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[128,4,4]_ttx[64,4,4]_seed1", ax=axes[3, 0])
    axes[3, 0].set_ylabel(ylabel="transformer-128", fontsize=fontsize)

    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[64,2]_lstm[64,2]_seed1", ax=axes[0, 1])
    axes[0, 1].set_ylabel(ylabel="lstm-64", fontsize=fontsize)
    axes[0, 1].set_title("Gridworld", fontdict={'fontsize': title_font, 'fontweight': 'medium'})
    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[128,2]_lstm[64,2]_seed1", ax=axes[1, 1])
    axes[1, 1].set_ylabel(ylabel="lstm-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[64,8,8]_ttx[64,4,8]_seed1", ax=axes[2, 1])
    axes[2, 1].set_ylabel(ylabel="transformer-64", fontsize=fontsize)
    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[128,8,8]_ttx[64,4,8]_seed1", ax=axes[3, 1])
    axes[3, 1].set_ylabel(ylabel="transformer-128", fontsize=fontsize)

    plt.savefig(plot_dirpath + filename)


def save_large_dist(iter_df, grid_df):
    def plot_embedding_space_as_boxplot(df, model_name, ax):
        percentile_order = 2
        embeddings = np.vstack(df[(df["model_name"] == model_name) & (df["n_past"] == 5)].char_embedding.tolist())
        percentiles = np.percentile(embeddings, [0, 25, 50, 75, 100], axis=0)
        embeddings = embeddings[:, np.argsort(percentiles[percentile_order, :])]
        sns.boxplot(data=embeddings, ax=ax, fliersize=.5)
        ax.set(xticks=[], ylim=(-1.5, 1.5))

    filename = "large_dist.png"
    fig, axes = plt.subplots(4, 1, figsize=(4 * img_width, 4 * img_height))

    fontsize = 18
    title_font = 18
    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[512,1]_lstm[64,1]_seed1", ax=axes[0])
    axes[0].set_ylabel(ylabel="lstm-512", fontsize=fontsize)
    axes[0].set_title("Iterative Action",  fontdict={'fontsize': title_font, 'fontweight': 'medium'})
    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[512,4,4]_ttx[64,4,4]_seed1", ax=axes[1])
    axes[1].set_ylabel(ylabel="transformer-512", fontsize=fontsize)
    
    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[512,2]_lstm[64,2]_seed1", ax=axes[2])
    axes[2].set_ylabel(ylabel="lstm-512", fontsize=fontsize)
    axes[2].set_title("Gridworld", fontdict={'fontsize': title_font, 'fontweight': 'medium'})

    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[512,8,8]_ttx[64,4,8]_seed1", ax=axes[3])
    axes[3].set_ylabel(ylabel="transformer-512", fontsize=fontsize)

    plt.savefig(plot_dirpath + filename)


def save_wide_dist(iter_df, grid_df):
    def plot_embedding_space_as_boxplot(df, model_name, ax):
        percentile_order = 2
        embeddings = np.vstack(df[(df["model_name"] == model_name) & (df["n_past"] == 5)].char_embedding.tolist())
        percentiles = np.percentile(embeddings, [0, 25, 50, 75, 100], axis=0)
        embeddings = embeddings[:, np.argsort(percentiles[percentile_order, :])]
        sns.boxplot(data=embeddings, ax=ax, fliersize=.5)
        ax.set(xticks=[], ylim=(-1.5, 1.5))

    filename = "iter_dist.png"
    fig, axes = plt.subplots(6, 1, figsize=(4 * img_width, 6 * img_height))

    fontsize = 18
    title_font = 18

    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[64,1]_lstm[64,1]_seed1", ax=axes[0])
    axes[0].set_ylabel(ylabel="lstm-64", fontsize=fontsize)

    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[128,1]_lstm[64,1]_seed1", ax=axes[1])
    axes[1].set_ylabel(ylabel="lstm-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(iter_df, model_name="lstm[512,1]_lstm[64,1]_seed1", ax=axes[2])
    axes[2].set_ylabel(ylabel="lstm-512", fontsize=fontsize)

    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[64,4,4]_ttx[64,4,4]_seed1", ax=axes[3])
    axes[3].set_ylabel(ylabel="transformer-64", fontsize=fontsize)
    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[128,4,4]_ttx[64,4,4]_seed1", ax=axes[4])
    axes[4].set_ylabel(ylabel="transformer-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(iter_df, model_name="ttx[512,4,4]_ttx[64,4,4]_seed1", ax=axes[5])
    axes[5].set_ylabel(ylabel="transformer-512", fontsize=fontsize)

    plt.savefig(plot_dirpath + filename)


    filename = "grid_dist.png"
    fig, axes = plt.subplots(6, 1, figsize=(4 * img_width, 6 * img_height))

    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[64,2]_lstm[64,2]_seed1", ax=axes[0])
    axes[0].set_ylabel(ylabel="lstm-64", fontsize=fontsize)

    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[128,2]_lstm[64,2]_seed1", ax=axes[1])
    axes[1].set_ylabel(ylabel="lstm-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(grid_df, model_name="lstm[512,2]_lstm[64,2]_seed1", ax=axes[2])
    axes[2].set_ylabel(ylabel="lstm-512", fontsize=fontsize)

    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[64,8,8]_ttx[64,4,8]_seed1", ax=axes[3])
    axes[3].set_ylabel(ylabel="transformer-64", fontsize=fontsize)
    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[128,8,8]_ttx[64,4,8]_seed1", ax=axes[4])
    axes[4].set_ylabel(ylabel="transformer-128", fontsize=fontsize)
    plot_embedding_space_as_boxplot(grid_df, model_name="ttx[512,8,8]_ttx[64,4,8]_seed1", ax=axes[5])
    axes[5].set_ylabel(ylabel="transformer-512", fontsize=fontsize)

    plt.savefig(plot_dirpath + filename)


def save_all_results():
    if os.path.isdir(plot_dirpath):
        shutil.rmtree(plot_dirpath)
    os.makedirs(plot_dirpath)

    iter_agent_specific_df = load_iter_data("agent_specific")
    iter_model_specific_df = load_iter_data("model_specific")

    grid_agent_specific_df = load_grid_data("agent_specific")
    grid_model_specific_df = load_grid_data("model_specific")
    #
    save_iter_performance_plot(iter_model_specific_df)
    save_grid_performance_plot(grid_model_specific_df)
    save_disentanglement(iter_model_specific_df, grid_model_specific_df)
    save_cluster(iter_model_specific_df, grid_model_specific_df)
    save_pca(iter_model_specific_df, grid_model_specific_df)
    save_appendix_pca(iter_model_specific_df, grid_model_specific_df)
    save_convergence(iter_model_specific_df, grid_model_specific_df)
    save_stability(iter_agent_specific_df, grid_agent_specific_df)
    save_dist(iter_agent_specific_df, grid_agent_specific_df)
    save_large_dist(iter_agent_specific_df, grid_agent_specific_df)
    save_wide_dist(iter_agent_specific_df, grid_agent_specific_df)
