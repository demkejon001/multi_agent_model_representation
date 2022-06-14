import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from tommas.viz.embedding_plot import hyp_plot, extract_embeddings_and_labels


def cosine_batch_sampler(label_cluster, tensor_embeddings, batch_size=64):
    n_embeddings = len(tensor_embeddings)
    matching_matrix = np.ones((n_embeddings, n_embeddings)) * -1
    for label, cluster in label_cluster.items():
        ixgrid = np.ix_(cluster, cluster)
        matching_matrix[ixgrid] = 1

    while True:
        indices = np.random.choice(n_embeddings, size=(batch_size, 2))
        # Get all index pairs that are the same
        exact_indices = indices[:, 0] == indices[:, 1]
        num_exact_indices = np.sum(exact_indices)
        # Set all index pairs that were the same to a different random value
        index_offset = np.random.randint(1, n_embeddings, size=num_exact_indices)
        indices[exact_indices, 1] = (indices[exact_indices, 0] + index_offset) % n_embeddings

        x = tensor_embeddings[indices[:, 0]]
        y = tensor_embeddings[indices[:, 1]]
        matches = torch.from_numpy(matching_matrix[indices[:, 0], indices[:, 1]])
        yield x, y, matches


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.euc_dist = nn.PairwiseDistance()

    def forward(self, x1, x2, y):
        distances = self.euc_dist(x1, x2)
        margin = torch.zeros(x1.size(0))
        margin[y == -1] = self.margin
        distances = F.relu(distances - margin)
        return torch.mean(distances * y)


# def cosine_loss(embeddings, labels, epochs=1000, mask_threshold=.85, verbose=False, weight_decay=.05):
#     label_cluster = dict()
#     for i, label in enumerate(labels):
#         if label not in label_cluster:
#             label_cluster[label] = []
#         label_cluster[label].append(i)
#
#     tensor_embeddings = torch.from_numpy(embeddings)
#
#     # criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)
#     criterion = ContrastiveLoss()
#
#     embedding_mask = torch.zeros((1, embeddings.shape[-1]))
#     embedding_mask.requires_grad = True
#     optimizer = Adam([embedding_mask], .01)
#     batch_sampler = cosine_batch_sampler(label_cluster, tensor_embeddings)
#     for i in range(epochs):
#         optimizer.zero_grad()
#         x, y, matches = next(batch_sampler)
#         masked_x = x * torch.sigmoid(embedding_mask)
#         masked_y = y * torch.sigmoid(embedding_mask)
#         # relued_embedding = F.relu(embedding_mask)
#         loss = criterion(masked_x, masked_y, matches) #+ weight_decay * (torch.inner(relued_embedding, relued_embedding) / masked_x.size(0))
#         # print(loss.item())
#         loss.backward()
#         optimizer.step()
#
#     # embedding_responsibility = torch.sigmoid(embedding_mask.detach()).numpy()[0] > mask_threshold
#     best_values = torch.sigmoid(embedding_mask.detach()).numpy()[0].argsort()[::-1]
#     # print(best_values)
#
#     i = 1
#     ctr = 0
#     best_val = -1
#     best_embed = [False for _ in range(len(best_values))]
#
#     while ctr < 3:
#         embedding_responsibility = np.array([False for _ in range(len(best_values))])
#         # print(best_values[:i])
#         embedding_responsibility[best_values[:i]] = True
#         # embedding_responsiblity[best_values[i]] = True
#         sil_score = get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility)
#         # print(sil_score)
#         if sil_score > best_val:
#             best_val = sil_score
#             best_embed = embedding_responsibility.copy()
#         else:
#             ctr += 1
#         i += 1
#         # return silhouette_score(embeddings[:, embedding_responsibility], labels)
#     # print('best val', best_val)
#     embedding_responsibility = best_embed.copy()
#
#     # print(np.mean(torch.sigmoid(embedding_mask.detach()).numpy()[0][embedding_responsibility]))
#     # print(sum(embedding_responsibility))
#     # if sum(embedding_responsibility) > 10:
#     #     # print(embedding_responsibility)
#     #     # print(sum(embedding_responsibility))
#     #     # print()
#     #     responsible_indices = np.argpartition(torch.sigmoid(embedding_mask.detach()).numpy()[0], -10)[-10:] #np.sort(torch.sigmoid(embedding_mask.detach()).numpy()[0])
#     #
#     #     embedding_responsibility = np.zeros_like(embedding_responsibility).astype(bool)
#     #     embedding_responsibility[responsible_indices] = True
#     #     # raise ValueError
#     if verbose:
#         print(f"Embedding Mask: \n{torch.sigmoid(embedding_mask.detach()).numpy()[0]}")
#     return embedding_responsibility


def cosine_loss(embeddings, labels, epochs=1000, mask_threshold=.85, verbose=False, weight_decay=.05):
    label_cluster = dict()
    for i, label in enumerate(labels):
        if label not in label_cluster:
            label_cluster[label] = []
        label_cluster[label].append(i)

    tensor_embeddings = torch.from_numpy(embeddings)

    # criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)
    criterion = ContrastiveLoss()

    embedding_mask = torch.zeros((1, embeddings.shape[-1]))
    embedding_mask.requires_grad = True
    optimizer = Adam([embedding_mask], .01)
    batch_sampler = cosine_batch_sampler(label_cluster, tensor_embeddings)
    for i in range(epochs):
        optimizer.zero_grad()
        x, y, matches = next(batch_sampler)
        masked_x = x * torch.sigmoid(embedding_mask)
        masked_y = y * torch.sigmoid(embedding_mask)
        # relued_embedding = F.relu(embedding_mask)
        loss = criterion(masked_x, masked_y, matches) #+ weight_decay * (torch.inner(relued_embedding, relued_embedding) / masked_x.size(0))
        # print(loss.item())
        loss.backward()
        optimizer.step()

    # embedding_responsibility = torch.sigmoid(embedding_mask.detach()).numpy()[0] > mask_threshold
    best_values = torch.sigmoid(embedding_mask.detach()).numpy()[0].argsort()[::-1]
    # print(best_values)

    i = 1
    ctr = 0
    best_val = -1
    best_embed = [False for _ in range(len(best_values))]

    while ctr < 3:
        embedding_responsibility = np.array([False for _ in range(len(best_values))])
        # print(best_values[:i])
        embedding_responsibility[best_values[:i]] = True
        # embedding_responsiblity[best_values[i]] = True
        sil_score = get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility)
        # print(sil_score)
        if sil_score > best_val:
            best_val = sil_score
            best_embed = embedding_responsibility.copy()
        else:
            ctr += 1
        i += 1
        # return silhouette_score(embeddings[:, embedding_responsibility], labels)
    # print('best val', best_val)
    embedding_responsibility = best_embed.copy()

    # print(np.mean(torch.sigmoid(embedding_mask.detach()).numpy()[0][embedding_responsibility]))
    # print(sum(embedding_responsibility))
    # if sum(embedding_responsibility) > 10:
    #     # print(embedding_responsibility)
    #     # print(sum(embedding_responsibility))
    #     # print()
    #     responsible_indices = np.argpartition(torch.sigmoid(embedding_mask.detach()).numpy()[0], -10)[-10:] #np.sort(torch.sigmoid(embedding_mask.detach()).numpy()[0])
    #
    #     embedding_responsibility = np.zeros_like(embedding_responsibility).astype(bool)
    #     embedding_responsibility[responsible_indices] = True
    #     # raise ValueError
    if verbose:
        print(f"Embedding Mask: \n{torch.sigmoid(embedding_mask.detach()).numpy()[0]}")
    return embedding_responsibility


def linear_contrastive_loss(embeddings, labels, epochs=1000, verbose=False, weight_decay=.05):
    label_cluster = dict()
    for i, label in enumerate(labels):
        if label not in label_cluster:
            label_cluster[label] = []
        label_cluster[label].append(i)

    tensor_embeddings = torch.from_numpy(embeddings).float()

    # criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)
    # criterion = ContrastiveLoss(margin=1.5)
    criterion = ContrastiveLoss(margin=1.5)

    embedding_mask = nn.Linear(embeddings.shape[-1], embeddings.shape[-1], bias=False) #torch.zeros((embeddings.shape[-1], embeddings.shape[-1]))
    # embedding_mask.requires_grad = True
    # optimizer = Adam([embedding_mask], .01)
    optimizer = Adam(embedding_mask.parameters(), .01)
    batch_sampler = cosine_batch_sampler(label_cluster, tensor_embeddings)
    for i in range(epochs):
        optimizer.zero_grad()
        x, y, matches = next(batch_sampler)
        masked_x = embedding_mask(x)
        masked_y = embedding_mask(y)
        # relued_embedding = F.relu(embedding_mask)
        loss = criterion(masked_x, masked_y, matches) #+ weight_decay * (torch.inner(relued_embedding, relued_embedding) / masked_x.size(0))
        # print(loss.item())
        loss.backward()
        optimizer.step()

    # print(embedding_mask.weight.data)

    return silhouette_score(embedding_mask(tensor_embeddings).detach().numpy(), labels)

    # print(np.mean(torch.sigmoid(embedding_mask.detach()).numpy()[0][embedding_responsibility]))
    # print(sum(embedding_responsibility))
    # if sum(embedding_responsibility) > 10:
    #     # print(embedding_responsibility)
    #     # print(sum(embedding_responsibility))
    #     # print()
    #     responsible_indices = np.argpartition(torch.sigmoid(embedding_mask.detach()).numpy()[0], -10)[-10:] #np.sort(torch.sigmoid(embedding_mask.detach()).numpy()[0])
    #
    #     embedding_responsibility = np.zeros_like(embedding_responsibility).astype(bool)
    #     embedding_responsibility[responsible_indices] = True
    #     # raise ValueError
    # if verbose:
    #     print(f"Embedding Mask: \n{torch.sigmoid(embedding_mask.detach()).numpy()[0]}")
    # return embedding_responsibility


def get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility):
    return silhouette_score(embeddings[:, embedding_responsibility], labels)


def get_responsible_embeddings_centroid(embeddings, labels, embedding_responsibility):
    embedding_centroids = []
    for label in np.unique(labels):
        embedding_centroids.append(np.mean(embeddings[labels == label], axis=0))
    embedding_centroids = np.array(embedding_centroids)
    return embedding_centroids[:, embedding_responsibility]


def labeled_embeddings_to_cluster(embeddings, labels):
    embedding_clusters = []
    for label in set(labels):
        embedding_clusters.append(embeddings[labels == label])
    return embedding_clusters


def plot_embedding_responsibility(df, metadata, label_by="", remove_missing_labels=False,
                                  responsible_embedding_cells=None, n_past=5, current_traj_len=19, mask_threshold=.5,
                                  verbose=False, plot_char_embedding=True, agent_type=None, short_title=False,
                                  short_output=False, **kwargs):
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

    embedding_name = ""
    if len(embeddings.shape) == 3:
        embedding_idx = 1
        embedding_name = "Mental-"
        if plot_char_embedding:
            embedding_idx = 0
            embedding_name = "Character-"
        embeddings = embeddings[:, embedding_idx]
        embeddings = np.stack(embeddings, axis=0).astype(float)

    if responsible_embedding_cells is None:
        embedding_responsibility = cosine_loss(embeddings, labels, mask_threshold=mask_threshold, verbose=verbose)
    else:
        embedding_responsibility = np.zeros(embeddings.shape[-1])
        for cell in responsible_embedding_cells:
            embedding_responsibility[cell] = 1
        embedding_responsibility = embedding_responsibility.astype(bool)

    if np.any(embedding_responsibility):
        silhouette = get_responsible_embeddings_silhouette_score(embeddings, labels, embedding_responsibility)
        if short_output:
            print(f"Responsible {embedding_name}Embeddings: {embedding_responsibility.nonzero()[0].tolist()}"
                  f" -> Silhouette Score: {silhouette:.3f}")
        else:
            print(f"Responsible {embedding_name}Embeddings: {embedding_responsibility.nonzero()[0].tolist()}"
                  f" -> Silhouette Score: {silhouette:.3f}")
            responsible_centroid = get_responsible_embeddings_centroid(embeddings, labels, embedding_responsibility)
            print(f"Centroids:")
            for centroid in responsible_centroid:
                print(f"\t{centroid}")
        title = f"Responsible {embedding_name}Embeddings grouped by: {label_by}"
        if short_title:
            title = f"{label_by}"
        hyp_plot(embeddings[:, embedding_responsibility], labels, title=title, **kwargs)
    else:
        print(f"No embedding responsible")


def plot_embedding_responsibility_distribution(df, model_name="lstm[128,1]_lstm[48,2]_seed1", label_by="starting_action"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    specific_df = df[(df["model_name"] == model_name) & (df["n_past"] == 9)]
    embeddings = []
    labels = []
    for label, grouped_df in specific_df.groupby(label_by):
        group_embeddings = np.vstack(np.vstack(grouped_df.char_embedding.tolist()))
        embeddings.append(group_embeddings)
        labels.extend([label for _ in range(len(group_embeddings))])

    embeddings = np.concatenate(embeddings)
    sorting = np.argsort(np.var(embeddings, axis=0))
    embeddings = embeddings[:, sorting]
    embedding_responsibility = cosine_loss(embeddings, labels).nonzero()[0]

    if np.any(embedding_responsibility):
        cols = 8
        rows = embeddings.shape[1] // cols
        col_size = 12
        row_size = int(col_size * (rows / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(col_size, row_size))
        for i in range(rows):
            for j in range(cols):
                idx = 8*i+j
                if idx in embedding_responsibility:
                    g = sns.kdeplot(x=embeddings[:, idx], hue=labels, ax=axes[i, j], legend=False)
                else:
                    g = sns.kdeplot(embeddings[:, idx], ax=axes[i, j])
                g.set(xlim=(-1.5, 1.5), ylabel='')
        plt.plot()
    else:
        print(f"No embedding responsible")
