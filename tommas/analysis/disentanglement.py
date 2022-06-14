# import itertools
# import numpy as np
# from numpy.linalg import norm
# from scipy.spatial.distance import cosine
# from scipy.spatial.transform import Rotation as R
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
#
# from tommas.agents.create_iterative_action_agents import get_random_iterative_action_agent, RandomStrategySampler
# from tommas.data.gridworld_transforms import IterativeActionFullPastCurrentSplit
# from tommas.data.iterative_action_dataset import IterativeActionTrajectory, play_episode
# from tommas.analysis.representation_metrics import _get_model_char_embeddings
#
# '''
# Issues with DisLoss:
# 1. If there are certain representation values that don't change then embedding mask will favor them because the z_diff will be zero.
#     For example if there is absolute no disentanglement between two variables.
# 2. When comparing a_diff and b_diff, if one of the diffs have a column of zeros, then that is accepted column.
# '''
#
#
# class OrthoDisentanglementLoss(nn.Module):
#     def __init__(self, alpha=15):
#         super().__init__()
#         self.alpha = alpha
#         self.cos_sim = nn.CosineSimilarity()
#
#     def forward(self, a, b):
#         abs_sims = torch.abs(self.cos_sim(a, b))
#         loss = torch.mean(abs_sims)
#         loss += self.alpha * torch.var(abs_sims)
#         return loss
#
#
# def dis_batch_sampler(label_cluster, tensor_embeddings, batch_size=64):
#     while True:
#         indices_a = np.random.choice(label_cluster[1], size=batch_size)
#         indices_b = np.random.choice(label_cluster[-1], size=batch_size)
#         x = tensor_embeddings[indices_a]
#         y = tensor_embeddings[indices_b]
#         yield x, y
#
#
# def get_ortho_disentanglement_scores(embeddings, labels, epochs=3000, mask_threshold=.75, verbose=False):
#     label_cluster = dict()
#     for i, label in enumerate(labels):
#         if label not in label_cluster:
#             label_cluster[label] = []
#         label_cluster[label].append(i)
#
#     tensor_embeddings = torch.from_numpy(embeddings)
#     criterion = OrthoDisentanglementLoss()
#
#     embedding_mask = torch.zeros((1, embeddings.shape[-1]))
#     embedding_mask.requires_grad = True
#     optimizer = Adam([embedding_mask], .001)
#     batch_sampler = dis_batch_sampler(label_cluster, tensor_embeddings, batch_size=1280)
#     for i in range(epochs):
#         optimizer.zero_grad()
#         x, y = next(batch_sampler)
#         masked_x = x * torch.sigmoid(embedding_mask)
#         masked_y = y * torch.sigmoid(embedding_mask)
#         loss = criterion(masked_x, masked_y)
#         loss.backward()
#         optimizer.step()
#
#     embedding_responsibility = torch.sigmoid(embedding_mask.detach()) > mask_threshold
#     embedding_responsibility = embedding_responsibility.float()
#     masked_embeddings = tensor_embeddings * embedding_responsibility
#
#     a = masked_embeddings[labels == 1].repeat_interleave(masked_embeddings.shape[0] // 2, dim=0)
#     b = masked_embeddings[labels == -1].repeat((masked_embeddings.shape[0] // 2, 1))
#
#     cos_sim = torch.mean(torch.abs(F.cosine_similarity(a, b))).item()
#     var_score = torch.var(torch.abs(F.cosine_similarity(a, b))).item()
#
#     print(torch.sigmoid(embedding_mask.detach()))
#     if verbose:
#         print(f"Embedding Mask: \n{torch.sigmoid(embedding_mask.detach()).numpy()[0]}")
#     return 1 - cos_sim, var_score, embedding_responsibility
#
#
# def generate_random_traj(agent_type, agent_kwargs, seed=42, n_past=5, num_agents=2, opponent_idx=1, opponent_seed=42):
#     if opponent_idx < 1:
#         raise ValueError(f"opponent_idx needs to be greater than 1")
#     rss = RandomStrategySampler(seed, True)
#     opponent_rss = RandomStrategySampler(opponent_seed, True)
#     np.random.seed(opponent_seed)
#     agents = [get_random_iterative_action_agent(agent_type, rss, -1, 0, **agent_kwargs)[0]]
#     for i in range(1, num_agents):
#         agents.append(get_random_iterative_action_agent("mixed_strategy", opponent_rss, -1, i)[0])
#     traj = np.array([play_episode(agents) for _ in range(n_past + 1)], dtype=np.int64)
#
#     agent_indices = list(range(num_agents))
#     agent_indices[1], agent_indices[opponent_idx] = agent_indices[opponent_idx], agent_indices[1]
#     return traj[:, :, agent_indices]
#
#
# def get_factors_diff_params(match_params, mismatch_params, num_diff_vectors):
#     num_prior_diff = num_diff_vectors // 2
#
#     match1_indices = np.random.choice(len(match_params), size=num_prior_diff)
#     match1_params = np.array(match_params)[match1_indices]
#
#     mismatch1_indices = np.random.choice(len(mismatch_params), size=num_prior_diff)
#     index_offset = np.random.randint(1, len(mismatch_params), size=num_prior_diff)
#     mismatch2_indices = (mismatch1_indices + index_offset) % len(mismatch_params)
#
#     mismatch1_params = np.array(mismatch_params)[mismatch1_indices]
#     mismatch2_params = np.array(mismatch_params)[mismatch2_indices]
#
#     return match1_params, mismatch1_params, mismatch2_params
#
# #
# # def get_diff_dataset(num_diff_vectors=100, seed=314):
# #     starting_actions = [0, 1]
# #     mixed_strategies = [[1., 0], [0, 1.]]
# #     trigger_actions = [0, 1]
# #     opponent_indices = [1, 2, 3]
# #
# #     # grim_starting_action = [0, 1]
# #     # mirror_starting_action = [0, 1]
# #     # wsls_starting_action = [0, 1]
# #     # mixed_strategies = [[1., 0], [0, 1.]]
# #     #
# #     # grim_trigger_actions = [0, 1]
# #     # wsls_trigger_actions = [0, 1]
# #     # mtp_trigger_actions = [0, 1]
# #
# #     # params = [
# #     # "grim_starting_action", [0, 1],
# #     # "mirror_starting_action", [0, 1]
# #     # "wsls_starting_action", [0, 1]
# #     # "mixed_strategies", [[1., 0], [0, 1.]]
# #     # "grim_trigger_actions", [0, 1]
# #     # "wsls_trigger_actions", [0, 1]
# #     # "mtp_trigger_actions", [0, 1]
# #     # ]
# #
# #     params = [
# #         ("grim", "starting_action", [0, 1]),
# #         ("mirror", "starting_action", [0, 1]),
# #         ("wsls", "starting_action", [0, 1]),
# #         ("mixed_trigger_pattern", "mixed_strategies", [[1., 0], [0, 1.]]),
# #         ("grim", "trigger_action", [0, 1]),
# #         ("wsls", "trigger_action", [0, 1]),
# #         ("mixed_trigger_pattern", "trigger_action", [0, 1]), # TODO: Opponent indices?
# #     ]
# #
# #     import itertools
# #     for a, b in itertools.product(params):
# #         agent_type_a, agent_kwarg_a, agent_param_a = a
# #         agent_type_b, agent_kwarg_b, agent_param_b = b
# #
# #         delta_a_factors = get_factors_diff_params(agent_param_b, agent_param_a, num_diff_vectors)
# #         delta_b_factors = get_factors_diff_params(agent_param_a, agent_param_b, num_diff_vectors)
# #
# #         delta_trajs1 = []
# #         delta_trajs2 = []
# #         for a_match, delta_b1, delta_b2 in zip(delta_b_factors):
# #             delta_trajs1.append(generate_random_traj(agent_type_a, {agent_kwarg_a: a_match}, seed=seed, num_agents=4,
# #                                                      opponent_idx=np.random.randint(1, 4), n_past=9))
# #             seed += 1
# #             delta_trajs2.append(generate_random_traj(agent_type_a, {agent_kwarg_a: a_match}, seed=seed, num_agents=4,
# #                                                      opponent_idx=np.random.randint(1, 4), n_past=9))
# #
# #         for model_name, model_info in zip(models_names, models_info):
# #             char_embeddings, losses, accs = _get_model_prediction_data(models_dict[model_name], trajs)
# #             for n_p in range(n_past):
# #                 char_embedding = char_embeddings[:, n_p]
# #                 loss = losses[:, n_p]
# #                 acc = losses[:, n_p]
# #                 initial_representation_data.append(
# #                     {"model_name": model_name, **model_info, "agent_label": agent_label, "agent_type": agent_type,
# #                      **filled_agent_info, "opponent_idx": opponent_idx, "n_past": n_p,
# #                      "char_embedding": char_embedding, "loss": loss, "acc": acc}
# #                 )
# #
# #
# #
# #     agent_infos = dict()
# #     agent_infos["mirror"] = [{"starting_action": 0}, {"starting_action": 1}]
# #     agent_clusters = []
# #     for starting_action, opponent_idx in itertools.product(starting_actions, opponent_indices):
# #         agent_type = "mirror"
# #         agent_info = {"starting_action": starting_action}
# #         agent_clusters.append((agent_type, agent_info, opponent_idx))
# #
# #     for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions,
# #                                                                            opponent_indices):
# #         agent_type = "grim_trigger"
# #         agent_info = {"starting_action": starting_action, "trigger_action": [trigger_action]}
# #         agent_clusters.append((agent_type, agent_info, opponent_idx))
# #
# #     for starting_action, trigger_action, opponent_idx in itertools.product(starting_actions, trigger_actions,
# #                                                                            opponent_indices):
# #         agent_type = "wsls"
# #         agent_info = {"starting_action": starting_action, "win_trigger": [trigger_action]}
# #         agent_clusters.append((agent_type, agent_info, opponent_idx))
# #
# #     for mixed_strategy, trigger_action, trigger_action_pattern, opponent_idx in itertools.product(mixed_strategies,
# #                                                                                                   trigger_actions,
# #                                                                                                   trigger_action_patterns,
# #                                                                                                   opponent_indices):
# #         agent_type = "mixed_trigger_pattern"
# #         agent_info = {"mixed_strategy": mixed_strategy, "trigger_action": [trigger_action],
# #                       "action_pattern": trigger_action_pattern}
# #         agent_clusters.append((agent_type, agent_info, opponent_idx))
# #     return agent_clusters
#
#
# def get_opponent_idx_disentanglement_scores(models, num_diff_vectors=100, seed=314):
#     opponent_indices = [1, 2, 3]
#     params = [
#         ("grim_trigger", "starting_action", [0, 1]),
#         ("mirror", "starting_action", [0, 1]),
#         ("wsls", "starting_action", [0, 1]),
#         ("mixed_trigger_pattern", "mixed_strategy", [[1., 0], [0, 1.]]),
#         ("grim_trigger", "trigger_action", [[0], [1]]),
#         ("wsls", "win_trigger", [[0], [1]]),
#         ("mixed_trigger_pattern", "trigger_action", [[0], [1]]),
#     ]
#
#     models_ortho_disentanglement_scores = [dict() for _ in models]
#
#     opponent_seed = seed
#
#     for agent_type, agent_kwarg_name, agent_kwarg_params in params:
#         models_opp_idx_diff_vectors = []
#         models_agent_param_diff_vectors = []
#
#         delta_opp_indices = get_factors_diff_params(agent_kwarg_params, opponent_indices, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#
#         for agent_kwarg_match, opp_idx1, opp_idx2 in zip(*delta_opp_indices):
#             traj = generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
#                                         num_agents=4, opponent_idx=opp_idx1, n_past=9, opponent_seed=opponent_seed)
#             delta_trajs1.append(traj)
#             seed += 1
#             swapped_opponent_indices = list(range(4))
#             swapped_opponent_indices[opp_idx1], swapped_opponent_indices[opp_idx2] = swapped_opponent_indices[opp_idx2], swapped_opponent_indices[opp_idx1]
#             delta_trajs2.append(traj[:, swapped_opponent_indices])
#             opponent_seed += 1
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_opp_idx_diff_vectors.append(a - b)
#
#         delta_agent_param = get_factors_diff_params(opponent_indices, agent_kwarg_params, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#         for opp_idx, agent_kwarg_a, agent_kwarg_b in zip(*delta_agent_param):
#             delta_trajs1.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_a}, seed=seed,
#                                 opponent_seed=opponent_seed, num_agents=4, opponent_idx=opp_idx, n_past=9))
#             seed += 1
#             delta_trajs2.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_b}, seed=seed,
#                                 opponent_seed=opponent_seed, num_agents=4, opponent_idx=opp_idx, n_past=9))
#             seed += 1
#             opponent_seed += 1
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_agent_param_diff_vectors.append(a - b)
#
#         for i, (opp_idx_diff_vectors, agent_param_diff_vectors) in enumerate(zip(models_opp_idx_diff_vectors, models_agent_param_diff_vectors)):
#             embeddings = np.concatenate((opp_idx_diff_vectors, agent_param_diff_vectors), axis=0)
#             labels = np.array([1 for _ in range(len(opp_idx_diff_vectors))] + [-1 for _ in range(len(agent_param_diff_vectors))])
#             ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
#             models_ortho_disentanglement_scores[i][agent_type + "_" + agent_kwarg_name] = (ortho_score, var_score, embedding_mask)
#
#     return models_ortho_disentanglement_scores
#
# # def get_opponent_idx_disentanglement_scores(models, num_diff_vectors=100, seed=314):
# #     opponent_indices = [1, 2, 3]
# #     params = [
# #         ("grim_trigger", "starting_action", [0, 1]),
# #         ("mirror", "starting_action", [0, 1]),
# #         ("wsls", "starting_action", [0, 1]),
# #         ("mixed_trigger_pattern", "mixed_strategy", [[1., 0], [0, 1.]]),
# #         ("grim_trigger", "trigger_action", [[0], [1]]),
# #         ("wsls", "win_trigger", [[0], [1]]),
# #         ("mixed_trigger_pattern", "trigger_action", [[0], [1]]),
# #     ]
# #
# #     models_ortho_disentanglement_scores = [dict() for _ in models]
# #
# #     opponent_seed = seed
# #
# #     for agent_type, agent_kwarg_name, agent_kwarg_params in params:
# #         models_opp_idx_diff_vectors = []
# #         models_agent_param_diff_vectors = []
# #
# #         delta_opp_indices = get_factors_diff_params(agent_kwarg_params, opponent_indices, num_diff_vectors)
# #         delta_trajs1 = []
# #         delta_trajs2 = []
# #
# #         for agent_kwarg_match, opp_idx1, opp_idx2 in zip(*delta_opp_indices):
# #             traj = generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
# #                                         num_agents=4, opponent_idx=opp_idx1, n_past=9, opponent_seed=opponent_seed)
# #             delta_trajs1.append(traj)
# #             seed += 1
# #             swapped_opponent_indices = list(range(4))
# #             swapped_opponent_indices[opp_idx1], swapped_opponent_indices[opp_idx2] = swapped_opponent_indices[opp_idx2], swapped_opponent_indices[opp_idx1]
# #             delta_trajs2.append(traj[:, swapped_opponent_indices])
# #
# #             # delta_trajs1.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
# #             #                                          num_agents=4, opponent_idx=opp_idx1, n_past=9))
# #             # seed += 1
# #             # delta_trajs2.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
# #             #                                          num_agents=4, opponent_idx=opp_idx2, n_past=9))
# #             # seed += 1
# #
# #         for model in models:
# #             a = _get_model_char_embeddings(model, delta_trajs1)
# #             b = _get_model_char_embeddings(model, delta_trajs2)
# #             models_opp_idx_diff_vectors.append(a - b)
# #
# #         delta_agent_param = get_factors_diff_params(opponent_indices, agent_kwarg_params, num_diff_vectors)
# #         delta_trajs1 = []
# #         delta_trajs2 = []
# #         for opp_idx, agent_kwarg_a, agent_kwarg_b in zip(*delta_agent_param):
# #             delta_trajs1.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_a}, seed=seed,
# #                                                      num_agents=4, opponent_idx=opp_idx, n_past=9))
# #             seed += 1
# #             delta_trajs2.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_b}, seed=seed,
# #                                                      num_agents=4, opponent_idx=opp_idx, n_past=9))
# #             seed += 1
# #
# #         for model in models:
# #             a = _get_model_char_embeddings(model, delta_trajs1)
# #             b = _get_model_char_embeddings(model, delta_trajs2)
# #             models_agent_param_diff_vectors.append(a - b)
# #
# #         for i, (opp_idx_diff_vectors, agent_param_diff_vectors) in enumerate(zip(models_opp_idx_diff_vectors, models_agent_param_diff_vectors)):
# #             embeddings = np.concatenate((opp_idx_diff_vectors, agent_param_diff_vectors), axis=0)
# #             labels = np.array([1 for _ in range(len(opp_idx_diff_vectors))] + [-1 for _ in range(len(agent_param_diff_vectors))])
# #             ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
# #             models_ortho_disentanglement_scores[i][agent_type + "_" + agent_kwarg_name] = (ortho_score, var_score, embedding_mask)
# #
# #     return models_ortho_disentanglement_scores
#
#
# def get_opponent_idx_baseline_disentanglement_scores(models, num_diff_vectors=100, seed=314):
#     opponent_indices = [1, 2, 3]
#     params = [
#         ("grim_trigger", "starting_action", [0, 1]),
#         ("mirror", "starting_action", [0, 1]),
#         ("wsls", "starting_action", [0, 1]),
#         ("mixed_trigger_pattern", "mixed_strategy", [[1., 0], [0, 1.]]),
#         ("grim_trigger", "trigger_action", [[0], [1]]),
#         ("wsls", "win_trigger", [[0], [1]]),
#         ("mixed_trigger_pattern", "trigger_action", [[0], [1]]),
#     ]
#
#     def rand_agent_type(): return np.random.choice(["grim_trigger", "wsls", "mixed_trigger_pattern", "mirror"])
#
#     models_ortho_disentanglement_scores = [dict() for _ in models]
#
#     for agent_type, agent_kwarg_name, agent_kwarg_params in params:
#         models_opp_idx_diff_vectors = []
#         models_agent_param_diff_vectors = []
#
#         delta_opp_indices = get_factors_diff_params(agent_kwarg_params, opponent_indices, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#
#         for agent_kwarg_match, opp_idx1, opp_idx2 in zip(*delta_opp_indices):
#             traj = generate_random_traj(rand_agent_type(), {}, seed=seed,
#                                         num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9)
#             delta_trajs1.append(traj)
#             seed += 1
#             # swapped_opponent_indices = list(range(4))
#             # swapped_opponent_indices[opp_idx1], swapped_opponent_indices[opp_idx2] = swapped_opponent_indices[opp_idx2], swapped_opponent_indices[opp_idx1]
#             delta_trajs2.append(generate_random_traj(rand_agent_type(), {}, seed=seed,
#                                         num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#             seed += 1
#
#             # delta_trajs1.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
#             #                                          num_agents=4, opponent_idx=opp_idx1, n_past=9))
#             # seed += 1
#             # delta_trajs2.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_match}, seed=seed,
#             #                                          num_agents=4, opponent_idx=opp_idx2, n_past=9))
#             # seed += 1
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_opp_idx_diff_vectors.append(a - b)
#
#         delta_agent_param = get_factors_diff_params(opponent_indices, agent_kwarg_params, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#         for opp_idx, agent_kwarg_a, agent_kwarg_b in zip(*delta_agent_param):
#             delta_trajs1.append(generate_random_traj(rand_agent_type(), {}, seed=seed,
#                                                      num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#             seed += 1
#             delta_trajs2.append(generate_random_traj(rand_agent_type(), {}, seed=seed,
#                                                      num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#             seed += 1
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_agent_param_diff_vectors.append(a - b)
#
#         for i, (opp_idx_diff_vectors, agent_param_diff_vectors) in enumerate(zip(models_opp_idx_diff_vectors, models_agent_param_diff_vectors)):
#             embeddings = np.concatenate((opp_idx_diff_vectors, agent_param_diff_vectors), axis=0)
#             labels = np.array([1 for _ in range(len(opp_idx_diff_vectors))] + [-1 for _ in range(len(agent_param_diff_vectors))])
#             ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
#             models_ortho_disentanglement_scores[i][agent_type + "_" + agent_kwarg_name] = (ortho_score, var_score, embedding_mask)
#
#     return models_ortho_disentanglement_scores
#
#
# def get_randomly_swapped_baseline_disentanglement_scores(models, num_diff_vectors=100, seed=314):
#     opponent_indices = [1, 2, 3]
#     params = [
#         ("grim_trigger", "starting_action", [0, 1]),
#         ("mirror", "starting_action", [0, 1]),
#         ("wsls", "starting_action", [0, 1]),
#         ("mixed_trigger_pattern", "mixed_strategy", [[1., 0], [0, 1.]]),
#         ("grim_trigger", "trigger_action", [[0], [1]]),
#         ("wsls", "win_trigger", [[0], [1]]),
#         ("mixed_trigger_pattern", "trigger_action", [[0], [1]]),
#     ]
#
#     def rand_agent_type(): return np.random.choice(["grim_trigger", "wsls", "mixed_trigger_pattern", "mirror"])
#
#     models_ortho_disentanglement_scores = [dict() for _ in models]
#
#     for agent_type, agent_kwarg_name, agent_kwarg_params in params:
#         models_opp_idx_diff_vectors = []
#         models_agent_param_diff_vectors = []
#
#         delta_opp_indices = get_factors_diff_params(agent_kwarg_params, opponent_indices, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#
#         for agent_kwarg_match, opp_idx1, opp_idx2 in zip(*delta_opp_indices):
#             traj = generate_random_traj(rand_agent_type(), {}, seed=seed,
#                                         num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9)
#             delta_trajs1.append(traj)
#             seed += 1
#             swapped_opponent_indices = list(range(4))
#             swapped_opponent_indices[opp_idx1], swapped_opponent_indices[opp_idx2] = swapped_opponent_indices[opp_idx2], swapped_opponent_indices[opp_idx1]
#
#             delta_trajs2.append(traj[:, swapped_opponent_indices])
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_opp_idx_diff_vectors.append(a - b)
#
#         delta_agent_param = get_factors_diff_params(opponent_indices, agent_kwarg_params, num_diff_vectors)
#         delta_trajs1 = []
#         delta_trajs2 = []
#
#         # for opp_idx, agent_kwarg_a, agent_kwarg_b in zip(*delta_agent_param):
#         #     delta_trajs1.append(generate_random_traj(rand_agent_type(), {}, seed=seed,
#         #                                              num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         #     seed += 1
#         #     delta_trajs2.append(generate_random_traj(rand_agent_type(), {}, seed=seed,
#         #                                              num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         #     seed += 1
#         for opp_idx, agent_kwarg_a, agent_kwarg_b in zip(*delta_agent_param):
#             delta_trajs1.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_a}, seed=seed,
#                                 opponent_seed=seed, num_agents=4, opponent_idx=opp_idx, n_past=9))
#             seed += 1
#             delta_trajs2.append(generate_random_traj(agent_type, {agent_kwarg_name: agent_kwarg_b}, seed=seed,
#                                 opponent_seed=seed, num_agents=4, opponent_idx=opp_idx, n_past=9))
#             seed += 1
#
#         for model in models:
#             a = _get_model_char_embeddings(model, delta_trajs1)
#             b = _get_model_char_embeddings(model, delta_trajs2)
#             models_agent_param_diff_vectors.append(a - b)
#
#         for i, (opp_idx_diff_vectors, agent_param_diff_vectors) in enumerate(zip(models_opp_idx_diff_vectors, models_agent_param_diff_vectors)):
#             embeddings = np.concatenate((opp_idx_diff_vectors, agent_param_diff_vectors), axis=0)
#             labels = np.array([1 for _ in range(len(opp_idx_diff_vectors))] + [-1 for _ in range(len(agent_param_diff_vectors))])
#             ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
#             models_ortho_disentanglement_scores[i][agent_type + "_" + agent_kwarg_name] = (ortho_score, var_score, embedding_mask)
#
#     return models_ortho_disentanglement_scores
#
#
#
# def get_fully_random_disentanglement_baseline(models, num_diff_vectors=100, seed=314):
#
#     def rand_agent_type(): return np.random.choice(["grim_trigger", "wsls", "mixed_trigger_pattern", "mirror"])
#
#     models_ortho_disentanglement_scores = [dict() for _ in models]
#
#     models_opp_idx_diff_vectors = []
#     models_agent_param_diff_vectors = []
#
#     delta_trajs1 = []
#     delta_trajs2 = []
#
#     for _ in range(num_diff_vectors):
#         delta_trajs1.append(generate_random_traj(rand_agent_type(), {}, seed=seed, opponent_seed=seed,
#                                                  num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         seed += 1
#         delta_trajs2.append(generate_random_traj(rand_agent_type(), {}, seed=seed, opponent_seed=seed,
#                                                  num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         seed += 1
#
#     for model in models:
#         a = _get_model_char_embeddings(model, delta_trajs1)
#         b = _get_model_char_embeddings(model, delta_trajs2)
#         models_opp_idx_diff_vectors.append(a - b)
#
#     delta_trajs1 = []
#     delta_trajs2 = []
#     for _ in range(num_diff_vectors):
#         delta_trajs1.append(generate_random_traj(rand_agent_type(), {}, seed=seed, opponent_seed=seed,
#                                                  num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         seed += 1
#         delta_trajs2.append(generate_random_traj(rand_agent_type(), {}, seed=seed, opponent_seed=seed,
#                                                  num_agents=4, opponent_idx=np.random.randint(1, 4), n_past=9))
#         seed += 1
#
#     for model in models:
#         a = _get_model_char_embeddings(model, delta_trajs1)
#         b = _get_model_char_embeddings(model, delta_trajs2)
#         models_agent_param_diff_vectors.append(a - b)
#
#     for i, (opp_idx_diff_vectors, agent_param_diff_vectors) in enumerate(zip(models_opp_idx_diff_vectors, models_agent_param_diff_vectors)):
#         embeddings = np.concatenate((opp_idx_diff_vectors, agent_param_diff_vectors), axis=0)
#         labels = np.array([1 for _ in range(len(opp_idx_diff_vectors))] + [-1 for _ in range(len(agent_param_diff_vectors))])
#         ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
#         models_ortho_disentanglement_scores[i]["baseline"] = (ortho_score, var_score, embedding_mask)
#
#     return models_ortho_disentanglement_scores
#
#
#
# def get_models_disentanglement_scores(models, num_diff_vectors=100, seed=314):
#     params = [
#         ("grim_trigger", "starting_action", [0, 1]),
#         # ("mirror", "starting_action", [0, 1]),
#         # ("wsls", "starting_action", [0, 1]),
#         # ("mixed_trigger_pattern", "mixed_strategy", [[1., 0], [0, 1.]]),
#         ("grim_trigger", "trigger_action", [[0], [1]]),
#         # ("wsls", "win_trigger", [[0], [1]]),
#         # ("mixed_trigger_pattern", "trigger_action", [[0], [1]]),
#     ]
#
#     models_ortho_disentanglement_scores = [dict() for _ in models]
#
#     opponent_seed = seed
#
#     models_a_param_diff_vectors = []
#     models_b_param_diff_vectors = []
#
#     a_type, a_kwarg_name, a_kwarg_params = params[0]
#     b_type, b_kwarg_name, b_kwarg_params = params[1]
#
#     delta_b_params = get_factors_diff_params(a_kwarg_params, b_kwarg_params, num_diff_vectors)
#     delta_trajs1 = []
#     delta_trajs2 = []
#
#     for a_kwarg_match, delta_b1, delta_b2 in zip(*delta_b_params):
#         traj = generate_random_traj(a_type, {a_kwarg_name: a_kwarg_match, b_kwarg_name: delta_b1}, seed=seed,
#                                     num_agents=4, opponent_idx=1, n_past=9, opponent_seed=opponent_seed)
#         delta_trajs1.append(traj)
#         seed += 1
#
#         traj = generate_random_traj(a_type, {a_kwarg_name: a_kwarg_match, b_kwarg_name: delta_b2}, seed=seed,
#                                     num_agents=4, opponent_idx=1, n_past=9, opponent_seed=opponent_seed)
#         delta_trajs2.append(traj)
#         seed += 1
#         opponent_seed += 1
#
#     for model in models:
#         a = _get_model_char_embeddings(model, delta_trajs1)
#         b = _get_model_char_embeddings(model, delta_trajs2)
#         models_b_param_diff_vectors.append(a - b)
#
#     delta_a_params = get_factors_diff_params(b_kwarg_params, a_kwarg_params, num_diff_vectors)
#     delta_trajs1 = []
#     delta_trajs2 = []
#     for b_kwarg_match, delta_a1, delta_a2 in zip(*delta_a_params):
#         traj = generate_random_traj(a_type, {a_kwarg_name: delta_a1, b_kwarg_name: b_kwarg_match}, seed=seed,
#                                     num_agents=4, opponent_idx=1, n_past=9, opponent_seed=opponent_seed)
#         delta_trajs1.append(traj)
#         seed += 1
#
#         traj = generate_random_traj(a_type, {a_kwarg_name: delta_a2, b_kwarg_name: b_kwarg_match}, seed=seed,
#                                     num_agents=4, opponent_idx=1, n_past=9, opponent_seed=opponent_seed)
#         delta_trajs2.append(traj)
#         seed += 1
#         opponent_seed += 1
#
#     for model in models:
#         a = _get_model_char_embeddings(model, delta_trajs1)
#         b = _get_model_char_embeddings(model, delta_trajs2)
#         models_a_param_diff_vectors.append(a - b)
#
#     for i, (a_diff_vectors, b_diff_vectors) in enumerate(zip(models_a_param_diff_vectors, models_b_param_diff_vectors)):
#         embeddings = np.concatenate((a_diff_vectors, b_diff_vectors), axis=0)
#         labels = np.array([1 for _ in range(len(a_diff_vectors))] + [-1 for _ in range(len(b_diff_vectors))])
#         ortho_score, var_score, embedding_mask = get_ortho_disentanglement_scores(embeddings, labels)
#         models_ortho_disentanglement_scores[i]["grim_trigger_params"] = (ortho_score, var_score, embedding_mask)
#
#     return models_ortho_disentanglement_scores
#


# Taken from https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/dci.py
# Modified to handle discrete data in RandomForestRegressor
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale


def dci(factors, codes):
    ''' DCI metrics from C. Eastwood and C. K. I. Williams,
        “A framework for the quantitative evaluation of disentangled representations,”
        in ICLR, 2018.

    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param model:                           model to use for score computation
                                            either lasso or random_forest
    '''

    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]

    # normalize in [0, 1] all columns
    # factors = minmax_scale(factors)
    codes = minmax_scale(codes)

    # compute entropy matrix and informativeness per factor
    e_matrix = np.zeros((nb_factors, nb_codes))
    informativeness = np.zeros((nb_factors,))
    for f in range(nb_factors):
        factors_f = minmax_scale(np.vstack(factors[:, f]))
        informativeness[f], weights = _fit_random_forest(factors_f, codes)
        e_matrix[f, :] = weights

    # compute disentanglement per code
    rho = np.zeros((nb_codes,))
    disentanglement = np.zeros((nb_codes,))
    for c in range(nb_codes):
        # get importance weight for code c
        rho[c] = np.sum(e_matrix[:, c])
        if rho[c] == 0:
            disentanglement[c] = 0
            break

        # transform weights in probabilities
        prob = e_matrix[:, c] / rho[c]

        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        disentanglement[c] = 1 - H

    # compute final disentanglement
    if np.sum(rho):
        rho = rho / np.sum(rho)
    else:
        rho = rho * 0

    # compute completeness
    completeness = np.zeros((nb_factors,))
    for f in range(nb_factors):
        if np.sum(e_matrix[f, :]) != 0:
            prob = e_matrix[f, :] / np.sum(e_matrix[f, :])
        else:
            prob = np.ones((len(e_matrix[f, :]), 1)) / len(e_matrix[f, :])

            # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))

        # get disentanglement score
        completeness[f] = 1 - H

    # average all results
    disentanglement = np.dot(disentanglement, rho)
    completeness = np.mean(completeness)
    # informativeness = np.mean(informativeness)

    return disentanglement, completeness, informativeness


def _fit_random_forest(factors, codes):
    ''' Fit a Random Forest regressor on the data

    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    max_depth = [8, 16, 32, 64, 128]
    max_features = [0.2, 0.4, 0.8, 1.0]

    # make sure factors are N by 0
    if factors.shape[1] == 1:
        factors = np.ravel(factors)

    # find the optimal alpha regularization parameter
    best_mse = 10e10
    best_mf = 0
    best_md = 0
    for md in max_depth:
        for mf in max_features:
            # perform cross validation on the tree classifiers
            clf = RandomForestRegressor(n_estimators=10, max_depth=md, max_features=mf)
            mse = cross_val_score(clf, codes, factors, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
            mse = -mse.mean()

            if mse < best_mse:
                best_mse = mse
                best_mf = mf
                best_md = md

    # train the model using the best performing parameter
    clf = RandomForestRegressor(n_estimators=10, max_depth=best_md, max_features=best_mf)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12)
    mse = mean_squared_error(y_pred, factors)
    informativeness = max(1 - 12 * mse, 0)

    # get the weight from the regressor
    predictor_weights = clf.feature_importances_

    return informativeness, predictor_weights
