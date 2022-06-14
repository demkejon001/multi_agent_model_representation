# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import AgglomerativeClustering, DBSCAN
# # from skimage.metrics import structural_similarity as ssim
# import pickle
from typing import Tuple, List
# from itertools import combinations
#
from tommas.agents.hand_coded_agents import PathfindingAgent
from tommas.agents.destination_selector import CollaborativeStatePartitionFilter
# from tommas.agents.create_handcodedagents import create_agent
# from tommas.agents.reward_function import get_cooperative_reward_functions
# from tommas.env.gridworld_env import GridworldEnv
# # from tommas.data.gridworld_dataset import BehaviorGridworldDataset, GridworldTrajectory, recursively_split_dataset
# # from tommas.data.gridworld_datamodule import MultiFileGridworldDataModule
# from tommas.data.dataset_base import save_dataset as base_save_dataset, load_dataset as base_load_dataset, dataset_extension
#
#
# # Environment parameters
# num_agents_per_episode = 4
# num_goals = 12
# num_actions = 5
# horizon = 30
# dim = (21, 21)
# max_num_walls = 6
#
# # Dataset parameters
# num_train_dataset_agents = 1000
# num_test_dataset_agents = 100
# min_num_episodes_per_agent = 11
#
# destination_selector_dissimilarity_filename = '../../data/destination_selector_comparison.pickle'
# dataset_dirpath = "data/datasets/cooperative_rewards/"
# dataset_base_name = "cooperative_rewards"
#
# dataset_types = ['coop', 'coop_small', 'coop_collaborative', 'coop_collaborative_small']
#
#
# def play_episode(env, agents, create_new_world=True):
#     trajectory = []
#     state = env.reset(create_new_world)
#     for agent_idx, agent in enumerate(agents):
#         agent.reset(state, agent_idx)
#     done = False
#     while not done:
#         actions = [agent.action(state) for agent in agents]
#         new_state, reward, done, info = env.step(actions)
#         trajectory.append((state, reward, actions, info))
#         state = new_state
#     agent_ids = [agent.id for agent in agents]
#     return GridworldTrajectory(trajectory, env, agent_ids)
#
#
# def srs_dissimilarity(srs_a, srs_b):
#     return 1 - ssim(srs_a, srs_b, data_range=1)
#
#
# def get_random_goal_rewards():
#     goal_rewards = np.random.dirichlet(alpha=np.ones(num_goals), size=1)[0]
#     num_negative_goals = np.random.choice(range(0, 4), size=None, p=[.4, .3, .2, .1]) * 3 + np.random.randint(0, 3)
#     goal_rewards[np.argsort(goal_rewards)[:num_negative_goals]] *= -1
#     # goal_rewards -= np.random.random() / num_goals  # Another potential way to have random negative rewards
#     return goal_rewards * 2  # Multiply by 2 so chance of having goals == 1. This helps to destination selector parameters such as distance_penalty
#
#
# def get_random_coop_reward_funcs():
#     goal_rewards = get_random_goal_rewards()
#     return get_cooperative_reward_functions(num_agents_per_episode, goal_rewards), goal_rewards
#
#
# def get_random_env(reward_funcs):
#     env_seed = np.random.randint(0, 1e10)
#     return GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=max_num_walls,
#                         seed=env_seed)
#
#
# # This shows 2, 3, 4, and 6 are good filter sizes for all shapes
# def analyze_difference_in_local_state_filter_sizes():
#     np.random.seed(42)
#     goal_rewards = np.random.random(size=num_goals)
#     reward_funcs = get_cooperative_reward_functions(num_agents_per_episode, goal_rewards)
#     num_trials = 20
#     max_filter_radius = 20
#     num_filters = max_filter_radius - 1
#
#     fig, axes = plt.subplots(1, 3)
#     for shape_idx, filter_type in enumerate(['static_square', 'static_circle', 'static_diamond']):
#         agents_error_matrix = np.zeros((num_filters, num_filters))
#
#         for experiment_env_seed in range(num_trials):
#             env = GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=0,
#                                seed=experiment_env_seed)
#
#             agents_srs = [[] for _ in range(num_agents_per_episode)]
#             for filter_radius in range(1, max_filter_radius):
#                 filter_args = (filter_radius, )
#                 agents, _, _ = create_agent(reward_funcs, goal_ranker_type="highest", state_filter_type=filter_type,
#                                             state_filter_args=filter_args)
#                 gridworld_trajectory = play_episode(env, agents, create_new_world=False)
#                 for agent_idx in range(num_agents_per_episode):
#                     agents_srs[agent_idx].append(gridworld_trajectory.get_successor_representations(0, agent_idx=agent_idx)[0])
#
#             for agent_idx in range(num_agents_per_episode):
#                 error_matrix = np.zeros((num_filters, num_filters))
#                 for i in range(num_filters):
#                     for j in range(num_filters):
#                         error_matrix[i][j] = srs_dissimilarity(agents_srs[agent_idx][i], agents_srs[agent_idx][j])
#                 agents_error_matrix += error_matrix
#
#         axes[shape_idx].imshow(agents_error_matrix, cmap='hot', interpolation='nearest')
#         axes[shape_idx].set_title(filter_type)
#     plt.show()
#
#     # UNCOMMENT IF YOU WANT TO SEE THE DIFFERENCES
#     # fig, axes = plt.subplots(6, 3)
#     # for filter_size in range(2, 20):
#     #     summed_sr_representation = np.zeros(dim)
#     #     agents = get_highest_goal_preference_agents(reward_funcs, local_state_filter=shape + '_' + str(filter_size))
#     #     gridworld_trajectory = play_episode(env, agents, create_new_world=False)
#     #     for agent_idx in range(num_agents_per_episode):
#     #         summed_sr_representation += gridworld_trajectory.get_successor_representations(0, agent_idx=agent_idx)[0]
#     #
#     #     axes[np.unravel_index(filter_size - 2, (6, 3))].imshow(summed_sr_representation, cmap='hot', interpolation='nearest')
#     # plt.show()
#
#
# def analyze_difference_between_filter_shapes():
#     np.random.seed(42)
#     goal_rewards = np.random.random(size=num_goals)
#     reward_funcs = get_cooperative_reward_functions(num_agents_per_episode, goal_rewards)
#     num_trials = 20
#     shapes = ['square', 'circle', 'diamond']
#     filter_sizes = list(range(0, 15))
#     num_filters = len(filter_sizes)
#
#     shapes_error_matrix = np.zeros((len(shapes), num_filters))
#     for experiment_env_seed in range(num_trials):
#         env = GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=0,
#                            seed=experiment_env_seed)
#
#         for filter_idx, radius in enumerate(filter_sizes):
#             if radius == 0:
#                 agents_s, _, _ = create_agent(reward_funcs, goal_ranker_type="highest")
#                 agents_c, _, _ = create_agent(reward_funcs, goal_ranker_type="highest")
#                 agents_d, _, _ = create_agent(reward_funcs, goal_ranker_type="highest")
#             else:
#                 agents_s, _, _ = create_agent(reward_funcs, goal_ranker_type="highest",
#                                               state_filter_type="static_square", state_filter_args=(radius,))
#                 agents_c, _, _ = create_agent(reward_funcs, goal_ranker_type="highest",
#                                               state_filter_type="static_circle", state_filter_args=(radius,))
#                 agents_d, _, _ = create_agent(reward_funcs, goal_ranker_type="highest",
#                                               state_filter_type="static_diamond", state_filter_args=(radius,))
#             gridworld_trajectory_s = play_episode(env, agents_s, create_new_world=False)
#             gridworld_trajectory_c = play_episode(env, agents_c, create_new_world=False)
#             gridworld_trajectory_d = play_episode(env, agents_d, create_new_world=False)
#             for agent_idx in range(num_agents_per_episode):
#                 srs_s = gridworld_trajectory_s.get_successor_representations(0, agent_idx=agent_idx)[0]
#                 srs_c = gridworld_trajectory_c.get_successor_representations(0, agent_idx=agent_idx)[0]
#                 srs_d = gridworld_trajectory_d.get_successor_representations(0, agent_idx=agent_idx)[0]
#
#                 square_circle_distance = srs_dissimilarity(srs_s, srs_c)
#                 circle_diamond_distance = srs_dissimilarity(srs_c, srs_d)
#                 diamond_square_distance = srs_dissimilarity(srs_d, srs_s)
#
#                 shapes_error_matrix[0][filter_idx] += square_circle_distance
#                 shapes_error_matrix[1][filter_idx] += circle_diamond_distance
#                 shapes_error_matrix[2][filter_idx] += diamond_square_distance
#
#     plt.imshow(shapes_error_matrix, cmap='hot', interpolation='nearest')
#     plt.yticks(range(3), ['square vs circle', 'circle vs diamond', 'diamond vs square'])
#     plt.xticks(range(num_filters), filter_sizes)
#     plt.show()
#
#
# # # tracking goals attained / max goals ratio, approx num player collisions, timesteps to finish, approx computation time
# # def get_destination_selector_metrics(env_seed, goal_rewards, agents):
# #     def get_all_reachable_goals():
# #         start_state = env.reset(create_new_world=False)
# #         goal_positions = env.world.goal_positions
# #         agent_positions = env.world.agent_positions
# #         agent_reachable_positions = set()
# #         for agent_position in agent_positions:
# #             agent_reachable_positions = agent_reachable_positions.union(set(maze_traversal(start_state[0], agent_position)))
# #         return [int(goal) for goal_position, goal in goal_positions.items() if goal_position in agent_reachable_positions]
# #
# #     def calculate_max_possible_reward():
# #         reachable_goals = get_all_reachable_goals()
# #         return sum([goal_rewards[goal] for goal in reachable_goals if goal_rewards[goal] > 0])
# #
# #     def play_episode_track_metrics(env, agents, create_new_world=True):
# #         trajectory = []
# #         state = env.reset(create_new_world)
# #         for agent_idx, agent in enumerate(agents):
# #             agent.reset(state, agent_idx)
# #         done = False
# #         timestep = 0
# #         finish_timesteps = horizon
# #         while not done:
# #             actions = [agent.action(state) for agent in agents]
# #             new_state, reward, done, info = env.step(actions)
# #             trajectory.append((state, reward, actions, info))
# #             state = new_state
# #             timestep += 1
# #             if actions == [4 for _ in range(num_agents_per_episode)]:
# #                 if finish_timesteps == horizon:
# #                     finish_timesteps = timestep
# #                 else:
# #                     continue
# #         agent_ids = [agent.id for agent in agents]
# #         return GridworldTrajectory(trajectory, env, agent_ids), finish_timesteps
# #
# #     reward_funcs = get_cooperative_reward_functions(num_agents_per_episode, goal_rewards, player_collision_penalty=0,
# #                                                     movement_penalty=0)
# #     reward_funcs_player_collision = get_cooperative_reward_functions(num_agents_per_episode, goal_rewards,
# #                                                                      player_collision_penalty=-1, movement_penalty=0)
# #     env = GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=6,
# #                        seed=env_seed)
# #     start_time = time.time()
# #     gridworld_trajectory, finish_timesteps = play_episode_track_metrics(env, agents, create_new_world=False)
# #     computation_time = (time.time() - start_time) / num_agents_per_episode  # approximation
# #     attained_reward = np.sum(gridworld_trajectory.get_return()) / num_agents_per_episode
# #     max_reward = calculate_max_possible_reward()
# #     print('seed', env_seed, 'rew', attained_reward, 'max_rew', max_reward)
# #     assert attained_reward <= (max_reward + .000001)
# #     if max_reward == 0:
# #         max_reward_ratio = 1
# #     else:
# #         max_reward_ratio = attained_reward / max_reward
# #     env_player_collisions = GridworldEnv(num_agents_per_episode, reward_funcs_player_collision, num_goals, horizon,
# #                                          dim, max_num_walls=6, seed=env_seed)
# #     gridworld_trajectory = play_episode(env_player_collisions, agents, create_new_world=False)
# #     reward_with_collisions = np.sum(gridworld_trajectory.get_return()) / num_agents_per_episode
# #     num_player_collisions = 2 * (attained_reward - reward_with_collisions)  # approximation
# #     print('finish', finish_timesteps)
# #     return max_reward_ratio, num_player_collisions, finish_timesteps, computation_time
#
#
# # # stats about goals attained / max goals ratio, approx num player collisions, finish_timesteps, approx computation time
# # def get_destination_selector_stats(destination_selector_type, num_trials=20, local_state_filter='none', discount=1.,
# #                                    distance_penalty=0., optimize=False, sub_destination_selector_type='highest'):
# #     def get_destination_selector_name():
# #         destination_selector_name = destination_selector_type
# #         if destination_selector_type == 'highest':
# #             if local_state_filter != 'none':
# #                 destination_selector_name += '_' + local_state_filter
# #         elif destination_selector_type == 'distance':
# #             if discount < 1:
# #                 destination_selector_name += '_discount=' + str(discount)
# #             if distance_penalty < 0:
# #                 destination_selector_name += '_dist-penalty=' + str(distance_penalty)
# #             if local_state_filter != 'none':
# #                 destination_selector_name += '_' + local_state_filter
# #         elif destination_selector_type == 'colab_distance':
# #             if discount < 1:
# #                 destination_selector_name += '_discount=' + str(discount)
# #             if distance_penalty < 0:
# #                 destination_selector_name += '_dist-penalty=' + str(distance_penalty)
# #             destination_selector_name += '_optimize=' + ('T' if optimize else 'F')
# #         elif destination_selector_type == 'colab_state_partition':
# #             destination_selector_name += '_sub_ds=' + sub_destination_selector_type
# #             if discount < 1:
# #                 destination_selector_name += '_discount=' + str(discount)
# #             if distance_penalty < 0:
# #                 destination_selector_name += '_dist-penalty=' + str(distance_penalty)
# #         return destination_selector_name
# #
# #     destination_selector_name = get_destination_selector_name()
# #     np.random.seed(42)
# #
# #     finish_timesteps = []
# #     max_reward_ratios = []
# #     num_player_collisions = []
# #     computation_times = []
# #
# #     for env_seed in range(num_trials):
# #         reward_funcs, goal_rewards = get_random_coop_reward_funcs()
# #         if destination_selector_type == 'highest':
# #             agents, _, _ = get_highest_goal_preference_agents(reward_funcs, local_state_filter)
# #         elif destination_selector_type == 'distance':
# #             agents, _, _ = get_distance_based_goal_preference_agents(reward_funcs, local_state_filter, discount, distance_penalty)
# #         elif destination_selector_type == 'colab_distance':
# #             agents, _, _ = get_collaborative_distance_based_goal_preference_agents(reward_funcs, discount, distance_penalty,
# #                                                                              optimize)
# #         elif destination_selector_type == 'colab_state_partition':
# #             agents, _, _ = get_collaborative_state_partition_agents(reward_funcs, sub_destination_selector_type, discount,
# #                                                               distance_penalty)
# #         else:
# #             raise ValueError(destination_selector_type + ' is not a valid destination_selector_type')
# #
# #         max_reward_ratio, collisions, timesteps, computation_time = \
# #             get_destination_selector_metrics(env_seed, goal_rewards, agents)
# #         finish_timesteps.append(timesteps)
# #         max_reward_ratios.append(max_reward_ratio)
# #         num_player_collisions.append(collisions)
# #         computation_times.append(computation_time)
# #
# #     metric_labels = ['% of max reward', 'collisions', 'finish timesteps', 'time']
# #     metrics = [max_reward_ratios, num_player_collisions, finish_timesteps, computation_times]
# #     fig, axes = plt.subplots(2, 2)
# #     fig.suptitle(destination_selector_name)
# #     for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
# #         axes[np.unravel_index(i, (2, 2))].hist(metric)
# #         axes[np.unravel_index(i, (2, 2))].title.set_text(metric_label)
# #     plt.show()
# #
#
# def save_destination_selector_type_dissimilarity_data(num_trials=20):
#     """
#     Destination Selector Types:
#     Highest goal
#     Highest goal filter [square_x, dimaond_x, circle_x]
#     Closest goal
#     Closest goal filter [square_x, dimaond_x, circle_x]
#     Discount goal [.5, .9, .95, .99]
#     Discount goal filter [.5, .9, .95, .99] [square_x, dimaond_x, circle_x]
#     Distance penalty goal [.01, .05, .1, .2]
#     Distance penalty goal filter [.01, .05, .1, .2] [square_x, dimaond_x, circle_x]
#
#     Collaborative closest goal unoptimized
#     Collaborative closest goal optimized
#     Collaborative discount goal unoptimized [.5, .9, .95, .99]
#     Collaborative discount goal optimized [.5, .9, .95, .99]
#     Collaborative distance penalty goal unoptimized [.01, .05, .1]
#     Collaborative distance penalty goal optimized [.01, .05, .1]
#
#     Collaborative static grid state partition highest goal
#     Collaborative static grid state partition closest goal
#     Collaborative static grid state partition discount goal [.5, .9, .95, .99]
#     Collaborative static grid state partition distance penalty goal [.01, .05, .1]
#     """
#     def get_agents_srs(env, agents):
#         gridworld_traj = play_episode(env, agents, create_new_world=False)
#         agents_srs = []
#         for agent_idx in range(num_agents_per_episode):
#             agents_srs.append(gridworld_traj.get_successor_representations(0, agent_idx=agent_idx)[0])
#         return agents_srs
#
#     np.random.seed(42)
#
#     dataset_goal_rewards = [get_random_goal_rewards() for _ in range(num_trials)]
#
#     goal_ranker_types = ['highest', 'closest_distance', 'discount_distance', 'distance_penalty']
#     goal_ranker_type_args = [[None], [None], [.5, .9, .95, .99], [-.01, -.05, -.1]]
#     state_filter_types = ['none', 'static_circle', 'static_diamond', 'static_square', "colab_state_partition"]
#
#     static_shape_radiuses = [2, 3, 4, 6]
#
#     all_static_filter_types_and_args = [(None, None)]
#     for state_filter_type in ['static_circle', 'static_diamond', 'static_square']:
#         for radius in static_shape_radiuses:
#             all_static_filter_types_and_args.append((state_filter_type, (radius,)))
#
#     all_independent_goal_ranker_types_and_args = []
#     for goal_ranker_type, goal_ranker_args in zip(goal_ranker_types, goal_ranker_type_args):
#         for arg in goal_ranker_args:
#             goal_ranker_arg = None
#             if arg is not None:
#                 goal_ranker_arg = (arg, )
#             all_independent_goal_ranker_types_and_args.append((goal_ranker_type, goal_ranker_arg))
#
#     overall_agent_names = []
#     destination_selectors_comparison_matrix = None
#     for trial in range(num_trials):
#         trial_srs_comparisons = []
#         agent_names = []
#
#         goal_rewards = dataset_goal_rewards[trial]
#         reward_funcs = get_cooperative_reward_functions(num_agents_per_episode, goal_rewards)
#
#         env = GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, max_num_walls=6, seed=trial)
#
#         for state_partition_seed, (goal_ranker_type, goal_ranker_args) in enumerate(all_independent_goal_ranker_types_and_args):
#             for static_filter_type, filter_args in all_static_filter_types_and_args:
#                 agents, _, agent_name = create_agent(reward_funcs, goal_ranker_type, goal_ranker_args,
#                                                      static_filter_type, filter_args)
#                 trial_srs_comparisons.append(get_agents_srs(env, agents))
#                 agent_names.append(agent_name[0])
#             agents, _, agent_name = create_agent(reward_funcs, goal_ranker_type, goal_ranker_args, is_collaborative=True)
#             trial_srs_comparisons.append(get_agents_srs(env, agents))
#             agent_names.append(agent_name[0])
#
#             state_partition_args = (list(range(num_agents_per_episode)), 1 + num_goals, state_partition_seed)
#             agents, _, agent_name = create_agent(reward_funcs, goal_ranker_type, goal_ranker_args,
#                                                  state_filter_type="colab_state_partition",
#                                                  state_filter_args=state_partition_args,
#                                                  is_collaborative=True)
#             trial_srs_comparisons.append(get_agents_srs(env, agents))
#             agent_names.append(agent_name[0])
#
#         if trial == 0:
#             overall_agent_names += agent_names
#
#         if destination_selectors_comparison_matrix is None:
#             destination_selectors_comparison_matrix = np.zeros((len(overall_agent_names), len(overall_agent_names)))
#         for i, dest_selector_agent_srs_a in enumerate(trial_srs_comparisons):
#             for j, dest_selector_agent_srs_b in enumerate(trial_srs_comparisons):
#                 for agent_idx in range(num_agents_per_episode):
#                     print()
#                     score = srs_dissimilarity(dest_selector_agent_srs_a[agent_idx], dest_selector_agent_srs_b[agent_idx])
#                     destination_selectors_comparison_matrix[i][j] += score
#
#     destination_selectors_comparison_matrix /= num_trials
#
#     comparison_data = {'agent_names': overall_agent_names,
#                        'dissimilarity_matrix': destination_selectors_comparison_matrix}
#     with open(destination_selector_dissimilarity_filename, 'wb') as file:
#         pickle.dump(comparison_data, file)
#     print('Made dissimilarity matrix for (', len(overall_agent_names), ') agents:', overall_agent_names)
#
#
# def load_destination_selector_dissimilarity_data() -> Tuple[list, np.array]:
#     with open(destination_selector_dissimilarity_filename, 'rb') as file:
#         data = pickle.load(file)
#     return data['agent_names'], data['dissimilarity_matrix']
#
#
# def cluster_destination_selector_types(plot=False):
#     from scipy.cluster.hierarchy import dendrogram
#     def plot_dendrogram(model, **kwargs):
#         counts = np.zeros(model.children_.shape[0])
#         n_samples = len(model.labels_)
#         for i, merge in enumerate(model.children_):
#             current_count = 0
#             for child_idx in merge:
#                 if child_idx < n_samples:
#                     current_count += 1  # leaf node
#                 else:
#                     current_count += counts[child_idx - n_samples]
#             counts[i] = current_count
#
#         linkage_matrix = np.column_stack([model.children_, model.distances_,
#                                           counts]).astype(float)
#         dendrogram(linkage_matrix, **kwargs)
#
#     def get_ensembled_cluster_idx(dest_selector_idx):
#         for cluster_idx, cluster in enumerate(ensembled_cluster):
#             if dest_selector_idx in cluster:
#                 return cluster_idx
#
#     agent_names, dissimilarity_matrix = load_destination_selector_dissimilarity_data()
#     if plot:
#         model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
#                                         compute_full_tree=True, distance_threshold=0)
#         model = model.fit(dissimilarity_matrix)
#         plt.title('Hierarchical Clustering Dendrogram')
#         # plot the top three levels of the dendrogram
#         plot_dendrogram(model, truncate_mode=None, p=3)
#         plt.xlabel("Number of points in node (or index of point if no parenthesis).")
#         plt.show()
#     else:
#         cluster_vote = np.zeros((len(agent_names), len(agent_names)))
#         models = [
#             AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
#                                     compute_full_tree=True, distance_threshold=.02),
#             AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
#                                     compute_full_tree=True, distance_threshold=.04),
#             DBSCAN(1e-2, min_samples=2, metric='precomputed')
#         ]
#         for model in models:
#             model = model.fit(dissimilarity_matrix)
#             print(model, '-> num clusters:', np.max(model.labels_))
#             clustered_dest_selectors = [[] for _ in range(max(model.labels_) + 1)]
#             for dest_selector_idx, label in enumerate(model.labels_):
#                 clustered_dest_selectors[label].append((agent_names[dest_selector_idx], dest_selector_idx))
#             for cluster in clustered_dest_selectors:
#                 for _, dest_selector_idx_a in cluster:
#                     for _, dest_selector_idx_b in cluster:
#                         cluster_vote[dest_selector_idx_a][dest_selector_idx_b] += 1
#
#         ensembled_cluster = [{i} for i in range(len(agent_names))]
#         for i, j in zip(*np.where(cluster_vote > 1)):
#             if j <= i:
#                 continue
#             i_cluster_idx = get_ensembled_cluster_idx(i)
#             j_cluster_idx = get_ensembled_cluster_idx(j)
#             if i_cluster_idx != j_cluster_idx:
#                 ensembled_cluster[i_cluster_idx] = ensembled_cluster[i_cluster_idx].union(ensembled_cluster[j_cluster_idx])
#                 del ensembled_cluster[j_cluster_idx]
#
#         for cluster in ensembled_cluster:
#             cluster_names = []
#             for i in cluster:
#                 cluster_names.append(agent_names[i])
#             print(cluster_names)
#         print(len(ensembled_cluster))
#
#
def reassign_agent_indices(agents: List[PathfindingAgent]):
    collaborative_state_partition_agent_indices = []
    for agent_idx, agent in enumerate(agents):
        agent.destination_selector.agent_idx = agent_idx
        if agent.destination_selector.state_filters is not None:
            for state_filter in agent.destination_selector.state_filters:
                if isinstance(state_filter, CollaborativeStatePartitionFilter):
                    collaborative_state_partition_agent_indices.append(agent_idx)

    for agent_idx in collaborative_state_partition_agent_indices:
        for state_filter in agents[agent_idx].destination_selector.state_filters:
            if isinstance(state_filter, CollaborativeStatePartitionFilter):
                state_filter.agent_indices = collaborative_state_partition_agent_indices
                break
#
#
# def collect_3_group_1_group_data():
#     """
#     Left hand side is the group of 3, right hand side is group of 1
#     colab-state-partition_sub=[highest], highest
#     colab-state-partition_sub=[highest], distance
#     colab-state-partition_sub=[highest], distance_discount=.99
#     colab-state-partition_sub=[highest], distance_distance-penalty=-0.05
#     colab-state-partition_sub=[distance], highest
#     colab-state-partition_sub=[distance], distance
#     colab-state-partition_sub=[distance], distance_discount=.99
#     colab-state-partition_sub=[distance], distance_distance-penalty=-0.05
#     colab-distance_optimize=T, highest
#     colab-distance_optimize=T, distance
#     colab-distance_optimize=T, distance_discount=.99
#     colab-distance_optimize=T, distance_distance-penalty=-0.05
#     colab-distance_optimize=F, highest
#     colab-distance_optimize=F, distance
#     colab-distance_optimize=F, distance_discount=.99
#     colab-distance_optimize=F, distance_distance-penalty=-0.05
#     """
#     def get_colab_state_partition_group(reward_funcs, sub_destination_selector_type, agent_func, agent_kwargs):
#         group_agents, group_ids, group_names = get_collaborative_state_partition_agents(
#             reward_funcs[:num_agents_per_episode - 1], sub_destination_selector_type)
#         ungrouped_agents, ungrouped_ids, ungrouped_names = agent_func(reward_funcs[-1:], **agent_kwargs)
#         return group_agents + ungrouped_agents, group_ids + ungrouped_ids, group_names + ungrouped_names
#
#     def get_colab_distance_group(reward_funcs, optimize, agent_func, agent_kwargs):
#         group_agents, group_ids, group_names = get_collaborative_distance_based_goal_preference_agents(
#             reward_funcs[:num_agents_per_episode - 1], optimize)
#         ungrouped_agents, ungrouped_ids, ungrouped_names = agent_func(reward_funcs[-1:], **agent_kwargs)
#         return group_agents + ungrouped_agents, group_ids + ungrouped_ids, group_names + ungrouped_names
#
#     ungrouped_agent_funcs = [
#         (get_highest_goal_preference_agents, {}),
#         (get_distance_based_goal_preference_agents, {}),
#         (get_distance_based_goal_preference_agents, {'discount': .99}),
#         (get_distance_based_goal_preference_agents, {'distance_penalty': -0.05}),
#     ]
#
#     def create_agents_and_record_history(agent_group_func, group_func_param, agent_func, agent_kwargs):
#         reward_funcs, goals = get_random_coop_reward_funcs()
#         env = get_random_env(reward_funcs)
#         agents, agent_ids, agent_names = agent_group_func(reward_funcs, group_func_param, agent_func, agent_kwargs)
#         reassign_agent_indices(agents)
#         agent_histories.extend([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)])
#         for agent_id, agent_name in zip(agent_ids, agent_names):
#             all_agent_ids.append(agent_id)
#             agent_behaviors[agent_id] = agent_name
#
#     agent_histories = []
#     all_agent_ids = []
#     agent_behaviors = dict()
#     for agent_func, agent_kwargs in ungrouped_agent_funcs:
#         for sub_destination_selector_type in ['highest', 'distance']:
#             create_agents_and_record_history(get_colab_state_partition_group, sub_destination_selector_type,
#                                              agent_func, agent_kwargs)
#         for optimize in [True, False]:
#             create_agents_and_record_history(get_colab_distance_group, optimize, agent_func, agent_kwargs)
#     return agent_histories, all_agent_ids, agent_behaviors
#
#
# def collect_2_group_2_group_data():
#     """
#     Combination of following agents:
#         highest
#         distance
#         distance_discount=.99
#         distance_distance-penalty=-0.05
#         colab-distance_optimize=T
#         colab-distance_optimize=F
#         colab-state-partition_sub=[highest]
#         colab-state-partition_sub=[distance]
#     highest, distance,
#     highest, distance_discount=.99,
#     highest, distance_distance-penalty=-0.05,
#     highest, colab-distance_optimize=T,
#     highest, colab-distance_optimize=F,
#     highest, colab-state-partition_sub=[highest]
#     highest, colab-state-partition_sub=[distance]
#     distance, distance_discount=.99,
#     distance, distance_distance-penalty=-0.05,
#     distance, colab-distance_optimize=T,
#     distance, colab-distance_optimize=F,
#     distance, colab-state-partition_sub=[highest]
#     distance, colab-state-partition_sub=[distance]
#     distance_discount=.99, distance_distance-penalty=-0.05,
#     distance_discount=.99, colab-distance_optimize=T,
#     distance_discount=.99, colab-distance_optimize=F,
#     distance_discount=.99, colab-state-partition_sub=[highest]
#     distance_discount=.99, colab-state-partition_sub=[distance]
#     distance_distance-penalty=-0.05, colab-distance_optimize=T,
#     distance_distance-penalty=-0.05, colab-distance_optimize=F,
#     distance_distance-penalty=-0.05, colab-state-partition_sub=[highest]
#     distance_distance-penalty=-0.05, colab-state-partition_sub=[distance]
#     colab-distance_optimize=T, colab-distance_optimize=F,
#     colab-distance_optimize=T, colab-state-partition_sub=[highest]
#     colab-distance_optimize=T, colab-state-partition_sub=[distance]
#     colab-distance_optimize=F, colab-state-partition_sub=[highest]
#     colab-distance_optimize=F, colab-state-partition_sub=[distance]
#     colab-state-partition_sub=[highest], colab-state-partition_sub=[distance]
#     """
#     agent_funcs = [
#         (get_highest_goal_preference_agents, {}),
#         (get_distance_based_goal_preference_agents, {}),
#         (get_distance_based_goal_preference_agents, {'discount': .99}),
#         (get_distance_based_goal_preference_agents, {'distance_penalty': -0.05}),
#         (get_collaborative_distance_based_goal_preference_agents, {'optimize': True}),
#         (get_collaborative_distance_based_goal_preference_agents, {'optimize': False}),
#         (get_collaborative_state_partition_agents, {'sub_destination_selector_type': 'highest'}),
#         (get_collaborative_state_partition_agents, {'sub_destination_selector_type': 'distance'})
#     ]
#
#     def get_group(agent_func, reward_funcs, kwargs):
#         group_agents, group_ids, group_names = agent_func(reward_funcs, **kwargs)
#         agents.extend(group_agents)
#         for agent_id, agent_name in zip(group_ids, group_names):
#             agent_ids.append(agent_id)
#             agent_behaviors[agent_id] = agent_name
#
#     agent_histories = []
#     agent_ids = []
#     agent_behaviors = dict()
#     for two_groups in list(combinations(agent_funcs, 2)):
#         agents = []
#         reward_funcs, goal_rewards = get_random_coop_reward_funcs()
#         agent_func, agent_kwargs = two_groups[0]
#         get_group(agent_func, reward_funcs[:num_agents_per_episode // 2], agent_kwargs)
#         agent_func, agent_kwargs = two_groups[1]
#         get_group(agent_func, reward_funcs[num_agents_per_episode // 2:], agent_kwargs)
#         reassign_agent_indices(agents)
#         env = get_random_env(reward_funcs)
#         agent_histories.extend([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)])
#     return agent_histories, agent_ids, agent_behaviors
#
#
# def collect_no_group_data(num_agents):
#     """
#     Random sample without replacement from following agents:
#         'highest'
#         'highest_square2'
#         'highest_square3'
#         'highest_square4'
#         'highest_square6'
#         'highest_circle4'
#         'highest_circle6'
#         'highest_diamond4'
#         'highest_diamond6'
#         'distance'
#         'distance_square2'
#         'distance_square3'
#         'distance_square4'
#         'distance_square6'
#         'distance_circle4'
#         'distance_circle6'
#         'distance_diamond4'
#         'distance_diamond6'
#         'distance_discount=0.99_square3'
#         'distance_discount=0.99_square4'
#         'distance_discount=0.99_square6'
#         'distance_discount=0.99_circle4'
#         'distance_discount=0.99_circle6'
#         'distance_discount=0.5'
#         'distance_discount=0.9'
#         'distance_discount=0.95'
#         'distance_discount=0.99'
#         'distance_distance-penalty-0.01'
#         'distance_distance-penalty-0.05'
#         'distance_distance-penalty-0.1'
#     """
#     def get_highest_agents():
#         return [(get_highest_goal_preference_agents, {'local_state_filter': shape}) for shape in shapes]
#
#     def get_distance_agents():
#         nearest_agents = [(get_distance_based_goal_preference_agents, {'local_state_filter': shape}) for shape in shapes]
#         discount_shapes = ['square_3', 'square_4', 'square_6', 'circle_4', 'circle_6']
#         discount_shape_agents = [(get_distance_based_goal_preference_agents, {'discount': .99, 'local_state_filter': shape}) for shape in discount_shapes]
#         discount_agents = [(get_distance_based_goal_preference_agents, {'discount': discount}) for discount in discounts]
#         penalty_agents = [(get_distance_based_goal_preference_agents, {'distance_penalty': penalty}) for penalty in distance_penalties]
#         return nearest_agents + discount_shape_agents + discount_agents + penalty_agents
#
#     shapes = ['none', 'square_2', 'square_3', 'square_4', 'square_6', 'circle_4', 'circle_6', 'diamond_4',
#               'diamond_6']
#     discounts = [.5, .9, .95, .99]
#     distance_penalties = [-0.01, -0.05, -0.1]
#     all_agents = get_highest_agents() + get_distance_agents()
#
#     agent_histories = []
#     agent_ids = []
#     agent_behaviors = dict()
#     for _ in range(num_agents):
#         agent_func_indices = np.random.choice(len(all_agents), size=min_num_episodes_per_agent, replace=False)
#         sampled_agents = [all_agents[agent_func_idx] for agent_func_idx in agent_func_indices]
#         reward_funcs, goal_rewards = get_random_coop_reward_funcs()
#         agents = []
#         for i, (agent_create_func, kwargs) in enumerate(sampled_agents):
#             group_agents, group_ids, agent_names = agent_create_func(reward_funcs[i:i+1], **kwargs)
#             agents += group_agents
#             for agent_id, agent_name in zip(group_ids, agent_names):
#                 agent_ids.append(agent_id)
#                 agent_behaviors[agent_id] = agent_name
#         reassign_agent_indices(agents)
#         env = get_random_env(reward_funcs)
#         agent_histories.extend([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)])
#     return agent_histories, agent_ids, agent_behaviors
#
#
# def collect_pure_group_data():
#     """
#     ['highest']
#     ['highest_square2']
#     ['highest_square3']
#     ['highest_square4']
#     ['highest_square6']
#     ['highest_circle4']
#     ['highest_circle6']
#     ['highest_diamond4']
#     ['highest_diamond6']
#     ['distance']
#     ['distance_square2']
#     ['distance_square3']
#     ['distance_square4']
#     ['distance_square6']
#     ['distance_circle4']
#     ['distance_circle6']
#     ['distance_diamond4']
#     ['distance_diamond6']
#     ['distance_discount=0.99_square3']
#     ['distance_discount=0.99_square4']
#     ['distance_discount=0.99_square6']
#     ['distance_discount=0.99_circle4']
#     ['distance_discount=0.99_circle6']
#     ['distance_discount=0.5']
#     ['distance_discount=0.9']
#     ['distance_discount=0.95']
#     ['distance_discount=0.99']
#     ['distance_distance-penalty-0.01']
#     ['distance_distance-penalty-0.05']
#     ['distance_distance-penalty-0.1']
#     ['colab-distance_optimize=T']
#     ['colab-distance_optimize=F']
#     ['colab-distance_discount=0.5_optimize=T']
#     ['colab-distance_discount=0.5_optimize=F']
#     ['colab-distance_discount=0.9_optimize=T']
#     ['colab-distance_discount=0.9_optimize=F']
#     ['colab-distance_discount=0.95_optimize=T']
#     ['colab-distance_discount=0.95_optimize=F']
#     ['colab-distance_discount=0.99_optimize=T']
#     ['colab-distance_discount=0.99_optimize=F']
#     ['colab-distance_distance-penalty-0.05_optimize=T']
#     ['colab-distance_distance-penalty-0.05_optimize=F']
#     ['colab-distance_distance-penalty-0.1_optimize=T']
#     ['colab-distance_distance-penalty-0.1_optimize=F']
#     ['colab-distance_distance-penalty-0.01_optimize=T']
#     ['colab-distance_distance-penalty-0.01_optimize=F']
#     ['colab-state-partition_sub=[highest]']
#     ['colab-state-partition_sub=[distance]']
#     """
#     def create_agents_and_record_history(agent_func, kwargs):
#         reward_funcs, goals = get_random_coop_reward_funcs()
#         env = get_random_env(reward_funcs)
#         agents, agent_ids, agent_names = agent_func(reward_funcs, **kwargs)
#         agent_histories.extend([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)])
#         goal_rewards.append(goals)
#         for agent_id, agent_name in zip(agent_ids, agent_names):
#             all_agent_ids.append(agent_id)
#             agent_behaviors[agent_id] = agent_name
#
#     distance_penalties = [-.01, -.05, -.1]
#     discounts = [.5, .9, .95, .99]
#     optimizations = [True, False]
#     sub_destination_selector_types = ['highest', 'distance']
#     filters = ['none', 'square_2', 'square_3', 'square_4', 'square_6', 'circle_4', 'circle_6', 'diamond_4', 'diamond_6']
#
#     agent_histories, goal_rewards, all_agent_ids = [], [], []
#     agent_behaviors = dict()
#
#     for dest_selector_type in ['highest', 'distance', 'discount']:
#         for filter in filters:
#             kwargs = {'local_state_filter': filter}
#             if dest_selector_type == 'highest':
#                 create_agents_and_record_history(get_highest_goal_preference_agents, kwargs)
#             elif dest_selector_type == 'distance':
#                 create_agents_and_record_history(get_distance_based_goal_preference_agents, kwargs)
#             else:
#                 kwargs['discount'] = .99
#                 create_agents_and_record_history(get_distance_based_goal_preference_agents, kwargs)
#
#     for discount in discounts:
#         kwargs = {'discount': discount}
#         create_agents_and_record_history(get_distance_based_goal_preference_agents, kwargs)
#
#     for distance_penalty in distance_penalties:
#         kwargs = {'distance_penalty': distance_penalty}
#         create_agents_and_record_history(get_distance_based_goal_preference_agents, kwargs)
#
#     for optimize in optimizations:
#         for discount in discounts:
#             kwargs = {'optimize': optimize, 'discount': discount}
#             create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#         for distance_penalty in distance_penalties:
#             kwargs = {'optimize': optimize, 'distance_penalty': distance_penalty}
#             create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#         kwargs = {'optimize': optimize}
#         create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#
#     for sub_destination_selector_type in sub_destination_selector_types:
#         kwargs = {'sub_destination_selector_type': sub_destination_selector_type}
#         create_agents_and_record_history(get_collaborative_state_partition_agents, kwargs)
#
#     return agent_histories, all_agent_ids, agent_behaviors
#
#
# def create_dataset(seed):
#     """
#     ['highest']
#     ['highest_square2']
#     ['highest_square3']
#     ['highest_square4']
#     ['highest_square6']
#     ['highest_circle4']
#     ['highest_circle6']
#     ['highest_diamond4']
#     ['highest_diamond6']
#     ['distance']
#     ['distance_square2']
#     ['distance_square3']
#     ['distance_square4']
#     ['distance_square6']
#     ['distance_circle4']
#     ['distance_circle6']
#     ['distance_diamond4']
#     ['distance_diamond6']
#     ['distance_discount=0.99_square3']
#     ['distance_discount=0.99_square4']
#     ['distance_discount=0.99_square6']
#     ['distance_discount=0.99_circle4']
#     ['distance_discount=0.99_circle6']
#     ['distance_discount=0.5']
#     ['distance_discount=0.9']
#     ['distance_discount=0.95']
#     ['distance_discount=0.99']
#     ['distance_distance-penalty-0.01']
#     ['distance_distance-penalty-0.05']
#     ['distance_distance-penalty-0.1']
#     ['colab-distance_optimize=T']
#     ['colab-distance_optimize=F']
#     ['colab-distance_discount=0.5_optimize=T']
#     ['colab-distance_discount=0.5_optimize=F']
#     ['colab-distance_discount=0.9_optimize=T']
#     ['colab-distance_discount=0.9_optimize=F']
#     ['colab-distance_discount=0.95_optimize=T']
#     ['colab-distance_discount=0.95_optimize=F']
#     ['colab-distance_discount=0.99_optimize=T']
#     ['colab-distance_discount=0.99_optimize=F']
#     ['colab-distance_distance-penalty-0.05_optimize=T']
#     ['colab-distance_distance-penalty-0.05_optimize=F']
#     ['colab-distance_distance-penalty-0.1_optimize=T']
#     ['colab-distance_distance-penalty-0.1_optimize=F']
#     ['colab-distance_distance-penalty-0.01_optimize=T']
#     ['colab-distance_distance-penalty-0.01_optimize=F']
#     ['colab-state-partition_sub=[highest]']
#     ['colab-state-partition_sub=[distance]']
#     """
#     def collect_group_data(group_data_func):
#         history, ids, behaviors = group_data_func()
#         agent_histories.extend(history)
#         agent_ids.extend(ids)
#         agent_behaviors.update(behaviors)
#
#     np.random.seed(seed)
#     agent_histories, agent_ids, agent_behaviors = collect_no_group_data(num_agents=40)
#     collect_group_data(collect_pure_group_data)
#     collect_group_data(collect_2_group_2_group_data)
#     collect_group_data(collect_3_group_1_group_data)
#     return BehaviorGridworldDataset(agent_histories, agent_behaviors)
#
#
# def create_collaborative_dataset(seed):
#     """
#     ['colab-distance_optimize=T']
#     ['colab-distance_optimize=F']
#     ['colab-distance_discount=0.5_optimize=T']
#     ['colab-distance_discount=0.5_optimize=F']
#     ['colab-distance_discount=0.9_optimize=T']
#     ['colab-distance_discount=0.9_optimize=F']
#     ['colab-distance_discount=0.95_optimize=T']
#     ['colab-distance_discount=0.95_optimize=F']
#     ['colab-distance_discount=0.99_optimize=T']
#     ['colab-distance_discount=0.99_optimize=F']
#     ['colab-distance_distance-penalty-0.05_optimize=T']
#     ['colab-distance_distance-penalty-0.05_optimize=F']
#     ['colab-distance_distance-penalty-0.1_optimize=T']
#     ['colab-distance_distance-penalty-0.1_optimize=F']
#     ['colab-distance_distance-penalty-0.01_optimize=T']
#     ['colab-distance_distance-penalty-0.01_optimize=F']
#     ['colab-state-partition_sub=[highest]']
#     ['colab-state-partition_sub=[distance]']
#     """
#
#     def create_agents_and_record_history(agent_func, kwargs):
#         reward_funcs, goals = get_random_coop_reward_funcs()
#         env = get_random_env(reward_funcs)
#         agents, agent_ids, agent_names = agent_func(reward_funcs, **kwargs)
#         agent_histories.extend([play_episode(env, agents) for _ in range(min_num_episodes_per_agent)])
#         goal_rewards.append(goals)
#         for agent_id, agent_name in zip(agent_ids, agent_names):
#             all_agent_ids.append(agent_id)
#             agent_behaviors[agent_id] = agent_name
#
#     np.random.seed(seed)
#     distance_penalties = [-.01, -.05, -.1]
#     discounts = [.5, .9, .95, .99]
#     optimizations = [True, False]
#     sub_destination_selector_types = ['highest', 'distance']
#
#     agent_histories, goal_rewards, all_agent_ids = [], [], []
#     agent_behaviors = dict()
#
#     for optimize in optimizations:
#         for discount in discounts:
#             kwargs = {'optimize': optimize, 'discount': discount}
#             create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#         for distance_penalty in distance_penalties:
#             kwargs = {'optimize': optimize, 'distance_penalty': distance_penalty}
#             create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#         kwargs = {'optimize': optimize}
#         create_agents_and_record_history(get_collaborative_distance_based_goal_preference_agents, kwargs)
#
#     for sub_destination_selector_type in sub_destination_selector_types:
#         kwargs = {'sub_destination_selector_type': sub_destination_selector_type}
#         create_agents_and_record_history(get_collaborative_state_partition_agents, kwargs)
#
#     return BehaviorGridworldDataset(agent_histories, agent_behaviors)
#
#
# def get_dataset_filename(is_collaborative_dataset: bool, is_train_dataset: bool, iteration: int):
#     dataset_filename = dataset_base_name
#     if is_collaborative_dataset:
#         dataset_filename += "_collaborative"
#     dataset_type = '_test'
#     if is_train_dataset:
#         dataset_type = '_train'
#     return dataset_filename + dataset_type + "-" + str(iteration)
#
#
# def get_all_existing_dataset_filepaths(is_collaborative_dataset: bool, is_train_dataset: bool):
#     filename = get_dataset_filename(is_collaborative_dataset, is_train_dataset, 0)
#     filename = filename.split('-')[0]
#     return glob.glob(dataset_dirpath + filename + '*' + dataset_extension)
#
#
# def get_dataset_filepath(is_collaborative_dataset: bool, is_train_dataset: bool, iteration: int):
#     return dataset_dirpath + get_dataset_filename(is_collaborative_dataset, is_train_dataset, iteration) + dataset_extension
#
#
# def save_dataset(dataset: BehaviorGridworldDataset, is_collaborative_dataset: bool, is_train_dataset: bool, iteration: int):
#     base_save_dataset(dataset, dataset_dirpath, get_dataset_filename(is_collaborative_dataset, is_train_dataset, iteration))
#
#
# def create_and_save_all_datasets(dataset_type='', seed=42, n_recursive_splits=2):
#     if dataset_type not in dataset_types:
#         raise ValueError("%s is not a valid dataset type. Possible dataset types %s" % (dataset_type, dataset_types))
#     is_small_dataset = False
#     if 'small' in dataset_type:
#         is_small_dataset = True
#     collaborative_only = False
#     if 'collaborative' in dataset_type:
#         collaborative_only = True
#     for is_training_dataset in [True, False]:
#         num_current_dataset_agents = 0
#         dataset_iteration = 0
#         while num_current_dataset_agents < (num_train_dataset_agents if is_training_dataset else num_test_dataset_agents):
#             create_dataset_fn = create_collaborative_dataset if collaborative_only else create_dataset
#             dataset = create_dataset_fn(seed)
#             num_current_dataset_agents += len(dataset)
#             dataset_splits = recursively_split_dataset(dataset, n_recursive_splits)
#             for dataset_split in dataset_splits:
#                 save_dataset(dataset_split, collaborative_only, is_training_dataset, dataset_iteration)
#                 dataset_iteration += 1
#                 if is_small_dataset:
#                     break
#             del dataset  # To save space
#             del dataset_splits[:]  # To save space
#             seed += 1
#
#
# def load_dataset(is_collaborative_dataset, is_train_dataset=True, iteration=0) -> BehaviorGridworldDataset:
#     return base_load_dataset(get_dataset_filepath(is_collaborative_dataset, is_train_dataset, iteration))
#
#
# def get_datamodule(is_collaborative_dataset, is_small_dataset, *args, **kwargs):
#     train_dataset_filepaths = get_all_existing_dataset_filepaths(is_collaborative_dataset, is_train_dataset=True)
#     test_dataset_filepaths = get_all_existing_dataset_filepaths(is_collaborative_dataset, is_train_dataset=False)
#     if is_small_dataset:
#         train_dataset_filepaths = np.sort(train_dataset_filepaths)[:4]  # [np.sort(train_dataset_filepaths)[0]]
#         test_dataset_filepaths = np.sort(test_dataset_filepaths)[:4]  # [np.sort(test_dataset_filepaths)[0]]
#     return MultiFileGridworldDataModule(train_dataset_filepaths, test_dataset_filepaths, *args, **kwargs)
#
#
# if __name__=='__main__':
#     cluster_destination_selector_types(plot=False)