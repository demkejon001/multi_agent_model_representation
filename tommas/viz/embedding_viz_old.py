import matplotlib.axes
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

from tommas.agent_modellers.tommas import TOMMAS
from tommas.data.gridworld_dataset import GridworldDataset, collate_mgdataset, GridworldDatasetBatchSampler, \
    collate_current_traj_batch_get
from tommas.env.gridworld_env import GridworldEnv
from tommas.agents.reward_function import get_zeros_reward_functions
from tommas.helper_code.metrics import action_acc, goal_acc, successor_representation_acc


embeddings_filepath = "data/viz/embeddings.txt"


def visualize_agent_embeddings(agent_modeller: TOMMAS, dataset: GridworldDataset, n_past, empty_current_traj, agent_idx_to_behavior, split_behaviors=False, additional_dataset_kwargs=None):
    def embedding_is_graphable(network_zero_embedding_func):
        zero_embedding = network_zero_embedding_func()
        if zero_embedding is not None:
            if zero_embedding.dim() == 2 and zero_embedding.shape[1] == 2:
                return True
        return False

    def get_filtered_graphable_embeddings():
        new_embedding_lists = []
        new_embedding_types = []
        for embedding_list, embedding_type, is_graphable in zip(embedding_lists, embedding_types, graphable_embeddings):
            if is_graphable:
                new_embedding_lists.append(embedding_list)
                new_embedding_types.append(embedding_type)
        return new_embedding_lists, new_embedding_types

    def get_behavior_to_colors():
        behavior_to_colors = dict.fromkeys(unique_behaviors, None)
        cm = plt.get_cmap('gist_rainbow')
        num_behaviors = len(unique_behaviors)
        for i, behavior in enumerate(unique_behaviors):
            behavior_to_colors[behavior] = cm(1. * i / num_behaviors)
        return behavior_to_colors

    def get_behavior_to_agent_indices():
        behavior_to_agent_indices = dict()
        for agent_idx, behavior in agent_idx_to_behavior.items():
            if behavior not in behavior_to_agent_indices:
                behavior_to_agent_indices[behavior] = []
            behavior_to_agent_indices[behavior].append(agent_idx)
        return behavior_to_agent_indices


    batch_size = 10
    sampler = SequentialSampler(dataset)
    batch_sampler = GridworldDatasetBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False,
                                                 min_num_episodes=dataset.min_num_episodes_per_agent, n_past=n_past,
                                                 empty_current_traj_probability=1 if empty_current_traj else 1/35,
                                                 additional_dataset_kwargs=additional_dataset_kwargs)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_mgdataset, pin_memory=False,
                            num_workers=4)

    embedding_types = ["IA Char", "IA Mental", "JA Char", "JA Mental"]
    embedding_lists = [[], [], [], []]

    graph_ia_char = embedding_is_graphable(agent_modeller.get_ia_char_zero_embedding)
    graph_ia_mental = embedding_is_graphable(agent_modeller.get_ia_mental_zero_embedding)
    graph_ja_char = embedding_is_graphable(agent_modeller.get_ja_char_zero_embedding)
    graph_ja_mental = embedding_is_graphable(agent_modeller.get_ja_mental_zero_embedding)

    graphable_embeddings = [graph_ia_char, graph_ia_mental, graph_ja_char, graph_ja_mental]

    for agent_idx, batch in enumerate(dataloader):
        past_traj, current_traj, state, action, goal_consumption, srs = batch
        _, _, _, embeddings = agent_modeller(past_traj, current_traj, state, return_embeddings=True)
        for embedding, embedding_list, is_graphable in zip(embeddings, embedding_lists, graphable_embeddings):
            if is_graphable:
                embedding_list += embedding.squeeze(-1).squeeze(-1).tolist()

    unique_behaviors = set(agent_idx_to_behavior.values())
    num_behaviors = len(unique_behaviors)

    embedding_lists, embedding_types = get_filtered_graphable_embeddings()
    num_embedding_types = len(embedding_types)
    num_embedding_row = (num_behaviors if split_behaviors else 1)
    fig, axes = plt.subplots(num_embedding_row, num_embedding_types, sharex="col", sharey="col")

    behavior_to_agent_indices = get_behavior_to_agent_indices()
    behavior_to_colors = get_behavior_to_colors()

    handles = []
    labels = []
    for embedding_type_idx, embedding_type in enumerate(embedding_types):
        embeddings = np.array(embedding_lists[embedding_type_idx])
        for behavior_idx, behavior in enumerate(unique_behaviors):
            if num_embedding_types == num_embedding_row == 1:
                ax = axes
            elif num_embedding_types == 1:
                ax = axes[behavior_idx]
            elif num_embedding_row == 1:
                ax = axes[embedding_type_idx]
            else:
                ax = axes[behavior_idx, embedding_type_idx]
            agent_indices = behavior_to_agent_indices[behavior]
            scatter_element = ax.scatter(embeddings[agent_indices, 0], embeddings[agent_indices, 1], c=np.array([behavior_to_colors[behavior]]), label=behavior)
            if behavior_idx == 0:
                ax.set_title(embedding_type)
            if embedding_type_idx == 0:
                handles.append(scatter_element)
                labels.append(behavior)

    fig.legend(handles, labels)
    plt.show()


def srs_heatmap(srs: np.array):
    plt.imshow(srs, cmap='hot', interpolation='nearest')
    plt.show()


def predict_with_no_embeddings(agent_modeller: TOMMAS, dataset: GridworldDataset):
    def model_different_agent(agent_idx):
        agent_offset = 1 + num_goals
        state[[agent_offset, agent_idx + agent_offset]] = state[[agent_idx + agent_offset, agent_offset]]

    def is_occupied(position):
        if len(np.where(state[:, position[0], position[1]] == 1)[0]) > 0:
            return True
        return False

    def shift_agent_position():
        def shift_position():
            if agent_position[1] < (world_dim[1] - 1):
                agent_position[1] += 1
            elif agent_position[0] < (world_dim[0] - 1):
                agent_position[1] = 1
                agent_position[0] += 1
            else:
                agent_position[0] = 1
                agent_position[1] = 1
        agent_offset = 1 + num_goals
        agent_position = np.array(np.where(state[agent_offset] == 1)).flatten()
        print('agent pos', agent_position)
        # agent_position = [agent_position[0][0], agent_offset[1][0]]
        state[(agent_offset, *agent_position)] = 0
        shift_position()
        while is_occupied(agent_position):
            shift_position()
        state[(agent_offset, *agent_position)] = 1




    gridworld_traj = dataset.gridworld_trajectories[0]
    world_dim = gridworld_traj.dim
    num_agents = gridworld_traj.num_agents
    num_goals = gridworld_traj.num_goals

    env = GridworldEnv(num_agents, get_zeros_reward_functions(num_agents, num_goals), num_goals, 30, world_dim, max_num_walls=0)#, seed=42)
    state = env.reset()
    initial_state = np.copy(state)

    print(env.world)
    for idx in range(0, num_agents):
        state = np.copy(initial_state)
        print('MODDELLING AGENT', idx)
        model_different_agent(idx)
        for i in range(0, 20):
            shift_agent_position()
            actions, goals, srs = agent_modeller(None, None, torch.from_numpy(state).unsqueeze(0).float())
            policy = torch.softmax(actions, dim=1)
            goals = torch.sigmoid(goals)
            print('policy', policy)
            print('goals', goals)
            srs_heatmap(srs[0, 2])

    # print(env.world)
    # for i in range(0, num_agents):
    #     model_different_agent(i)
    #     actions, goals, srs = agent_modeller(None, None, torch.from_numpy(state).unsqueeze(0).float())
    #     policy = torch.softmax(actions, dim=1)
    #     goals = torch.sigmoid(goals)
    #     print('policy', policy)
    #     print('goals', goals)
    #     srs_heatmap(srs[0, 0])
        # print('srs', srs)


def interactive_embedding_modeller_prediction(agent_modeller: TOMMAS, dataset: GridworldDataset, n_past, empty_current_traj, additional_dataset_kwargs=None):
    def save_original_embeddings():
        # with open('data.json', 'w', encoding='utf-8') as f:
        #     json.dump(original_embeddings, f, ensure_ascii=False, indent=4)
        with open(embeddings_filepath, 'w') as f:
            json.dump(original_embeddings, f)

    def load_embeddings():
        with open(embeddings_filepath, 'r') as f:
            return json.load(f)

    def convert_embeddings_to_numpy():
        for i in range(len(embeddings)):
            embeddings[i] = np.array(embeddings)

    def user_gave_new_embedding():
        answer = input("Have you changed the embeddings in " + embeddings_filepath + "?   [Y/n]:")
        if answer.lower() in ['n', 'no']:
            return False
        return True

    batch_sampler = GridworldDatasetBatchSampler(sampler=RandomSampler(dataset), batch_size=1, drop_last=False,
                                                 min_num_episodes=dataset.min_num_episodes_per_agent, n_past=n_past,
                                                 empty_current_traj_probability=1 if empty_current_traj else 1 / 35,
                                                 additional_dataset_kwargs=additional_dataset_kwargs)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_mgdataset, pin_memory=False,
                            num_workers=4)

    past_traj, current_traj, state, action, goal, srs = next(iter(dataloader))
    _, _, _, original_embeddings = agent_modeller(past_traj, current_traj, state, return_embeddings=True)
    save_original_embeddings()

    while user_gave_new_embedding():
        embeddings = load_embeddings()
        convert_embeddings_to_numpy()
        action_pred, goal_pred, srs_pred = agent_modeller.forward_given_embeddings(past_traj, current_traj, state, user_embeddings=embeddings)
        print('policy', torch.softmax(action_pred, dim=1))
        print('goal', torch.sigmoid(goal_pred))


def plot_increase_in_past_trajectories(agent_modeller: TOMMAS, dataset: GridworldDataset, min_npast=0, max_npast=10, additional_dataset_kwargs=None):
    n_past_vals = [i for i in range(min_npast, max_npast+1)]
    n_past_action_acc, n_past_goal_acc, n_past_srs_acc = [], [], []

    for n_past in n_past_vals:
        sampler = SequentialSampler(dataset)
        batch_sampler = GridworldDatasetBatchSampler(sampler=sampler, batch_size=10, drop_last=False,
                                                     min_num_episodes=dataset.min_num_episodes_per_agent, n_past=n_past,
                                                     additional_dataset_kwargs=additional_dataset_kwargs)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_mgdataset, pin_memory=False,
                                num_workers=4)
        action_accs, goal_accs, srs_accs = [], [], []
        for batch in dataloader:
            past_traj, current_traj, state, action, goal, srs = batch
            action_pred, goal_pred, srs_pred = agent_modeller(past_traj, current_traj, state)
            action_accs.append(action_acc(action_pred, action))
            goal_accs.append(goal_acc(goal_pred, goal))
            srs_accs.append(successor_representation_acc(srs_pred, srs))

        n_past_action_acc.append(np.average(action_accs))
        n_past_goal_acc.append(np.average(goal_accs))
        n_past_srs_acc.append(np.average(srs_accs))

    plt.plot(n_past_vals, n_past_action_acc)
    plt.plot(n_past_vals, n_past_goal_acc)
    plt.plot(n_past_vals, n_past_srs_acc)
    plt.legend(['Action Acc', 'Goal Acc', 'SRS Acc'])
    plt.show()


def visualize_predictions(visualization_name, agent_modeller: TOMMAS, dataset: GridworldDataset, n_past, empty_current_traj, additional_dataset_kwargs=None):
    def plot_actions(ax: matplotlib.axes.Axes, is_prediction):
        action_list = list(range(len(actions_to_plot[0])))
        bar = ax.bar(action_list, [0 for _ in range(len(action_list))])
        ax.set_ylim([0, 1])
        ax.set_xticks(action_list)
        ax.set_xticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY'])
        title = ("Predicted " if is_prediction else "True ") + "Actions"
        ax.set_title(title)
        return bar

    def plot_goals(ax, is_prediction):
        goal_list = list(range(len(goals_to_plot[0])))
        bar = ax.bar(goal_list, [0 for _ in range(len(goal_list))])
        ax.set_ylim([0, 1])
        ax.set_xticks(goal_list)
        ax.set_xticklabels(goal_list)
        title = ("Predicted " if is_prediction else "True ") + "Goals"
        ax.set_title(title)
        return bar

    def plot_sr(ax, is_prediction):
        fake_sr = np.zeros_like(srss_to_plot[0])
        fake_sr[0, 0] = 1
        image = ax.imshow(fake_sr, cmap='hot', interpolation='nearest')
        ax.grid(False)
        title = ("Predicted " if is_prediction else "True ") + "SR"
        ax.set_title(title)
        return image

    if additional_dataset_kwargs is None:
        additional_dataset_kwargs = dict()
    additional_dataset_kwargs["current_traj_batch_get"] = True

    batch_size = 1
    sampler = SequentialSampler(dataset)
    batch_sampler = GridworldDatasetBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False,
                                                 min_num_episodes=dataset.min_num_episodes_per_agent, n_past=n_past,
                                                 empty_current_traj_probability=1 if empty_current_traj else 1 / 35,
                                                 additional_dataset_kwargs=additional_dataset_kwargs)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_current_traj_batch_get,
                            pin_memory=False, num_workers=4)

    for agent_idx, batch in enumerate(dataloader):
        past_traj, current_traj, states, actions, goal_consumptions, srss = batch
        states = states.transpose(0, 1)
        actions = actions.transpose(0, 1)
        goal_consumptions = goal_consumptions.transpose(0, 1)
        srss = srss.transpose(0, 1)

        actions_to_plot = []
        action_hats_to_plot = []
        goals_to_plot = []
        goal_hats_to_plot = []
        srss_to_plot = []
        srs_hats_to_plot = []
        seq_len = len(states)
        for t, (state, action, goal_consumption, srs) in enumerate(zip(states, actions, goal_consumptions, srss)):
            if t == 0:
                current_traj_t = None
            else:
                current_traj_t = current_traj[0:t]
            action_hat, goal_hat, srs_hat, = agent_modeller(past_traj, current_traj_t, state)

            action_hat = torch.softmax(action_hat.squeeze(), dim=0).numpy()
            action_taken = action.squeeze().item()
            one_hot_action = np.zeros(len(action_hat))
            one_hot_action[action_taken] = 1
            action = one_hot_action

            goal_consumption = goal_consumption.squeeze().numpy()
            goal_hat = torch.sigmoid(goal_hat.squeeze()).numpy()

            srs = srs[0][-2].numpy()
            _, _, row, col = srs_hat.shape
            srs_hat = torch.softmax(srs_hat[0][-2].view(row * col), dim=0).view(row, col).numpy()

            actions_to_plot.append(action)
            action_hats_to_plot.append(action_hat)
            goals_to_plot.append(goal_consumption)
            goal_hats_to_plot.append(goal_hat)
            srss_to_plot.append(srs)
            srs_hats_to_plot.append(srs_hat)

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        action_bar = plot_actions(axes[0][0], False)
        action_hat_bar = plot_actions(axes[0][1], True)
        goal_bar = plot_goals(axes[1][0], False)
        goal_hat_bar = plot_goals(axes[1][1], True)
        srs_image = plot_sr(axes[2][0], False)
        srs_hat_image = plot_sr(axes[2][1], True)

        data = [action_bar, action_hat_bar, goal_bar, goal_hat_bar, srs_image, srs_hat_image]
        def run(t):
            def set_bar_data(bar, data):
                for i, b in enumerate(bar):
                    b.set_height(data[t][i])

            def set_image_data(image, data):
                sr = data[t]
                image.set_data(sr / np.max(sr))
                image.set_cmap("hot")

            set_bar_data(data[0], actions_to_plot)
            set_bar_data(data[1], action_hats_to_plot)
            set_bar_data(data[2], goals_to_plot)
            set_bar_data(data[3], goal_hats_to_plot)
            set_image_data(data[4], srss_to_plot)
            set_image_data(data[5], srs_hats_to_plot)
            return data

        anim = FuncAnimation(fig, func=run, frames=seq_len, repeat=False)
        anim.save("data/results/" + visualization_name + ".mp4", fps=1)
        return




