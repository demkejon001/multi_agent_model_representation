import pickle
import numpy as np
from typing import List, Union

from tommas.agents.agents import GoalDirectedAgent
from tommas.agents.reward_function import get_random_reward_functions, get_zeros_reward_functions
from tommas.env.gridworld_env import GridworldEnv
from tommas.data.gridworld_dataset import GridworldDataset
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.agent_ids import get_agent_ids

num_dataset_agents = 120
num_actions = 5
num_goals = 4
min_num_episodes_per_agent = 11
dataset_filename = "data/datasets/tomnet_goal_directed"
file_extension = "pickle"
dataset_types = ['', 'dynamic']


def play_episode(env: GridworldEnv, agent: GoalDirectedAgent):
    trajectory = []
    done = False
    state = env.reset()
    agent.reset(env)
    while not done:
        action = agent.action(state)
        obs, reward, done, info = env.step(action)
        trajectory.append((obs, [reward], [action], info))
        state = obs
    return trajectory


def get_agent_history(env: GridworldEnv, agent: GoalDirectedAgent, agent_ids: List[int]):
    agent_history = []
    while len(agent_history) < min_num_episodes_per_agent:
        trajectory = play_episode(env, agent)
        if len(trajectory) > 1:
            agent_history.append(GridworldTrajectory(trajectory, env, agent_ids))
    return agent_history


def create_dataset(seed=42):
    np.random.seed(seed)
    history = []
    agent_ids = get_agent_ids(num_dataset_agents)
    env = GridworldEnv(1, get_zeros_reward_functions(1, num_goals), num_goals=num_goals, seed=seed)
    for i in range(num_dataset_agents):
        reward_function = get_random_reward_functions(1, num_goals)[0]
        agent = GoalDirectedAgent(i, num_actions, None, reward_function, discount=1)
        env.reset(agent_reward_functions=[reward_function])
        history += get_agent_history(env, agent, [agent_ids[i]])
    return GridworldDataset(history)


def create_dynamic_dataset(seed=42):
    np.random.seed(seed)
    history = []
    agents = []
    agent_ids = get_agent_ids(num_dataset_agents)
    env = GridworldEnv(1, get_zeros_reward_functions(1, num_goals), num_goals=num_goals, seed=seed)
    for i in range(num_dataset_agents):
        reward_function = get_random_reward_functions(1, num_goals)[0]
        agent = GoalDirectedAgent(i, num_actions, None, reward_function, discount=1)
        agents.append(agent)
        env.reset(agent_reward_functions=[reward_function])
        history += get_agent_history(env, agent, [agent_ids[i]])
    return DynamicGridworldDataset(history, agents,
                                   GridworldEnv(1, get_random_reward_functions(1, num_goals), num_goals=num_goals,
                                                seed=seed + 1),
                                   update_ctr=30)


def create_dynamic_train_and_test_set():
    return create_dynamic_dataset(), create_dynamic_dataset()


class DynamicGridworldDataset(GridworldDataset):
    def __init__(self, gridworld_trajectories, agents, env,
                 update_ctr=None):
        super(DynamicGridworldDataset, self).__init__(gridworld_trajectories)
        self.agents = agents
        self.env = env
        self.agent_history_accessed_ctr = np.zeros(self.num_agents)
        self.update_ctr = update_ctr
        if self.update_ctr is None:
            self.update_ctr = self.min_num_episodes_per_agent

    def play_episode(self, agent):
        trajectory = []
        done = False
        state = self.env.reset()
        agent.reset(self.env)
        while not done:
            action = agent.action(state)
            obs, reward, done, info = self.env.step(action)
            trajectory.append((obs, [reward], [action], info))
            state = obs
        return trajectory

    def update_agent_history(self, agent_idx):
        agent = self.agents[agent_idx]
        agent_id = self.agent_ids[agent_idx]
        agent_history = []
        for gridworld_idx in self.agent_gridworld_indices[agent_id]:
            trajectory = self.play_episode(agent)
            self.gridworld_trajectories[gridworld_idx] = trajectory
        return agent_history

    def __getitem__(self, item):
        agent_idx, _, _ = item
        self.agent_history_accessed_ctr[agent_idx] += 1
        if self.agent_history_accessed_ctr[agent_idx] > self.update_ctr:
            self.agent_history_accessed_ctr[agent_idx] = 0
            self.update_agent_history(agent_idx)
        return super(DynamicGridworldDataset, self).__getitem__(item)


def check_dataset_type(dataset_type):
    if dataset_type not in dataset_types:
        raise ValueError("Invalid dataset type. Expected one of: %s" % dataset_types)


def get_dataset_filename(is_train_dataset, dataset_type):
    check_dataset_type(dataset_type)
    return dataset_filename + ("_train" if is_train_dataset else "_test") + dataset_type + file_extension


def save_dataset(dataset, is_train_dataset, dataset_type=''):
    filename = get_dataset_filename(is_train_dataset, dataset_type)
    file = open(filename, "wb")
    pickle.dump(dataset, file)
    file.close()


def load_dataset(is_train_dataset=True, dataset_type='') -> Union[GridworldDataset, DynamicGridworldDataset]:
    filename = get_dataset_filename(is_train_dataset, dataset_type)
    file = open(filename, "rb")
    dataset = pickle.load(file)
    file.close()
    return dataset


def create_and_save_all_datasets(dataset_type='', seed=42):
    if dataset_type == "all":
        for dataset_type in dataset_types:
            create_and_save_all_datasets(dataset_type)
    else:
        check_dataset_type(dataset_type)
        if dataset_type == '':
            train = create_dataset(seed=seed)
            test = create_dataset(seed=seed+1)
        elif dataset_type == 'dynamic':
            train = create_dynamic_dataset(seed=seed)
            test = create_dynamic_dataset(seed=seed+1)
        else:
            raise ValueError("Dataset Type " + dataset_type + " is not accounted for")
        save_dataset(train, is_train_dataset=True, dataset_type=dataset_type)
        save_dataset(test, is_train_dataset=False, dataset_type=dataset_type)

