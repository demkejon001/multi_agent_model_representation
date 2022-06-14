import numpy as np
from typing import Union, List, Dict

from tommas.agents.agents import RandomAgent
from tommas.env.gridworld_env import GridworldEnv
from tommas.agents.reward_function import get_zeros_reward_functions
from tommas.data.agent_ids import get_agent_ids
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.dataset_base import save_dataset as base_save_dataset, load_dataset as base_load_dataset, \
    dataset_extension, get_dataset_filepaths as base_get_dataset_filepaths, seed_everything, delete_dirpath


num_agents = 4
num_dataset_agents = 3000
num_goals = 4
num_actions = 5
horizon = 1
min_num_episodes_per_agent = 30
dim = (11, 11)
default_alphas = [.01, .03, .1, .3, 1.0, 3.0]


def play_episode(env, agents):
    trajectory = []
    done = False
    state = env.reset()
    while not done:
        actions = [agent.action(state) for agent in agents]
        new_state, reward, done, info = env.step(actions)
        trajectory.append((state, reward, actions, info))
        state = new_state
    agent_ids = [agent.id for agent in agents]
    return GridworldTrajectory(trajectory, env, agent_ids)


def create_dataset_and_policies(alpha, is_train_dataset, is_single_agent, seed=42):
    dataset_dirpath = get_dataset_dirpath(alpha, is_train_dataset, is_single_agent)
    delete_dirpath(dataset_dirpath)
    seed_everything(seed)
    agent_reward_funcs = get_zeros_reward_functions(num_agents, num_goals)
    env = GridworldEnv(num_agents, agent_reward_funcs, num_goals, horizon=1, dim=dim, seed=seed)
    agent_id_idx = 0
    agent_ids = get_agent_ids(num_dataset_agents)
    policies = {}

    for file_idx in range(num_dataset_agents//num_agents):
        agents = []
        for _ in range(num_agents):
            agent_id = agent_ids[agent_id_idx]
            agent = RandomAgent(agent_id, num_actions, alpha)
            agents.append(agent)
            policies[agent_id] = agent.policy
            agent_id_idx += 1
        history = [play_episode(env, agents) for _ in range(min_num_episodes_per_agent)]
        filename = ("train" if is_train_dataset else "test") + str(file_idx)
        base_save_dataset(history, dataset_dirpath, filename)
    base_save_dataset(policies, dataset_dirpath, "metadata")


def get_policy_filepath(alpha, is_training_data, is_single_agent):
    dataset_dirpath = get_dataset_dirpath(alpha, is_training_data, is_single_agent)
    return dataset_dirpath + "metadata" + dataset_extension


def get_dataset_dirpath(alpha: float, is_train_dataset, is_single_agent):
    dataset_dirpath = "data/datasets/random_policy/"
    dataset_dirpath += ("single_agent/" if is_single_agent else "multi_agent/")
    dataset_dirpath += "alpha_" + str(alpha) + "/"
    dataset_dirpath += ("train/" if is_train_dataset else "test/")
    return dataset_dirpath


def load_policies(alphas: Union[List[float], float],
                  is_train_dataset: bool,
                  single_agent: bool = False) -> Dict[int, np.array]:
    def load_single_dataset_and_policies(alpha):
        return base_load_dataset(get_policy_filepath(alpha, is_train_dataset, single_agent))

    if type(alphas) is float:
        return load_single_dataset_and_policies(alphas)
    if len(alphas) == 1:
        return load_single_dataset_and_policies(alphas[0])

    policies = {}
    for i, alpha in enumerate(alphas):
        if i == 0:
            policy = load_single_dataset_and_policies(alpha)
            policies.update(policy)
        else:
            policy = load_single_dataset_and_policies(alpha)
            policies.update(policy)

    return policies


def create_dataset(seed=42):
    create_and_save_all_datasets(seed=seed)


def create_single_agent_dataset(seed=42):
    create_and_save_all_datasets(single_agent=True, seed=seed)


def create_and_save_all_datasets(alphas=None, single_agent=False, seed=42):
    global num_agents
    if single_agent:
        num_agents = 1
    if alphas is None:
        alphas = default_alphas.copy()
    for is_training_dataset in [True, False]:
        for alpha in alphas:
            seed += 1
            create_dataset_and_policies(alpha, is_training_dataset, single_agent, seed)


def get_dataset_filepaths(is_train_dataset, alpha=None, is_single_agent=False):
    if alpha is None:
        filepaths = []
        for alpha in default_alphas:
            filepaths.extend(get_dataset_filepaths(is_train_dataset, alpha, is_single_agent))
        return filepaths

    dataset_dirpath = get_dataset_dirpath(alpha, is_train_dataset, is_single_agent)
    dataset_filename = ("train" if is_train_dataset else "test")
    return base_get_dataset_filepaths(dataset_dirpath, dataset_filename)
