import numpy as np
from typing import List, Dict
from tqdm import tqdm

from tommas.agents.gridworld_agents import PathfindingAgent
from tommas.agents.create_gridworld_agents import get_random_gridworld_agent, RandomAgentParamSampler
from tommas.env.gridworld_env import GridworldEnv
from tommas.data.gridworld_trajectory import UnspatialisedGridworldTrajectory
from tommas.data.dataset_base import save_dataset, get_dataset_filepaths as base_get_dataset_filepaths, \
    delete_dirpath, seed_everything


# Environment parameters
num_agents_per_episode = 4
num_goals = 12
num_actions = 5
horizon = 20
dim = (21, 21)
min_num_walls = 1
max_num_walls = 4

# Dataset parameters
num_episodes_per_agent = 10
dirpath = "data/datasets/cooperative_gridworld/"


def play_episode(env, agents, create_new_world=True):
    trajectory = []
    state = env.reset(create_new_world)
    for agent_idx, agent in enumerate(agents):
        agent.reset(state, agent_idx)
    done = False
    timestep = 0
    while not done:
        actions = [agent.action(state) for agent in agents]
        if timestep == 0 and 4 in actions:
            return play_episode(env, agents, create_new_world)
        if actions == [4, 4, 4, 4] and timestep < (horizon * .5):
            return play_episode(env, agents, create_new_world)
        new_state, reward, done, info = env.step(actions)
        trajectory.append((state, reward, actions, info))
        state = new_state
        timestep += 1
    agent_ids = [agent.id for agent in agents]
    return UnspatialisedGridworldTrajectory(trajectory, env, agent_ids)


def get_random_env(reward_funcs):
    env_seed = np.random.randint(0, 1e10)
    return GridworldEnv(num_agents_per_episode, reward_funcs, num_goals, horizon, dim, min_num_walls=min_num_walls,
                        max_num_walls=max_num_walls, seed=env_seed)


def reassign_agent_indices(agents: List[PathfindingAgent]):
    for agent_idx, agent in enumerate(agents):
        agent.destination_selector.agent_idx = agent_idx


def get_agent_ids(num_agents, is_training_dataset: bool, seed):
    base_agent_id = ("train" if is_training_dataset else "test") + "_"
    base_agent_id += "_seed" + str(seed) + "-"
    return [base_agent_id + str(i) for i in range(num_agents)]


def _create_dataset(seed, is_training_dataset, num_agent_types):
    if is_training_dataset:
        dataset_file_type = "train"
    else:
        dataset_file_type = "test"

    raps = RandomAgentParamSampler(seed, is_training_dataset)
    agents_metadata = dict()

    num_dataset_agents = sum(num_agent_types.values()) * num_agents_per_episode
    agent_ids = np.array(get_agent_ids(num_dataset_agents, is_training_dataset, seed))
    agent_ids = np.split(agent_ids, len(agent_ids) // num_agents_per_episode)

    filename_idx = 0
    agent_ids_idx = 0
    for agent_type, num_agents in num_agent_types.items():
        for _ in tqdm(range(num_agents)):
            agents, metadata = get_random_gridworld_agent(agent_type, raps, agent_ids[agent_ids_idx])
            for gridworld_agent_id, data in zip(agent_ids[agent_ids_idx], metadata):
                agents_metadata[gridworld_agent_id] = data
            reassign_agent_indices(agents)
            agent_history = []
            env = get_random_env(reward_funcs=[agent.reward_function for agent in agents])
            for _ in range(num_episodes_per_agent):
                agent_history.append(play_episode(env, agents))
            save_dataset(agent_history, dirpath + dataset_file_type + '/', dataset_file_type + str(filename_idx))

            agent_ids_idx += 1
            filename_idx += 1
    save_dataset(agents_metadata, dirpath + dataset_file_type + '/', "metadata")


def create_dataset(seed=42, num_train_agent_types: Dict[str, int] = None, num_test_agent_types: Dict[str, int] = None):
    delete_dirpath(dirpath)
    seed_everything(seed)
    _create_dataset(seed, True, num_train_agent_types)
    _create_dataset(seed, False, num_test_agent_types)


def get_dataset_filepaths(is_train_dataset=True):
    filename = ("train" if is_train_dataset else "test")
    return base_get_dataset_filepaths(dirpath + filename + '/', filename)
