from typing import List

from tommas.agents.agents import MovementPatternAgent
from tommas.agents.reward_function import get_zeros_reward_functions
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.env.gridworld_env import GridworldEnv
from tommas.data.dataset_base import save_dataset as base_save_dataset, delete_dirpath, seed_everything, \
    get_dataset_filepaths as base_get_dataset_filepaths


num_dataset_agents = 16
world_dim = (11, 11)
num_actions = 5
num_goals = 4
min_num_episodes_per_agent = 5
horizon = 10
dataset_dirpath = "data/datasets/debug/"


def play_episode(env: GridworldEnv, agents: List[MovementPatternAgent]):
    trajectory = []
    done = False
    state = env.reset()
    for agent in agents:
        agent.reset()
    while not done:
        actions = [agent.action(state) for agent in agents]
        obs, reward, done, info = env.step(actions)
        trajectory.append((obs, [reward], actions, info))
        state = obs
    return trajectory


def get_agents_history(env: GridworldEnv, agents: List[MovementPatternAgent]):
    agent_history = []
    agent_ids = [agent.id for agent in agents]
    while len(agent_history) < min_num_episodes_per_agent:
        trajectory = play_episode(env, agents)
        agent_history.append(GridworldTrajectory(trajectory, env, agent_ids))
    return agent_history


def create_dataset(single_agent, seed=42):
    delete_dirpath(dataset_dirpath)
    seed_everything(seed)
    i = 0
    num_agents = 1 if single_agent else 4
    for j in range(num_dataset_agents):
        reward_functions = get_zeros_reward_functions(num_agents, num_goals)
        agents = []
        for _ in range(num_agents):
            agents.append(MovementPatternAgent(i, num_actions))
            i += 1
        env = GridworldEnv(num_agents, reward_functions, num_goals, horizon=horizon, dim=world_dim, seed=seed)
        history = get_agents_history(env, agents)
        base_save_dataset(history, dataset_dirpath, get_dataset_filename(single_agent) + str(j))


def get_dataset_filename(single_agent):
    if single_agent:
        return "debug_single"
    return "debug"


def get_dataset_filepaths(is_training_dataset=True):
    filename = get_dataset_filename(False)
    return base_get_dataset_filepaths(dataset_dirpath, filename)


def get_dataset(is_training_dataset: bool):
    filepaths = get_dataset_filepaths(is_training_dataset)
