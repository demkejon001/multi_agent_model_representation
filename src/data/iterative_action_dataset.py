from typing import List, Dict
import numpy as np

from src.agents.iterative_action_agents import IterativeActionAgent
from src.agents.create_iterative_action_agents import get_random_iterative_action_agent, RandomStrategySampler
from src.data.dataset_base import save_dataset, get_dataset_filepaths as base_get_dataset_filepaths, \
    seed_everything, delete_dirpath


def get_agent_ids(num_agents, num_opponents, is_training_dataset: bool, seed, is_ia: bool, tags=None):
    base_agent_id = ("train" if is_training_dataset else "test") + "_"
    if tags is not None and tags != "":
        base_agent_id += tags + "_"
    base_agent_id += ("ia" if is_ia else "ja")
    base_agent_id += str(num_opponents + 1)
    base_agent_id += "_seed" + str(seed) + "-"
    return [base_agent_id + str(i) for i in range(num_agents)]


class IterativeActionTrajectory:
    def __init__(self, agent_ids: List[int], trajectory: List[List[int]]):
        self.agent_ids = agent_ids
        self.trajectory = trajectory

    def __len__(self):
        return len(self.trajectory)


def play_episode(agents: List[IterativeActionAgent], horizon=20):
    state = [[-1 for _ in range(len(agents))]]
    for agent in agents:
        agent.reset()
    for _ in range(horizon):
        actions = [agent.action(state) for agent in agents]
        state.append(actions.copy())
    return state


def get_agent_type_pairs(num_agent_types: Dict[str, int], num_opponents=1):
    def get_random_agent_type(agent_group_idx):
        for group_idx in agent_group_indices:
            if agent_group_idx < group_idx:
                return agent_group_indices[group_idx]

    agent_group_indices = dict()
    running_sum_agent_type = 0
    for agent_type, num_agent_type in num_agent_types.items():
        agent_group_indices[running_sum_agent_type + num_agent_type] = agent_type
        running_sum_agent_type += num_agent_type

    num_agents = sum(list(num_agent_types.values()))
    agent_types = np.arange(num_agents)
    agent_type_pairs = []
    for agent_type in agent_types:
        agent_type_pairs.append([get_random_agent_type(agent_type)] + ["mixed_strategy" for _ in range(num_opponents)])
    return agent_type_pairs


def create_fixed_opponent_dataset(agent_type_pairs, num_episode, horizon, get_agent_fn,
                                  rss: RandomStrategySampler, is_training_dataset: bool, dirpath, agent_ids):
    if is_training_dataset:
        dataset_file_type = "train"
    else:
        dataset_file_type = "test"

    agents_metadata = dict()
    agent_pairs_ids = np.split(agent_ids, len(agent_type_pairs))
    filename_idx = 0
    for agent_type_pair, agent_pair_ids in zip(agent_type_pairs, agent_pairs_ids):
        agents = []
        for agent_idx, (agent_type, agent_id) in enumerate(zip(agent_type_pair, agent_pair_ids)):
            agent, agent_metadata = get_agent_fn(rss=rss, agent_type=agent_type, agent_id=agent_id,
                                                 agent_idx=agent_idx)
            agents.append(agent)
            agents_metadata[agent_id] = agent_metadata

        agent_history = []
        for _ in range(num_episode):
            agent_history.append(IterativeActionTrajectory(agent_pair_ids, play_episode(agents, horizon)))

        save_dataset(agent_history, dirpath + dataset_file_type + '/', dataset_file_type + str(filename_idx))
        filename_idx += 1
    save_dataset(agents_metadata, dirpath + dataset_file_type + '/', "metadata")


def create_action_dataset(seed, num_agent_types, is_training_dataset, dirpath, num_opponents, is_ia, horizon=20, tags=None):
    num_episode = 10
    rss = RandomStrategySampler(seed, is_train_sampler=is_training_dataset)
    agent_type_pairs = get_agent_type_pairs(num_agent_types, num_opponents)
    num_agents = len(agent_type_pairs) * len(agent_type_pairs[0])
    agent_ids = np.array(get_agent_ids(num_agents, num_opponents, is_training_dataset, seed, is_ia, tags))
    create_fixed_opponent_dataset(agent_type_pairs, num_episode=num_episode, horizon=horizon,
                                  get_agent_fn=get_random_iterative_action_agent, rss=rss,
                                  is_training_dataset=is_training_dataset, dirpath=dirpath, agent_ids=agent_ids)


def create_dataset(seed=42, num_train_agent_types: Dict[str, int] = None, num_test_agent_types: Dict[str, int] = None,
                   num_opponents=1, is_ia=True, tags=None, horizon=20):
    dataset_dirpath = get_dataset_dirpath(num_opponents=num_opponents, is_ia=is_ia, tags=tags)
    delete_dirpath(dataset_dirpath)
    seed_everything(seed)

    create_action_dataset(seed, num_train_agent_types, is_training_dataset=True,
                          dirpath=dataset_dirpath, num_opponents=num_opponents, is_ia=is_ia, horizon=horizon, tags=tags)

    create_action_dataset(seed, num_test_agent_types, is_training_dataset=False,
                          dirpath=dataset_dirpath, num_opponents=num_opponents, is_ia=is_ia, horizon=horizon, tags=tags)


def get_dataset_dirpath(num_opponents, is_ia, tags=None):
    tag_info = ""
    if tags is not None:
        if type(tags) is list:
            tag_info = "_".join(tags) + "_"
        else:
            tag_info = tags + "_"

    agent_types = ("ia" if is_ia else "ja")
    return "data/datasets/action_" + tag_info + agent_types + str(num_opponents+1) + "/"


def get_dataset_filepaths(is_train_dataset=True, num_opponents=1, is_ia=True, tags=None):
    filename = ("train" if is_train_dataset else "test")
    dataset_dirpath = get_dataset_dirpath(num_opponents=num_opponents, is_ia=is_ia, tags=tags)
    return base_get_dataset_filepaths(dataset_dirpath + filename + '/', filename)
