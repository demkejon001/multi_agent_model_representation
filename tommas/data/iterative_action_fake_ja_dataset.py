from typing import List, Dict
import numpy as np
import random

from tommas.agents.iterative_action_agents import IterativeActionAgent
from tommas.agents.create_iterative_action_agents import get_random_iterative_action_agent, RandomStrategySampler
from tommas.data.dataset_base import save_dataset, seed_everything, delete_dirpath
from tommas.data.iterative_action_dataset import get_agent_ids, IterativeActionTrajectory, get_agent_type_pairs, \
    get_dataset_dirpath, get_dataset_filepaths


def play_episode(agents: List[IterativeActionAgent], horizon=20):
    state = [[-1 for _ in range(len(agents))]]
    for agent in agents:
        agent.reset()
    for _ in range(horizon):
        actions = [agent.action(state) for agent in agents]
        state.append(actions.copy())
    for step in state:
        step[1] = random.randint(0, 1)
    return state


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


def create_action_dataset(seed, num_agent_types, is_training_dataset, dirpath, num_opponents, is_ia, horizon=20):
    num_episode = 10
    rss = RandomStrategySampler(seed, is_train_sampler=is_training_dataset)
    agent_type_pairs = get_agent_type_pairs(num_agent_types, num_opponents)
    num_agents = len(agent_type_pairs) * len(agent_type_pairs[0])
    agent_ids = np.array(get_agent_ids(num_agents, num_opponents, is_training_dataset, seed, is_ia))
    create_fixed_opponent_dataset(agent_type_pairs, num_episode=num_episode, horizon=horizon,
                                  get_agent_fn=get_random_iterative_action_agent, rss=rss,
                                  is_training_dataset=is_training_dataset, dirpath=dirpath, agent_ids=agent_ids)


def create_dataset(seed=42, num_train_agent_types: Dict[str, int] = None, num_test_agent_types: Dict[str, int] = None,
                   num_opponents=1, is_ia=True, tags=None, horizon=20):
    dataset_dirpath = get_dataset_dirpath(num_opponents=num_opponents, is_ia=is_ia, tags=tags)
    delete_dirpath(dataset_dirpath)
    seed_everything(seed)

    create_action_dataset(seed, num_train_agent_types, is_training_dataset=True,
                          dirpath=dataset_dirpath, num_opponents=num_opponents, is_ia=is_ia, horizon=horizon)

    create_action_dataset(seed, num_test_agent_types, is_training_dataset=False,
                          dirpath=dataset_dirpath, num_opponents=num_opponents, is_ia=is_ia, horizon=horizon)
