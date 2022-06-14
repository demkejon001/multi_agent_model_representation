from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.dataset_base import load_dataset


class AgentTrajectoryFetcher(ABC):
    min_num_episodes_per_agent: int

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item) -> Tuple[List[GridworldTrajectory], int]:
        pass


class AgentGridworldTrajectories(AgentTrajectoryFetcher):
    def __init__(self, gridworld_trajectories: List[GridworldTrajectory]):
        self.agent_ids = set()
        for gridworld_trajectory in gridworld_trajectories:
            for agent_id in gridworld_trajectory.agent_ids:
                self.agent_ids.add(agent_id)
        self.agent_ids = np.sort(list(self.agent_ids))
        self.num_agents = len(self.agent_ids)
        self.agent_gridworld_indices = self._get_agent_gridworld_indices(gridworld_trajectories)
        self.min_num_episodes_per_agent = np.inf
        for agent_id in self.agent_ids:
            self.min_num_episodes_per_agent = min(self.min_num_episodes_per_agent,
                                                  len(self.agent_gridworld_indices[agent_id]))
        self.gridworld_trajectories = np.array(gridworld_trajectories)

    def _get_agent_gridworld_indices(self, gridworld_trajectories):
        agent_gridworld_indices = dict()
        for agent_id in self.agent_ids:
            agent_gridworld_indices[agent_id] = []
        for gridworld_idx, gridworld_trajectory in enumerate(gridworld_trajectories):
            for agent_id in gridworld_trajectory.agent_ids:
                agent_gridworld_indices[agent_id].append(gridworld_idx)
        for agent_id in self.agent_ids:
            agent_gridworld_indices[agent_id] = np.array(agent_gridworld_indices[agent_id])
        return agent_gridworld_indices

    def __len__(self):
        return self.num_agents

    def __getitem__(self, item) -> Tuple[List[GridworldTrajectory], int]:
        idx, n_past = item
        agent_id = self.agent_ids[idx]
        gridworld_indices = self.agent_gridworld_indices[agent_id]
        rand_indices = np.random.choice(gridworld_indices.shape[0], n_past + 1, replace=False)
        selected_gridworld_indices = gridworld_indices[rand_indices]
        trajectories = [self.gridworld_trajectories[gridworld_idx] for gridworld_idx in selected_gridworld_indices]
        traj_agent_idx = trajectories[0].agent_id_to_idx(agent_id)
        traj_agent_indices = np.array([trajectory.agent_id_to_idx(agent_id) for trajectory in trajectories])
        if not np.all(traj_agent_indices == traj_agent_idx):
            raise ValueError(f"agent_id {agent_id} doesn't have the same trajectory agent index across gridworlds "
                             f"({traj_agent_indices})")
        return trajectories, traj_agent_idx

    def append(self, trajectory_fetcher):
        assert type(trajectory_fetcher) is type(self)
        gridworld_offset = len(self.gridworld_trajectories)

        self.gridworld_trajectories = np.append(self.gridworld_trajectories, trajectory_fetcher.gridworld_trajectories)
        self.min_num_episodes_per_agent = min(self.min_num_episodes_per_agent,
                                              trajectory_fetcher.min_num_episodes_per_agent)
        self.num_agents += trajectory_fetcher.num_agents
        self.agent_ids = np.append(self.agent_ids, trajectory_fetcher.agent_ids)
        # offsetting the agent's gridworld indices since dataset.gridworld_trajectories are no longer at the beginning
        for agent_id in trajectory_fetcher.agent_ids:
            offset_gridworld_idx = trajectory_fetcher.agent_gridworld_indices[agent_id] + gridworld_offset
            self.agent_gridworld_indices[agent_id] = offset_gridworld_idx


class AgentGridworldFilepaths(AgentTrajectoryFetcher):
    def __init__(self, gridworld_trajectories_filepaths: List[str], rand=True):
        self.filepaths = gridworld_trajectories_filepaths
        self.rand = rand
        self.min_num_episodes_per_agent = len(load_dataset(self.filepaths[0]))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item) -> Tuple[List[GridworldTrajectory], int]:
        file_idx, n_past = item
        gridworld_trajectories = load_dataset(self.filepaths[file_idx])
        if self.rand:
            selected_gridworld_indices = np.random.choice(len(gridworld_trajectories), n_past + 1, replace=False)
        else:
            selected_gridworld_indices = np.arange(n_past + 1)
        trajectories = [gridworld_trajectories[gridworld_idx] for gridworld_idx in selected_gridworld_indices]
        traj_agent_idx = np.random.choice(trajectories[0].num_agents)
        return trajectories, traj_agent_idx


class IterativeActionTrajectoryFilepaths(AgentTrajectoryFetcher):
    def __init__(self, gridworld_trajectories_filepaths: List[str], rand=True):
        self.filepaths = gridworld_trajectories_filepaths
        self.rand = rand
        self.min_num_episodes_per_agent = len(load_dataset(self.filepaths[0]))

    def __len__(self):
        return len(self.filepaths)

    def get_random_agent_idx(self):
        return 0

    def __getitem__(self, item):
        file_idx, n_past = item
        gridworld_trajectories = load_dataset(self.filepaths[file_idx])
        if self.rand:
            selected_gridworld_indices = np.random.choice(len(gridworld_trajectories), n_past + 1, replace=False)
        else:
            selected_gridworld_indices = np.arange(n_past + 1)
        trajectories = [gridworld_trajectories[gridworld_idx] for gridworld_idx in selected_gridworld_indices]
        agent_idx = self.get_random_agent_idx()
        return trajectories, agent_idx
