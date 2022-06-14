import numpy as np
import random

from torch.utils.data import Dataset, BatchSampler

from src.data.agent_trajectory_fetcher import AgentTrajectoryFetcher
from src.data.gridworld_transforms import GridworldTrajectoryTransform


class GridworldDatasetBatchSampler(BatchSampler):
    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 min_num_episodes,
                 current_traj_len=-1,
                 empty_current_traj_probability=1 / 35,
                 n_past=-1):
        super(GridworldDatasetBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.min_num_episodes = min_num_episodes
        self.current_traj_len = current_traj_len
        self.empty_current_traj_probability = empty_current_traj_probability
        self.n_past = n_past

    def get_n_past(self):
        if self.n_past < 0:
            return random.randint(0, self.min_num_episodes - 1)
        return self.n_past

    def get_current_traj_len(self):
        if random.random() < self.empty_current_traj_probability:
            return 0
        return self.current_traj_len

    def __iter__(self):
        batch = []
        n_past = self.get_n_past()
        current_traj_len = self.get_current_traj_len()
        for idx in self.sampler:
            batch.append((idx, n_past, current_traj_len))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                n_past = self.get_n_past()
                current_traj_len = self.get_current_traj_len()
        if len(batch) > 0 and not self.drop_last:
            yield batch


class GridworldDataset(Dataset):
    def __init__(self,
                 agent_trajectory_fetcher: AgentTrajectoryFetcher,
                 trajectory_transform: GridworldTrajectoryTransform):
        self.agent_trajectory_fetcher = agent_trajectory_fetcher
        self.trajectory_transform = trajectory_transform
        self.min_num_episodes_per_agent = agent_trajectory_fetcher.min_num_episodes_per_agent
        self.observation_space = None
        self.action_space = None
        self._init_io_space()

    def _init_io_space(self):
        n_past = 2
        trajectory, agent_idx = self.agent_trajectory_fetcher.__getitem__((0, n_past))
        collate_fn = self.trajectory_transform.get_collate_fn()
        inputs, predictions = collate_fn([self.trajectory_transform(trajectory, agent_idx, n_past, np.inf)])
        self.observation_space = inputs.shape()
        self.action_space = predictions.shape()

    def __len__(self):
        return len(self.agent_trajectory_fetcher)

    def __getitem__(self, item):
        idx, n_past, current_traj_len = item
        gridworld_trajectories, agent_idx = self.agent_trajectory_fetcher.__getitem__((idx, n_past))
        return self.trajectory_transform(gridworld_trajectories, agent_idx, n_past, current_traj_len)
