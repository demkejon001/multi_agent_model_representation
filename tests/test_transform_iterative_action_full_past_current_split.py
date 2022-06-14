import numpy as np
import torch

from tommas.data.gridworld_transforms import IterativeActionFullPastCurrentSplit
from tommas.data.datamodule_factory import get_agent_trajectory_fetcher, IterativeActionTrajectoryFilepaths


class DummyDatamoduleNamespace:
    def __init__(self):
        self.dataset = ["action_mirror_ja4"]
        self.model = "iterative_action_past_current"
        self.batch_size = 2
        self.num_workers = 0
        self.log_eval_interval = 50
        self.seed = 411352


def test_transform():
    agent_trajectory_fetcher, _ = get_agent_trajectory_fetcher(DummyDatamoduleNamespace())

    trajectory_transform = IterativeActionFullPastCurrentSplit()
    collate_fn = trajectory_transform.get_collate_fn()

    num_agents = 4

    for n_past in range(4):
        batch = []

        trajectories, agent_idx = agent_trajectory_fetcher.__getitem__((0, n_past))
        past_trajectories, current_trajectory, hidden_state_indices, actions = trajectory_transform(trajectories, agent_idx, n_past, np.inf)
        batch.append([past_trajectories, current_trajectory, hidden_state_indices, actions])
        assert past_trajectories.shape[0] == len(hidden_state_indices)
        assert current_trajectory.shape[0] == actions.shape[0]

        trajectories, agent_idx = agent_trajectory_fetcher.__getitem__((1, n_past))
        past_trajectories, current_trajectory, hidden_state_indices, actions = trajectory_transform(trajectories, agent_idx, n_past, np.inf)
        batch.append([past_trajectories, current_trajectory, hidden_state_indices, actions])
        assert past_trajectories.shape[0] == len(hidden_state_indices)
        assert current_trajectory.shape[0] == actions.shape[0]

        batch_size = 2

        x, y = collate_fn(batch)
        assert x.past_trajectories.shape == (batch_size, *past_trajectories.shape)
        assert x.current_trajectory.shape == (batch_size, *current_trajectory.shape)
        assert x.hidden_state_indices.shape == (len(hidden_state_indices), )
        assert y.action.shape == (batch_size * actions.shape[0], )

        assert x.past_trajectories[:, x.hidden_state_indices].shape == (batch_size, (n_past+1), num_agents)
        assert torch.all(x.past_trajectories[:, x.hidden_state_indices] == -1)
