from abc import ABC, abstractmethod
import numpy as np
from typing import List, Callable, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.gridworld_trajectory import GridworldTrajectory, UnspatialisedGridworldTrajectory
from src.data.iterative_action_dataset import IterativeActionTrajectory
from src.agent_modellers.modeller_inputs import IterativeActionToMnetInput, GridworldToMnetInput
from src.agent_modellers.modeller_outputs import AgentObjectives, ActionClassification


class TrajectoryTransform(ABC):
    @abstractmethod
    def get_collate_fn(self) -> Callable:
        pass

    @abstractmethod
    def __call__(self, gridworld_trajectories: Union[List[GridworldTrajectory], List[list]], agent_idx: int,
                 n_past: int, current_traj_len: int):
        pass

    @staticmethod
    def get_state_and_action(current_traj, current_traj_len, agent_idx=0, attach_agent_ids=False,
                             remove_other_agents=False):
        state = current_traj.get_state(current_traj_len, agent_idx=agent_idx, attach_agent_ids=attach_agent_ids,
                                       remove_other_agents=remove_other_agents)
        action = current_traj.get_action(current_traj_len, agent_idx=agent_idx)
        return state, action

    @staticmethod
    def get_goal_consumption(current_traj, current_traj_len, agent_idx=0):
        return current_traj.get_future_goals_consumed(start_index=current_traj_len, agent_idx=agent_idx)

    @staticmethod
    def get_srs(current_traj, current_traj_len, agent_idx=0):
        return current_traj.get_successor_representations(start_index=current_traj_len, agent_idx=agent_idx)

    @staticmethod
    def get_current_traj_len(gridworld_trajectory: GridworldTrajectory, current_traj_len: int):
        current_traj_len = min(len(gridworld_trajectory)-1, current_traj_len)
        if current_traj_len == -1:
            current_traj_len = np.random.randint(1, len(gridworld_trajectory)-1)
        return current_traj_len


class GridworldTrajectoryTransform(TrajectoryTransform):
    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            past_trajectory_batch = pad_sequence([torch.tensor(batch[i][0]) for i in range(len(batch))],
                                                 batch_first=True).float()
            past_action_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(len(batch))],
                                             batch_first=True).float()
            current_trajectory_batch = pad_sequence([torch.tensor(batch[i][2]) for i in range(len(batch))],
                                                    batch_first=True).float()
            current_action_batch = pad_sequence([torch.tensor(batch[i][3]) for i in range(len(batch))],
                                                batch_first=True).float()
            action_batch = torch.from_numpy(np.concatenate([batch[i][4] for i in range(len(batch))]))
            goal_consumption_batch = torch.from_numpy(np.concatenate([batch[i][5] for i in range(len(batch))]))
            hidden_state_indices = torch.tensor(batch[0][6], dtype=torch.bool)

            modeller_input = GridworldToMnetInput(past_trajectory_batch, past_action_batch, current_trajectory_batch,
                                                  current_action_batch, hidden_state_indices)
            true_output = AgentObjectives(action_batch, goal_consumption_batch, None)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[UnspatialisedGridworldTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int):
        actions = []
        goals = []

        reordered_gridworld_trajectories = []
        for i in range(1, n_past+1):
            reordered_gridworld_trajectories.append(gridworld_trajectories[i])
        reordered_gridworld_trajectories.append(gridworld_trajectories[0])

        past_trajectories = []
        past_actions = []
        hidden_state_indices = []
        for gridworld_idx in range(n_past):
            trajectory = reordered_gridworld_trajectories[gridworld_idx]
            t, a = trajectory.get_trajectory(agent_idx=agent_idx, add_start_token=True)
            past_trajectories.append(t)
            past_actions.append(a)
            traj_hidden_state_indices = [False for _ in range(len(trajectory) + 1)]
            traj_hidden_state_indices[0] = True
            hidden_state_indices += traj_hidden_state_indices
        past_trajectories.append(np.expand_dims(reordered_gridworld_trajectories[0].get_start_state_token(), 0))
        past_actions.append(np.expand_dims(reordered_gridworld_trajectories[0].get_start_action_token(), 0))

        past_trajectories = np.concatenate(past_trajectories, axis=0)
        past_actions = np.concatenate(past_actions, axis=0)
        hidden_state_indices += [True]

        current_gridworld = reordered_gridworld_trajectories[-1]
        current_trajectory, current_actions = current_gridworld.get_trajectory(agent_idx=agent_idx, add_start_token=True)

        for timestep in range(len(current_trajectory) - 1):
            actions.append(current_gridworld.get_action(timestep, agent_idx=agent_idx))
            goals.append(self.get_goal_consumption(current_gridworld, timestep, agent_idx))

        current_trajectory = current_trajectory
        current_actions = current_actions
        actions = np.tile(actions, n_past + 1)
        goals = np.tile(goals, (n_past + 1, 1))

        return past_trajectories, past_actions, current_trajectory, current_actions, actions, goals, hidden_state_indices


class IterativeActionTrajectoryTransform(TrajectoryTransform):
    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            past_trajectory_batch = pad_sequence([torch.tensor(batch[i][0]) for i in range(len(batch))],
                                                 batch_first=True).float()
            current_trajectory_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(len(batch))],
                                                    batch_first=True).float()
            hidden_state_indices = torch.tensor(batch[0][2], dtype=torch.bool)
            action_batch = torch.from_numpy(np.concatenate([batch[i][3] for i in range(len(batch))]))
            modeller_input = IterativeActionToMnetInput(past_trajectory_batch, current_trajectory_batch, hidden_state_indices)
            true_output = ActionClassification(action_batch)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[IterativeActionTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int, shuffled_agent_indices=None):

        reordered_gridworld_trajectories = []
        for i in range(1, n_past+1):
            reordered_gridworld_trajectories.append(gridworld_trajectories[i])
        reordered_gridworld_trajectories.append(gridworld_trajectories[0])

        past_trajectories = []
        hidden_state_indices = []
        for i in range(n_past):
            trajectory = np.array(reordered_gridworld_trajectories[i].trajectory, dtype=int)
            traj_hidden_state_indices = [False for _ in range(trajectory.shape[0])]
            traj_hidden_state_indices[0] = True
            hidden_state_indices += traj_hidden_state_indices
            if agent_idx == 1:
                trajectory = trajectory[:, [1, 0]]
            past_trajectories.append(trajectory)
        past_trajectories.append(reordered_gridworld_trajectories[-1].trajectory[0:1])
        past_trajectories = np.concatenate(past_trajectories)
        hidden_state_indices += [True]

        current_trajectory = np.array(reordered_gridworld_trajectories[-1].trajectory, dtype=int)
        if agent_idx == 1:
            current_trajectory = current_trajectory[:, [1, 0]]
        actions = current_trajectory[1:, 0]
        current_trajectory = current_trajectory[:-1]

        # Shuffle Opponents
        num_agents = len(current_trajectory[0])
        if num_agents > 2:
            if shuffled_agent_indices is None:
                shuffled_opponent_indices = np.arange(num_agents-1) + 1
                np.random.shuffle(shuffled_opponent_indices)
                shuffled_agent_indices = np.concatenate(([0], shuffled_opponent_indices))
            else:
                if len(shuffled_agent_indices) != num_agents:
                    raise ValueError(f"shuffled_opponent_indices should be length {num_agents}. "
                                     f"Current length is {len(shuffled_agent_indices)}")
            past_trajectories = past_trajectories[:, shuffled_agent_indices]
            current_trajectory = current_trajectory[:, shuffled_agent_indices]

        current_trajectory = np.tile(current_trajectory, (n_past+1, 1))
        actions = np.tile(actions, n_past+1)

        return past_trajectories, current_trajectory, hidden_state_indices, actions

    @staticmethod
    def get_current_traj_len(gridworld_trajectory: np.ndarray, current_traj_len: int):
        current_traj_len = min(len(gridworld_trajectory)-1, current_traj_len)
        if current_traj_len == 0:
            return 1
        if current_traj_len == -1:
            current_traj_len = np.random.randint(1, len(gridworld_trajectory)-1)
        return current_traj_len
