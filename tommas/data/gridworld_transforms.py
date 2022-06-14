from abc import ABC, abstractmethod
import numpy as np
from typing import List, Callable, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from tommas.data.gridworld_trajectory import GridworldTrajectory, UnspatialisedGridworldTrajectory
from tommas.data.iterative_action_dataset import IterativeActionTrajectory
from tommas.agent_modellers.modeller_inputs import TOMMASInput, TOMMASTransformerInput, IterativeActionTOMMASInput, \
    IterativeActionTransformerInput, IterativeActionToMnetInput, GridworldToMnetInput
from tommas.agent_modellers.modeller_outputs import AgentWeightedObjectives, AgentObjectives, ActionClassification


class GridworldTrajectoryTransform(ABC):
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

#
# class CoopGridworldTrajectory(GridworldTrajectoryTransform):
#     def get_collate_fn(self) -> Callable:
#         def collate_fn(batch):
#             past_trajectory_batch = pad_sequence([torch.tensor(batch[i][0]) for i in range(len(batch))],
#                                                  batch_first=True).float()
#             current_trajectory_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(len(batch))],
#                                                     batch_first=True).float()
#             state_batch = torch.from_numpy(np.concatenate([batch[i][2] for i in range(len(batch))])).float()
#             action_batch = torch.from_numpy(np.concatenate([batch[i][3] for i in range(len(batch))]))
#             goal_consumption_batch = torch.from_numpy(np.concatenate([batch[i][4] for i in range(len(batch))]))
#             srs_batch = torch.from_numpy(np.concatenate([batch[i][5] for i in range(len(batch))]))
#             hidden_state_indices = torch.tensor(batch[0][6], dtype=torch.bool)
#
#             modeller_input = GridworldToMnetInput(past_trajectory_batch, current_trajectory_batch, state_batch,
#                                                   hidden_state_indices)
#             true_output = AgentObjectives(action_batch, goal_consumption_batch, srs_batch)
#             return modeller_input, true_output
#         return collate_fn
#
#     def __call__(self, gridworld_trajectories: List[GridworldTrajectory], agent_idx: int, n_past: int,
#                  current_traj_len: int):
#         states = []
#         actions = []
#         goals = []
#         srs = []
#
#         reordered_gridworld_trajectories = []
#         for i in range(1, n_past+1):
#             reordered_gridworld_trajectories.append(gridworld_trajectories[i])
#         reordered_gridworld_trajectories.append(gridworld_trajectories[0])
#
#         past_trajectories = []
#         hidden_state_indices = []
#         for gridworld_idx in range(n_past):
#             trajectory = reordered_gridworld_trajectories[gridworld_idx]
#             past_trajectories.append(trajectory.get_spatialised_trajectory(agent_idx=agent_idx, add_start_token=True))
#             traj_hidden_state_indices = [False for _ in range(len(trajectory) + 1)]
#             traj_hidden_state_indices[0] = True
#             hidden_state_indices += traj_hidden_state_indices
#         past_trajectories.append(np.expand_dims(reordered_gridworld_trajectories[0].get_start_token(), 0))
#
#         past_trajectories = np.concatenate(past_trajectories, axis=0)
#         hidden_state_indices += [True]
#
#         current_gridworld = reordered_gridworld_trajectories[-1]
#         current_trajectory = current_gridworld.get_spatialised_trajectory(end_index=-1, agent_idx=agent_idx,
#                                                                           add_start_token=True)
#         for timestep in range(len(current_trajectory)):
#             states.append(current_gridworld.get_state(timestep, agent_idx=agent_idx))
#             actions.append(current_gridworld.get_action(timestep, agent_idx=agent_idx))
#             goals.append(self.get_goal_consumption(current_gridworld, timestep, agent_idx))
#             srs.append(self.get_srs(current_gridworld, timestep, agent_idx))
#
#         current_trajectory = np.tile(current_trajectory, (n_past + 1, 1, 1, 1))
#         states = np.tile(states, (n_past + 1, 1, 1, 1))
#         actions = np.tile(actions, n_past + 1)
#         goals = np.tile(goals, (n_past + 1, 1))
#         srs = np.tile(srs, (n_past + 1, 1, 1, 1))
#
#         return past_trajectories, current_trajectory, states, actions, goals, srs, hidden_state_indices


class CoopGridworldTrajectory(GridworldTrajectoryTransform):
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
        # actions = np.array(actions)
        # goals = np.array(goals)

        return past_trajectories, past_actions, current_trajectory, current_actions, actions, goals, hidden_state_indices


class PastCurrentStateSplit(GridworldTrajectoryTransform):
    def __init__(self, remove_actions=False, concat_past_traj=False, remove_other_agents=False, attach_agent_ids=False,
                 keep_goals=False):
        self.remove_actions = remove_actions
        self.concat_past_traj = concat_past_traj
        self.remove_other_agents = remove_other_agents
        self.attach_agent_ids = attach_agent_ids
        self.keep_goals = keep_goals

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            past_traj_batch = None
            if batch[0][0] is not None:
                past_traj_batch = pad_sequence(
                    [torch.tensor(batch[i // len(batch[0][0])][0][i % len(batch[0][0])]) for i in
                     range(len(batch) * len(batch[0][0]))]).float()
            current_traj_batch = None
            if batch[0][1] is not None:
                current_traj_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(len(batch))]).float()
            state_batch = torch.tensor([batch[i][2] for i in range(len(batch))]).float()
            action_batch = torch.tensor(np.array([batch[i][3] for i in range(len(batch))]))
            goal_consumption_batch = torch.tensor([batch[i][4] for i in range(len(batch))])
            srs_batch = torch.tensor([batch[i][5] for i in range(len(batch))])
            ia_features = torch.tensor(batch[0][6])
            modeller_input = TOMMASInput(past_traj_batch, current_traj_batch, state_batch, ia_features)
            true_output = AgentObjectives(action_batch, goal_consumption_batch, srs_batch)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[GridworldTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int):
        if self.keep_goals:
            for traj in gridworld_trajectories:
                traj.keep_goals = True

        past_trajectories = None
        if n_past > 0:
            past_trajectories = []
            for i in range(1, n_past + 1):
                past_trajectory = gridworld_trajectories[i]
                past_trajectories.append(past_trajectory.get_spatialised_trajectory(agent_idx=agent_idx,
                                                                                    attach_agent_ids=self.attach_agent_ids,
                                                                                    remove_actions=self.remove_actions,
                                                                                    remove_other_agents=self.remove_other_agents))
                if self.concat_past_traj:
                    past_trajectories = [np.concatenate(past_trajectories)]

        current_gridworld_trajectory = gridworld_trajectories[0]
        start_trajectory = None

        current_traj_len = GridworldTrajectoryTransform.get_current_traj_len(current_gridworld_trajectory, current_traj_len)
        if current_traj_len > 0:
            start_trajectory = current_gridworld_trajectory.get_spatialised_trajectory(end_index=current_traj_len,
                                                                                       agent_idx=agent_idx,
                                                                                       attach_agent_ids=self.attach_agent_ids,
                                                                                       remove_actions=self.remove_actions,
                                                                                       remove_other_agents=self.remove_other_agents)

        state, action = self.get_state_and_action(current_gridworld_trajectory, current_traj_len, agent_idx,
                                                  attach_agent_ids=self.attach_agent_ids,
                                                  remove_other_agents=self.remove_other_agents)
        goal_consumption = self.get_goal_consumption(current_gridworld_trajectory, current_traj_len, agent_idx)
        srs = self.get_srs(current_gridworld_trajectory, current_traj_len, agent_idx)
        ia_features = current_gridworld_trajectory.get_independent_agent_features(remove_actions=self.remove_actions)
        return past_trajectories, start_trajectory, state, action, goal_consumption, srs, ia_features


class FullTrajectory(GridworldTrajectoryTransform):
    def __init__(self, remove_actions=False, remove_other_agents=False, attach_agent_ids=False, keep_goals=False):
        self.remove_actions = remove_actions
        self.remove_other_agents = remove_other_agents
        self.attach_agent_ids = attach_agent_ids
        self.keep_goals = keep_goals

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            attention_mask_batch = pad_sequence([torch.tensor(batch[i][0][0]) for i in range(len(batch))],
                                                batch_first=True).float()
            embedding_positions = pad_sequence([torch.tensor(batch[i][0][1]) for i in range(len(batch))],
                                               batch_first=True).bool()
            trajectory_batch = pad_sequence([torch.tensor(batch[i][1][0]) for i in range(len(batch))],
                                            batch_first=True).float()
            state_batch = torch.from_numpy(np.concatenate([batch[i][2] for i in range(len(batch))])).float()
            action_batch = torch.from_numpy(np.concatenate([batch[i][3][0] for i in range(len(batch))]))
            batch_sequence_weighting = np.concatenate([batch[i][3][1] for i in range(len(batch))]) / len(batch)
            action_weighting = torch.from_numpy(batch_sequence_weighting).float()
            goal_consumption_batch = torch.from_numpy(np.concatenate([batch[i][4] for i in range(len(batch))]))
            num_goals = goal_consumption_batch.size(1)
            goal_weighting = torch.from_numpy(batch_sequence_weighting).float() / num_goals
            srs_batch = torch.from_numpy(np.concatenate([batch[i][5] for i in range(len(batch))]))
            srs_weighting = torch.from_numpy(batch_sequence_weighting).float()
            modeller_input = TOMMASTransformerInput(trajectory_batch, state_batch, attention_mask_batch,
                                                    embedding_positions)
            true_output = AgentWeightedObjectives(action_batch, action_weighting, goal_consumption_batch,
                                                  goal_weighting,
                                                  srs_batch, srs_weighting)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[GridworldTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int):
        if self.keep_goals:
            for traj in gridworld_trajectories:
                traj.keep_goals = True

        full_trajectory = []
        states = []
        actions = []
        goals = []
        srs = []

        reordered_gridworld_trajectories = []
        for i in range(1, n_past+1):
            reordered_gridworld_trajectories.append(gridworld_trajectories[i])
        reordered_gridworld_trajectories.append(gridworld_trajectories[0])

        embedding_positions = []  # where we get the embeddings for the current state
        for trajectory in reordered_gridworld_trajectories:
            full_trajectory.append(trajectory.get_spatialised_trajectory(agent_idx=agent_idx,
                                                                         attach_agent_ids=self.attach_agent_ids,
                                                                         remove_actions=self.remove_actions,
                                                                         remove_other_agents=self.remove_other_agents,
                                                                         add_start_token=True))
            for timestep in range(len(trajectory)):
                states.append(trajectory.get_state(timestep, agent_idx=agent_idx,
                                                   attach_agent_ids=self.attach_agent_ids,
                                                   remove_other_agents=self.remove_other_agents))
                actions.append(trajectory.get_action(timestep, agent_idx=agent_idx))
                goals.append(self.get_goal_consumption(trajectory, timestep, agent_idx))
                srs.append(self.get_srs(trajectory, timestep, agent_idx))
                embedding_positions.append(1)
            embedding_positions.append(0)  # We don't include last state for the past/cur embeddings

        full_trajectory = np.concatenate(full_trajectory, axis=0)
        attention_mask = [1 for _ in range(len(full_trajectory))]

        sequence_weighting = np.ones((len(actions))) / len(actions)
        return (attention_mask, embedding_positions), [full_trajectory], states, (actions, sequence_weighting), \
            goals, srs


class IterativeActionPastCurrentSplit(GridworldTrajectoryTransform):
    def __init__(self, concat_past_traj=False):
        self.concat_past_traj = concat_past_traj

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            past_traj_batch = None
            batch_size = len(batch)
            if batch[0][0] is not None:
                num_seq = len(batch[0][0])
                past_traj_batch = pad_sequence([torch.tensor(batch[i // num_seq][0][i % num_seq]) for i in
                                                range(len(batch) * num_seq)]).float()

            current_traj_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(batch_size)],
                                              padding_value=-1.).float()
            action_batch = torch.from_numpy(np.array([batch[i][2] for i in range(len(batch))]))
            ia_features = torch.tensor(batch[0][3])
            modeller_input = IterativeActionTOMMASInput(past_traj_batch, current_traj_batch, ia_features)
            true_output = ActionClassification(action_batch)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[IterativeActionTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int, shuffled_agent_indices=None):
        past_trajectories = None
        if n_past > 0:
            past_trajectories = []
            for i in range(1, n_past + 1):
                past_trajectory = np.array(gridworld_trajectories[i].trajectory, dtype=int)
                if agent_idx == 1:
                    past_trajectory = past_trajectory[:, [1, 0]]
                past_trajectories.append(past_trajectory)

                if self.concat_past_traj:
                    past_trajectories = [np.concatenate(past_trajectories)]

        current_action_trajectory = np.array(gridworld_trajectories[0].trajectory, dtype=int)
        if agent_idx == 1:
            current_action_trajectory = current_action_trajectory[:, [1, 0]]

        current_traj_len = IterativeActionPastCurrentSplit.get_current_traj_len(current_action_trajectory,
                                                                                current_traj_len)
        start_trajectory = current_action_trajectory[:current_traj_len]
        action = current_action_trajectory[current_traj_len][0]
        ia_features = [0]

        # Shuffle Opponents
        num_agents = len(start_trajectory[0])
        if num_agents > 2:
            if shuffled_agent_indices is None:
                shuffled_opponent_indices = np.arange(num_agents-1) + 1
                np.random.shuffle(shuffled_opponent_indices)
                shuffled_agent_indices = np.concatenate(([0], shuffled_opponent_indices))
            else:
                if len(shuffled_agent_indices) != num_agents:
                    raise ValueError(f"shuffled_opponent_indices should be length {num_agents}. "
                                     f"Current length is {len(shuffled_agent_indices)}")
            if past_trajectories is not None:
                for traj_idx, past_trajectory in enumerate(past_trajectories):
                    past_trajectories[traj_idx] = past_trajectory[:, shuffled_agent_indices]
            start_trajectory = start_trajectory[:, shuffled_agent_indices]

        return past_trajectories, start_trajectory, action, ia_features

    @staticmethod
    def get_current_traj_len(gridworld_trajectory: np.ndarray, current_traj_len: int):
        current_traj_len = min(len(gridworld_trajectory)-1, current_traj_len)
        if current_traj_len == 0:
            return 1
        if current_traj_len == -1:
            current_traj_len = np.random.randint(1, len(gridworld_trajectory)-1)
        return current_traj_len


class IterativeActionFullPastCurrentSplit(GridworldTrajectoryTransform):
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
            # past_trajectories.append(trajectory[:len(trajectory)-1])
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


class IterativeActionFullTrajectory(GridworldTrajectoryTransform):
    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            trajectory_batch = pad_sequence([torch.tensor(batch[i][0]) for i in range(len(batch))],
                                            batch_first=True).float()
            action_batch = torch.from_numpy(np.concatenate([batch[i][1] for i in range(len(batch))]))
            modeller_input = IterativeActionTransformerInput(trajectory_batch, None, None)
            true_output = ActionClassification(action_batch)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[IterativeActionTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int, shuffled_agent_indices=None):
        current_trajectory = []
        actions = []

        reordered_gridworld_trajectories = []
        for i in range(1, n_past+1):
            reordered_gridworld_trajectories.append(gridworld_trajectories[i])
        reordered_gridworld_trajectories.append(gridworld_trajectories[0])

        for i in range(0, n_past + 1):
            trajectory = np.array(reordered_gridworld_trajectories[i].trajectory, dtype=int)
            if agent_idx == 1:
                trajectory = trajectory[:, [1, 0]]
            current_trajectory.append(trajectory[:len(trajectory)-1])
            actions.append(trajectory[1:, 0])

        current_trajectory = np.concatenate(current_trajectory)
        actions = np.concatenate(actions)

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
            current_trajectory = current_trajectory[:, shuffled_agent_indices]

        return current_trajectory, actions


class TokenIDTrajectory(GridworldTrajectoryTransform):
    def get_collate_fn(self) -> Callable:
        def collate_fn(batch):
            trajectory_batch = pad_sequence([torch.tensor(batch[i][0]) for i in range(len(batch))],
                                            batch_first=True).long()
            action_batch = pad_sequence([torch.tensor(batch[i][1]) for i in range(len(batch))],
                                            batch_first=True).long()
            # action_batch = torch.from_numpy(np.concatenate([batch[i][1] for i in range(len(batch))]))
            modeller_input = IterativeActionTransformerInput(trajectory_batch, None, None)
            true_output = ActionClassification(action_batch)
            return modeller_input, true_output
        return collate_fn

    def __call__(self, gridworld_trajectories: List[IterativeActionTrajectory], agent_idx: int, n_past: int,
                 current_traj_len: int, shuffled_agent_indices=None):
        current_trajectory = []
        actions = []

        reordered_gridworld_trajectories = []
        for i in range(1, n_past+1):
            reordered_gridworld_trajectories.append(gridworld_trajectories[i])
        reordered_gridworld_trajectories.append(gridworld_trajectories[0])

        for i in range(0, n_past + 1):
            trajectory = np.array(reordered_gridworld_trajectories[i].trajectory, dtype=int)
            if agent_idx == 1:
                trajectory = trajectory[:, [1, 0]]
            current_trajectory.append(trajectory[:len(trajectory)-1])
            actions.append(trajectory[1:, 0])

        current_trajectory = np.concatenate(current_trajectory)
        actions = np.concatenate(actions)

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
            current_trajectory = current_trajectory[:, shuffled_agent_indices]

        seq_len, _ = current_trajectory.shape
        zeros = np.zeros((seq_len, 8 - num_agents), dtype=int)
        stack_traj = np.hstack([zeros, current_trajectory])
        current_trajectory = np.packbits(stack_traj, axis=1).flatten()
        individual_seq_len = seq_len // (n_past + 1)
        current_trajectory[[individual_seq_len*i for i in range(n_past+1)]] = (2**num_agents)
        new_actions = np.append([-1000], actions[:-1])
        return current_trajectory, new_actions
