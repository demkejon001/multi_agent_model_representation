from abc import ABC, abstractmethod
from typing import Dict, Union
import torch


class TensorClass(ABC):
    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def shape(self) -> Dict[str, torch.Size]:
        pass


class IterativeActionToMnetInput(TensorClass):
    def __init__(self, past_trajectories: torch.Tensor, current_trajectory: torch.Tensor,
                 hidden_state_indices: torch.Tensor):
        self.past_trajectories = past_trajectories
        self.current_trajectory = current_trajectory
        self.hidden_state_indices = hidden_state_indices

    def to(self, device):
        if self.past_trajectories is not None:
            self.past_trajectories = self.past_trajectories.to(device)
        self.current_trajectory = self.current_trajectory.to(device)
        self.hidden_state_indices = self.hidden_state_indices.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['past_trajectories'] = (None if self.past_trajectories is None else self.past_trajectories.shape)
        shape_dict['current_trajectory'] = (None if self.current_trajectory is None else self.current_trajectory.shape)
        shape_dict['hidden_state_indices'] = (None if self.hidden_state_indices is None else self.hidden_state_indices.shape)
        return shape_dict


class GridworldToMnetInput(TensorClass):
    def __init__(self, past_trajectories: torch.Tensor, past_actions: torch.Tensor, current_trajectory: torch.Tensor,
                 current_actions: torch.Tensor, hidden_state_indices: torch.Tensor):
        self.past_trajectories = past_trajectories
        self.past_actions = past_actions
        self.current_trajectory = current_trajectory
        self.current_actions = current_actions
        self.hidden_state_indices = hidden_state_indices

    def to(self, device):
        self.past_trajectories = self.past_trajectories.to(device)
        self.past_actions = self.past_actions.to(device)
        self.current_trajectory = self.current_trajectory.to(device)
        self.current_actions = self.current_actions.to(device)
        self.hidden_state_indices = self.hidden_state_indices.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['past_trajectories'] = (None if self.past_trajectories is None else self.past_trajectories.shape)
        shape_dict['past_actions'] = (None if self.past_actions is None else self.past_actions.shape)
        shape_dict['current_trajectory'] = (None if self.current_trajectory is None else self.current_trajectory.shape)
        shape_dict['current_actions'] = (None if self.current_actions is None else self.current_actions.shape)
        shape_dict['hidden_state_indices'] = (
            None if self.hidden_state_indices is None else self.hidden_state_indices.shape)
        return shape_dict
