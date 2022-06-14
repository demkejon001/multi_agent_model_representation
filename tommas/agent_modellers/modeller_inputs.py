from abc import ABC, abstractmethod
from typing import Dict, Union, Optional
import torch


class TensorClass(ABC):
    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def shape(self) -> Dict[str, torch.Size]:
        pass


class TOMMASInput(TensorClass):
    def __init__(self, past_trajectories: torch.Tensor, current_trajectory: torch.Tensor, query_state: torch.Tensor,
                 independent_agent_features: torch.Tensor):
        self.past_trajectories = past_trajectories
        self.current_trajectory = current_trajectory
        self.query_state = query_state
        self.independent_agent_features = independent_agent_features

    def to(self, device):
        if self.past_trajectories is not None:
            self.past_trajectories = self.past_trajectories.to(device)
        if self.current_trajectory is not None:
            self.current_trajectory = self.current_trajectory.to(device)
        self.query_state = self.query_state.to(device)
        self.independent_agent_features = self.independent_agent_features.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['past_trajectories'] = (None if self.past_trajectories is None else self.past_trajectories.shape)
        shape_dict['current_trajectory'] = (None if self.current_trajectory is None else self.current_trajectory.shape)
        shape_dict['query_state'] = self.query_state.shape
        shape_dict['independent_agent_features'] = self.independent_agent_features.shape
        return shape_dict


class TOMMASTransformerInput(TensorClass):
    # current_trajectory: [seq length, batch, num_features, gridworld rows, gridworld columns]
    # query_state: [batch, num_features_without_actions, gridworld rows, gridworld cols]
    # attention_mask: [batch, num_sequences]
    # embedding_positions: [batch, num_sequences]
    def __init__(self, trajectory: torch.Tensor, query_state: torch.Tensor, attention_mask: torch.Tensor,
                 embedding_positions: torch.Tensor):
        self.trajectory = trajectory
        self.query_state = query_state
        self.attention_mask = attention_mask
        self.embedding_positions = embedding_positions

    def to(self, device):
        self.trajectory = self.trajectory.to(device)
        self.query_state = self.query_state.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.embedding_positions = self.embedding_positions.to(device)

    def shape(self) -> Dict[str, torch.Size]:
        return {
            'trajectory': self.trajectory.shape,
            'query_state': self.query_state.shape,
            'attention_mask': self.attention_mask.shape,
            'embedding_positions': self.embedding_positions.shape
        }


class IterativeActionTOMMASInput(TensorClass):
    def __init__(self, past_trajectories: torch.Tensor, current_trajectory: torch.Tensor,
                 independent_agent_features: torch.Tensor):
        self.past_trajectories = past_trajectories
        self.current_trajectory = current_trajectory
        self.independent_agent_features = independent_agent_features

    def to(self, device):
        if self.past_trajectories is not None:
            self.past_trajectories = self.past_trajectories.to(device)
        self.current_trajectory = self.current_trajectory.to(device)
        self.independent_agent_features = self.independent_agent_features.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['past_trajectories'] = (None if self.past_trajectories is None else self.past_trajectories.shape)
        shape_dict['current_trajectory'] = (None if self.current_trajectory is None else self.current_trajectory.shape)
        shape_dict['independent_agent_features'] = self.independent_agent_features.shape
        return shape_dict


class IterativeActionTransformerInput(TensorClass):
    # trajectory: [seq length, batch, num_features, gridworld rows, gridworld columns]
    def __init__(self, trajectory: torch.Tensor, attention_mask: Optional[torch.Tensor],
                 embedding_positions: Optional[torch.Tensor]):
        self.trajectory = trajectory
        self.attention_mask = attention_mask
        self.embedding_positions = embedding_positions

    def to(self, device):
        self.trajectory = self.trajectory.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.embedding_positions is not None:
            self.embedding_positions = self.embedding_positions.to(device)

    def shape(self) -> Dict[str, torch.Size]:
        return {
            'trajectory': self.trajectory.shape,
            'attention_mask': (None if self.attention_mask is None else self.attention_mask.shape),
            'embedding_positions': (None if self.embedding_positions is None else self.embedding_positions.shape),
        }


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
