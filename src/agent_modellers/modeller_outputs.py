from typing import Union, List, Tuple, Optional, Dict
import torch
from src.agent_modellers.modeller_inputs import TensorClass


class AgentObjectives(TensorClass):
    def __init__(self,
                 action: Optional[torch.Tensor] = None,
                 goal_consumption: Optional[torch.Tensor] = None,
                 successor_representation: Optional[torch.Tensor] = None):
        self.action = action
        self.goal_consumption = goal_consumption
        self.successor_representation = successor_representation

    def to(self, device):
        if self.action is not None:
            self.action = self.action.to(device)
        if self.goal_consumption is not None:
            self.goal_consumption = self.goal_consumption.to(device)
        if self.successor_representation is not None:
            self.successor_representation = self.successor_representation.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['action'] = torch.Size([-1, 5])
        shape_dict['goal_consumption'] = (None if self.goal_consumption is None else self.goal_consumption.shape)
        shape_dict['successor_representation'] = (None if self.successor_representation is None else self.successor_representation.shape)
        return shape_dict


class ToMnetPredictions(AgentObjectives):
    def __init__(self,
                 action: torch.Tensor,
                 goal_consumption: torch.Tensor,
                 successor_representation: torch.Tensor,
                 embeddings: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], None] = None):
        super().__init__(action, goal_consumption, successor_representation)
        self.embeddings = embeddings

    def to(self, device):
        super().to(device)
        if isinstance(self.embeddings, torch.Tensor):
            self.embeddings = self.embeddings.to(device)
        else:
            for i in range(len(self.embeddings)):
                self.embeddings[i] = self.embeddings[i].to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = super().shape()
        if isinstance(self.embeddings, torch.Tensor):
            shape_dict['embeddings'] = self.embeddings.shape
        else:
            embeddings_shape = []
            for i in range(len(self.embeddings)):
                embeddings_shape.append(self.embeddings[i].shape)
            shape_dict['embeddings'] = embeddings_shape
        return shape_dict


class ActionClassification(TensorClass):
    def __init__(self,
                 action: torch.Tensor):
        self.action = action

    def to(self, device):
        self.action = self.action.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = dict()
        shape_dict['action'] = torch.Size([-1, 2])
        return shape_dict


class IterativeActionToMnetPredictions(ActionClassification):
    def __init__(self,
                 action: torch.Tensor,
                 embeddings: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], None] = None,
                 attentions: Optional[Tuple[torch.FloatTensor]] = None,
                 hidden_states: Optional[Tuple[torch.FloatTensor]] = None):
        super().__init__(action)
        self.embeddings = embeddings
        self.attentions = attentions
        self.hidden_states = hidden_states

    def to(self, device):
        super().to(device)
        if isinstance(self.embeddings, torch.Tensor):
            self.embeddings = self.embeddings.to(device)
        else:
            for i in range(len(self.embeddings)):
                self.embeddings[i] = self.embeddings[i].to(device)
        if self.attentions is not None:
            self.attentions = self.attentions.to(device)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device)

    def shape(self) -> Dict[str, Union[torch.Size, None]]:
        shape_dict = super().shape()
        if isinstance(self.embeddings, torch.Tensor):
            shape_dict['embeddings'] = self.embeddings.shape
        else:
            embeddings_shape = []
            for i in range(len(self.embeddings)):
                embeddings_shape.append(self.embeddings[i].shape)
            shape_dict['embeddings'] = embeddings_shape
        if self.attentions is not None:
            shape_dict["attentions"] = self.attentions.shape
        if self.hidden_states is not None:
            shape_dict["hidden_states"] = self.hidden_states.shape
        return shape_dict
