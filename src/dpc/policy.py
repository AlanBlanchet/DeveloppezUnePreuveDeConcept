from abc import abstractmethod
from typing import Any, Literal, Mapping, TypeVar

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from .collect.collector import Collector, DistributedCollector
from .collect.state import StateDict

T = TypeVar("T")
STACK_TYPE = Literal["frame", "mask"]


class Policy(nn.Module):
    def __init__(
        self,
        collector: Collector | DistributedCollector,
        stack_type: STACK_TYPE = None,
    ):
        super().__init__()
        self.collector = collector
        self.logger: SummaryWriter = None
        self.in_state = self.collector.state_shape
        self.out_actions = self.collector.out_action
        self.stack_type = stack_type
        self.states = {}

    def setup(self):
        ...

    @abstractmethod
    def act(self, obs: torch.Tensor):
        """
        Base act method for RL algorithms.

        ## Parameters:
        - `obs` (any): The current observation of the environment.

        ## Returns:
        - `action` (any): The action taken in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Perform batch action selection based on the given states.

        ## Parameters:
        - `obs` (`torch.Tensor`): The observations for which to select actions.

        ## Returns:
        - `actions` (`torch.Tensor`): The selected actions for each observation.
        """
        ...

    @abstractmethod
    def forward(self, batch: StateDict) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self) -> StateDict:
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def global_step(self):
        return self.collector._n_steps

    def set_logger(self, logger):
        self.logger = logger

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update(self.states)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [
            setattr(self, k, state_dict.pop(k))
            for k in self.states.keys()
            if k in state_dict
        ]
        super().load_state_dict(state_dict, strict)

    def register_state(self, name: str, variable: T):
        self.states.update({name: variable})
        return variable


class DistributedPolicy(nn.Module):
    def __init__(
        self,
        collector: Collector,
        stack_type: STACK_TYPE = None,
    ):
        super().__init__()
        self.collector = collector
        self.logger: SummaryWriter = None
        self.in_state = self.collector.state_shape
        self.out_actions = self.collector.out_action
        self.stack_type = stack_type
        self.states = {}

    def setup(self):
        ...

    @abstractmethod
    def act(self, obs: torch.Tensor):
        """
        Base act method for RL algorithms.

        ## Parameters:
        - `obs` (any): The current observation of the environment.

        ## Returns:
        - `action` (any): The action taken in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Perform batch action selection based on the given states.

        ## Parameters:
        - `obs` (`torch.Tensor`): The observations for which to select actions.

        ## Returns:
        - `actions` (`torch.Tensor`): The selected actions for each observation.
        """
        ...

    @abstractmethod
    def forward(self, batch: StateDict) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self) -> StateDict:
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def global_step(self):
        return self.collector._n_steps

    def set_logger(self, logger):
        self.logger = logger

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update(self.states)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [
            setattr(self, k, state_dict.pop(k))
            for k in self.states.keys()
            if k in state_dict
        ]
        super().load_state_dict(state_dict, strict)

    def register_state(self, name: str, variable: T):
        self.states.update({name: variable})
        return variable
