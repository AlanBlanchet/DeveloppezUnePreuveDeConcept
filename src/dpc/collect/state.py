from __future__ import annotations

from typing import Literal, get_args

import torch
from torch._tensor import Tensor

from ..utils.func import parse_tensor

DEFAULT_KEYS_TYPE = Literal["obs", "action", "reward", "next_state", "done"]
DEFAULT_KEYS = get_args(DEFAULT_KEYS_TYPE)


class StateDict(dict[DEFAULT_KEYS_TYPE, torch.Tensor | list]):
    def __init__(self, L=[], device="cpu", **kwargs):
        super().__init__(**kwargs)

        self.device = device

        if not isinstance(L, list):
            L = self._parse_list(L)

        if len(L) > 0:
            self.add_defaults(L)

    def __setitem__(self, __key: str, __value: Tensor | list) -> None:
        return super().__setitem__(__key, __value)

    def _parse_list(self, L: list):
        L = list(L)
        assert len(L) <= len(
            DEFAULT_KEYS
        ), f"Must have the same length as {DEFAULT_KEYS}"
        return L

    def add_defaults(self, t: list):
        t = self._parse_list(t)
        for key, v in zip(DEFAULT_KEYS, t):
            self.__setitem__(key, v)
        self._to_tensors()

    def get_defaults(self):
        return [self.__getitem__(key) for key in DEFAULT_KEYS]

    def episode_split(self):
        """
        Splits the current state into a list of states each containing terminated episodes and padding
        Creates a mask to be used for the loss function
        """
        states: list[StateDict] = []
        # Check if mask is already present. In that case the split has already been done
        if "mask" in self.keys():
            states.append(self)
            return states

        dones = self["done"]
        n = len(dones)
        dones_idx = (dones == 1).nonzero().flatten()

        if len(dones_idx) == 0:
            mask = torch.ones(n)
            self["mask"] = mask
            states.append(self)
            return states

        # Split each dones
        dones_idx += 1
        if dones_idx[-1] != n:
            dones_idx = torch.cat([dones_idx, torch.tensor([n])])
        dones_idx = torch.cat([torch.tensor([0]), dones_idx])
        dones_sizes = [
            len(dones[dones_idx[i] : dones_idx[i + 1]])
            for i in range(len(dones_idx) - 1)
        ]

        if len(dones_sizes) == 1:
            mask = torch.ones(n)
            self["mask"] = mask
            states.append(self)
            return states

        splitted_states = [torch.split(v, dones_sizes) for v in self.values()]

        for item_states_tuple in zip(*splitted_states):
            # One mask for every splits
            mask = torch.zeros(n)
            split_len = len(item_states_tuple[0])
            new_state_list = []
            for item in item_states_tuple:
                padded = torch.zeros(n, *item.shape[1:])
                padded[:split_len] = item
                new_state_list.append(padded)
            mask[:split_len] = 1
            new_state = StateDict()
            new_state.add_defaults(new_state_list)
            new_state["mask"] = mask
            new_state._to_tensors()
            states.append(new_state)

        return states

    @staticmethod
    def batch(states: list[StateDict]):
        """
        Batch a list of states into a single state
        """
        batched_state = StateDict()

        for key in states[0].keys():
            # batched_state[key] = torch.stack([state[key] for state in states])
            batched_state[key] = torch.cat([state[key] for state in states])
        return batched_state

    def _to_tensors(self):
        for k in self.keys():
            if not torch.is_tensor(self[k]):
                self[k] = parse_tensor(self[k], device=self.device)
            else:
                self[k] = self[k].to(self.device)
        return self

    def set_device(self, device: str):
        self.device = device
        self._to_tensors()
        return self

    def __repr__(self) -> str:
        vals = ",".join([f"{k}={self[k].shape}" for k in self.keys()])
        return f"StateDict({vals})"
