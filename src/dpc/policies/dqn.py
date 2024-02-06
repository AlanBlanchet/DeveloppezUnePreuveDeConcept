from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torch._tensor import Tensor
from torch.optim import Adam

from ..collect.collector import Collector
from ..policy import Policy
from ..utils.encode import IMAGE_TYPE, Encoder


class DQN(Policy):
    def __init__(
        self,
        collector: Collector,
        batch_size: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.95,
        tau: float = 5e-3,
        update_target: int = None,
        stacks=4,
        image_type: IMAGE_TYPE = "rgb",
    ):
        super().__init__(collector, stack_type="mask")

        self.lr = self.register_state("lr", lr)
        self.gamma = self.register_state("gamma", gamma)
        self.tau = self.register_state("tau", tau)
        self.batch_size = self.register_state("batch_size", batch_size)
        self.update_target = self.register_state(
            "update_target", update_target or batch_size
        )
        self.stacks = self.register_state("stacks", stacks)
        self.image_type = self.register_state("image_type", image_type)

        self.network = Encoder(
            self.in_state, self.out_actions, stacks=stacks, image_type=image_type
        )

        self.target_network = deepcopy(self.network)

        self._train_steps = self.register_state("_train_steps", 0)
        self._network_syncs = self.register_state("_network_syncs", 0)

        # Adam optimizers
        self._optim = Adam(self.network.parameters(), lr=self.lr)

    def setup(self):
        self.collector.set_mode("keep")
        self.collector.fill(400)

    def preprocess_state(self, state: torch.Tensor):
        if state.ndim < 3:
            assert self.stacks == 1, "Stacks must be 1 if state is 1D"
            return state
        else:
            state = state.permute(0, -1, -3, -2).float() / 255.0

        if state.shape[1] == 3 and self.image_type == "grayscale":
            state = TF.rgb_to_grayscale(state)
        if state.shape[0] == 1:
            return state
        return rearrange(state, "(b s) c h w -> b (s c) h w", s=self.stacks)

    def preprocess(self, batch: list[torch.Tensor]):
        # (B * S, C, H, W)
        states, actions, rewards, next_states, dones = batch

        # Correct channel layout and normalized
        # (B, S * C, H, W)
        states = self.preprocess_state(states)
        next_states = self.preprocess_state(next_states)

        # All we care about is the current taken action/rewards/dones
        s = self.stacks

        # (B, ...)
        actions = actions[s - 1 :: s]
        rewards = rewards[s - 1 :: s]
        dones = dones[s - 1 :: s]

        return states, actions, rewards, next_states, dones

    def act(self, state):
        # State should be stacked (B*S, C, H, W)
        return self.batch_act(state).squeeze(dim=0).detach()

    def batch_act(self, states) -> torch.Tensor:
        states = self.preprocess_state(states)
        return self.network(states.to(self.device)).softmax(dim=-1)

    def step(self):
        if self.training:
            return self.collector.sample(
                self.batch_size, self.device, stacks=self.stacks
            )
        else:
            return self.collector.sample(1, self.device, stacks=self.stacks)

    def forward(self, batch) -> Tensor:
        states, actions, rewards, next_states, dones = self.preprocess(batch)

        # Get chosen actions at states
        chosen_actions = actions.argmax(dim=-1)

        # Calculate Q values for state / next state
        Q = (
            self.network(states)
            .gather(1, chosen_actions.unsqueeze(dim=-1))
            .squeeze(dim=-1)
        )

        # The next rewards are not learnable, they are our targets
        with torch.no_grad():
            Q_next = self.target_network(next_states)
            # Long term reward function
            expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (1 - dones)
            self.logger.add_scalar(
                "policy/expected_Q", expected_Q.mean(), self.global_step
            )

        loss = F.smooth_l1_loss(Q, expected_Q)

        self._optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self._optim.step()

        self.logger.add_scalar("policy/loss", loss, self.global_step)

        self._train_steps += self.batch_size

        # Update target network with current network
        # Prevents network to follow a moving target
        if self._train_steps >= self.update_target:
            # We do a soft update to prevent sudden changes
            state_dict = self.network.state_dict()
            target_state_dict = self.target_network.state_dict()
            for key in state_dict.keys():
                target_state_dict[key] = (
                    self.tau * state_dict[key] + (1 - self.tau) * target_state_dict[key]
                )
            self.target_network.load_state_dict(target_state_dict)
            self._train_steps = 0
            self._network_syncs += 1
            self.logger.add_scalar(
                "policy/network_sync", self._network_syncs, self.global_step
            )

        self.collector.collect_steps(self.batch_size // 4)
