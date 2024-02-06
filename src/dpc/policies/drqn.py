import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from torch._tensor import Tensor
from torch.optim import Adam

from ..collect.collector import Collector
from ..collect.state import StateDict
from ..policy import Policy
from ..utils.encode import IMAGE_TYPE, Encoder


class DRQN(Policy):
    def __init__(
        self,
        collector: Collector,
        batch_size: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.95,
        tau: float = 5e-3,
        update_target: int = None,
        image_type: IMAGE_TYPE = "rgb",
        # Sequence length isn't the same as stacks from DQN !
        sequence_length: int = 4,
        collect=True,
    ):
        super().__init__(collector, stack_type="mask")

        self.lr = self.register_state("lr", lr)
        self.gamma = self.register_state("gamma", gamma)
        self.tau = self.register_state("tau", tau)
        self.batch_size = self.register_state("batch_size", batch_size)
        self.update_target = self.register_state(
            "update_target", update_target or batch_size
        )
        self.stacks = self.register_state("stacks", sequence_length)
        self.image_type = self.register_state("image_type", image_type)
        self.collect = collect

        def _encoder():
            return Encoder(
                self.in_state,
                self.out_actions,
                last_layer="lstm",
                image_type=image_type,
                stacks=self.stacks,
            )

        self.network = _encoder()
        self.target_network = _encoder()

        self._train_steps = self.register_state("_train_steps", 0)
        self._network_syncs = self.register_state("_network_syncs", 0)

        # Adam optimizers
        self._optim = Adam(self.network.parameters(), lr=self.lr)

    def setup(self):
        self.collector.set_mode("keep")
        self.collector.fill(1000)

    def preprocess_obs(self, state: torch.Tensor, stacks=None):
        if state.ndim < 3:
            assert stacks == 1, "Stacks must be 1 if state is 1D"
            return state
        elif state.ndim == 3:
            state = (state.permute(-1, -3, -2).float() / 255.0).unsqueeze(dim=0)
        else:
            state = state.permute(0, -1, -3, -2).float() / 255.0

        if state.shape[1] == 3 and self.image_type == "grayscale":
            state = TF.rgb_to_grayscale(state)
        if stacks is None:
            return state.unsqueeze(dim=0)
        return rearrange(state, "(b s) c h w -> b s c h w", s=self.stacks)

    def preprocess(self, batch: StateDict):
        # (B*S, C, H, W)
        obs, actions, rewards, next_states, dones, masks = batch.values()

        # Correct channel layout and normalized
        # (B, S, C, H, W)
        obs = self.preprocess_obs(obs, stacks=self.stacks)
        next_states = self.preprocess_obs(next_states, stacks=self.stacks)

        # Make (B, S, N)
        pattern = "(b s) n -> b s n"
        actions = rearrange(actions, pattern, s=self.stacks)
        pattern = "(b s) -> b s"
        rewards = rearrange(rewards, pattern, s=self.stacks)
        dones = rearrange(dones, pattern, s=self.stacks)
        masks = rearrange(masks, pattern, s=self.stacks)

        return obs, actions, rewards, next_states, dones, masks

    def act(self, obs):
        x = self.batch_act(obs.unsqueeze(dim=0))
        return x.squeeze(dim=0).detach()

    def batch_act(self, obs) -> torch.Tensor:
        # Obs should be stacked (B*S, C, H, W)
        obs = self.preprocess_obs(obs)
        x = self.network(obs.to(self.device))
        x = x[-1]  # Get the last hidden state actions
        return x.softmax(dim=-1)

    def step(self):
        # Sample batch_size steps inside an episode
        # We can't be sure to get a full batch_size of steps !
        return self.collector.sample(
            self.batch_size,
            self.device,
            stacks=self.stacks,
            rand_type="episode",
        )

    def forward(self, batch) -> Tensor:
        obs, actions, rewards, next_states, dones, masks = self.preprocess(batch)

        # Get chosen actions at states
        chosen_actions = actions.argmax(dim=-1)

        # Calculate Q values for state / next state
        Q = (
            self.network(obs, masks)
            .gather(1, chosen_actions.unsqueeze(dim=-1))
            .squeeze(dim=-1)
        )

        # The next rewards are not learnable, they are our targets
        with torch.no_grad():
            Q_next = self.target_network(next_states, masks)
            # Long term reward function
            expected_Q = rewards + self.gamma * Q_next.max(dim=-1).values * (1 - dones)
            self.logger.add_scalar(
                "policy/expected_Q", expected_Q.mean(), self.global_step
            )

        td_err = Q - expected_Q.detach()
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

        return td_err
