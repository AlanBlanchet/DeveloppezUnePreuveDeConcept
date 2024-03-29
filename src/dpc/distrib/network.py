"""
Implementation of the Recurrent Replay Distributed DQN policy.

Paper: https://openreview.net/pdf?id=r1lyTjAqYX
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..policy import Policy


class R2D2(Policy):
    def __init__(self, action_dim, obs_shape=10, hidden_dim=512):
        super().__init__()

        # 84 x 84 input

        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        self.max_forward_steps = 5

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(True),
        )

        self.recurrent = nn.LSTM(
            512 + self.action_dim + 1, self.hidden_dim, batch_first=True
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, episodes):
        latent = self.feature(state.obs / 255)

        recurrent_input = torch.cat(
            (latent, state.last_action, state.last_reward), dim=1
        )

        _, recurrent_output = self.recurrent(recurrent_input, state.hidden_state)

        hidden = recurrent_output[0]

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value, recurrent_output

    def calculate_q_(
        self,
        obs,
        last_action,
        last_reward,
        hidden_state,
        burn_in_steps,
        learning_steps,
        forward_steps,
    ):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.in_state)
        last_action = last_action.view(-1, self.in_action)
        last_reward = last_reward.view(-1, 1)
        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps + forward_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)

        recurrent_input = pack_padded_sequence(
            recurrent_input, seq_len, batch_first=True, enforce_sorted=False
        )

        self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        seq_start_idx = burn_in_steps + self.max_forward_steps
        forward_pad_steps = torch.minimum(
            self.max_forward_steps - forward_steps, learning_steps
        )

        hidden = []
        for hidden_seq, start_idx, end_idx, padding_length in zip(
            recurrent_output, seq_start_idx, seq_len, forward_pad_steps
        ):
            hidden.append(hidden_seq[start_idx:end_idx])
            if padding_length > 0:
                hidden.append(
                    hidden_seq[end_idx - 1 : end_idx].repeat(padding_length, 1)
                )

        hidden = torch.cat(hidden)

        assert hidden.size(0) == torch.sum(learning_steps)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value

    def calculate_q(
        self, obs, last_action, last_reward, hidden_state, burn_in_steps, learning_steps
    ):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.in_state)
        last_action = last_action.view(-1, self.in_action)
        last_reward = last_reward.view(-1, 1)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(
            recurrent_input, seq_len, batch_first=True, enforce_sorted=False
        )

        # self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        hidden = torch.cat(
            [
                output[burn_in : burn_in + learning]
                for output, burn_in, learning in zip(
                    recurrent_output, burn_in_steps, learning_steps
                )
            ],
            dim=0,
        )

        adv = self.advantage(hidden)
        val = self.value(hidden)

        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value
