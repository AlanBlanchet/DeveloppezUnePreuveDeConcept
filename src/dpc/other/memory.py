from collections import deque, namedtuple

import numpy as np
import torch

from dpc.other.config import config

Transition = namedtuple(
    "Transition",
    ("state", "next_state", "action", "reward", "mask", "step", "rnn_state"),
)


class LocalBuffer(object):
    def __init__(self):
        self.n_step_memory = []
        self.local_memory = []
        self.memory = []
        self.over_lapping_from_prev = []

    def push(self, state, next_state, action, reward, mask, rnn_state):
        state = state.cpu()
        next_state = next_state.cpu()
        reward = reward
        mask = mask
        rnn_state = rnn_state

        self.n_step_memory.append([state, next_state, action, reward, mask, rnn_state])
        if len(self.n_step_memory) == config.n_step or mask == 0:
            [state, _, action, _, _, rnn_state] = self.n_step_memory[0]
            [_, next_state, _, _, mask, _] = self.n_step_memory[-1]

            sum_reward = 0
            for t in reversed(range(len(self.n_step_memory))):
                [_, _, _, reward, _, _] = self.n_step_memory[t]
                sum_reward += reward + config.gamma * sum_reward
            reward = sum_reward
            step = len(self.n_step_memory)
            self.push_local_memory(
                state, next_state, action, reward, mask, step, rnn_state
            )
            self.n_step_memory = []

    def push_local_memory(
        self, state, next_state, action, reward, mask, step, rnn_state
    ):
        self.local_memory.append(
            Transition(
                state,
                next_state,
                action,
                reward,
                mask,
                step,
                torch.stack(rnn_state).view(2, -1).cpu(),
            )
        )
        if (
            len(self.local_memory) + len(self.over_lapping_from_prev)
        ) == config.sequence_length or mask == 0:
            self.local_memory = self.over_lapping_from_prev + self.local_memory
            length = len(self.local_memory)
            while len(self.local_memory) < config.sequence_length:
                self.local_memory.append(
                    Transition(
                        torch.Tensor([0, 0]),
                        torch.Tensor([0, 0]),
                        0,
                        0,
                        0,
                        0,
                        torch.zeros([2, 1, config.hidden_size]).view(2, -1),
                    )
                )
            self.memory.append([self.local_memory, length])
            if mask == 0:
                self.over_lapping_from_prev = []
            else:
                self.over_lapping_from_prev = self.local_memory[
                    len(self.local_memory) - config.over_lapping_length :
                ]
            self.local_memory = []

    def sample(self):
        episodes = self.memory
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_mask,
            batch_step,
            batch_rnn_state,
        ) = ([], [], [], [], [], [], [])
        lengths = []
        for episode, length in episodes:
            batch = Transition(*zip(*episode))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_mask.append(torch.Tensor(list(batch.mask)))
            batch_step.append(torch.Tensor(list(batch.step)))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))

            lengths.append(length)
        self.memory = []
        return (
            Transition(
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
                batch_mask,
                batch_step,
                batch_rnn_state,
            ),
            lengths,
        )


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)

    def td_error_to_priority(self, td_error, lengths):
        abs_td_error_sum = (
            td_error.abs().sum(dim=1, keepdim=True).view(-1).detach().cpu().numpy()
        )
        lengths_burn = [length - config.burn_in_length + 1 for length in lengths]

        prior_max = (
            td_error.abs().max(dim=1, keepdim=True)[0].view(-1).detach().cpu().numpy()
        )

        prior_mean = abs_td_error_sum / lengths_burn
        prior = config.eta * prior_max + (1 - config.eta) * prior_mean
        return prior

    def push(self, td_error, batch, lengths):
        # batch.state[local_mini_batch, config.sequence_length, item]
        prior = self.td_error_to_priority(td_error, lengths)

        for i in range(len(batch)):
            self.memory.append(
                [
                    Transition(
                        batch.state[i].cpu(),
                        batch.next_state[i].cpu(),
                        batch.action[i].cpu(),
                        batch.reward[i].cpu(),
                        batch.mask[i].cpu(),
                        batch.step[i].cpu(),
                        batch.rnn_state[i].cpu(),
                    ),
                    lengths[i],
                ]
            )
            self.memory_probability.append(prior[i])

    def sample(self, batch_size):
        probability = np.array(self.memory_probability)
        probability = probability / probability.sum()

        indexes = np.random.choice(
            range(len(self.memory_probability)), batch_size, p=probability
        )
        episodes = [self.memory[idx][0] for idx in indexes]
        lengths = [self.memory[idx][1] for idx in indexes]

        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_mask,
            batch_step,
            batch_rnn_state,
        ) = ([], [], [], [], [], [], [])
        for episode in episodes:
            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)
            batch_mask.append(episode.mask)
            batch_step.append(episode.step)
            batch_rnn_state.append(episode.rnn_state)

        return (
            Transition(
                batch_state,
                batch_next_state,
                batch_action,
                batch_reward,
                batch_mask,
                batch_step,
                batch_rnn_state,
            ),
            indexes,
            lengths,
        )

    def update_priority(self, indexes, td_error, lengths):
        prior = self.td_error_to_priority(td_error, lengths)
        priors_idx = 0
        for idx in indexes:
            self.memory_probability[idx] = prior[priors_idx]
            priors_idx += 1

    def __len__(self):
        return len(self.memory)
