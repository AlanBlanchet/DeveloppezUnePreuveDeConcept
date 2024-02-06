import gymnasium
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from gymnasium import Wrapper
from tensorboardX import SummaryWriter

from dpc.other.config import config
from dpc.other.embedding_model import EmbeddingModel, compute_intrinsic_reward
from dpc.other.memory import LocalBuffer, Memory
from dpc.other.model import R2D2
from dpc.utils.func import random_run_name


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


class Montezuma(Wrapper):
    def step(self, action: int):
        obs, rew, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        # Only used for counting thankfully
        self.set.add(hash(obs.data.tobytes()))
        if rew > 0:
            rew = 10
        return obs / 10, rew, done, info

    def reset(self):
        self.set = set()
        return super().reset()


def format_state(state, device=None):
    if device is None:
        state = torch.Tensor(state)
    else:
        state = torch.Tensor(state).to(device)
    state = state.permute(2, 0, 1)
    state = TF.rgb_to_grayscale(state)
    return state


def main():
    env = Montezuma(gymnasium.make("ALE/MontezumaRevenge-v5"))

    name = random_run_name()
    logger = SummaryWriter(f"runs/{name}")

    torch.manual_seed(config.random_seed)
    env.seed(config.random_seed)
    np.random.seed(config.random_seed)
    env.action_space.seed(config.random_seed)

    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
    update_target_model(online_net, target_net)
    embedding_model = EmbeddingModel(obs_size=num_inputs, num_outputs=num_actions)
    embedding_loss = 0

    optimizer = optim.Adam(online_net.parameters(), lr=config.lr)

    online_net.to(config.device)
    target_net.to(config.device)
    embedding_model.to(config.device)
    online_net.train()
    target_net.train()
    memory = Memory(config.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer()
    sum_reward = 0
    sum_augmented_reward = 0
    sum_obs_set = 0
    video_buffer = []

    for episode in range(30000):
        can_log = episode > 0 and episode % config.log_interval == 0
        done = False
        state, _ = env.reset()

        if can_log:
            video_buffer.append(state)

        state = format_state(state, config.device)

        hidden = (
            torch.Tensor().new_zeros(1, 1, config.hidden_size),
            torch.Tensor().new_zeros(1, 1, config.hidden_size),
        )

        episodic_memory = [embedding_model.embedding(state.unsqueeze(0)).cpu()]

        episode_steps = 0
        horizon = 100
        while not done:
            steps += 1
            episode_steps += 1

            action, new_hidden = get_action(
                state.to(config.device), target_net, epsilon, env, hidden
            )

            next_state, env_reward, done, _ = env.step(action)
            if can_log:
                video_buffer.append(next_state)

            next_state = torch.Tensor(next_state)
            next_state = format_state(next_state).to(config.device)

            augmented_reward = env_reward
            if config.enable_ngu:
                next_state_emb = embedding_model.embedding(
                    next_state.unsqueeze(dim=0)
                ).cpu()
                intrinsic_reward = compute_intrinsic_reward(
                    episodic_memory, next_state_emb
                )
                episodic_memory.append(next_state_emb)
                beta = 0.1
                augmented_reward = env_reward + beta * intrinsic_reward

            mask = 0 if done else 1

            local_buffer.push(state, next_state, action, augmented_reward, mask, hidden)
            hidden = new_hidden
            if len(local_buffer.memory) == config.local_mini_batch:
                batch, lengths = local_buffer.sample()
                td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)

            sum_reward += env_reward
            state = next_state
            sum_augmented_reward += augmented_reward

            if steps > config.initial_exploration and len(memory) > config.batch_size:
                epsilon -= config.epsilon_decay
                epsilon = max(epsilon, 0.4)

                batch, indexes, lengths = memory.sample(config.batch_size)
                loss, td_error = R2D2.train_model(
                    online_net, target_net, optimizer, batch, lengths
                )

                if config.enable_ngu:
                    embedding_loss = embedding_model.train_model(batch)

                memory.update_priority(indexes, td_error, lengths)

                if steps % config.update_target == 0:
                    update_target_model(online_net, target_net)

            if episode_steps >= horizon or done:
                sum_obs_set += len(env.set)
                break

        if can_log:
            mean_reward = sum_reward / config.log_interval
            mean_augmented_reward = sum_augmented_reward / config.log_interval
            metrics = {
                "episode": episode,
                "mean_reward": mean_reward,
                "epsilon": epsilon,
                "embedding_loss": embedding_loss,
                "loss": loss,
                "mean_augmented_reward": mean_augmented_reward,
                "steps": steps,
                "sum_obs_set": sum_obs_set / config.log_interval,
            }

            print(metrics)

            for key, value in metrics.items():
                logger.add_scalar(f"metrics/{key}", value, episode)

            logger.add_video(
                "collector/video",
                torch.from_numpy(np.array(video_buffer))
                .permute(0, -1, 1, 2)
                .unsqueeze(0),
                episode,
                fps=30,
            )

            sum_reward = 0
            sum_augmented_reward = 0
            sum_obs_set = 0


if __name__ == "__main__":
    main()
