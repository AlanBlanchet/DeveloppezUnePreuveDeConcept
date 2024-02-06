import os
from time import sleep

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from torchvision.ops import MLP

from ..collect.buffer import SharedStateBuffer, StateBuffer
from ..collect.collector import Collector, DistributedCollector
from ..collect.state import StateDict
from ..policy import Policy
from ..utils.encode import Encoder
from .drqn import DRQN


class NeverGiveUp(Policy):
    def __init__(self, collector: Collector):
        super().__init__(collector)

        # Checking type to strictly enforce the use of a distributed collector
        assert (
            type(collector) == DistributedCollector
        ), "R2D2 requires a DistributedCollector"

        self.rnd = RND(self.in_state)

        # self.gamma = gamma
        # self.burn = burn_in
        # self.nstep = nstep
        self.curiosity_factor = 0.1

        # Our main non-distributed DRQN
        self.network = DRQN(
            collector,
            image_type="grayscale",
            sequence_length=8,
            batch_size=8,
            collect=False,
        )

        self.siamese = Siamese(self.in_state, self.out_action)
        self.episodic_emb = EpisodicEmbedding(self.in_state)

        self.episodic_novelty = EpisodicNovelty(self.in_state, memory=20)

        self.shared_buffer = SharedStateBuffer(20)

        self._optim = optim.Adam(self.network.parameters(), lr=self.lr)

    # Train embedding clf

    def forward(self, steps: int):
        num = self.collector.num_collectors

        self.network = self.network.cpu()
        # Start from nothing
        self.network = self.network.share_memory()
        self.episodic_novelty = self.episodic_novelty.share_memory()
        self.network.set_logger(self.logger)
        torch.set_num_threads(1)

        # Start all the processes in the background
        processes: list[mp.Process] = []
        for proc_i in range(num):
            print("Starting process", proc_i, flush=True)
            p = mp.Process(
                target=_run_collection,
                args=(self.network, self.episodic_novelty, self.shared_buffer, steps),
                name="Collector_" + str(proc_i),
            )
            p.start()
            processes.append(p)

        shared_queue = self.shared_buffer.buffer

        main_steps = 0
        # Retrieve the data from the queue
        while True:
            state: StateDict = shared_queue.get()
            print(f"{main_steps=}", flush=True)
            # Compute all algo calculations
            states = state.episode_split()
            # Store in main queue
            for s in states:
                self.collector.buffer.push(s.set_device("cpu"))

            if main_steps % 4 == 0:
                self.try_forward()

            main_steps += 1

            if main_steps >= steps * self.collector.num_collectors:
                break

        for p in processes:
            print(f"Closing process {p.pid}")
            p.terminate()

    def try_forward(self):
        buffer: StateBuffer = self.collector.buffer

        if len(buffer) > 8:
            # Prioritized sampling
            idx, states = buffer.sample(8)

            state = StateDict.batch(states)
            # state = state.set_device("cuda")

            # EMBEDDING
            x_t = self.network.network.encoder(state["obs"])
            x_t1 = self.network.network.encoder(state["next_obs"])
            actions = self.siamese(x_t, x_t1, state["action"])

            # LONG TERM REWARD ----------------------
            # If the obs is complex, then the RND loss will be high
            # That is what we want since we use the loss to add bonus reward !
            bonus_reward: torch.Tensor = self.rnd(state["obs"])

            # Get the intrinsic reward
            r_i_t = bonus_reward * torch.clamp(self.curiosity_factor, 1, 5)
            # --------------------------------------

            # SHORT TERM REWARD ---------------------
            novelty_reward = self.episodic_novelty(state)

            final_reward = novelty_reward + r_i_t

            # COMPUTE MAIN R2D2 DRQN LOSS ------------

            # Loss calculations happen in the network
            td_errors = self.network(state)
            self.network.collector._n_steps += 1

            # Prioritized sampling -----------------
            td_mean = td_errors.mean(dim=1).abs()
            # Update priorities for sampling
            for i, m in zip(idx, td_mean):
                buffer.priorities[i] = m.item()


class EpisodicEmbedding(nn.Module):
    def __init__(self, in_state: int, memory):
        self.embedding_dim = 256

        self.embedding = Encoder(in_state, self.embedding_dim)

        self.memory = memory

    def forward(self, state: torch.Tensor):
        embedding = self.embedding(state)
        return embedding


class EpisodicNovelty(nn.Module):
    """
    Compute Episodic Intrinsic Reward

    Has an episodic memory M of 'Controllable states'

    Has an embedding function mapping obs -> 'Controllable states'

    Computation are done intra-episode and not inter-episode
    """

    ...

    # State -> Embeddings -> controllable states -> KNN -> Episodic Reward

    def __init__(self, in_state: int, memory):
        self.embedding_dim = 512

        self.embedding = Encoder(in_state, self.embedding_dim)

        self.k = 5

        self.knn = NearestNeighbors(n_neighbors=self.k)

    def compute_novelty(self, emb_state: torch.Tensor):
        # Fit nearest neighbors model
        self.knn.fit(self.memory)
        _, indices = self.knn.kneighbors(emb_state)
        neighbors = self.memory[indices]  # get the nearest neighbors

        distances = torch.norm(
            neighbors - emb_state, dim=1
        )  # Compute distances to the neighbors
        novelty_score = torch.mean(
            distances
        )  # Consider novelty as the mean distance to k nearest neighbors.
        return novelty_score

    def forward(self, batch: StateDict):
        obs = batch["obs"]

        controllable_states = self.embedding(obs)  # (B, E)

        novelty_reward = self.compute_novelty(controllable_states)

        return novelty_reward


class Siamese(nn.Module):
    """
    Siamese network

    Gives the action to take to get from s to s+1
    """

    def __init__(self, in_state, out_action):
        super().__init__()

        # Used to predict the action to go from s to s+1
        self.h = MLP(in_state * 2, out_action)

    def forward(self, state_emb, next_state_emb, action):
        embedding = torch.cat((state_emb, next_state_emb), dim=-1)

        actions = self.h(embedding).softmax(dim=-1)

        return actions


class RND(nn.Module):
    """
    Random Network Distillation network
    """

    def __init__(self, in_state):
        super().__init__()

        self.predictor = Encoder(in_state, 512, image_type="grayscale")

        self.target = Encoder(in_state, 512, image_type="grayscale")
        self.target.requires_grad_(False)

        self.register_buffer("r_mean", torch.tensor(0.0))
        self.register_buffer("r_std", torch.tensor(1.0))

        self._optim = optim.Adam(self.network.parameters(), lr=0.0001)

    def forward(self, state: torch.Tensor, next_state: torch.Tensor):
        # Calculate error
        error = (self.predictor(next_state) - self.conv(state)).pow(2)

        self.r_mean = self.r_mean * 0.99 + error.mean() * 0.01
        self.r_std = torch.clamp_min(self.r_std * 0.99 + error.std() * 0.01, 1e-8)

        reward_loss = 1 + (error - self.r_mean) / self.r_std

        loss = reward_loss.mean()
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return reward_loss


class NullLogger(object):
    """A fake logger with does nothing."""

    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass  # Do nothing

        return method


def _run_collection(
    shared_network: Policy,
    episodic_novelty: EpisodicNovelty,
    shared_buffer: SharedStateBuffer,
    steps: int,
):
    # Create a custom env here
    env_id = shared_network.collector.env_id
    collector = Collector(env_id)
    policy = DRQN(collector, image_type="grayscale", sequence_length=8, batch_size=8)
    # policy.eval()

    collector.set_policy(policy)
    # We don't need a tensorboard logger
    logger = NullLogger()
    collector.set_logger(logger)
    policy.set_logger(logger)
    # policy.setup()

    # Now collect steps from the seperate environment
    for step in range(steps):
        done = False
        episode_steps = []
        while not done:
            # Collect a step in the child
            step_data = collector.collect_step()
            # Reset buffer
            collector.buffer.reset()
            collector._episode_start = 0

            episodic_novelty.embedding()

            # Set done
            *_, done = step_data
            episode_steps.append(step_data)

        # Push in shared buffer
        shared_buffer.push(episode_steps)

        if step % 10 == 0:
            policy.load_state_dict(shared_network.cpu().state_dict())
            print(f"PID {os.getpid()} collected {step} steps")

    print("PROCESS EXIT", flush=True)

    while True:
        sleep(1)
