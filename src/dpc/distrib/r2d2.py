"""
Implementation of the Recurrent Replay Distributed DQN policy.

Paper: https://openreview.net/pdf?id=r1lyTjAqYX
"""


import os
from time import sleep

import torch
import torch.multiprocessing as mp

from ..collect.buffer import SharedStateBuffer, StateBuffer
from ..collect.collector import Collector, DistributedCollector
from ..collect.state import StateDict
from ..policies.drqn import DRQN
from ..policy import Policy


class NullLogger(object):
    """A fake logger with does nothing."""

    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass  # Do nothing

        return method


class R2D2(Policy):
    def __init__(
        self,
        # We do not just take a 'simple collector' but a distributed one
        collector: DistributedCollector,
        lr=1e-4,
        nu: int = 32,
        gamma=0.997,
        burn_in=40,
        nstep=5,
    ):
        super().__init__(collector)

        # Checking type to strictly enforce the use of a distributed collector
        assert (
            type(collector) == DistributedCollector
        ), "R2D2 requires a DistributedCollector"

        self.gamma = gamma
        self.burn = burn_in
        self.nstep = nstep

        # Our main non-distributed DRQN
        self.network = DRQN(
            collector,
            image_type="grayscale",
            sequence_length=8,
            batch_size=8,
            collect=False,
        )
        self.shared_buffer = SharedStateBuffer(20)

    def forward(self, steps: int):
        num = self.collector.num_collectors

        self.network = self.network.cpu()
        # Start from nothing
        self.network = self.network.share_memory()
        self.network.set_logger(self.logger)
        torch.set_num_threads(1)

        # Start all the processes in the background
        processes: list[mp.Process] = []
        for proc_i in range(num):
            print("Starting process", proc_i, flush=True)
            p = mp.Process(
                target=_run_collection,
                args=(self.network, self.shared_buffer, steps),
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
            td_errors = self.network(state)
            self.network.collector._n_steps += 1
            td_mean = td_errors.mean(dim=1).abs()

            # Update priorities
            for i, m in zip(idx, td_mean):
                buffer.priorities[i] = m.item()


def _run_collection(
    shared_network: Policy, shared_buffer: SharedStateBuffer, steps: int
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
        # Collect steps in children env
        state = collector.collect_steps(8)
        collector.buffer.reset()
        collector._episode_start = 0
        # Push in shared buffer
        shared_buffer.push(state)

        if step % 10 == 0:
            policy.load_state_dict(shared_network.cpu().state_dict())
            print(f"PID {os.getpid()} collected {step} steps")

    print("PROCESS EXIT", flush=True)

    while True:
        sleep(1)
