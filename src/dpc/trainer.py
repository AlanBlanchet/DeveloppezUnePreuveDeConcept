import json
import pickle
from pathlib import Path
from time import time
from typing import Literal

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ai.utils.paths import AIPaths

from .collect.collector import COLLECTION_OF, Collector, DistributedCollector
from .policy import Policy
from .utils.func import random_run_name


class Trainer:
    """
    Collects data from the environment and trains the agent.
    """

    def __init__(
        self,
        agent: Policy,
        reward_agg: Literal["sum", "mean"] = "mean",
        tqdm_reward_update_s: float = 0.2,
        run_name: str = random_run_name(),
    ):
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.tqdm_reward_update_s = tqdm_reward_update_s
        self.train_mode_ = "rgb_array"
        self.run_name = run_name
        self.logger = SummaryWriter(AIPaths.tensorboard / self.run_name)
        self.agent.set_logger(self.logger)
        self.collector.set_logger(self.logger)
        self.collector.set_policy(agent)
        self.reward_agg = reward_agg
        self._loaded = False
        self.eval_steps = 100

        self.setup = False

        torch.autograd.set_detect_anomaly(True)

    @property
    def collector(self):
        return self.agent.collector

    def _training(self):
        # Make sure we are in a train render mode
        if self.train_mode_ is None:
            self.train_mode_ = self.collector.render_mode
        if self.train_mode_ != self.collector.render_mode:
            self.collector.set_render(self.train_mode_)

        self.agent.to(self.device)

    def _setup(self):
        if not self.setup and not self._loaded:
            # self.agent.setup()
            self.setup = True

    def train(self, agent_steps=4):
        self._training()

        self.agent.train()
        self.collector.train()
        self.collector.reset()
        self._setup()

        if type(self.collector).__name__ == Collector.__name__:
            self._run(agent_steps, "train")
        elif type(self.collector).__name__ == DistributedCollector.__name__:
            self._run_distrib(agent_steps)
        else:
            raise NotImplementedError(
                f"Collector type {type(self.collector)} not implemented"
            )

    def eval(self, steps=32):
        self._training()

        self.agent.eval()
        self.collector.eval()
        self.collector.reset()

        assert self.setup, "Agent must be setup before evaluation"

        return self._run(steps, "eval")

    def test(self, collection_steps=4, of: COLLECTION_OF = "episodes"):
        self._setup()
        self.agent.eval()
        self.collector.set_render("human")
        self.collector.eval()
        self.collector.reset()

        episode = 0
        if of == "episodes":
            while episode < collection_steps:
                _, __, ___, ____, done = self.collector.collect_step(store=False)

                if done:
                    episode += 1
                    # self.rewards_.append(reward)
        elif of == "steps":
            for step in range(collection_steps):
                _, __, ___, ____, done = self.collector.collect_step(store=False)

                if done:
                    episode += 1
                    # self.rewards_.append(reward)
        self.collector.clean()

    def _run(self, agent_steps: int, mode="train"):
        start_time = time()
        delta_time = 0

        pbar = tqdm(range(int(agent_steps)))

        for agent_step in pbar:
            delta_time = time() - start_time

            data = self.agent.step()
            _, actions, rewards, _, _ = data.get_defaults()

            agg = torch.mean if self.reward_agg == "mean" else torch.sum
            reduced_reward = agg(rewards).item()

            self.logger.add_scalar(
                f"trainer_{mode}/reward", reduced_reward, self.agent.global_step
            )

            for idx, action in enumerate(self.collector.action_names):
                self.logger.add_scalar(
                    f"actions_{mode}/{action}",
                    (actions.argmax(-1) == idx).sum().item() / actions.shape[0],
                    self.agent.global_step,
                )

            if mode == "train":
                self.agent(data)

            if delta_time > self.tqdm_reward_update_s:
                start_time += self.tqdm_reward_update_s
                delta_time = 0
                pbar.set_description(
                    f"Step: {agent_step} | Reward: {reduced_reward:<.4f}"
                )

    def _run_distrib(self, agent_steps: int):
        self.agent(agent_steps)

    @classmethod
    def _resolve_path(cls, path: str | Path):
        if isinstance(path, str):
            if "/" not in path:
                path = AIPaths.cache / path
            else:
                path = Path(path).resolve()

        if path.is_file():
            path = path.with_suffix("")

        path.mkdir(exist_ok=True)

        return path

    def save(self, path: str | Path = AIPaths.cache):
        path = Trainer._resolve_path(path)

        name = type(self.agent).__name__

        params = {}
        params["env"] = self.collector._env.spec.id
        params["agent_mod"] = self.agent.__module__
        params["agent"] = name
        params["trainer"] = {
            "reward_mode": self.reward_agg,
            "tqdm_reward_update_s": self.tqdm_reward_update_s,
        }

        # Model
        torch.save(self.agent.state_dict(), path / "agent.pt")

        # Collector
        with open(path / "collector.pkl", "wb+") as f:
            f.write(pickle.dumps(self.collector))

        # Config
        with open(path / "params.json", "w+") as f:
            f.write(json.dumps(params))

    @classmethod
    def load(cls, path: str | Path, render_mode: str = "human"):
        import importlib

        path = cls._resolve_path(path)

        agent_state = torch.load(path / "agent.pt")

        with open(path / "params.json", "r") as f:
            params = json.loads(f.read())
            trainer_params = params["trainer"]

        with open(path / "collector.pkl", "rb") as f:
            collector: Collector = pickle.loads(f.read())
            collector.setup()

        # Load agent
        agent_name = str(params["agent"])
        agent_mod = str(params["agent_mod"])
        agent_mod = importlib.import_module(agent_mod)
        agent_class = getattr(agent_mod, agent_name)
        agent: Policy = agent_class(collector)
        agent.load_state_dict(agent_state)

        trainer = Trainer(agent, **trainer_params, run_name=random_run_name())
        trainer._loaded = True
        return trainer
