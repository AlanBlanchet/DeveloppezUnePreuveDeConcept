from dpc.collect.collector import DistributedCollector
from dpc.distrib.r2d2 import R2D2
from dpc.trainer import Trainer

if __name__ == "__main__":
    collector = DistributedCollector("ALE/SpaceInvaders-v5", num_collectors=2)
    agent = R2D2(collector, lr=0.001, gamma=1)
    trainer = Trainer(agent, reward_agg="sum")

    trainer.train(1000)
