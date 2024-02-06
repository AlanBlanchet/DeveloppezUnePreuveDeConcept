# AAA

## Youtube channels

- <https://www.youtube.com/@MachineLearningwithPhil>
- <https://www.youtube.com/@sentdex>
- <https://www.youtube.com/@Eigensteve>
- <https://www.youtube.com/@ArxivInsights>

## Medium

- [Vanilla Policy Gradients](https://towardsdatascience.com/policy-gradients-in-reinforcement-learning-explained-ecec7df94245)

## Blogs

- [Real World ML](https://www.realworldml.xyz/the-hands-on-reinforcement-learning-course)

## Modern RL implementations

- [DI-engine](https://github.com/opendilab/DI-engine)

## Papers (easy to hard)

- [Quality of Action Matrix (state -> action matrix)]()
- [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [PPO](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [SAC](https://arxiv.org/pdf/1801.01290.pdf)
- TD3
  - [Paper]()
  - [Youtube](https://www.youtube.com/watch?v=1lZOB2S17LU)
- A3C
  - [Paper](https://arxiv.org/pdf/1602.01783.pdf)
  - [Youtube](https://www.youtube.com/watch?v=OcIx_TBu90Q)
- [PPG](https://arxiv.org/pdf/2009.04416v1.pdf)
  - [Dataset](https://paperswithcode.com/dataset/procgen)
  - Sources : [OpenAI](https://github.com/openai/phasic-policy-gradient/), [DI-Engine](https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppg.py), [PBRL](https://github.com/jjccero/pbrl/blob/master/pbrl/algorithms/ppg/ppg.py)

## Vocabulary / Knowledge

### State / Observation

The current state received by the environment.

### Trajectory

In RL, a trajectory is a sequence of steps. After some steps, we might achieve a big change in our reward function. For example :

| Step 1 reward | Step 2 reward | Step 3 reward | Step 4 reward | Step 5 reward |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.1           | 0.05          | 0.01          | 0.07          | 40            |

What we have to do when our Q function is getting trained is to propagate this `+40` reward so that from the state of _Step 1_ the Q value can lead us through the intermediate states (Step 2/3/4).

We basically train our Q function to **maximise our futur reward**.

### Temporal Difference

We use temporal **difference** in order to compute an error from state `s` to state `s+1`.

The equation for calculating the difference is really simple :

$$ TD = Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) $$

The smaller $TD$ gets, the lower the update is going to be to our Q function (Q matrix / Deep Learning Neural Network). It means the algorithm has converged.

### SARSA

**S**tate **A**ction **R**eward **S**tate **A**ction is a way to update the Q function from $TD$ and the reward obtained in `s+1`

$$ Q(s_{t}, a_{t}) + \alpha(r_{t+1} + \gamma TD) $$

$\gamma$ is called the discount factor. It helps propagate the current reward to the next states.

$\alpha$ is the learning rate. It determines to what extent the newly acquired information will override the old information.

### On-Policy / Off-Policy

In on-policy problems, the agent picks up actions from the policy. It then updates it's own policy to make more rewarding actions.

In off-policy problems, the agent can't pick up actions. It learns from exploration only.

### Q-Learning

Q-Learning is nearly the same as SARSA, except we don't need to know the next action $a_{t+1}$.

$$ TD_{max} = \max_{a} Q(s_{t+1}) - Q(s_t, a_t) $$

$ \max_{a} Q(s_{t+1}) $ simply means that instead of taking the action, we just get the output `probabilities` for each actions of size `N` and we take the highest value.

Whereas in SARSA we already defined our next action.

Which changes the equation :

$$ Q(s_{t}, a_{t}) + \alpha(r_{t+1} + \gamma TD_{max}) $$

### Model-free / Model-based

Some experiences can be played with the agent having access to some extra post-states rather than just having a reward and having to work out how to maximise that reward.

In model-based, the agent get directly access to some environment states (ex: distance from objective, score, target).

Thus we already know what objective we have. Thus, it's easier to understand how to maximuse our reward.

### Value function

A prediction of future rewards that the agent expects to receive if it follows a certain policy from a given state or a state-action pair

#### State Value function

Quantifies how good a given state is by predicting the total reward that an agent can expect from that state

#### Action Value Function

Predicts what happens when the agent chooses an action in a certain state, then follows the policy.

### Transition

A transition is the data stored in a memory buffer of states. It will then be used to get samples of state/action/next action/reward in order to train a Q function.

## Unsorted

- Critic
- Target
- Value
- Actor = Policy
- Agent
- [VPG]()
- [TRPO](https://arxiv.org/pdf/1502.05477.pdf)
- [DDPG]()
- [R2D2](https://arxiv.org/pdf/1906.06195.pdf)
- [HAPPO](https://arxiv.org/pdf/2109.11251.pdf)
- [On Learning Intrinsic Rewards for Policy Gradient Methods](https://arxiv.org/pdf/1804.06459.pdf)
- Double Q Learning
