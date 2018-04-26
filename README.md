# rl_algorithms
Implementations of different off-policy reinforcement learning algorithms.

# Framework 

1. Module [methods.py](methods.py) contains [TensorFlow](https://www.tensorflow.org) implementations of various neural network architectures used in value-based deep reinforcement learning.

2. Module [agents.py](agents.py) contains general **Agent** class and various wrappers around it which represent corresponding deep RL algorithms.

3. Module [utils.py](utils.py) contains **Replay Buffer** implementation together with a wrapper around **OpenAI gym Atari 2600** environment necessary for reproducing original DeepMind results.

4. Jupyter notebook [train_agents.ipynb](train_agents.ipynb) contains examples of how to use the proposed framework to train deep RL agents on various environments.

# Available algorithms

- Deep Q-Network [Volodymyr Mnih et al. "Human-level control through deep reinforcement learning." Nature (2015)](https://pra.open.tips/storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
<p align="center">
<img src="/img/dqn_classic.png" width="80%">
</p>

- Dueling Deep Q-Network [Ziyu Wang et al. "Dueling network architectures for deep reinforcement learning." ICML (2016).](https://arxiv.org/pdf/1511.06581.pdf)
<p align="center">
<img src="/img/dqn_dueling.png" width="80%">
</p>

- Categorical Deep Q-Network [Marc G. Bellemare, Will Dabney, and Rémi Munos. "A distributional perspective on reinforcement learning." ICML (2017).](https://arxiv.org/pdf/1707.06887)
<p align="center">
<img src="/img/dqn_categorical.png" width="80%">
</p>

- Quantile Regression Deep Q-Network [Will Dabney, Mark Rowland, Marc G. Bellemare, and Rémi Munos. "Distributional Reinforcement Learning with Quantile Regression." AAAI (2018).](https://arxiv.org/pdf/1710.10044)
<p align="center">
<img src="/img/dqn_quantile.png" width="80%">
</p>

- Soft Actor-Critic [Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv preprint arXiv:1801.01290 (2018).](https://arxiv.org/pdf/1801.01290)

<p align="center">
<img src="/img/sac_v_network.png" width="80%">
<img src="/img/sac_q_network.png" width="80%">
<img src="/img/sac_p_network.png" width="80%">
</p>

**Note.** Images of different neural network architectures are based on the images from the [Dueling architectures](https://arxiv.org/pdf/1511.06581.pdf) paper. The original images were copied and adapted to reflect features of particular architectures and learning algorithms.
