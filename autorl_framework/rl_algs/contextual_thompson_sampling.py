import numpy as np
from scipy.stats import multivariate_normal
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class ContextualThompsonSampling(BaseRLAlgorithm):
    """
    Implementação do Thompson Sampling Contextual (Linear) estado da arte.
    """
    def __init__(self, n_arms: int, context_dim: int, v: float = 1.0):
        super().__init__(n_arms)
        self.context_dim = context_dim
        self.v = v
        self.arms = {}
        for i in range(self.n_arms):
            self.arms[i] = {
                'B': np.identity(self.context_dim),
                'f': np.zeros((self.context_dim, 1)),
                'mu': np.zeros((self.context_dim, 1))
            }

    def select_arm(self, context: np.ndarray) -> int:
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        sampled_rewards = []
        for i in range(self.n_arms):
            arm = self.arms[i]
            B_inv = np.linalg.inv(arm['B'])
            mu_tilde = multivariate_normal.rvs(
                mean=arm['mu'].flatten(),
                cov=(self.v**2) * B_inv
            ).reshape(-1, 1)
            predicted_reward = context.T @ mu_tilde
            sampled_rewards.append(predicted_reward.item())
        return np.argmax(sampled_rewards)

    def update(self, chosen_arm: int, reward: float, context: np.ndarray):
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        arm = self.arms[chosen_arm]
        arm['B'] += context @ context.T
        arm['f'] += context * reward
        B_inv = np.linalg.inv(arm['B'])
        arm['mu'] = B_inv @ arm['f'] 