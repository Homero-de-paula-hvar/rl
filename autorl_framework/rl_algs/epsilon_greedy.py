import random
import numpy as np
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class EpsilonGreedy(BaseRLAlgorithm):
    """
    Implementação do algoritmo Epsilon-Greedy.
    Explora aleatoriamente com probabilidade epsilon, senão explota o melhor braço conhecido.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon deve estar entre 0 e 1.")
        self.epsilon = epsilon
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_pulls = 0

    def select_arm(self) -> int:
        if random.random() > self.epsilon:
            max_value = max(self.values)
            best_arms = [i for i, v in enumerate(self.values) if v == max_value]
            return random.choice(best_arms)
        else:
            return random.randrange(self.n_arms)

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        self.total_pulls += 1
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        new_value = old_value + (1 / n) * (reward - old_value)
        self.values[chosen_arm] = new_value 