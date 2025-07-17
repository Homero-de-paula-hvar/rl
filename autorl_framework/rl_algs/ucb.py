# Conteúdo para autorl_framework/rl_algs/ucb.py

import math
import random
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class UCB1(BaseRLAlgorithm):
    """
    Implementação do algoritmo Upper Confidence Bound (UCB1).
    Baseado no princípio do "otimismo em face da incerteza".
    """

    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0

    def select_arm(self) -> int:
        # Primeiro, joga em cada braço uma vez para evitar divisão por zero
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Calcula o índice UCB para cada braço
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            # UCB1 formula: value + sqrt(2 * log(t) / n)
            bonus = math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        # Escolhe o braço com o maior índice UCB
        max_ucb = max(ucb_values)
        best_arms = [i for i, v in enumerate(ucb_values) if v == max_ucb]
        return random.choice(best_arms)

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        
        # Atualização incremental da média
        old_value = self.values[chosen_arm]
        new_value = old_value + (1 / n) * (reward - old_value)
        self.values[chosen_arm] = new_value