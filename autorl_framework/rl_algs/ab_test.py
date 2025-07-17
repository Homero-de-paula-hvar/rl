from autorl_framework.rl_algs.base import BaseRLAlgorithm

class ABTestAgent(BaseRLAlgorithm):
    """
    Agente que simula um teste A/B clássico: explora cada braço por um número fixo de rodadas, depois explota o melhor.
    """
    def __init__(self, n_arms: int, exploration_rounds: int = 10):
        super().__init__(n_arms)
        self.exploration_rounds = exploration_rounds
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_pulls = 0
        self.phase = 'exploration'

    def select_arm(self) -> int:
        if self.phase == 'exploration':
            for arm in range(self.n_arms):
                if self.counts[arm] < self.exploration_rounds:
                    return arm
            self.phase = 'exploitation'
        # Exploitation: escolhe o braço com maior média
        max_value = max(self.values)
        best_arms = [i for i, v in enumerate(self.values) if v == max_value]
        return best_arms[0]

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        self.total_pulls += 1
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        new_value = old_value + (1 / n) * (reward - old_value)
        self.values[chosen_arm] = new_value 