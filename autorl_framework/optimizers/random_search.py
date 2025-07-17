from autorl_framework.optimizers.base import BaseSimpleOptimizer
import random

class RandomSearch(BaseSimpleOptimizer):
    """
    Otimizador Random Search: avalia combinações aleatórias de hiperparâmetros.
    """
    def __init__(self, param_grid, eval_func, n_iter=10):
        super().__init__(param_grid, eval_func)
        self.n_iter = n_iter

    def optimize(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        best_score = None
        best_params = None
        for _ in range(self.n_iter):
            params = {k: random.choice(v) for k, v in self.param_grid.items()}
            score = self.eval_func(params)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_params = params
        return best_params, best_score 