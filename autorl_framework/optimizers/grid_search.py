from autorl_framework.optimizers.base import BaseSimpleOptimizer
import itertools

class GridSearch(BaseSimpleOptimizer):
    """
    Otimizador Grid Search: avalia todas as combinações possíveis de hiperparâmetros.
    """
    def __init__(self, param_grid, eval_func, n_iter=10):
        super().__init__(param_grid, eval_func)
        self.n_iter = n_iter

    def optimize(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        best_score = None
        best_params = None
        # Limitar o número de combinações testadas se n_iter < total
        combinations = list(itertools.product(*values))
        if self.n_iter < len(combinations):
            import random
            combinations = random.sample(combinations, self.n_iter)
        for combination in combinations:
            params = dict(zip(keys, combination))
            score = self.eval_func(params)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_params = params
        return best_params, best_score 