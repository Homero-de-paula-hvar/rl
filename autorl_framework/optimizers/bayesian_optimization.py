import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from autorl_framework.optimizers.base import BaseSimpleOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

class BayesianOptimization(BaseSimpleOptimizer):
    """
    Otimizador Bayesian Optimization: usa um modelo probabilístico para guiar a busca por hiperparâmetros ótimos.
    """
    def __init__(self, param_grid, eval_func, n_iter=10):
        super().__init__(param_grid, eval_func)
        self.n_iter = n_iter

    def optimize(self):
        keys = list(self.param_grid.keys())
        bounds = [ (min(v), max(v)) for v in self.param_grid.values() ]
        X = []
        y = []
        for _ in range(3):  # Inicialização com 3 pontos aleatórios
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            params = dict(zip(keys, x0))
            score = self.eval_func(params)
            X.append(x0)
            y.append(score)
        X = np.array(X)
        y = np.array(y)
        for _ in range(self.n_iter):
            gp = GaussianProcessRegressor(kernel=Matern(length_scale_bounds=(1e-6, 1e5)), n_restarts_optimizer=2)
            gp.fit(X, y)
            x_next = [np.random.uniform(b[0], b[1]) for b in bounds]
            params = dict(zip(keys, x_next))
            score = self.eval_func(params)
            X = np.vstack([X, x_next])
            y = np.append(y, score)
        best_idx = np.argmax(y)
        best_params = dict(zip(keys, X[best_idx]))
        best_score = y[best_idx]
        return best_params, best_score 