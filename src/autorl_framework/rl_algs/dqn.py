import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class DQNAgent(BaseRLAlgorithm):
    """
    Agente DQN (Deep Q-Network) para problemas de RL discreto.
    """
    
    def __init__(self, n_arms: int, learning_rate: float = 0.0001,
                 buffer_size: int = 1000000, learning_starts: int = 50000,
                 batch_size: int = 32, tau: float = 1.0, gamma: float = 0.99,
                 train_freq: int = 4, gradient_steps: int = 1,
                 target_update_interval: int = 10000, exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10.0, verbose: int = 0):
        super().__init__(n_arms)
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        
        # Parâmetros para MAB
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.model = None
        self.env = None
        
    def create_env(self, data, reward_col):
        """Cria ambiente personalizado para os dados"""
        from autorl_framework.rl_tasks.mab_env import MABEnvironment
        self.env = DummyVecEnv([lambda: MABEnvironment(data, reward_col, self.n_arms)])
        
    def train(self, total_timesteps: int = 10000):
        """Treina o modelo DQN"""
        if self.env is None:
            raise ValueError("Ambiente não criado. Chame create_env() primeiro.")
            
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            max_grad_norm=self.max_grad_norm,
            verbose=self.verbose
        )
        
        self.model.learn(total_timesteps=total_timesteps)
        
    def select_arm(self) -> int:
        """Seleciona braço usando política treinada ou fallback para epsilon-greedy"""
        if self.model is not None:
            try:
                action, _ = self.model.predict(self.env.reset(), deterministic=True)
                return action[0]
            except:
                pass
                
        # Fallback para epsilon-greedy
        if np.random.random() > 0.1:
            max_value = max(self.values)
            best_arms = [i for i, v in enumerate(self.values) if v == max_value]
            return np.random.choice(best_arms)
        else:
            return np.random.randint(self.n_arms)
            
    def update(self, chosen_arm: int, reward: float):
        """Atualiza estatísticas do braço escolhido"""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        
        old_value = self.values[chosen_arm]
        new_value = old_value + (1 / n) * (reward - old_value)
        self.values[chosen_arm] = new_value
        
    def get_hyperparameters(self):
        """Retorna hiperparâmetros atuais"""
        return {
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'learning_starts': self.learning_starts,
            'batch_size': self.batch_size,
            'tau': self.tau,
            'gamma': self.gamma,
            'train_freq': self.train_freq,
            'gradient_steps': self.gradient_steps,
            'target_update_interval': self.target_update_interval,
            'exploration_fraction': self.exploration_fraction,
            'exploration_initial_eps': self.exploration_initial_eps,
            'exploration_final_eps': self.exploration_final_eps,
            'max_grad_norm': self.max_grad_norm
        }
        
    def set_hyperparameters(self, **kwargs):
        """Define novos hiperparâmetros"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 