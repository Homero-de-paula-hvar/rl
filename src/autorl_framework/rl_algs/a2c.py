import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class A2CAgent(BaseRLAlgorithm):
    """
    Agente A2C (Advantage Actor-Critic) para problemas de RL.
    """
    
    def __init__(self, n_arms: int, learning_rate: float = 0.0007,
                 n_steps: int = 5, gamma: float = 0.99, gae_lambda: float = 1.0,
                 ent_coef: float = 0.01, vf_coef: float = 0.25,
                 max_grad_norm: float = 0.5, rms_prop_eps: float = 1e-05,
                 use_rms_prop: bool = True, use_sde: bool = False,
                 sde_sample_freq: int = -1, normalize_advantage: bool = False,
                 verbose: int = 0):
        super().__init__(n_arms)
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rms_prop_eps = rms_prop_eps
        self.use_rms_prop = use_rms_prop
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.normalize_advantage = normalize_advantage
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
        """Treina o modelo A2C"""
        if self.env is None:
            raise ValueError("Ambiente não criado. Chame create_env() primeiro.")
            
        self.model = A2C(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            rms_prop_eps=self.rms_prop_eps,
            use_rms_prop=self.use_rms_prop,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            normalize_advantage=self.normalize_advantage,
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
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'rms_prop_eps': self.rms_prop_eps,
            'use_rms_prop': self.use_rms_prop,
            'use_sde': self.use_sde,
            'sde_sample_freq': self.sde_sample_freq,
            'normalize_advantage': self.normalize_advantage
        }
        
    def set_hyperparameters(self, **kwargs):
        """Define novos hiperparâmetros"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value) 