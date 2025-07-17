#!/usr/bin/env python3
"""
Testes básicos para o Auto RL Framework
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autorl_framework.rl_algs import EpsilonGreedy, UCB1, ThompsonSampling
from autorl_framework.optimizers import GeneticAlgorithm, ParticleSwarmOptimization
from autorl_framework.simulation.auto_rl_engine import AutoRLEngine

class TestBasicAlgorithms:
    """Testes para algoritmos básicos"""
    
    def test_epsilon_greedy(self):
        """Testa algoritmo Epsilon-Greedy"""
        alg = EpsilonGreedy(n_arms=3, epsilon=0.1)
        
        # Testa seleção de braço
        arm = alg.select_arm()
        assert 0 <= arm < 3
        
        # Testa atualização
        alg.update(arm, 1.0)
        assert alg.counts[arm] == 1
        assert alg.values[arm] == 1.0
        
    def test_ucb1(self):
        """Testa algoritmo UCB1"""
        alg = UCB1(n_arms=3)
        
        # Testa seleção inicial (deve escolher braços não testados)
        for i in range(3):
            arm = alg.select_arm()
            alg.update(arm, 0.5)
            
        # Agora deve usar UCB
        arm = alg.select_arm()
        assert 0 <= arm < 3
        
    def test_thompson_sampling(self):
        """Testa algoritmo Thompson Sampling"""
        alg = ThompsonSampling(n_arms=3)
        
        # Testa seleção
        arm = alg.select_arm()
        assert 0 <= arm < 3
        
        # Testa atualização
        alg.update(arm, 1)
        assert alg.alpha[arm] == 2
        assert alg.beta[arm] == 1

class TestOptimizers:
    """Testes para otimizadores"""
    
    def test_genetic_algorithm(self):
        """Testa algoritmo genético"""
        def objective_function(params):
            return params['x'] + params['y']
            
        param_bounds = {
            'x': (0, 10),
            'y': (0, 10)
        }
        
        optimizer = GeneticAlgorithm(
            param_bounds=param_bounds,
            objective_function=objective_function,
            maximize=True,
            population_size=10,
            n_iterations=5
        )
        
        best_params, best_score = optimizer.optimize(n_iterations=5)
        
        assert 'x' in best_params
        assert 'y' in best_params
        assert best_score > 0
        
    def test_particle_swarm(self):
        """Testa PSO"""
        def objective_function(params):
            return params['x'] + params['y']
            
        param_bounds = {
            'x': (0, 10),
            'y': (0, 10)
        }
        
        optimizer = ParticleSwarmOptimization(
            param_bounds=param_bounds,
            objective_function=objective_function,
            maximize=True,
            n_particles=5
        )
        
        best_params, best_score = optimizer.optimize(n_iterations=5)
        
        assert 'x' in best_params
        assert 'y' in best_params
        assert best_score > 0

class TestAutoRLEngine:
    """Testes para o motor principal"""
    
    def create_sample_data(self):
        """Cria dados de exemplo para testes"""
        np.random.seed(42)
        
        data = []
        campaigns = ['Campaign A', 'Campaign B', 'Campaign C']
        
        for i in range(30):
            for campaign in campaigns:
                data.append({
                    'campaign_name': campaign,
                    'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                    'binary_conversion': np.random.choice([0, 1], p=[0.7, 0.3])
                })
                
        return pd.DataFrame(data)
    
    def test_engine_creation(self):
        """Testa criação do engine"""
        data = self.create_sample_data()
        engine = AutoRLEngine(data=data, arms_col='campaign_name', reward_col='binary_conversion')
        
        assert engine.n_arms == 3
        assert len(engine.algorithms) > 0
        assert len(engine.optimizers) > 0
        
    def test_algorithm_evaluation(self):
        """Testa avaliação de algoritmo"""
        data = self.create_sample_data()
        engine = AutoRLEngine(data=data, arms_col='campaign_name', reward_col='binary_conversion')
        
        score = engine.evaluate_algorithm('epsilon_greedy', {'epsilon': 0.1})
        assert isinstance(score, (int, float))
        
    @pytest.mark.slow
    def test_hyperparameter_optimization(self):
        """Testa otimização de hiperparâmetros"""
        data = self.create_sample_data()
        engine = AutoRLEngine(data=data, arms_col='campaign_name', reward_col='binary_conversion')
        
        best_params, best_score = engine.optimize_hyperparameters(
            algorithm_name='epsilon_greedy',
            optimizer_name='genetic',
            n_iterations=5
        )
        
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        assert 'epsilon' in best_params

if __name__ == "__main__":
    pytest.main([__file__]) 