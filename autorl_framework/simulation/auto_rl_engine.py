"""
Auto RL Engine - Integra algoritmos de RL com otimizadores evolutivos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import time
import os
from datetime import datetime

from .simulator import MABSimulator
from ..rl_algs import EpsilonGreedy, UCB1, ThompsonSampling
from ..optimizers import GeneticAlgorithm, PSO, DifferentialEvolution, BayesianOptimization, GridSearch, RandomSearch, BaseSimpleOptimizer


class AutoRLEngine:
    """Engine principal que integra algoritmos de RL com otimizadores evolutivos"""
    
    def __init__(self, data: pd.DataFrame, arms_col: str, reward_col: str):
        """
        Initialize the Auto RL Engine
        
        Args:
            data: Historical data for simulation
            arms_col: Column name containing arm/campaign names
            reward_col: Column name containing reward values
        """
        self.data = data
        self.arms_col = arms_col
        self.reward_col = reward_col
        
        # Initialize simulator
        self.simulator = MABSimulator(data, arms_col, reward_col)
        self.n_arms = self.simulator.n_arms
        
        # Define algorithm parameter bounds for optimization
        self.algorithm_bounds = {
            'epsilon_greedy': {
                'epsilon': (0.01, 0.5)
            },
            'ucb': {},  # UCB1 não tem parâmetros para otimizar
            'thompson': {}  # Thompson Sampling não tem parâmetros para otimizar
        }
        
        # Define optimizers
        self.optimizers = {
            'genetic': GeneticAlgorithm,
            'pso': PSO,
            'differential_evolution': DifferentialEvolution,
            'bayesian_optimization': BayesianOptimization,
            'grid_search': GridSearch,
            'random_search': RandomSearch
        }
    
    def create_algorithm(self, algorithm_type: str, params: Dict[str, float]):
        """Create RL algorithm instance with given parameters"""
        if algorithm_type == 'epsilon_greedy':
            return EpsilonGreedy(n_arms=self.n_arms, epsilon=params['epsilon'])
        elif algorithm_type == 'ucb':
            return UCB1(n_arms=self.n_arms)
        elif algorithm_type == 'thompson':
            return ThompsonSampling(n_arms=self.n_arms)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    def fitness_function(self, algorithm_type: str) -> Callable:
        """Create fitness function for a specific algorithm type"""
        def fitness(params: Dict[str, float]) -> float:
            try:
                # Create algorithm with parameters
                algorithm = self.create_algorithm(algorithm_type, params)
                
                # Run simulation
                results = self.simulator.run(algorithm, f"{algorithm_type}_optimized")
                
                # Calculate fitness based on total reward and regret
                total_reward = results['reward'].sum()
                total_regret = results['regret'].sum()
                
                # Fitness = total reward - penalty for regret
                # Queremos maximizar recompensa e minimizar regret
                fitness_score = total_reward - (total_regret * 0.1)  # Penalty factor
                
                return fitness_score
            except Exception as e:
                print(f"Error in fitness evaluation: {e}")
                return float('-inf')
        
        return fitness
    
    def optimize_algorithm(self, algorithm_type: str, optimizer_type: str, 
                          max_generations: int = 50) -> Tuple[Dict[str, float], float]:
        """
        Optimize hyperparameters for a specific algorithm using evolutionary optimizer
        
        Args:
            algorithm_type: Type of RL algorithm to optimize
            optimizer_type: Type of evolutionary optimizer to use
            max_generations: Maximum number of generations for optimization
            
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        if algorithm_type not in self.algorithm_bounds:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Get parameter bounds for the algorithm
        param_bounds = self.algorithm_bounds[algorithm_type]
        
        # If no parameters to optimize, return default parameters
        if not param_bounds:
            default_params = {}
            # Run a single simulation to get baseline performance
            algorithm = self.create_algorithm(algorithm_type, default_params)
            results = self.simulator.run(algorithm, f"{algorithm_type}_baseline")
            total_reward = results['reward'].sum()
            total_regret = results['regret'].sum()
            fitness_score = total_reward - (total_regret * 0.1)
            
            print(f"\n🔧 {algorithm_type} não tem parâmetros para otimizar")
            print(f"🎯 Performance baseline: {fitness_score:.4f}")
            print(f"📊 Recompensa total: {total_reward:.4f}")
            print(f"📊 Regret total: {total_regret:.4f}")
            
            return default_params, fitness_score
        
        # Create optimizer
        optimizer_class = self.optimizers[optimizer_type]
        if issubclass(optimizer_class, BaseSimpleOptimizer):
            # Para otimizadores simples, gerar param_grid
            param_grid = {k: [v[0], v[1]] for k, v in param_bounds.items()}
            optimizer = optimizer_class(
                param_grid=param_grid,
                eval_func=self.fitness_function(algorithm_type),
                n_iter=max_generations
            )
        else:
            optimizer = optimizer_class(
                param_bounds=param_bounds,
                max_generations=max_generations,
                population_size=30
            )
        
        # Create fitness function
        fitness_func = self.fitness_function(algorithm_type)
        
        print(f"\n🔧 Otimizando {algorithm_type} com {optimizer_type}...")
        print(f"Parâmetros a otimizar: {list(param_bounds.keys())}")
        
        # Run optimization
        start_time = time.time()
        if issubclass(optimizer_class, BaseSimpleOptimizer):
            best_params, best_fitness = optimizer.optimize()
        else:
            best_params, best_fitness = optimizer.optimize(fitness_func)
        end_time = time.time()
        
        print(f"⏱️  Tempo de otimização: {end_time - start_time:.2f}s")
        print(f"🎯 Melhor fitness: {best_fitness:.4f}")
        print(f"📊 Melhores parâmetros: {best_params}")
        
        return best_params, best_fitness
    
    def run_comparison(self, algorithms: List[str] = None, 
                      optimizers: List[str] = None,
                      n_iterations: int = 10) -> pd.DataFrame:
        """
        Run comparison between different algorithms and optimizers
        
        Args:
            algorithms: List of algorithm types to compare
            optimizers: List of optimizer types to use
            n_iterations: Number of optimization runs per combination
            
        Returns:
            DataFrame with comparison results
        """
        if algorithms is None:
            algorithms = ['epsilon_greedy', 'ucb', 'thompson']
        
        if optimizers is None:
            optimizers = ['genetic', 'pso', 'differential_evolution']
        
        results = []
        
        for algorithm in algorithms:
            for optimizer in optimizers:
                print(f"\n{'='*60}")
                print(f"🔄 Testando {algorithm} + {optimizer}")
                print(f"{'='*60}")
                
                # Run optimization multiple times
                for i in range(n_iterations):
                    print(f"\n--- Iteração {i+1}/{n_iterations} ---")
                    
                    try:
                        best_params, best_fitness = self.optimize_algorithm(
                            algorithm, optimizer, max_generations=30
                        )
                        
                        # Run final simulation with best parameters
                        best_algorithm = self.create_algorithm(algorithm, best_params)
                        final_results = self.simulator.run(best_algorithm, f"{algorithm}_{optimizer}_{i}")
                        
                        total_reward = final_results['reward'].sum()
                        total_regret = final_results['regret'].sum()
                        
                        results.append({
                            'algorithm': algorithm,
                            'optimizer': optimizer,
                            'iteration': i,
                            'best_params': best_params,
                            'total_reward': total_reward,
                            'total_regret': total_regret,
                            'best_fitness': best_fitness
                        })
                        
                    except Exception as e:
                        print(f"❌ Erro na iteração {i+1}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def run_single_optimization(self, algorithm_type: str, optimizer_type: str,
                               max_generations: int = 100) -> Dict[str, Any]:
        """
        Run a single optimization and return detailed results
        
        Args:
            algorithm_type: Type of RL algorithm to optimize
            optimizer_type: Type of evolutionary optimizer to use
            max_generations: Maximum number of generations
            
        Returns:
            Dictionary with optimization results
        """
        best_params, best_fitness = self.optimize_algorithm(
            algorithm_type, optimizer_type, max_generations
        )
        
        # Run final simulation with best parameters
        best_algorithm = self.create_algorithm(algorithm_type, best_params)
        final_results = self.simulator.run(best_algorithm, f"{algorithm_type}_optimized")
        
        return {
            'algorithm_type': algorithm_type,
            'optimizer_type': optimizer_type,
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'total_reward': final_results['reward'].sum(),
            'total_regret': final_results['regret'].sum(),
            'simulation_results': final_results
        } 

    def run_full_test_and_save(self, n_iterations=3):
        """
        Executa todos os algoritmos com todos os otimizadores e salva o resultado em um TXT.
        """
        algorithms = list(self.algorithm_bounds.keys())
        optimizers = list(self.optimizers.keys())
        results = []
        log_lines = []
        log_lines.append(f"===== TESTE AUTOMÁTICO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for alg in algorithms:
            for opt in optimizers:
                log_lines.append(f"\n{'='*60}")
                log_lines.append(f"🔄 Testando {alg} + {opt}")
                log_lines.append(f"{'='*60}")
                for i in range(n_iterations):
                    log_lines.append(f"\n--- Iteração {i+1}/{n_iterations} ---")
                    try:
                        best_params, best_fitness = self.optimize_algorithm(alg, opt, max_generations=30)
                        best_algorithm = self.create_algorithm(alg, best_params)
                        final_results = self.simulator.run(best_algorithm, f"{alg}_{opt}_{i}")
                        total_reward = final_results['reward'].sum()
                        total_regret = final_results['regret'].sum()
                        log_lines.append(f"🎯 Melhor fitness: {best_fitness:.4f}")
                        log_lines.append(f"📊 Melhores parâmetros: {best_params}")
                        log_lines.append(f"📊 Recompensa total: {total_reward:.4f}")
                        log_lines.append(f"📊 Regret total: {total_regret:.4f}")
                        results.append({
                            'algorithm': alg,
                            'optimizer': opt,
                            'iteration': i,
                            'best_params': best_params,
                            'total_reward': total_reward,
                            'total_regret': total_regret,
                            'best_fitness': best_fitness
                        })
                    except Exception as e:
                        log_lines.append(f"❌ Erro na iteração {i+1}: {e}")
        # Salvar log
        filename = f"teste_automatico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for line in log_lines:
                f.write(line + '\n')
        return results, filename 