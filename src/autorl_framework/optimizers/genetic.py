"""
Genetic Algorithm optimizer for hyperparameter optimization
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable
from .base import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """Genetic Algorithm optimizer for hyperparameter optimization"""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50, max_generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 tournament_size: int = 3):
        """
        Initialize Genetic Algorithm optimizer
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for parent selection
        """
        super().__init__(param_bounds, population_size, max_generations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                individual[param_name] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Dict[str, float], 
                        fitness_function: Callable) -> float:
        """Evaluate fitness of an individual"""
        try:
            return fitness_function(individual)
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return float('-inf')
    
    def tournament_selection(self, population: List[Dict[str, float]], 
                           fitnesses: List[float]) -> Dict[str, float]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def select_parents(self, population: List[Dict[str, float]], 
                      fitnesses: List[float]) -> List[Dict[str, float]]:
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.population_size):
            parent = self.tournament_selection(population, fitnesses)
            parents.append(parent)
        return parents
    
    def crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Perform uniform crossover"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = {}, {}
                
                for param_name in self.param_names:
                    if random.random() < 0.5:
                        child1[param_name] = parent1[param_name]
                        child2[param_name] = parent2[param_name]
                    else:
                        child1[param_name] = parent2[param_name]
                        child2[param_name] = parent1[param_name]
                
                offspring.extend([child1, child2])
            else:
                # No crossover, just copy parents
                offspring.append(parents[i].copy())
                if i + 1 < len(parents):
                    offspring.append(parents[i + 1].copy())
        
        return offspring
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Perform Gaussian mutation"""
        mutated = individual.copy()
        
        for param_name in self.param_names:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.param_bounds[param_name]
                current_val = mutated[param_name]
                
                # Gaussian mutation with bounds
                std = (max_val - min_val) * 0.1  # 10% of parameter range
                new_val = current_val + random.gauss(0, std)
                new_val = np.clip(new_val, min_val, max_val)
                
                mutated[param_name] = new_val
        
        return mutated 