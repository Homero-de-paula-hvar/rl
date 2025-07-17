"""
Base classes for evolutionary optimizers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import random


class BaseOptimizer(ABC):
    """Base class for all evolutionary optimizers"""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50, max_generations: int = 100):
        """
        Initialize the optimizer
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
        """
        self.param_bounds = param_bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.param_names = list(param_bounds.keys())
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []
        
    @abstractmethod
    def initialize_population(self) -> List[Dict[str, float]]:
        """Initialize the population"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, individual: Dict[str, float], 
                        fitness_function: Callable) -> float:
        """Evaluate fitness of an individual"""
        pass
    
    @abstractmethod
    def select_parents(self, population: List[Dict[str, float]], 
                      fitnesses: List[float]) -> List[Dict[str, float]]:
        """Select parents for reproduction"""
        pass
    
    @abstractmethod
    def crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Perform crossover operation"""
        pass
    
    @abstractmethod
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Perform mutation operation"""
        pass
    
    def optimize(self, fitness_function: Callable) -> Tuple[Dict[str, float], float]:
        """
        Run the optimization process
        
        Args:
            fitness_function: Function that takes parameters and returns fitness score
            
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, fitness_function)
                fitnesses.append(fitness)
                
                # Update best solution
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = individual.copy()
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            
            # Select parents
            parents = self.select_parents(population, fitnesses)
            
            # Generate new population
            new_population = []
            while len(new_population) < self.population_size:
                # Crossover
                offspring = self.crossover(parents)
                
                # Mutation
                for child in offspring:
                    mutated_child = self.mutate(child)
                    new_population.append(mutated_child)
                    
                    if len(new_population) >= self.population_size:
                        break
            
            population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness
