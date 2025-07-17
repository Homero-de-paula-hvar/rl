"""
Differential Evolution (DE) for hyperparameter optimization
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable
from .base import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution optimizer for hyperparameter optimization"""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 50, max_generations: int = 100,
                 F: float = 0.5, CR: float = 0.7):
        """
        Initialize Differential Evolution optimizer
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
            F: Differential weight (mutation factor)
            CR: Crossover probability
        """
        super().__init__(param_bounds, population_size, max_generations)
        self.F = F
        self.CR = CR
    
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
    
    def select_parents(self, population: List[Dict[str, float]], 
                      fitnesses: List[float]) -> List[Dict[str, float]]:
        """DE doesn't use traditional parent selection, return population as is"""
        return population
    
    def crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """DE doesn't use traditional crossover, return parents as is"""
        return parents
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """DE doesn't use traditional mutation, return individual as is"""
        return individual
    
    def differential_mutation(self, target: Dict[str, float], 
                            population: List[Dict[str, float]]) -> Dict[str, float]:
        """Perform differential mutation"""
        # Select three random individuals (different from target)
        indices = list(range(len(population)))
        indices.remove(population.index(target))
        a, b, c = random.sample(indices, 3)
        
        # Create mutant vector
        mutant = {}
        for param_name in self.param_names:
            mutant[param_name] = (population[a][param_name] + 
                                self.F * (population[b][param_name] - population[c][param_name]))
        
        return mutant
    
    def crossover_de(self, target: Dict[str, float], 
                    mutant: Dict[str, float]) -> Dict[str, float]:
        """Perform DE crossover"""
        trial = {}
        j_rand = random.choice(self.param_names)  # Ensure at least one parameter changes
        
        for param_name in self.param_names:
            if param_name == j_rand or random.random() < self.CR:
                trial[param_name] = mutant[param_name]
            else:
                trial[param_name] = target[param_name]
        
        return trial
    
    def selection_de(self, target: Dict[str, float], trial: Dict[str, float],
                    target_fitness: float, trial_fitness: float) -> Tuple[Dict[str, float], float]:
        """Perform DE selection"""
        if trial_fitness > target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness
    
    def optimize(self, fitness_function: Callable) -> Tuple[Dict[str, float], float]:
        """Run Differential Evolution optimization"""
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitnesses = []
        for individual in population:
            fitness = self.evaluate_fitness(individual, fitness_function)
            fitnesses.append(fitness)
            
            # Update best solution
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = individual.copy()
        
        for generation in range(self.max_generations):
            new_population = []
            new_fitnesses = []
            
            for i, target in enumerate(population):
                # Differential mutation
                mutant = self.differential_mutation(target, population)
                
                # Crossover
                trial = self.crossover_de(target, mutant)
                
                # Ensure bounds
                for param_name, (min_val, max_val) in self.param_bounds.items():
                    trial[param_name] = np.clip(trial[param_name], min_val, max_val)
                
                # Selection
                trial_fitness = self.evaluate_fitness(trial, fitness_function)
                selected, selected_fitness = self.selection_de(target, trial, 
                                                             fitnesses[i], trial_fitness)
                
                new_population.append(selected)
                new_fitnesses.append(selected_fitness)
                
                # Update best solution
                if selected_fitness > self.best_fitness:
                    self.best_fitness = selected_fitness
                    self.best_solution = selected.copy()
            
            # Update population
            population = new_population
            fitnesses = new_fitnesses
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_solution': self.best_solution.copy() if self.best_solution else None
            })
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")
        
        return self.best_solution, self.best_fitness 