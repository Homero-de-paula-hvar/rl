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
                 tournament_size: int = 3, elite_size: int = 2):
        """
        Initialize Genetic Algorithm optimizer
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for parent selection
            elite_size: Number of best individuals to preserve
        """
        super().__init__(param_bounds, population_size, max_generations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """Initialize random population with some diversity"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                # Use different sampling strategies for diversity
                if random.random() < 0.3:
                    # Uniform random
                    individual[param_name] = random.uniform(min_val, max_val)
                elif random.random() < 0.5:
                    # Gaussian around center
                    center = (min_val + max_val) / 2
                    std = (max_val - min_val) / 6
                    individual[param_name] = np.clip(
                        random.gauss(center, std), min_val, max_val
                    )
                else:
                    # Boundary values
                    individual[param_name] = random.choice([min_val, max_val])
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
        """Tournament selection with pressure"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Tournament with pressure (fitness proportional selection)
        total_fitness = sum(max(0, f) for f in tournament_fitnesses)
        if total_fitness == 0:
            winner_idx = tournament_indices[random.randint(0, len(tournament_indices)-1)]
        else:
            probabilities = [max(0, f) / total_fitness for f in tournament_fitnesses]
            winner_idx = tournament_indices[np.random.choice(len(tournament_indices), p=probabilities)]
        
        return population[winner_idx]
    
    def select_parents(self, population: List[Dict[str, float]], 
                      fitnesses: List[float]) -> List[Dict[str, float]]:
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.population_size - self.elite_size):
            parent = self.tournament_selection(population, fitnesses)
            parents.append(parent)
        return parents
    
    def crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Perform arithmetic crossover for continuous parameters"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = {}, {}
                
                for param_name in self.param_names:
                    min_val, max_val = self.param_bounds[param_name]
                    
                    # Arithmetic crossover
                    alpha = random.uniform(0, 1)
                    val1 = alpha * parent1[param_name] + (1 - alpha) * parent2[param_name]
                    val2 = (1 - alpha) * parent1[param_name] + alpha * parent2[param_name]
                    
                    # Ensure bounds
                    child1[param_name] = np.clip(val1, min_val, max_val)
                    child2[param_name] = np.clip(val2, min_val, max_val)
                
                offspring.extend([child1, child2])
            else:
                # No crossover, just copy parents
                offspring.append(parents[i].copy())
                if i + 1 < len(parents):
                    offspring.append(parents[i + 1].copy())
        
        return offspring
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Perform adaptive Gaussian mutation"""
        mutated = individual.copy()
        
        for param_name in self.param_names:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.param_bounds[param_name]
                current_val = mutated[param_name]
                
                # Adaptive mutation based on parameter range
                range_size = max_val - min_val
                std = range_size * 0.1  # 10% of parameter range
                
                # Gaussian mutation
                new_val = current_val + random.gauss(0, std)
                new_val = np.clip(new_val, min_val, max_val)
                
                mutated[param_name] = new_val
        
        return mutated
    
    def optimize(self, fitness_function: Callable) -> Tuple[Dict[str, float], float]:
        """
        Run the optimization process with elitism
        
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
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            elite_population = [population[i] for i in elite_indices]
            
            # Select parents for reproduction
            parents = self.select_parents(population, fitnesses)
            
            # Generate new population
            new_population = elite_population.copy()  # Start with elite
            
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