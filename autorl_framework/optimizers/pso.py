"""
Particle Swarm Optimization (PSO) for hyperparameter optimization
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable
from .base import BaseOptimizer


class Particle:
    """Represents a particle in PSO"""
    
    def __init__(self, position: Dict[str, float], velocity: Dict[str, float]):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()
        self.best_fitness = float('-inf')


class PSO(BaseOptimizer):
    """Particle Swarm Optimization for hyperparameter optimization"""
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 30, max_generations: int = 100,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Initialize PSO optimizer
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            population_size: Size of the swarm
            max_generations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive learning factor
            c2: Social learning factor
        """
        super().__init__(param_bounds, population_size, max_generations)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.particles = []
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """Initialize particle swarm"""
        self.particles = []
        
        for _ in range(self.population_size):
            # Initialize position randomly
            position = {}
            velocity = {}
            
            for param_name, (min_val, max_val) in self.param_bounds.items():
                position[param_name] = random.uniform(min_val, max_val)
                # Initialize velocity as small random values
                velocity[param_name] = random.uniform(-0.1 * (max_val - min_val), 
                                                     0.1 * (max_val - min_val))
            
            particle = Particle(position, velocity)
            self.particles.append(particle)
        
        # Return positions for compatibility with base class
        return [p.position for p in self.particles]
    
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
        """PSO doesn't use parent selection, return current positions"""
        return [p.position for p in self.particles]
    
    def crossover(self, parents: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """PSO doesn't use crossover, return parents as is"""
        return parents
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """PSO doesn't use mutation, return individual as is"""
        return individual
    
    def update_particles(self, fitnesses: List[float]):
        """Update particle positions and velocities"""
        for i, particle in enumerate(self.particles):
            fitness = fitnesses[i]
            
            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocity and position
            for param_name in self.param_names:
                min_val, max_val = self.param_bounds[param_name]
                
                # Velocity update
                r1, r2 = random.random(), random.random()
                
                cognitive_component = (self.c1 * r1 * 
                                     (particle.best_position[param_name] - 
                                      particle.position[param_name]))
                
                social_component = (self.c2 * r2 * 
                                  (self.global_best_position[param_name] - 
                                   particle.position[param_name]))
                
                particle.velocity[param_name] = (self.w * particle.velocity[param_name] + 
                                               cognitive_component + social_component)
                
                # Position update
                particle.position[param_name] += particle.velocity[param_name]
                
                # Clamp to bounds
                particle.position[param_name] = np.clip(particle.position[param_name], 
                                                      min_val, max_val)
    
    def optimize(self, fitness_function: Callable) -> Tuple[Dict[str, float], float]:
        """Run PSO optimization"""
        # Initialize swarm
        self.initialize_population()
        
        for generation in range(self.max_generations):
            # Evaluate fitness for all particles
            fitnesses = []
            for particle in self.particles:
                fitness = self.evaluate_fitness(particle.position, fitness_function)
                fitnesses.append(fitness)
            
            # Update particles
            self.update_particles(fitnesses)
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_solution': self.global_best_position.copy() if self.global_best_position else None
            })
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.global_best_fitness:.4f}")
        
        return self.global_best_position, self.global_best_fitness 