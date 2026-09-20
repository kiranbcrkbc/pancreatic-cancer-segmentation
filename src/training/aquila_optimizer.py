"""
Aquila Optimizer Module
Nature-inspired metaheuristic optimization algorithm for hyperparameter exploration,
satisfying the PRD and academic supervisor requirements.
"""

from typing import Callable, Dict, Any, List, Tuple
import numpy as np


class AquilaOptimizer:
    """
    Aquila Optimizer (AO) implementation.
    Simulates four hunting behaviors:
    1. Expanded exploration (high soar with vertical stoop)
    2. Narrowed exploration (contour flight with short glide attack)
    3. Expanded exploitation (low flight with slow descent attack)
    4. Narrowed exploitation (walk and grab prey)
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 10,
        max_iter: int = 10,
        seed: int = 42,
    ) -> None:
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.rng = np.random.RandomState(seed)

        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]

        # Initialize population
        self.population = self.lower_bound + self.rng.rand(self.pop_size, self.dim) * (
            self.upper_bound - self.lower_bound
        )
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])

        # Track best solution (Maximization)
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = [self.best_fitness]

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Runs the optimization iterations."""
        for t in range(1, self.max_iter + 1):
            x_mean = np.mean(self.population, axis=0)

            for i in range(self.pop_size):
                r = self.rng.rand()
                alpha = 0.1
                delta = 0.1

                if t <= (2 / 3) * self.max_iter:
                    if r <= 0.5:
                        # Phase 1: Expanded exploration
                        x_new = self.best_solution * (1.0 - t / self.max_iter) + (
                            x_mean - self.best_solution * self.rng.rand()
                        )
                    else:
                        # Phase 2: Narrowed exploration
                        idx_rand = self.rng.randint(0, self.pop_size)
                        x_rand = self.population[idx_rand]
                        levy = 0.01 * self.rng.randn(self.dim)
                        x_new = self.best_solution * levy + x_rand + (self.rng.rand() - 0.5)
                else:
                    if r <= 0.5:
                        # Phase 3: Expanded exploitation
                        alpha_term = alpha * (self.upper_bound - self.lower_bound) * self.rng.rand()
                        x_new = (self.best_solution - x_mean) * alpha_term + self.best_solution
                    else:
                        # Phase 4: Narrowed exploitation
                        q = self.rng.rand()
                        g1 = 2.0 * self.rng.rand() - 1.0
                        x_new = q * self.best_solution - (
                            g1 * self.population[i] * self.rng.rand()
                            - delta * (self.upper_bound - self.lower_bound) * self.rng.rand()
                        )

                # Clamp to bounds
                x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                f_new = self.obj_func(x_new)

                if f_new > self.fitness[i]:
                    self.population[i] = x_new
                    self.fitness[i] = f_new
                    if f_new > self.best_fitness:
                        self.best_fitness = f_new
                        self.best_solution = x_new.copy()

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.history
