from typing import Dict, List, Tuple
import random
import time
import numpy as np

from models.puzzle import KenKenPuzzle

class CulturalAlgorithmSolver:
    """Implements an Improved Cultural Algorithm for KenKen puzzles"""

    def __init__(self, population_size=100, generations=1000, mutation_rate=0.1):
        self.base_population_size = population_size
        self.base_generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_history = []
        self.best_fitness_history = []
        
        # dynamic parameters
        self.population_size = population_size
        self.generations = generations

    def solve(self, puzzle: KenKenPuzzle) -> Tuple[bool, Dict]:
        """Solve using Cultural Algorithm with Auto-Tuning"""
        self.fitness_history = []
        self.best_fitness_history = []
        start_time = time.time()
        
        # 1. AUTO-TUNE PARAMETERS BASED ON SIZE
        # Larger puzzles require exponentially more effort
        if puzzle.size >= 6:
            self.population_size = max(self.base_population_size, 300)
            self.generations = max(self.base_generations, 3000)
            self.mutation_rate = 0.35 # Higher mutation for larger grids
        elif puzzle.size >= 5:
            self.population_size = max(self.base_population_size, 200)
            self.generations = max(self.base_generations, 1500)
            self.mutation_rate = 0.2
        else:
            self.population_size = self.base_population_size
            self.generations = self.base_generations
            self.mutation_rate = 0.1

        # Initialize population
        population = self._initialize_population(puzzle)
        belief_space = self._initialize_belief_space(puzzle)

        best_individual = None
        best_fitness = -1.0
        
        # Stagnation counters
        stagnation_counter = 0

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._calculate_fitness(ind, puzzle) for ind in population]
            
            # Find best in this generation
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            # Update global best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
                stagnation_counter = 0 # Reset stagnation
            else:
                stagnation_counter += 1

            # Update belief space
            belief_space = self._update_belief_space(population, fitness_scores, belief_space)

            # History tracking
            self.fitness_history.append(np.mean(fitness_scores))
            self.best_fitness_history.append(best_fitness)

            # Check for success
            if best_fitness >= 1.0: # 1.0 is normalized max fitness
                break
                
            # --- STAGNATION HANDLING (Cataclysm) ---
            # If stuck for 10% of generations, shake things up
            is_stagnant = stagnation_counter > (self.generations * 0.1)
            current_mutation_rate = 0.8 if is_stagnant else self.mutation_rate

            # Create new generation
            new_population = []
            
            # ELITISM: Keep the absolute best 2 solutions no matter what
            new_population.append(best_individual.copy())
            if len(population) > 1:
                # Add second best
                sorted_indices = np.argsort(fitness_scores)
                new_population.append(population[sorted_indices[-2]].copy())

            # Fill the rest
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2, puzzle)

                # Mutation
                child1 = self._smart_mutate(child1, puzzle, belief_space, current_mutation_rate)
                child2 = self._smart_mutate(child2, puzzle, belief_space, current_mutation_rate)

                new_population.extend([child1, child2])
            
            # Trim to exact size
            population = new_population[:self.population_size]
            
            # If we did a cataclysm, reset counter
            if is_stagnant:
                stagnation_counter = 0

        # Apply best solution
        success = best_fitness >= 1.0
        if best_individual is not None:
            puzzle.grid = best_individual

        return success, self._get_performance_metrics(
            puzzle, success, best_fitness, start_time
        )

    def _initialize_population(self, puzzle: KenKenPuzzle) -> List[np.ndarray]:
        """Initialize population with randomized Latin squares"""
        population = []
        for _ in range(self.population_size):
            individual = self._generate_latin_square(puzzle.size)
            population.append(individual)
        return population

    def _generate_latin_square(self, size: int) -> np.ndarray:
        """Generate a random Latin square (Valid Row/Col uniqueness)"""
        base = list(range(1, size + 1))
        random.shuffle(base)
        square = []
        for i in range(size):
            row = base[i:] + base[:i]
            square.append(row)
        arr = np.array(square)
        np.random.shuffle(arr) # Shuffle rows
        # Shuffle cols
        arr = arr.T
        np.random.shuffle(arr)
        arr = arr.T
        return arr

    def _calculate_fitness(self, individual: np.ndarray, puzzle: KenKenPuzzle) -> float:
        """Calculate normalized fitness score (0.0 to 1.0)"""
        # We assume Row/Col constraints are met by the Latin Square structure
        # We only need to check Cages
        
        cage_matches = 0
        total_cages = len(puzzle.cages)
        
        # Use existing puzzle logic to evaluate cages
        original_grid = puzzle.grid
        puzzle.grid = individual # Temporarily set grid
        
        for cage in puzzle.cages:
            if puzzle._is_cage_valid(cage):
                cage_matches += 1
                
        puzzle.grid = original_grid # Restore
        
        return cage_matches / total_cages

    def _max_possible_fitness(self, puzzle: KenKenPuzzle) -> float:
        return 1.0

    def _initialize_belief_space(self, puzzle: KenKenPuzzle) -> Dict:
        return {
            "best_individuals": [],
            "cell_preferences": np.zeros((puzzle.size, puzzle.size, puzzle.size)),
        }

    def _update_belief_space(self, population, fitness_scores, belief_space):
        # Keep top 5%
        top_count = max(1, len(population) // 20)
        top_indices = np.argsort(fitness_scores)[-top_count:]
        belief_space["best_individuals"] = [population[i].copy() for i in top_indices]
        
        # Simple frequency count of numbers in cells for best solutions
        for idx in top_indices:
            ind = population[idx]
            for r in range(ind.shape[0]):
                for c in range(ind.shape[1]):
                    val_idx = ind[r, c] - 1
                    belief_space["cell_preferences"][r, c, val_idx] += 1
        return belief_space

    def _tournament_selection(self, population, fitness_scores, size=3):
        indices = random.sample(range(len(population)), size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def _crossover(self, p1, p2, puzzle):
        # Row-based crossover preserves Latin Square property (mostly)
        point = random.randint(1, puzzle.size - 1)
        c1 = np.vstack([p1[:point], p2[point:]])
        c2 = np.vstack([p2[:point], p1[point:]])
        return c1, c2

    def _smart_mutate(self, individual: np.ndarray, puzzle: KenKenPuzzle, belief_space: Dict, rate: float) -> np.ndarray:
        """
        Smart Mutation:
        1. Identifies a row that contains an invalid cage.
        2. Swaps two numbers in that row to try and fix it.
        """
        if random.random() > rate:
            return individual

        size = puzzle.size
        
        # 1. Identify invalid cages
        original_grid = puzzle.grid
        puzzle.grid = individual
        
        invalid_cages = []
        for i, cage in enumerate(puzzle.cages):
            if not puzzle._is_cage_valid(cage):
                invalid_cages.append(cage)
        
        puzzle.grid = original_grid # Restore
        
        # 2. Pick a row to mutate
        # If we have invalid cages, pick a row involved in one of them
        target_row = random.randint(0, size - 1)
        
        if invalid_cages and random.random() < 0.7: # 70% chance to target a specific error
            target_cage = random.choice(invalid_cages)
            # Pick a cell from this cage
            cell = random.choice(target_cage['cells'])
            target_row = cell[0]

        # 3. Perform Swap within the row (Preserves Latin Square Row property)
        # We swap two columns in this row
        c1, c2 = random.sample(range(size), 2)
        
        # Belief Space Influence:
        # If belief space suggests a number is good at c1, try to find that number in the row and swap it to c1
        if random.random() < 0.3:
            # Find preferred value for c1
            prefs = belief_space["cell_preferences"][target_row, c1]
            if np.sum(prefs) > 0:
                best_val = np.argmax(prefs) + 1
                # Find where best_val is currently located in this row
                current_locs = np.where(individual[target_row] == best_val)[0]
                if len(current_locs) > 0:
                    c2 = current_locs[0] # Set c2 to the column where the desired number is
        
        # Execute swap
        individual[target_row, c1], individual[target_row, c2] = (
            individual[target_row, c2],
            individual[target_row, c1],
        )

        return individual

    def _get_performance_metrics(self, puzzle, success, best_fitness, start_time):
        return {
            "solved": success,
            "execution_time": time.time() - start_time,
            "best_fitness": best_fitness,
            "generations": len(self.fitness_history),
            "fitness_history": self.fitness_history,
            "best_fitness_history": self.best_fitness_history,
            "solution": puzzle.grid.copy() if success else None,
        }