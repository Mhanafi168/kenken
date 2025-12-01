from typing import Dict, Optional
import copy
import numpy as np

from models.puzzle import KenKenPuzzle, KenKenGenerator
from algorithms.backtracking import BacktrackingSolver
from algorithms.cultural_algorithm import CulturalAlgorithmSolver


class KenKenSolver:
    """Main class that integrates both algorithms"""

    def __init__(self):
        self.puzzle = None
        self.backtracking_solver = BacktrackingSolver()
        self.ca_solver = CulturalAlgorithmSolver()

    def generate_random_puzzle(self, size: int = 4):
        """Generate a new random puzzle - DIFFERENT EACH TIME"""
        self.puzzle = KenKenGenerator.generate_puzzle(size)
        return self.puzzle

    def create_sample_puzzle_4x4(self):
        """Create a sample 4x4 KenKen puzzle for testing"""
        # Now generates a RANDOM puzzle each time
        return self.generate_random_puzzle(4)

    def create_simple_test_puzzle(self):
        """Create a simpler 3x3 puzzle for debugging"""
        return self.generate_random_puzzle(3)

    def solve_with_backtracking(self) -> Dict:
        """Solve using backtracking algorithm"""
        if not self.puzzle:
            raise ValueError("No puzzle loaded")

        # Reset puzzle but keep the same cage structure (deep copy to ensure independence)
        temp_puzzle = KenKenPuzzle(self.puzzle.size)
        temp_puzzle.cages = copy.deepcopy(self.puzzle.cages)

        success, metrics = self.backtracking_solver.solve(temp_puzzle)
        return metrics

    def solve_with_cultural_algorithm(
        self, population_size=100, generations=1000, seed: Optional[int] = None
    ) -> Dict:
        """Solve using cultural algorithm"""
        if not self.puzzle:
            raise ValueError("No puzzle loaded")

        # Reset puzzle but keep the same cage structure (deep copy to ensure independence)
        temp_puzzle = KenKenPuzzle(self.puzzle.size)
        temp_puzzle.cages = copy.deepcopy(self.puzzle.cages)

        # Configure solver
        self.ca_solver.population_size = population_size
        self.ca_solver.generations = generations

        # Set seed for reproducibility if provided
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

        success, metrics = self.ca_solver.solve(temp_puzzle)
        return metrics

    def compare_algorithms(self, seed: Optional[int] = 42) -> Dict:
        """Compare both algorithms on the same puzzle
        
        Args:
            seed: Random seed for reproducible results. If None, results will vary each run.
        """
        print("Solving with Backtracking...")
        bt_results = self.solve_with_backtracking()

        print("Solving with Cultural Algorithm...")
        # Use seed for reproducible comparison results
        ca_results = self.solve_with_cultural_algorithm(seed=seed)

        return {"backtracking": bt_results, "cultural_algorithm": ca_results}

    def print_puzzle_info(self):
        """Print information about the current puzzle"""
        if not self.puzzle:
            print("No puzzle loaded")
            return

        print(f"=== Puzzle Information ===")
        print(f"Size: {self.puzzle.size}x{self.puzzle.size}")
        print(f"Number of cages: {len(self.puzzle.cages)}")
        print("Cages:")
        for i, cage in enumerate(self.puzzle.cages):
            cells_str = ", ".join([f"({r},{c})" for r, c in cage['cells']])
            operation = cage['operation'] if cage['operation'] else "none"
            print(f"  Cage {i+1}: cells=[{cells_str}], operation='{operation}', target={cage['target']}")
        
        if hasattr(self.puzzle, 'solution') and self.puzzle.solution is not None:
            print(f"Solution available: Yes")
        else:
            print(f"Solution available: No")

    def verify_solution(self, solution_grid):
        """Verify if a solution is correct against the stored solution"""
        if not hasattr(self.puzzle, 'solution') or self.puzzle.solution is None:
            return None  # No solution stored for verification
        
        return np.array_equal(solution_grid, self.puzzle.solution)