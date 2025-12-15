from typing import Dict, Optional
import copy, numpy as np
from models.puzzle import KenKenPuzzle, KenKenGenerator
from algorithms.backtracking import BacktrackingSolver
from algorithms.cultural_algorithm import CulturalAlgorithmSolver, HybridKenKenSolver

class KenKenSolver:
    def __init__(self):
        self.puzzle = None
        self.backtracking_solver = BacktrackingSolver()
        self.ca_solver = CulturalAlgorithmSolver()
        self.hybrid_solver = HybridKenKenSolver()

    def generate_random_puzzle(self, size: int = 4):
        self.puzzle = KenKenGenerator.generate_puzzle(size)
        return self.puzzle

    def create_sample_puzzle_4x4(self):
        return self.generate_random_puzzle(4)

    def create_simple_test_puzzle(self):
        return self.generate_random_puzzle(3)

    def solve_with_backtracking(self) -> Dict:
        if not self.puzzle:
            raise ValueError("No puzzle loaded")
        temp_puzzle = KenKenPuzzle(self.puzzle.size)
        temp_puzzle.cages = copy.deepcopy(self.puzzle.cages)
        return self.backtracking_solver.solve(temp_puzzle)[1]

    def solve_with_cultural_algorithm(self, population_size=100, generations=1000, seed: Optional[int] = None) -> Dict:
        if not self.puzzle:
            raise ValueError("No puzzle loaded")
        temp_puzzle = KenKenPuzzle(self.puzzle.size)
        temp_puzzle.cages = copy.deepcopy(self.puzzle.cages)
        self.hybrid_solver.ca_solver.base_population_size = population_size
        self.hybrid_solver.ca_solver.base_generations = generations
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
        metrics = self.hybrid_solver.solve(temp_puzzle)
        if metrics.get("solved") and metrics.get("solution") is not None:
            self.puzzle.grid = metrics["solution"].copy()
        return metrics

    def compare_algorithms(self, seed: Optional[int] = 42) -> Dict:
        print("Solving with Backtracking...")
        bt_results = self.solve_with_backtracking()
        print("Solving with Cultural Algorithm...")
        return {"backtracking": bt_results, "cultural_algorithm": self.solve_with_cultural_algorithm(seed=seed)}

    def print_puzzle_info(self):
        if not self.puzzle:
            print("No puzzle loaded")
            return
        print(f"=== Puzzle Information ===\nSize: {self.puzzle.size}x{self.puzzle.size}\nNumber of cages: {len(self.puzzle.cages)}\nCages:")
        for i, cage in enumerate(self.puzzle.cages):
            cells_str = ", ".join([f"({r},{c})" for r, c in cage['cells']])
            print(f"  Cage {i+1}: cells=[{cells_str}], operation='{cage['operation'] or 'none'}', target={cage['target']}")
        print(f"Solution available: {'Yes' if hasattr(self.puzzle, 'solution') and self.puzzle.solution is not None else 'No'}")

    def verify_solution(self, solution_grid):
        return None if not hasattr(self.puzzle, 'solution') or self.puzzle.solution is None else np.array_equal(solution_grid, self.puzzle.solution)