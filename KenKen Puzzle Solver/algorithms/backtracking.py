from typing import Dict, Optional, Tuple
import time
import random

from models.puzzle import KenKenPuzzle


class BacktrackingSolver:
    """Implements backtracking algorithm for KenKen puzzles"""

    def __init__(self):
        self.steps = 0
        self.backtracks = 0
        self.start_time = 0

    def solve(self, puzzle: KenKenPuzzle) -> Tuple[bool, Dict]:
        """Solve using backtracking algorithm"""
        self.steps = 0
        self.backtracks = 0
        self.start_time = time.time()

        # Find empty cell to start
        empty_cell = self._find_empty_cell(puzzle)
        if not empty_cell:
            return True, self._get_performance_metrics(puzzle, True)

        success = self._backtrack(puzzle, empty_cell[0], empty_cell[1])

        return success, self._get_performance_metrics(puzzle, success)

    def _backtrack(self, puzzle: KenKenPuzzle, row: int, col: int) -> bool:
        """Recursive backtracking function"""
        self.steps += 1

        # Create a list of possible numbers and SHUFFLE them
        # This prevents the solver from always picking 1, 2, 3...
        possible_numbers = list(range(1, puzzle.size + 1))
        random.shuffle(possible_numbers) 

        # Try all possible numbers in random order
        for num in possible_numbers:
            if puzzle.is_valid_number(row, col, num):
                # Place number
                puzzle.grid[row, col] = num

                # Find next empty cell
                next_cell = self._find_empty_cell(puzzle)

                if not next_cell:
                    return True  # Puzzle solved!

                # Recursively try to solve rest of puzzle
                if self._backtrack(puzzle, next_cell[0], next_cell[1]):
                    return True

                # Backtrack
                puzzle.grid[row, col] = 0
                self.backtracks += 1

        return False

    def _find_empty_cell(self, puzzle: KenKenPuzzle) -> Optional[Tuple[int, int]]:
        """Find next empty cell"""
        for i in range(puzzle.size):
            for j in range(puzzle.size):
                if puzzle.grid[i, j] == 0:
                    return (i, j)
        return None

    def _get_performance_metrics(self, puzzle: KenKenPuzzle, success: bool) -> Dict:
        """Collect performance metrics"""
        return {
            "solved": success,
            "execution_time": time.time() - self.start_time,
            "steps": self.steps,
            "backtracks": self.backtracks,
            "solution": puzzle.grid.copy() if success else None,
        }