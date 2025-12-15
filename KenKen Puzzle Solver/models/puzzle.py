import numpy as np
import random
from typing import List, Tuple, Dict

class KenKenPuzzle:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.cages = []
        self.solution = None

    def add_cage(self, cells: List[Tuple[int, int]], operation: str, target: int):
        self.cages.append({'cells': cells, 'operation': operation, 'target': target})

    def is_valid_number(self, row: int, col: int, num: int) -> bool:
        if num in self.grid[row, :] or num in self.grid[:, col]:
            return False
        original_value = self.grid[row, col]
        self.grid[row, col] = num
        valid = self._check_cage_constraints(row, col)
        self.grid[row, col] = original_value
        return valid

    def _check_cage_constraints(self, row: int, col: int) -> bool:
        return all(self._is_cage_valid(cage) for cage in self.cages if (row, col) in cage['cells'])

    def _evaluate_cage_op(self, values: List[int], op: str) -> int:
        if not values: return 0
        if op == '+': return sum(values)
        if op == '×': return int(np.prod(values))
        if op == '-': return abs(values[0] - values[1]) if len(values) == 2 else 0
        if op == '÷': return max(values) // min(values) if len(values) == 2 and min(values) != 0 else 0
        return values[0]

    def _is_cage_valid(self, cage: Dict) -> bool:
        values = [self.grid[r, c] for r, c in cage['cells'] if self.grid[r, c] != 0]
        if not values:
            return True
        if len(values) < len(cage['cells']):
            if cage['operation'] == '+': return sum(values) < cage['target']
            if cage['operation'] == '×': return np.prod(values) <= cage['target']
            return cage['operation'] in ['-', '÷'] and len(values) <= 2
        return self._evaluate_cage_op(values, cage['operation']) == cage['target']

    def is_complete(self) -> bool:
        return 0 not in self.grid and all(self._is_cage_valid(cage) for cage in self.cages)

    def __str__(self):
        return f"KenKen {self.size}x{self.size} with {len(self.cages)} cages"

class KenKenGenerator:
    @staticmethod
    def generate_puzzle(size: int = 4) -> KenKenPuzzle:
        puzzle = KenKenPuzzle(size)
        solved_grid = KenKenGenerator._generate_latin_square(size)
        for cage_cells, operation, target in KenKenGenerator._generate_cages(size, solved_grid):
            puzzle.add_cage(cage_cells, operation, target)
        puzzle.solution = solved_grid
        puzzle.grid = np.zeros((size, size), dtype=int)
        return puzzle

    @staticmethod
    def _generate_latin_square(size: int) -> np.ndarray:
        base = list(range(1, size + 1))
        random.shuffle(base)
        arr = np.array([base[i:] + base[:i] for i in range(size)])
        np.random.shuffle(arr)
        arr = arr.T
        np.random.shuffle(arr)
        return arr.T

    @staticmethod
    def _generate_cages(size: int, solved_grid: np.ndarray) -> List[Tuple[List[Tuple[int, int]], str, int]]:
        unassigned, cages = {(i, j) for i in range(size) for j in range(size)}, []
        while unassigned:
            start_cell = random.choice(tuple(unassigned))
            unassigned.remove(start_cell)
            cage_cells = [start_cell]
            cage_size = random.randint(1, min(4, len(unassigned) + 1))

            while len(cage_cells) < cage_size:
                neighbors = [adj for cell in cage_cells for adj in KenKenGenerator._adjacent_cells(cell, size)
                           if adj in unassigned and adj not in cage_cells]
                if not neighbors:
                    break
                next_cell = random.choice(neighbors)
                unassigned.remove(next_cell)
                cage_cells.append(next_cell)

            cage_cells.sort()
            operation = KenKenGenerator._choose_operation(cage_cells, solved_grid)
            target = KenKenGenerator._calculate_cage_target(cage_cells, operation, solved_grid)
            cages.append((cage_cells, operation, target))
        return cages

    @staticmethod
    def _choose_operation(cage_cells: List[Tuple[int, int]], solved_grid: np.ndarray) -> str:
        values = [solved_grid[r, c] for r, c in cage_cells]
        if len(cage_cells) == 1: return ''
        if len(cage_cells) == 2:
            ops = ['+', '×']
            if values[0] != values[1]: ops.append('-')
            if min(values) != 0 and max(values) % min(values) == 0: ops.append('÷')
            return random.choice(ops)
        return random.choice(['+', '×'])

    @staticmethod
    def _adjacent_cells(cell: Tuple[int, int], size: int) -> List[Tuple[int, int]]:
        r, c = cell
        return [n for n in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] if 0 <= n[0] < size and 0 <= n[1] < size]

    @staticmethod
    def _calculate_cage_target(cells: List[Tuple[int, int]], operation: str, solved_grid: np.ndarray) -> int:
        values = [solved_grid[r, c] for r, c in cells]
        if operation == '': return values[0]
        if operation == '+': return sum(values)
        if operation == '×': return int(np.prod(values))
        if operation == '-': return abs(values[0] - values[1])
        if operation == '÷': return max(values) // min(values)
        return 0
