import numpy as np
import random
from typing import List, Tuple, Dict, Optional

class KenKenPuzzle:
    """Represents a KenKen puzzle with grid and cage constraints"""
    
    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.cages = []
        self.solution = None
    
    def add_cage(self, cells: List[Tuple[int, int]], operation: str, target: int):
        """Add a cage to the puzzle"""
        self.cages.append({
            'cells': cells,
            'operation': operation,
            'target': target
        })
    
    def is_valid_number(self, row: int, col: int, num: int) -> bool:
        """Check if a number can be placed at (row, col)"""
        if num in self.grid[row, :] or num in self.grid[:, col]:
            return False
        
        original_value = self.grid[row, col]
        self.grid[row, col] = num
        
        valid = self._check_cage_constraints(row, col)
        
        self.grid[row, col] = original_value
        return valid
    
    def _check_cage_constraints(self, row: int, col: int) -> bool:
        """Check cage constraints for cages containing (row, col)"""
        for cage in self.cages:
            if (row, col) in cage['cells']:
                if not self._is_cage_valid(cage):
                    return False
        return True
    
    def _is_cage_valid(self, cage: Dict) -> bool:
        """Check if a cage satisfies its constraints"""
        cells = cage['cells']
        operation = cage['operation']
        target = cage['target']
        
        values = [self.grid[r, c] for r, c in cells if self.grid[r, c] != 0]
        
        if not values:
            return True
        
        if len(values) < len(cells):
            return self._is_partial_cage_valid(values, operation, target)
        
        return self._evaluate_cage(values, operation) == target
    
    def _is_partial_cage_valid(self, values: List[int], operation: str, target: int) -> bool:
        """Check if partially filled cage could still be valid"""
        if operation == '+':
            return sum(values) < target
        elif operation == '×':
            product = 1
            for v in values:
                product *= v
            return product <= target
        elif operation == '-':
            if len(values) > 2: return False
            return True
        elif operation == '÷':
            if len(values) > 2: return False
            return True
        return True
    
    def _evaluate_cage(self, values: List[int], operation: str) -> int:
        """Evaluate a fully filled cage"""
        if not values: return 0
        
        if operation == '+':
            return sum(values)
        elif operation == '×':
            product = 1
            for v in values:
                product *= v
            return product
        elif operation == '-':
            v_sorted = sorted(values, reverse=True)
            return v_sorted[0] - v_sorted[1]
        elif operation == '÷':
            v_sorted = sorted(values, reverse=True)
            big, small = v_sorted[0], v_sorted[1]
            return big // small if small != 0 else 0
        return values[0]
    
    def is_complete(self) -> bool:
        """Check if puzzle is completely and correctly filled"""
        if 0 in self.grid:
            return False
        for cage in self.cages:
            if not self._is_cage_valid(cage):
                return False
        return True

    def __str__(self):
        return f"KenKen {self.size}x{self.size} with {len(self.cages)} cages"

class KenKenGenerator:
    """Generates random KenKen puzzles that are guaranteed to be solvable"""
    
    @staticmethod
    def generate_puzzle(size: int = 4) -> KenKenPuzzle:
        puzzle = KenKenPuzzle(size)
        solved_grid = KenKenGenerator._generate_latin_square(size)
        cages = KenKenGenerator._generate_cages(size, solved_grid)
        
        for cage_cells, operation, target in cages:
            puzzle.add_cage(cage_cells, operation, target)
        
        puzzle.solution = solved_grid
        puzzle.grid = np.zeros((size, size), dtype=int)
        return puzzle
    
    @staticmethod
    def _generate_latin_square(size: int) -> np.ndarray:
        """Generate a RANDOMIZED valid Latin square"""
        # 1. Start with a shuffled base (e.g. 3, 1, 4, 2 instead of 1, 2, 3, 4)
        base = list(range(1, size + 1))
        random.shuffle(base)
        
        square = []
        for i in range(size):
            # 2. Create standard cyclic shift
            row = base[i:] + base[:i]
            square.append(row)
            
        arr = np.array(square)
        
        # 3. Shuffle Rows
        np.random.shuffle(arr)
        
        # 4. Shuffle Columns (Transpose -> Shuffle -> Transpose)
        arr = arr.T
        np.random.shuffle(arr)
        arr = arr.T
        
        return arr
    
    @staticmethod
    def _generate_cages(size: int, solved_grid: np.ndarray) -> List[Tuple[List[Tuple[int, int]], str, int]]:
        unassigned = {(i, j) for i in range(size) for j in range(size)}
        cages = []
        
        while unassigned:
            start_cell = random.choice(tuple(unassigned))
            unassigned.remove(start_cell)
            cage_cells = [start_cell]
            possible_size = min(4, len(unassigned) + 1)
            cage_size = random.randint(1, possible_size)
            
            while len(cage_cells) < cage_size:
                neighbors = []
                for cell in cage_cells:
                    for adj in KenKenGenerator._adjacent_cells(cell, size):
                        if adj in unassigned and adj not in cage_cells and adj not in neighbors:
                            neighbors.append(adj)
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
            big, small = max(values), min(values)
            if small != 0 and big % small == 0: ops.append('÷')
            return random.choice(ops)
        
        return random.choice(['+', '×'])
    
    @staticmethod
    def _adjacent_cells(cell: Tuple[int, int], size: int) -> List[Tuple[int, int]]:
        r, c = cell
        neighbors = []
        if r > 0: neighbors.append((r - 1, c))
        if r < size - 1: neighbors.append((r + 1, c))
        if c > 0: neighbors.append((r, c - 1))
        if c < size - 1: neighbors.append((r, c + 1))
        return neighbors
    
    @staticmethod
    def _calculate_cage_target(cells: List[Tuple[int, int]], operation: str, solved_grid: np.ndarray) -> int:
        values = [solved_grid[r, c] for r, c in cells]
        if operation == '': return values[0]
        if operation == '+': return sum(values)
        elif operation == '×':
            product = 1
            for v in values: product *= v
            return product
        elif operation == '-': return abs(values[0] - values[1])
        elif operation == '÷': return max(values) // min(values)
        return 0