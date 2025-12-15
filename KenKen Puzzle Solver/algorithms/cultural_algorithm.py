from typing import Dict, List, Tuple
import random, time, copy
import numpy as np
from models.puzzle import KenKenPuzzle
from algorithms.backtracking import BacktrackingSolver

class CulturalAlgorithmSolver:
    def __init__(self, population_size=120, generations=1200, mutation_rate=0.18):
        self.base_population_size = self.population_size = population_size
        self.base_generations = self.generations = generations
        self.base_mutation_rate = self.mutation_rate = mutation_rate
        self.restarts, self.time_budget_seconds = 1, 0.0
        self.fitness_history: List[float] = []
        self.best_fitness_history: List[float] = []

    def solve(self, puzzle: KenKenPuzzle) -> Tuple[bool, Dict]:
        self._reset_histories()
        start_time = time.time()
        self._tune_parameters(puzzle.size)
        if puzzle.size == 1:
            puzzle.grid[0, 0] = 1
            return True, self._get_performance_metrics(puzzle, True, 1.0, start_time)

        runs, best_overall, best_overall_fit, start_overall = self.restarts, None, -1.0, time.time()
        for _ in range(runs):
            population = self._initialize_population(puzzle)
            belief_space = self._initialize_belief_space(puzzle)
            cage_cache = self._build_cage_cache(puzzle)
            best_individual, best_fitness, stagnation = population[0].copy(), self._calculate_fitness(population[0], cage_cache), 0

            for gen in range(self.generations):
                fitness_scores = [self._calculate_fitness(ind, cage_cache) for ind in population]
                gen_best_idx = int(np.argmax(fitness_scores))
                gen_best_fit = fitness_scores[gen_best_idx]
                if gen_best_fit > best_fitness:
                    best_fitness, best_individual, stagnation = gen_best_fit, population[gen_best_idx].copy(), 0
                else:
                    stagnation += 1

                self.fitness_history.append(float(np.mean(fitness_scores)))
                self.best_fitness_history.append(best_fitness)

                if best_fitness >= 1.0 or (self.time_budget_seconds and (time.time() - start_overall) > self.time_budget_seconds):
                    break

                belief_space = self._update_belief_space(population, fitness_scores, belief_space)
                new_population: List[np.ndarray] = []
                elite_count = min(3, len(population))
                sorted_idx = np.argsort(fitness_scores)
                for i in range(elite_count):
                    new_population.append(population[sorted_idx[-(i + 1)]].copy())

                while len(new_population) < self.population_size:
                    p1, p2 = self._tournament_selection(population, fitness_scores), self._tournament_selection(population, fitness_scores)
                    c1, c2 = self._crossover(p1, p2, puzzle)
                    for c in [c1, c2]:
                        c = self._smart_mutate(c, puzzle, belief_space, self.mutation_rate, cage_cache)
                        c = self._repair_rows_and_cols(c)
                        new_population.append(c)

                stagnation_threshold = max(50, self.generations // 10)
                if stagnation > stagnation_threshold:
                    replace_count = max(2, self.population_size // 6)
                    for i in range(replace_count):
                        if (target_idx := elite_count + i) < len(new_population):
                            new_population[target_idx] = self._generate_latin_square(puzzle.size)
                    stagnation = max(0, stagnation - stagnation_threshold // 2)

                if puzzle.size >= 7 and (gen + 1) % 150 == 0 and best_fitness < 0.92:
                    replace_count = max(2, self.population_size // 12)
                    for i in range(replace_count):
                        if (target_idx := elite_count + i) < len(new_population):
                            new_population[target_idx] = self._generate_latin_square(puzzle.size)

                if stagnation > stagnation_threshold // 2 and best_fitness > 0.7:
                    if (improved := self._hill_climb(best_individual.copy(), cage_cache, puzzle)) is not None:
                        if (new_fit := self._calculate_fitness(improved, cage_cache)) > best_fitness:
                            best_fitness, best_individual, stagnation = new_fit, improved.copy(), 0

                population = new_population[:self.population_size]

            if best_fitness > best_overall_fit:
                best_overall_fit, best_overall = best_fitness, best_individual.copy()
            if best_overall_fit >= 1.0:
                break

        success = best_overall_fit >= 1.0
        if success and best_overall is not None:
            puzzle.grid = best_overall
        return success, self._get_performance_metrics(puzzle, success, best_overall_fit, start_time, best_overall)

    def _reset_histories(self):
        self.fitness_history, self.best_fitness_history = [], []

    def _tune_parameters(self, size: int):
        params = {1: (10, 50, 0.05), 2: (30, 200, 0.1), 3: (60, 500, 0.14), 4: (90, 900, 0.16),
                 5: (140, 1400, 0.2), 6: (200, 2200, 0.24), 7: (220, 2600, 0.32, 3, 12), 8: (260, 3200, 0.34, 3, 18)}
        size_key = 8 if size >= 8 else size
        p = params.get(size_key, params[8])
        self.population_size, self.generations, self.mutation_rate = p[0], p[1], p[2]
        if size >= 7:
            self.restarts, self.time_budget_seconds = p[3], p[4]

    def _initialize_population(self, puzzle: KenKenPuzzle) -> List[np.ndarray]:
        return [self._generate_latin_square(puzzle.size) for _ in range(self.population_size)]

    def _generate_latin_square(self, size: int) -> np.ndarray:
        base = list(range(1, size + 1))
        random.shuffle(base)
        arr = np.array([base[i:] + base[:i] for i in range(size)])
        np.random.shuffle(arr)
        arr = arr.T
        np.random.shuffle(arr)
        return arr.T

    def _calculate_fitness(self, individual: np.ndarray, cage_cache: List[Dict]) -> float:
        if not cage_cache:
            return 1.0
        cage_matches = sum(1 for cage in cage_cache if self._is_cage_satisfied(individual, cage))
        cage_score = cage_matches / len(cage_cache)
        if cage_score >= 1.0:
            return 1.0
        return 0.85 * cage_score + 0.15 * self._latin_consistency_score(individual)

    def _build_cage_cache(self, puzzle: KenKenPuzzle) -> List[Dict]:
        return [{"cells": tuple(cage["cells"]), "operation": cage["operation"], "target": cage["target"]} for cage in puzzle.cages]

    def _evaluate_cage_operation(self, values: List[int], op: str, target: int) -> bool:
        if op == "+": return sum(values) == target
        if op == "×": return np.prod(values) == target
        if op == "-": return len(values) == 2 and max(values) - min(values) == target
        if op == "÷":
            if len(values) != 2: return False
            big, small = max(values), min(values)
            return small != 0 and big % small == 0 and big // small == target
        return values[0] == target

    def _is_cage_satisfied(self, individual: np.ndarray, cage: Dict) -> bool:
        return self._evaluate_cage_operation([individual[r, c] for r, c in cage["cells"]], cage["operation"], cage["target"])

    def _latin_consistency_score(self, individual: np.ndarray) -> float:
        size = individual.shape[0]
        dups = sum(size - len(np.unique(row)) for row in individual) + sum(size - len(np.unique(col)) for col in individual.T)
        return max(0.0, 1.0 - dups / (2 * size * size))

    def _repair_line(self, individual: np.ndarray, is_row: bool):
        size = individual.shape[0]
        for idx in range(size):
            line = individual[idx, :] if is_row else individual[:, idx]
            counts = np.bincount(line, minlength=size + 1)
            missing = [v for v in range(1, size + 1) if counts[v] == 0]
            if not missing:
                continue
            dup_positions = [(idx, c) if is_row else (r, idx) for c, v in enumerate(line) if counts[v] > 1] if is_row else \
                           [(r, idx) for r, v in enumerate(line) if counts[v] > 1]
            random.shuffle(dup_positions)
            for pos in dup_positions:
                if not missing:
                    break
                individual[pos] = missing.pop()

    def _repair_rows_and_cols(self, individual: np.ndarray) -> np.ndarray:
        self._repair_line(individual, True)
        self._repair_line(individual, False)
        return individual

    def _hill_climb(self, individual: np.ndarray, cage_cache: List[Dict], puzzle: KenKenPuzzle) -> np.ndarray:
        best, best_fit, size = individual.copy(), self._calculate_fitness(individual, cage_cache), puzzle.size
        for _ in range(min(30, size * 4)):
            cand = best.copy()
            if random.random() < 0.5:
                r, c1, c2 = random.randint(0, size - 1), *random.sample(range(size), 2)
                cand[r, c1], cand[r, c2] = cand[r, c2], cand[r, c1]
            else:
                c, r1, r2 = random.randint(0, size - 1), *random.sample(range(size), 2)
                cand[r1, c], cand[r2, c] = cand[r2, c], cand[r1, c]
            cand = self._repair_rows_and_cols(cand)
            if (fit := self._calculate_fitness(cand, cage_cache)) > best_fit:
                best_fit, best = fit, cand
                if best_fit >= 1.0:
                    break
        return best if best_fit > self._calculate_fitness(individual, cage_cache) else None

    def _initialize_belief_space(self, puzzle: KenKenPuzzle) -> Dict:
        return {"best_individuals": [], "cell_preferences": np.zeros((puzzle.size, puzzle.size, puzzle.size))}

    def _update_belief_space(self, population, fitness_scores, belief_space):
        top_count = max(1, len(population) // 10)
        top_indices = np.argsort(fitness_scores)[-top_count:]
        belief_space["best_individuals"] = [population[i].copy() for i in top_indices]
        belief_space["cell_preferences"].fill(0)
        for idx in top_indices:
            for r in range(population[idx].shape[0]):
                for c in range(population[idx].shape[1]):
                    belief_space["cell_preferences"][r, c, population[idx][r, c] - 1] += 1
        return belief_space

    def _tournament_selection(self, population, fitness_scores, size=3):
        idxs = random.sample(range(len(population)), size)
        return population[max(idxs, key=lambda i: fitness_scores[i])].copy()

    def _crossover(self, p1, p2, puzzle):
        point = random.randint(1, puzzle.size - 1)
        return np.vstack([p1[:point], p2[point:]]), np.vstack([p2[:point], p1[point:]])

    def _smart_mutate(self, individual: np.ndarray, puzzle: KenKenPuzzle, belief_space: Dict, rate: float, cage_cache: List[Dict]) -> np.ndarray:
        if random.random() > rate:
            return individual
        size = puzzle.size
        invalid = [c for c in cage_cache if not self._is_cage_satisfied(individual, c)]
        use_col = size >= 6 and random.random() < 0.25
        if use_col:
            target_col = random.choice(random.choice(invalid)["cells"])[1] if invalid and random.random() < 0.6 else random.randint(0, size - 1)
            r1, r2 = random.sample(range(size), 2)
            if random.random() < 0.3 and np.sum(prefs := belief_space["cell_preferences"][:, target_col]) > 0:
                r2 = int(np.argmax(np.sum(prefs, axis=1)))
            individual[r1, target_col], individual[r2, target_col] = individual[r2, target_col], individual[r1, target_col]
        else:
            target_row = random.choice(random.choice(invalid)["cells"])[0] if invalid and random.random() < 0.7 else random.randint(0, size - 1)
            c1, c2 = random.sample(range(size), 2)
            if random.random() < 0.3 and np.sum(prefs := belief_space["cell_preferences"][target_row, c1]) > 0:
                if len(locs := np.where(individual[target_row] == int(np.argmax(prefs) + 1))[0]) > 0:
                    c2 = int(locs[0])
            individual[target_row, c1], individual[target_row, c2] = individual[target_row, c2], individual[target_row, c1]
        return individual

    def _get_performance_metrics(self, puzzle, success, best_fitness, start_time, best_candidate=None):
        return {"solved": success, "execution_time": time.time() - start_time, "best_fitness": best_fitness,
                "generations": len(self.fitness_history), "fitness_history": self.fitness_history,
                "best_fitness_history": self.best_fitness_history, "solution": puzzle.grid.copy() if success else None,
                "best_candidate": best_candidate.copy() if best_candidate is not None else None}


class HybridKenKenSolver:
    def __init__(self):
        self.ca_solver = CulturalAlgorithmSolver()
        self.bt_solver = BacktrackingSolver()

    def solve(self, puzzle: KenKenPuzzle) -> Dict:
        start_time = time.time()
        ca_puzzle = KenKenPuzzle(puzzle.size)
        ca_puzzle.cages = copy.deepcopy(puzzle.cages)
        ca_success, ca_metrics = self.ca_solver.solve(ca_puzzle)
        ca_solution, ca_fitness, ca_best = ca_metrics.get("solution"), ca_metrics.get("best_fitness", 0.0), ca_metrics.get("best_candidate")

        if ca_success and ca_solution is not None:
            puzzle.grid = ca_solution.copy()
            return {"solved": True, "execution_time": ca_metrics["execution_time"], "phase": "CA-only",
                    "best_fitness": ca_fitness, "ca_metrics": ca_metrics, "backtracking_metrics": None,
                    "solution": ca_solution.copy(), "hybrid": True}

        candidates = []
        if puzzle.size >= 7:
            candidates.append(self._prepare_smart_start(ca_puzzle, ca_solution, ca_fitness))
        if ca_solution is not None:
            candidates.append(ca_solution.copy())
        if ca_best is not None:
            candidates.append(ca_best.copy())
        candidates.append(np.zeros((puzzle.size, puzzle.size), dtype=int))

        bt_success, bt_metrics = False, {}
        for grid in candidates:
            bt_puzzle = KenKenPuzzle(puzzle.size)
            bt_puzzle.cages = copy.deepcopy(puzzle.cages)
            bt_puzzle.grid = grid
            success, metrics = self.bt_solver.solve(bt_puzzle)
            if success:
                bt_success, bt_metrics, puzzle.grid = True, metrics, metrics["solution"].copy()
                break
            if not bt_metrics or metrics.get("steps", 0) > bt_metrics.get("steps", 0):
                bt_metrics = metrics

        result = {"solved": bt_success, "execution_time": time.time() - start_time,
                 "phase": "CA+BT" if ca_solution is not None else "BT-only", "best_fitness": ca_fitness,
                 "ca_metrics": ca_metrics, "backtracking_metrics": bt_metrics, "solution": bt_metrics.get("solution"), "hybrid": True}
        if ca_metrics.get("generations"):
            result["generations"] = ca_metrics["generations"]
        if bt_metrics.get("steps"):
            result["steps"] = bt_metrics["steps"]
        return result

    def _prepare_smart_start(self, ca_puzzle: KenKenPuzzle, ca_solution: np.ndarray, ca_fitness: float) -> np.ndarray:
        if ca_solution is None or ca_fitness < 0.7:
            return np.zeros((ca_puzzle.size, ca_puzzle.size), dtype=int)
        start_grid = ca_solution.copy()
        keep_cells = set()
        for cage in ca_puzzle.cages:
            values = [ca_solution[r, c] for r, c in cage["cells"]]
            if self._evaluate_cage_operation(values, cage["operation"], cage["target"]):
                keep_cells.update(cage["cells"])

        for r in range(ca_puzzle.size):
            for c in range(ca_puzzle.size):
                if (r, c) not in keep_cells:
                    if any(start_grid[r, c] == start_grid[r, kc] for kr, kc in keep_cells if kr == r) or \
                       any(start_grid[r, c] == start_grid[kr, c] for kr, kc in keep_cells if kc == c):
                        start_grid[r, c] = 0
        return start_grid

    def _evaluate_cage_operation(self, values: List[int], op: str, target: int) -> bool:
        if op == "+": return sum(values) == target
        if op == "×": return np.prod(values) == target
        if op == "-": return len(values) == 2 and max(values) - min(values) == target
        if op == "÷":
            if len(values) != 2: return False
            big, small = max(values), min(values)
            return small != 0 and big % small == 0 and big // small == target
        return values[0] == target