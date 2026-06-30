"""Algorithms for Homberger 1000 VRPTW: BIH, Standard ACO, GWO, ABC, and BIH-ACO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import time

import numpy as np

from vrptw_core import (
    Route,
    Solution,
    VRPTWInstance,
    clone_solution,
    compare_solutions,
    evaluate_route,
    evaluate_solution,
)


# ----------------------------- BIH -----------------------------

def _ordered_customers(instance: VRPTWInstance, order_type: str) -> List[int]:
    customers = instance.customers
    depot_idx = instance.id_to_index[instance.depot.id]
    if order_type == "earliest_due":
        key = lambda c: (c.due_date, c.ready_time, c.id)
    elif order_type == "earliest_ready":
        key = lambda c: (c.ready_time, c.due_date, c.id)
    elif order_type == "largest_demand":
        key = lambda c: (-c.demand, c.due_date, c.id)
    elif order_type == "farthest":
        key = lambda c: (-instance.distance_matrix[depot_idx, instance.id_to_index[c.id]], c.due_date, c.id)
    else:
        raise ValueError(f"Unknown BIH order_type: {order_type}")
    return [c.id for c in sorted(customers, key=key)]


def best_insertion_position(instance: VRPTWInstance, routes: List[Route], customer_id: int) -> Tuple[Optional[int], Optional[int], float]:
    best_route_idx = None
    best_pos = None
    best_delta = float("inf")
    for r_idx, route in enumerate(routes):
        old_eval = evaluate_route(instance, route.customer_ids)
        seq = route.customer_ids
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [customer_id] + seq[pos:]
            new_eval = evaluate_route(instance, new_seq)
            if new_eval.feasible:
                delta = new_eval.distance - old_eval.distance
                if delta < best_delta:
                    best_delta = delta
                    best_route_idx = r_idx
                    best_pos = pos
    return best_route_idx, best_pos, best_delta


def solve_bih(instance: VRPTWInstance, order_type: str = "earliest_due") -> Solution:
    start = time.perf_counter()
    routes: List[Route] = []
    for cid in _ordered_customers(instance, order_type):
        r_idx, pos, _ = best_insertion_position(instance, routes, cid)
        if r_idx is None:
            new_route = evaluate_route(instance, [cid])
            routes.append(new_route)
        else:
            routes[r_idx].customer_ids.insert(pos, cid)
            routes[r_idx] = evaluate_route(instance, routes[r_idx].customer_ids)
    sol = Solution(routes=routes, algorithm=f"BIH_{order_type}")
    evaluate_solution(instance, sol)
    sol.algorithm = "BIH"
    sol.runtime_seconds = time.perf_counter() - start
    return sol


def solve_bih_multi_order(instance: VRPTWInstance) -> Solution:
    best = None
    total_start = time.perf_counter()
    for order in ["earliest_due", "earliest_ready", "largest_demand", "farthest"]:
        sol = solve_bih(instance, order)
        if compare_solutions(sol, best) < 0:
            best = sol
    best.algorithm = "BIH"
    best.runtime_seconds = time.perf_counter() - total_start
    return best


# --------------------------- Decoder ----------------------------

def split_sequence_to_routes(instance: VRPTWInstance, sequence: Sequence[int]) -> Solution:
    routes: List[Route] = []
    current: List[int] = []
    for cid in sequence:
        trial = current + [int(cid)]
        if evaluate_route(instance, trial).feasible:
            current = trial
        else:
            if current:
                routes.append(evaluate_route(instance, current))
            # Start a new route with cid. If infeasible alone, keep it and solution will be infeasible.
            current = [int(cid)]
    if current:
        routes.append(evaluate_route(instance, current))
    sol = Solution(routes=routes)
    return evaluate_solution(instance, sol)


def decode_random_keys(instance: VRPTWInstance, keys: np.ndarray) -> Solution:
    ids = np.array(instance.customer_ids)
    order = ids[np.argsort(keys)]
    return split_sequence_to_routes(instance, order.tolist())


# ------------------------- Local Search -------------------------

def _random_two_routes(sol: Solution, rng: np.random.Generator):
    nonempty = [i for i, r in enumerate(sol.routes) if r.customer_ids]
    if not nonempty:
        return None, None
    i = int(rng.choice(nonempty))
    j = int(rng.choice(nonempty))
    return i, j


def relocate_search(instance: VRPTWInstance, solution: Solution, max_trials: int, rng: np.random.Generator) -> Solution:
    best = clone_solution(solution)
    for _ in range(max_trials):
        i, j = _random_two_routes(best, rng)
        if i is None:
            break
        if not best.routes[i].customer_ids:
            continue
        a = int(rng.integers(0, len(best.routes[i].customer_ids)))
        cid = best.routes[i].customer_ids[a]
        new_routes = [Route(list(r.customer_ids)) for r in best.routes]
        new_routes[i].customer_ids.pop(a)
        pos = int(rng.integers(0, len(new_routes[j].customer_ids) + 1))
        new_routes[j].customer_ids.insert(pos, cid)
        new_routes = [r for r in new_routes if r.customer_ids]
        trial = evaluate_solution(instance, Solution(routes=new_routes, algorithm=best.algorithm, seed=best.seed))
        if trial.feasible and compare_solutions(trial, best) < 0:
            best = trial
    return best


def swap_search(instance: VRPTWInstance, solution: Solution, max_trials: int, rng: np.random.Generator) -> Solution:
    best = clone_solution(solution)
    for _ in range(max_trials):
        i, j = _random_two_routes(best, rng)
        if i is None or not best.routes[i].customer_ids or not best.routes[j].customer_ids:
            continue
        a = int(rng.integers(0, len(best.routes[i].customer_ids)))
        b = int(rng.integers(0, len(best.routes[j].customer_ids)))
        new_routes = [Route(list(r.customer_ids)) for r in best.routes]
        new_routes[i].customer_ids[a], new_routes[j].customer_ids[b] = new_routes[j].customer_ids[b], new_routes[i].customer_ids[a]
        trial = evaluate_solution(instance, Solution(routes=new_routes, algorithm=best.algorithm, seed=best.seed))
        if trial.feasible and compare_solutions(trial, best) < 0:
            best = trial
    return best


def two_opt_search(instance: VRPTWInstance, solution: Solution, max_trials: int, rng: np.random.Generator) -> Solution:
    best = clone_solution(solution)
    candidate_routes = [idx for idx, r in enumerate(best.routes) if len(r.customer_ids) >= 4]
    if not candidate_routes:
        return best
    for _ in range(max_trials):
        i = int(rng.choice(candidate_routes))
        seq = best.routes[i].customer_ids
        if len(seq) < 4:
            continue
        a, b = sorted(rng.choice(len(seq), size=2, replace=False).tolist())
        if b - a < 2:
            continue
        new_routes = [Route(list(r.customer_ids)) for r in best.routes]
        new_routes[i].customer_ids = seq[:a] + list(reversed(seq[a:b + 1])) + seq[b + 1:]
        trial = evaluate_solution(instance, Solution(routes=new_routes, algorithm=best.algorithm, seed=best.seed))
        if trial.feasible and compare_solutions(trial, best) < 0:
            best = trial
    return best


def improve_solution(instance: VRPTWInstance, solution: Solution, max_trials: int, rng: np.random.Generator) -> Solution:
    if max_trials <= 0:
        return solution
    per = max(1, max_trials // 3)
    sol = relocate_search(instance, solution, per, rng)
    sol = swap_search(instance, sol, per, rng)
    sol = two_opt_search(instance, sol, per, rng)
    sol.algorithm = solution.algorithm
    sol.seed = solution.seed
    return evaluate_solution(instance, sol)


# ----------------------------- ACO ------------------------------

@dataclass
class ACOConfig:
    number_of_ants: int = 10
    max_iterations: int = 20
    alpha: float = 1.0
    beta: float = 3.0
    evaporation_rate: float = 0.2
    pheromone_constant: float = 100.0
    runtime_limit: float = 60.0
    local_search_trials: int = 100
    random_seed: int = 2026
    mode: str = "standard_aco"


def _feasible_next_customers(instance: VRPTWInstance, current_route: List[int], unrouted: set) -> List[int]:
    feasible = []
    for cid in unrouted:
        if evaluate_route(instance, current_route + [cid]).feasible:
            feasible.append(cid)
    return feasible


def select_next_customer(instance: VRPTWInstance, current_id: int, feasible_customers: Sequence[int], pheromone: np.ndarray, config: ACOConfig, rng: np.random.Generator) -> int:
    eps = 1e-9
    i = instance.id_to_index[current_id]
    weights = []
    for cid in feasible_customers:
        j = instance.id_to_index[cid]
        tau = max(pheromone[i, j], eps)
        eta = 1.0 / (float(instance.distance_matrix[i, j]) + eps)
        weights.append((tau ** config.alpha) * (eta ** config.beta))
    weights = np.array(weights, dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return int(rng.choice(feasible_customers))
    probs = weights / weights.sum()
    return int(rng.choice(feasible_customers, p=probs))


def construct_ant_solution(instance: VRPTWInstance, pheromone: np.ndarray, config: ACOConfig, rng: np.random.Generator) -> Solution:
    unrouted = set(instance.customer_ids)
    routes: List[Route] = []
    while unrouted:
        current_route: List[int] = []
        current_id = instance.depot.id
        while unrouted:
            feasible = _feasible_next_customers(instance, current_route, unrouted)
            if not feasible:
                break
            next_id = select_next_customer(instance, current_id, feasible, pheromone, config, rng)
            current_route.append(next_id)
            unrouted.remove(next_id)
            current_id = next_id
        if not current_route:
            # Fallback: choose one unrouted customer to avoid infinite loop.
            cid = int(rng.choice(list(unrouted)))
            current_route = [cid]
            unrouted.remove(cid)
        routes.append(evaluate_route(instance, current_route))
    sol = evaluate_solution(instance, Solution(routes=routes, algorithm="Standard ACO" if config.mode == "standard_aco" else "BIH-ACO", seed=config.random_seed))
    return sol


def _edges_from_solution(instance: VRPTWInstance, sol: Solution):
    depot = instance.depot.id
    for r in sol.routes:
        prev = depot
        for cid in r.customer_ids:
            yield prev, cid
            prev = cid
        yield prev, depot


def _cost_for_pheromone(sol: Solution) -> float:
    return max(1.0, sol.vehicles * 1_000_000.0 + sol.total_distance + 0.001 * sol.total_travel_time)


def update_pheromone(pheromone: np.ndarray, iteration_best: Optional[Solution], global_best: Optional[Solution], instance: VRPTWInstance, config: ACOConfig) -> None:
    pheromone *= (1.0 - config.evaporation_rate)
    pheromone[:] = np.maximum(pheromone, 1e-6)
    for sol, multiplier in [(iteration_best, 1.0), (global_best, 2.0)]:
        if sol is None or not sol.feasible:
            continue
        delta = multiplier * config.pheromone_constant / _cost_for_pheromone(sol)
        for a, b in _edges_from_solution(instance, sol):
            i, j = instance.id_to_index[a], instance.id_to_index[b]
            pheromone[i, j] += delta
            pheromone[j, i] += delta


def _initialize_pheromone(instance: VRPTWInstance, initial_solution: Optional[Solution] = None) -> np.ndarray:
    n = len(instance.all_nodes)
    pheromone = np.ones((n, n), dtype=float)
    if initial_solution is not None:
        for a, b in _edges_from_solution(instance, initial_solution):
            i, j = instance.id_to_index[a], instance.id_to_index[b]
            pheromone[i, j] += 5.0
            pheromone[j, i] += 5.0
    return pheromone


def solve_standard_aco(instance: VRPTWInstance, config: ACOConfig) -> Solution:
    config.mode = "standard_aco"
    start = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    pheromone = _initialize_pheromone(instance)
    global_best = None
    history = []
    for it in range(1, config.max_iterations + 1):
        if time.perf_counter() - start >= config.runtime_limit:
            break
        iteration_best = None
        for _ in range(config.number_of_ants):
            if time.perf_counter() - start >= config.runtime_limit:
                break
            sol = construct_ant_solution(instance, pheromone, config, rng)
            sol = improve_solution(instance, sol, config.local_search_trials, rng)
            sol.algorithm = "Standard ACO"
            sol.seed = config.random_seed
            if compare_solutions(sol, iteration_best) < 0:
                iteration_best = sol
            if compare_solutions(sol, global_best) < 0:
                global_best = clone_solution(sol)
        update_pheromone(pheromone, iteration_best, global_best, instance, config)
        if global_best:
            history.append((it, global_best.vehicles, global_best.total_distance, global_best.total_travel_time))
    if global_best is None:
        global_best = construct_ant_solution(instance, pheromone, config, rng)
    global_best.algorithm = "Standard ACO"
    global_best.seed = config.random_seed
    global_best.runtime_seconds = time.perf_counter() - start
    global_best.convergence_history = history
    return evaluate_solution(instance, global_best)


def solve_bih_aco(instance: VRPTWInstance, config: ACOConfig) -> Solution:
    config.mode = "bih_aco"
    start = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    bih = solve_bih_multi_order(instance)
    bih.algorithm = "BIH-ACO"
    bih.seed = config.random_seed
    global_best = clone_solution(bih)
    pheromone = _initialize_pheromone(instance, bih)
    history = [(0, global_best.vehicles, global_best.total_distance, global_best.total_travel_time)]
    for it in range(1, config.max_iterations + 1):
        if time.perf_counter() - start >= config.runtime_limit:
            break
        iteration_best = None
        for _ in range(config.number_of_ants):
            if time.perf_counter() - start >= config.runtime_limit:
                break
            sol = construct_ant_solution(instance, pheromone, config, rng)
            sol = improve_solution(instance, sol, config.local_search_trials, rng)
            sol.algorithm = "BIH-ACO"
            sol.seed = config.random_seed
            if compare_solutions(sol, iteration_best) < 0:
                iteration_best = sol
            if compare_solutions(sol, global_best) < 0:
                global_best = clone_solution(sol)
        update_pheromone(pheromone, iteration_best, global_best, instance, config)
        history.append((it, global_best.vehicles, global_best.total_distance, global_best.total_travel_time))
    global_best.algorithm = "BIH-ACO"
    global_best.seed = config.random_seed
    global_best.runtime_seconds = time.perf_counter() - start
    global_best.convergence_history = history
    return evaluate_solution(instance, global_best)



# ----------------------------- ABC ------------------------------

@dataclass
class ABCConfig:
    colony_size: int = 20
    max_iterations: int = 20
    runtime_limit: float = 60.0
    local_search_trials: int = 100
    limit: int = 5
    random_seed: int = 2026


def _solution_key(sol: Solution):
    return (0 if sol.feasible else 1, sol.vehicles, sol.total_distance, sol.total_travel_time)


def _mutate_random_keys(keys: np.ndarray, rng: np.random.Generator, strength: float = 0.15) -> np.ndarray:
    """Create a neighboring food source for ABC using random-key perturbation."""
    new_keys = keys.copy()
    n = len(new_keys)
    if n == 0:
        return new_keys
    changes = max(1, int(0.05 * n))
    idx = rng.choice(n, size=changes, replace=False)
    noise = rng.normal(0.0, strength, size=changes)
    new_keys[idx] = np.clip(new_keys[idx] + noise, 0.0, 1.0)

    # Occasional swap helps explore order changes directly.
    if n >= 2 and rng.random() < 0.50:
        a, b = rng.choice(n, size=2, replace=False)
        new_keys[a], new_keys[b] = new_keys[b], new_keys[a]
    return new_keys


def _abc_fitness(sol: Solution) -> float:
    """Fitness is larger for better solutions; infeasible solutions are strongly penalized."""
    penalty = 0.0 if sol.feasible else 1_000_000_000.0
    cost = penalty + sol.vehicles * 1_000_000.0 + sol.total_distance + 0.001 * sol.total_travel_time
    return 1.0 / max(cost, 1e-9)


def solve_abc(instance: VRPTWInstance, config: ABCConfig) -> Solution:
    """Artificial Bee Colony using random-key encoding for customer ordering.

    Each food source is a random-key vector. Decoding sorts customers by key,
    then splits the sequence into feasible VRPTW routes. Employed/onlooker bees
    improve neighboring food sources; scout bees replace stagnant sources.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    n = len(instance.customer_ids)
    colony_size = max(2, int(config.colony_size))
    food_number = max(2, colony_size // 2)

    foods = rng.random((food_number, n))
    sols = []
    trials = np.zeros(food_number, dtype=int)
    for i in range(food_number):
        sol = decode_random_keys(instance, foods[i])
        sol.algorithm = "ABC"
        sol.seed = config.random_seed
        sol = improve_solution(instance, sol, config.local_search_trials, rng)
        sol.algorithm = "ABC"
        sol.seed = config.random_seed
        sols.append(sol)

    best = clone_solution(min(sols, key=_solution_key))
    history = [(0, best.vehicles, best.total_distance, best.total_travel_time)]

    for it in range(1, config.max_iterations + 1):
        if time.perf_counter() - start >= config.runtime_limit:
            break

        # Employed bee phase.
        for i in range(food_number):
            if time.perf_counter() - start >= config.runtime_limit:
                break
            candidate_keys = _mutate_random_keys(foods[i], rng)
            candidate = decode_random_keys(instance, candidate_keys)
            candidate.algorithm = "ABC"
            candidate.seed = config.random_seed
            candidate = improve_solution(instance, candidate, config.local_search_trials, rng)
            candidate.algorithm = "ABC"
            candidate.seed = config.random_seed
            if compare_solutions(candidate, sols[i]) < 0:
                foods[i] = candidate_keys
                sols[i] = candidate
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bee phase.
        fitness = np.array([_abc_fitness(s) for s in sols], dtype=float)
        if not np.isfinite(fitness).all() or fitness.sum() <= 0:
            probs = np.ones(food_number) / food_number
        else:
            probs = fitness / fitness.sum()
        for _ in range(food_number):
            if time.perf_counter() - start >= config.runtime_limit:
                break
            i = int(rng.choice(food_number, p=probs))
            candidate_keys = _mutate_random_keys(foods[i], rng, strength=0.10)
            candidate = decode_random_keys(instance, candidate_keys)
            candidate.algorithm = "ABC"
            candidate.seed = config.random_seed
            candidate = improve_solution(instance, candidate, config.local_search_trials, rng)
            candidate.algorithm = "ABC"
            candidate.seed = config.random_seed
            if compare_solutions(candidate, sols[i]) < 0:
                foods[i] = candidate_keys
                sols[i] = candidate
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bee phase.
        for i in range(food_number):
            if trials[i] >= config.limit:
                foods[i] = rng.random(n)
                sols[i] = decode_random_keys(instance, foods[i])
                sols[i].algorithm = "ABC"
                sols[i].seed = config.random_seed
                trials[i] = 0

        iteration_best = min(sols, key=_solution_key)
        if compare_solutions(iteration_best, best) < 0:
            best = clone_solution(iteration_best)
        history.append((it, best.vehicles, best.total_distance, best.total_travel_time))

    best.algorithm = "ABC"
    best.seed = config.random_seed
    best.runtime_seconds = time.perf_counter() - start
    best.convergence_history = history
    return evaluate_solution(instance, best)


# ----------------------------- GWO ------------------------------

@dataclass
class GWOConfig:
    population_size: int = 10
    max_iterations: int = 20
    runtime_limit: float = 60.0
    local_search_trials: int = 100
    random_seed: int = 2026


def update_wolf_position(wolf: np.ndarray, alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    def component(leader):
        r1 = rng.random(wolf.shape)
        r2 = rng.random(wolf.shape)
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leader - wolf)
        return leader - A * D
    new_pos = (component(alpha) + component(beta) + component(delta)) / 3.0
    return np.clip(new_pos, 0.0, 1.0)


def solve_gwo(instance: VRPTWInstance, config: GWOConfig) -> Solution:
    start = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    n = len(instance.customer_ids)
    pop = rng.random((config.population_size, n))
    sols = [decode_random_keys(instance, pop[i]) for i in range(config.population_size)]
    for s in sols:
        s.algorithm = "GWO"
        s.seed = config.random_seed
    best = min(sols, key=lambda s: (0 if s.feasible else 1, s.vehicles, s.total_distance, s.total_travel_time))
    best = clone_solution(best)
    history = []

    for it in range(1, config.max_iterations + 1):
        if time.perf_counter() - start >= config.runtime_limit:
            break
        order = sorted(range(config.population_size), key=lambda idx: (0 if sols[idx].feasible else 1, sols[idx].vehicles, sols[idx].total_distance, sols[idx].total_travel_time))
        alpha_idx, beta_idx, delta_idx = order[0], order[min(1, len(order)-1)], order[min(2, len(order)-1)]
        alpha, beta, delta = pop[alpha_idx].copy(), pop[beta_idx].copy(), pop[delta_idx].copy()
        a = 2.0 - 2.0 * (it / max(1, config.max_iterations))
        for i in range(config.population_size):
            if i == alpha_idx:
                continue
            pop[i] = update_wolf_position(pop[i], alpha, beta, delta, a, rng)
            sol = decode_random_keys(instance, pop[i])
            sol.algorithm = "GWO"
            sol.seed = config.random_seed
            sol = improve_solution(instance, sol, config.local_search_trials, rng)
            sol.algorithm = "GWO"
            sol.seed = config.random_seed
            sols[i] = sol
            if compare_solutions(sol, best) < 0:
                best = clone_solution(sol)
        history.append((it, best.vehicles, best.total_distance, best.total_travel_time))
    best.algorithm = "GWO"
    best.seed = config.random_seed
    best.runtime_seconds = time.perf_counter() - start
    best.convergence_history = history
    return evaluate_solution(instance, best)
